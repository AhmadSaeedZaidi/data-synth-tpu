import json
import struct

import numpy as np
import pytest
from glm53_offload.weights import BF16Reader, PackedExperts, RangeFile


def test_metadata_retries_are_bounded(monkeypatch):
    from io import BytesIO

    from glm53_offload import weights

    calls, waits = [], []

    def serve(url, timeout):
        calls.append(timeout)
        if len(calls) < 3:
            raise OSError("temporary upstream failure")
        return BytesIO(b'{"version":1}')

    monkeypatch.setattr(weights.urllib.request, "urlopen", serve)
    monkeypatch.setattr(weights.time, "sleep", waits.append)
    assert weights.fetch_json("https://example.invalid/config") == {"version": 1}
    assert calls == [30, 30, 30]
    assert waits == [1, 2]


def test_gather_deduplicates_routes_and_keeps_chip_order():
    raw = np.arange(2 * 7 * 3 * 4 * 4, dtype=np.uint8).reshape(2, 7, 3, 4, 4)
    store = PackedExperts({3: (raw, raw.copy(), raw.copy())}, {3: ("F32",) * 3}, 2)
    routes = np.array([[6, 2, 6], [1, 2, 1]], np.int32)
    arrays, remap, bucket = store.gather(3, routes)
    assert bucket == 4
    for chip in range(2):
        np.testing.assert_array_equal(arrays[0][chip][remap], raw[chip][routes])
    np.testing.assert_array_equal(arrays[0][:, 3], 0)
    assert store.stats["unique_experts"] == 3
    assert store.stats["bytes_transferred"] == sum(a.nbytes for a in arrays)
    with pytest.raises(ValueError, match="outside"):
        store.gather(3, np.array([[7]], np.int32))


def test_profile_loader_reads_only_requested_layer_and_expert_prefix(monkeypatch):
    from types import SimpleNamespace

    cfg = SimpleNamespace(mlp_types=("dense", "sparse", "sparse"), hidden=4, moe_inter=4, n_exp=5)
    arrays = np.arange(5 * 4 * 4, dtype=np.float32).reshape(5, 4, 4)
    reads = []

    class Source:
        def info(self, name):
            assert name.startswith("blk.2.")
            return {"dims": [4, 4, 5], "type": "F32", "nbytes": arrays.nbytes}

        def read(self, name, offset, length):
            reads.append((name, offset, length))
            return arrays.tobytes()[offset : offset + length]

    monkeypatch.setattr("glm53_offload.weights.available_ram", lambda: 10**9)
    store = PackedExperts.from_gguf(
        Source(), cfg, 2, reserve_bytes=0, log=lambda _: None, layers=[2], expert_limit=2
    )
    assert set(store.tables) == {2}
    assert len(reads) == 3
    assert all(offset == 0 and length == 2 * 4 * 4 * 4 for _, offset, length in reads)
    assert store.nbytes == 3 * 2 * 4 * 4 * 4
    gate, up, down = store.tables[2]
    for chip in range(2):
        for table in (gate, up):
            np.testing.assert_array_equal(
                table[chip].copy().view("<f4")[..., 0], arrays[:2, chip * 2 : (chip + 1) * 2]
            )
        np.testing.assert_array_equal(
            down[chip].copy().view("<f4")[..., 0], arrays[:2, :, chip * 2 : (chip + 1) * 2]
        )
    for settings in ({"layers": [0]}, {"layers": [3]}, {"expert_limit": 0}, {"expert_limit": 6}):
        with pytest.raises(ValueError):
            PackedExperts.from_gguf(Source(), cfg, 2, **settings)


class Response:
    def __init__(self, body, status, content_range):
        self.body, self.status = body, status
        self.headers = {"Content-Range": content_range}
        self.read_called = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read(self, amount):
        self.read_called = True
        return self.body[:amount]


def test_range_reads_are_chunked_and_validated(monkeypatch):
    calls = []

    def serve(request, timeout):
        low, high = map(int, request.get_header("Range")[6:].split("-"))
        calls.append((low, high))
        return Response(bytes(range(low, high + 1)), 206, f"bytes {low}-{high}/20")

    monkeypatch.setattr("urllib.request.urlopen", serve)
    file = RangeFile("https://example.invalid/weights", 20, chunk_bytes=4)
    assert file.pread(3, 9) == bytes(range(3, 12))
    assert sorted(calls) == [(3, 6), (7, 10), (11, 11)]
    assert file.bytes_read == 9
    with pytest.raises(ValueError, match="outside"):
        file.pread(19, 2)


@pytest.mark.parametrize("status,header", [(200, ""), (206, "bytes 0-3/999")])
def test_range_server_cannot_trigger_full_download(monkeypatch, status, header):
    response = Response(b"ignored", status, header)
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: response)
    with pytest.raises(ValueError, match="requested HTTP range"):
        RangeFile("https://example.invalid/weights", 20).pread(0, 4)
    assert not response.read_called


def test_bf16_reader_selects_tensor_ranges_and_preserves_router(monkeypatch):
    import ml_dtypes

    name = "model.language_model.layers.3.mlp.gate.weight"
    expected = np.array([[0.25, -0.5], [1, 2]], dtype=ml_dtypes.bfloat16)
    header = json.dumps({name: {"shape": [2, 2], "dtype": "BF16", "data_offsets": [0, 8]}}).encode()
    payload = struct.pack("<Q", len(header)) + header + expected.tobytes()
    file = RangeFile("https://example.invalid/shard", len(payload))
    calls = []

    def read(offset, length):
        calls.append((offset, length))
        return payload[offset : offset + length]

    monkeypatch.setattr(file, "_range", read)
    reader = BF16Reader.__new__(BF16Reader)
    reader.wm = {name: "part.safetensors"}
    reader.headers = {"part.safetensors": (json.loads(header), 8 + len(header))}
    reader.files = {"part.safetensors": file}
    reader.bytes_read = 0
    result = reader.get(name, ml_dtypes.bfloat16)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, expected.astype(np.float32))
    assert calls == [(8 + len(header), 8)]


def test_gguf_preload_splits_output_rows_and_input_blocks(monkeypatch):
    from types import SimpleNamespace

    cfg = SimpleNamespace(mlp_types=("sparse",), hidden=4, moe_inter=4, n_exp=3)
    arrays = {
        name: np.arange(3 * 4 * 4, dtype=np.float32).reshape(3, 4, 4) + offset
        for name, offset in (("gate", 0), ("up", 100), ("down", 200))
    }

    class Source:
        def info(self, name):
            return {"dims": [4, 4, 3], "type": "F32", "nbytes": 3 * 4 * 4 * 4}

        def read(self, name, offset, length):
            kind = name.split("ffn_")[1].split("_")[0]
            return arrays[kind].tobytes()[offset : offset + length]

    monkeypatch.setattr("glm53_offload.weights.available_ram", lambda: 10**9)
    store = PackedExperts.from_gguf(Source(), cfg, 2, reserve_bytes=0, log=lambda _: None)
    gate, up, down = store.tables[0]
    for chip in range(2):
        np.testing.assert_array_equal(
            gate[chip].copy().view("<f4")[..., 0], arrays["gate"][:, chip * 2 : (chip + 1) * 2]
        )
        np.testing.assert_array_equal(
            up[chip].copy().view("<f4")[..., 0], arrays["up"][:, chip * 2 : (chip + 1) * 2]
        )
        np.testing.assert_array_equal(
            down[chip].copy().view("<f4")[..., 0], arrays["down"][:, :, chip * 2 : (chip + 1) * 2]
        )
    monkeypatch.setattr("glm53_offload.weights.available_ram", lambda: 1)
    with pytest.raises(MemoryError, match="insufficient"):
        PackedExperts.from_gguf(Source(), cfg, 2, log=lambda _: None)
