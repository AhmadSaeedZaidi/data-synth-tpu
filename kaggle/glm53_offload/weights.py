"""Bounded range reads and a chip-major, lossless packed expert store.

No weight download is saved to Kaggle's small working disk. Only routed
expert payloads are copied into RAM; the BF16 reader fetches individual
non-expert tensors from immutable safetensors shards.
"""

from __future__ import annotations

import json
import re
import struct
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from types import SimpleNamespace

import numpy as np

from . import progress, tracing
from .gguf_quant import block_layout
from .routing import expert_bucket

EXPERT_REPO = "unsloth/GLM-5.3-Flash-GGUF"
EXPERT_REVISION = "621d456e93e926e4b52f85cff5f634358c1828f9"
BASE_REPO = "zai-org/GLM-5.3-Flash-BF16"
BASE_REVISION = "a5b45eb41df6402735dedc900be14a42e8d5e538"
CHUNK_BYTES = 16 << 20


def hf_url(repo, revision, filename):
    return f"https://huggingface.co/{repo}/resolve/{revision}/" + urllib.parse.quote(
        filename, safe="/"
    )


def fetch_json(url):
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                return json.load(response)
        except (OSError, urllib.error.URLError, json.JSONDecodeError):
            if attempt == 3:
                raise
            progress.update(retried=1)
            time.sleep(2**attempt)


class RangeFile:
    """pread interface with verified HTTP byte ranges and bounded retries.

    A server ignoring Range is rejected before its response body is read.
    Query parameters distinguish ranges in intermediary caches.
    """

    def __init__(self, url, size, chunk_bytes=CHUNK_BYTES, attempts=4):
        self.url = url
        self.size = int(size)
        self.chunk_bytes = int(chunk_bytes)
        self.attempts = int(attempts)
        if self.size <= 0 or self.chunk_bytes <= 0 or self.attempts <= 0:
            raise ValueError("size, chunk_bytes and attempts must be positive")
        self.bytes_read = 0
        self._counter_lock = Lock()

    def _range(self, offset, length):
        end = offset + length - 1
        url = self.url + ("&" if "?" in self.url else "?") + f"range={offset}-{end}"
        request = urllib.request.Request(
            url,
            headers={"Range": f"bytes={offset}-{end}", "Accept-Encoding": "identity"},
        )
        for attempt in range(self.attempts):
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    expected = f"bytes {offset}-{end}/{self.size}"
                    if response.status != 206 or response.headers.get("Content-Range") != expected:
                        raise ValueError("server did not return the requested HTTP range")
                    if response.headers.get("Content-Encoding", "identity") != "identity":
                        raise ValueError("compressed range response is unsupported")
                    data = response.read(length + 1)
                    if len(data) != length:
                        raise OSError(f"short range read: {len(data)} != {length}")
                    with self._counter_lock:
                        self.bytes_read += length
                    progress.update(downloaded=length)
                    return data
            except (OSError, urllib.error.URLError):
                if attempt + 1 == self.attempts:
                    raise
                progress.update(retried=1)
                time.sleep(min(2**attempt, 8))
        raise AssertionError("unreachable")

    def readinto(self, offset, target):
        view = memoryview(target).cast("B")
        if offset < 0 or offset + len(view) > self.size:
            raise ValueError("read outside file")
        # Four in-flight ranges amortize redirects/TLS latency. Submit only a
        # single bounded wave; a slow request cannot accumulate completed
        # payloads for the rest of a multi-gigabyte tensor.
        with ThreadPoolExecutor(max_workers=4) as pool:
            for wave in range(0, len(view), self.chunk_bytes * 4):
                jobs = []
                for start in range(
                    wave, min(len(view), wave + self.chunk_bytes * 4), self.chunk_bytes
                ):
                    length = min(self.chunk_bytes, len(view) - start)
                    jobs.append((start, length, pool.submit(self._range, offset + start, length)))
                for start, length, future in jobs:
                    view[start : start + length] = future.result()

    def pread(self, offset, length):
        data = bytearray(length)
        self.readinto(offset, data)
        return data

    def close(self):
        pass


def remote_gguf(quant, log=print):
    from glm53.gguf_reader import GGUFModel, GGUFShard

    if quant not in {"Q6_K_XL", "Q4_K_XL"}:
        raise ValueError("quant must be Q6_K_XL or Q4_K_XL")
    meta = fetch_json(
        f"https://huggingface.co/api/models/{EXPERT_REPO}/revision/{EXPERT_REVISION}?blobs=true"
    )
    files = {}
    for item in meta["siblings"]:
        name = item["rfilename"]
        if name.startswith(f"UD-{quant}/") and name.endswith(".gguf"):
            size = item.get("size") or item.get("lfs", {}).get("size")
            if not size:
                raise ValueError(f"missing size for {name}")
            files[name] = RangeFile(hf_url(EXPERT_REPO, EXPERT_REVISION, name), size)
    if not files:
        raise ValueError("no matching GGUF shards at pinned revision")
    log(f"Reading {len(files)} GGUF headers; weights stay off working disk")
    # The upstream parser requests 64 MiB even for small headers. Grow the
    # metadata window only when necessary, keeping remote header reads small.
    model = GGUFModel.__new__(GGUFModel)
    model.shards, model.where, model.kv = {}, {}, {}
    for name, file in sorted(files.items()):
        for window in (8, 16, 32, 64):
            header_view = SimpleNamespace(size=min(file.size, window << 20), pread=file.pread)
            try:
                shard = GGUFShard(header_view)
                shard.pf = file
                break
            except (struct.error, UnicodeDecodeError, IndexError):
                if window == 64:
                    raise
        model.shards[name] = shard
        model.kv.update(shard.kv)
        for tensor in shard.tensors:
            if tensor in model.where:
                raise ValueError(f"duplicate GGUF tensor: {tensor}")
            model.where[tensor] = name
    return model


def open_gguf(quant, dirs=(), log=print):
    if not dirs:
        return remote_gguf(quant, log)
    from glm53.gguf_reader import GGUFModel, open_mirror

    files = open_mirror(dirs)
    files = {n: f for n, f in files.items() if f"UD-{quant}-" in n}
    if not files:
        raise ValueError("mounted directories contain no files for the selected quant")
    return GGUFModel(files)


def available_ram():
    """Available physical memory, bounded by the process cgroup when present."""
    data = Path("/proc/meminfo").read_text()
    available = int(re.search(r"MemAvailable:\s+(\d+)", data)[1]) * 1024
    for root, limit_name, used_name in (
        ("/sys/fs/cgroup", "memory.max", "memory.current"),
        ("/sys/fs/cgroup/memory", "memory.limit_in_bytes", "memory.usage_in_bytes"),
    ):
        try:
            limit = (Path(root) / limit_name).read_text().strip()
            used = int((Path(root) / used_name).read_text())
            if limit != "max":
                available = min(available, max(0, int(limit) - used))
        except (OSError, ValueError):
            pass
    return available


class PackedExperts:
    """Host tables [chip, expert, output row, input block, block byte].

    Gate/up shard output rows; down shards input blocks. No block crosses a
    chip boundary. Decode staging deduplicates selected experts over a batch.
    """

    def __init__(self, tables, types, n_devices, tile=4):
        self.tables = tables
        self.types = types
        self.n_devices = n_devices
        self.tile = tile
        self.stats = {
            "gathers": 0,
            "bytes_transferred": 0,
            "unique_experts": 0,
            "useful_bytes_transferred": 0,
            "padding_bytes_transferred": 0,
            "host_gather_seconds": 0.0,
        }
        self.validate()

    def validate(self):
        if self.n_devices < 1 or self.tile < 1:
            raise ValueError("device count and tile must be positive")
        for layer, arrays in self.tables.items():
            if len(arrays) != 3 or len(self.types[layer]) != 3:
                raise ValueError("gate, up and down are required")
            counts = set()
            for array, qtype in zip(arrays, self.types[layer], strict=True):
                _, size = block_layout(qtype)
                if array.dtype != np.uint8 or array.ndim != 5:
                    raise ValueError("expert arrays must be five-dimensional uint8")
                if array.shape[0] != self.n_devices or array.shape[-1] != size:
                    raise ValueError("invalid chip count or block size")
                counts.add(array.shape[1])
            if len(counts) != 1:
                raise ValueError("expert counts differ")

    @property
    def nbytes(self):
        return sum(a.nbytes for arrays in self.tables.values() for a in arrays)

    def qtypes(self, layer):
        return self.types[layer]

    def gather(self, layer, idx):
        started = time.perf_counter()
        idx = np.asarray(idx)
        count = self.tables[layer][0].shape[1]
        if idx.size == 0 or not np.issubdtype(idx.dtype, np.integer):
            raise ValueError("integer expert IDs are required")
        if idx.min() < 0 or idx.max() >= count:
            raise ValueError("expert ID outside table")
        selected, inverse = np.unique(idx, return_inverse=True)
        bucket = expert_bucket(len(selected), count, self.tile)
        staged = []
        for table in self.tables[layer]:
            shape = (self.n_devices, bucket, *table.shape[2:])
            output = np.empty(shape, dtype=np.uint8)
            for chip in range(self.n_devices):
                # IDs were checked above. NumPy's default mode='raise' buffers
                # the entire output even with out=; clip avoids that extra copy.
                np.take(
                    table[chip], selected, axis=0, out=output[chip, : len(selected)], mode="clip"
                )
            # Only padding needs zeroing; useful rows were completely written.
            output[:, len(selected) :] = 0
            staged.append(output)
        self.stats["gathers"] += 1
        transferred = sum(a.nbytes for a in staged)
        useful = transferred // bucket * len(selected)
        self.stats["bytes_transferred"] += transferred
        self.stats["useful_bytes_transferred"] += useful
        self.stats["padding_bytes_transferred"] += transferred - useful
        self.stats["unique_experts"] += len(selected)
        self.stats["host_gather_seconds"] += time.perf_counter() - started
        return tuple(staged), inverse.reshape(idx.shape).astype(np.int32), bucket

    @classmethod
    @tracing.operation("load_experts")
    def from_gguf(
        cls,
        source,
        cfg,
        n_devices,
        tile=4,
        reserve_bytes=32 << 30,
        log=print,
        *,
        layers=None,
        expert_limit=None,
    ):
        limit = cfg.n_exp if expert_limit is None else expert_limit
        if not isinstance(limit, int) or not 1 <= limit <= cfg.n_exp:
            raise ValueError("expert_limit must be in [1, model expert count]")
        selected_layers = set(range(len(cfg.mlp_types))) if layers is None else set(layers)
        if layers is not None and (
            not selected_layers
            or any(
                not isinstance(i, int)
                or not 0 <= i < len(cfg.mlp_types)
                or cfg.mlp_types[i] != "sparse"
                for i in selected_layers
            )
        ):
            raise ValueError("layers must select existing sparse layers")
        descriptions = []
        required = 0
        for layer, mlp_type in enumerate(cfg.mlp_types):
            if mlp_type != "sparse" or layer not in selected_layers:
                continue
            for kind in ("gate", "up", "down"):
                name = f"blk.{layer}.ffn_{kind}_exps.weight"
                info = source.info(name)
                width, rows = (
                    (cfg.moe_inter, cfg.hidden) if kind == "down" else (cfg.hidden, cfg.moe_inter)
                )
                if info["dims"] != [width, rows, cfg.n_exp]:
                    raise ValueError(f"unexpected GGUF dimensions for {name}: {info['dims']}")
                block, size = block_layout(info["type"])
                if width % block or (kind == "down" and width % (block * n_devices)):
                    raise ValueError(f"tensor-parallel split crosses a quant block: {name}")
                if kind != "down" and rows % n_devices:
                    raise ValueError("gate/up rows are not divisible by device count")
                descriptions.append((layer, kind, name, width, rows, block, size, info["type"]))
                required += rows * (width // block) * size * limit
        available = available_ram()
        log(
            f"Packed routed experts: {required / 2**30:.2f} GiB; "
            f"available RAM {available / 2**30:.2f} GiB"
        )
        if required + reserve_bytes > available:
            raise MemoryError(
                "insufficient host RAM for packed experts plus staging reserve; try Q4_K_XL"
            )
        tables, types = {}, {}
        loaded = 0
        progress.update(expert_bytes_total=required, expert_bytes_loaded=0, force=True)
        for layer, kind, name, width, rows, block, size, qtype in descriptions:
            progress.update(detail=f"expert layer {layer:02d} {kind}", force=True)
            local_rows = rows if kind == "down" else rows // n_devices
            local_blocks = width // block // (n_devices if kind == "down" else 1)
            table = np.empty((n_devices, limit, local_rows, local_blocks, size), np.uint8)
            per_expert = rows * (width // block) * size
            # At most 64 MiB source scratch; individual expert matrices are small.
            step = max(1, (64 << 20) // per_expert)
            for start in range(0, limit, step):
                amount = min(step, limit - start)
                raw = np.frombuffer(
                    source.read(name, start * per_expert, amount * per_expert), np.uint8
                )
                if kind == "down":
                    raw = raw.reshape(amount, rows, n_devices, local_blocks, size).transpose(
                        2, 0, 1, 3, 4
                    )
                else:
                    raw = raw.reshape(amount, n_devices, local_rows, local_blocks, size).transpose(
                        1, 0, 2, 3, 4
                    )
                table[:, start : start + amount] = raw
                loaded += amount * per_expert
                progress.update(expert_bytes_loaded=loaded)
            tables.setdefault(layer, []).append(table)
            types.setdefault(layer, []).append(qtype)
            log(f"Loaded layer {layer:02d} {kind}: {qtype}, {table.nbytes / 2**30:.2f} GiB")
        progress.update(expert_bytes_loaded=loaded, force=True)
        return cls(tables, types, n_devices, tile)


class BF16Reader:
    """Minimal safetensors reader using ranges and no torch/full-shard cache."""

    def __init__(self, repo=BASE_REPO, revision=BASE_REVISION):
        self.repo, self.revision = repo, revision
        self.wm = fetch_json(hf_url(repo, revision, "model.safetensors.index.json"))["weight_map"]
        meta = fetch_json(
            f"https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true"
        )
        self.sizes = {
            x["rfilename"]: x.get("size") or x.get("lfs", {}).get("size") for x in meta["siblings"]
        }
        self.headers = {}
        self.files = {}
        self.bytes_read = 0

    def has(self, name):
        return name in self.wm

    def _header(self, filename):
        if filename not in self.headers:
            file = RangeFile(hf_url(self.repo, self.revision, filename), self.sizes[filename])
            length = struct.unpack("<Q", file.pread(0, 8))[0]
            if length > 32 << 20:
                raise ValueError("unexpectedly large safetensors header")
            self.headers[filename] = (json.loads(file.pread(8, length)), 8 + length)
            self.files[filename] = file
        return self.headers[filename]

    def get(self, name, dtype=np.float32):
        filename = self.wm[name]
        header, data_start = self._header(filename)
        info = header[name]
        start, end = info["data_offsets"]
        shape, kind = tuple(info["shape"]), info["dtype"]
        if kind not in {"BF16", "F16", "F32"}:
            raise ValueError(f"BF16 source contains unexpected dtype {kind} for {name}")
        expected = int(np.prod(shape)) * (4 if kind == "F32" else 2)
        if end - start != expected or data_start + end > self.files[filename].size:
            raise ValueError("invalid safetensors payload bounds")
        # Preserve sensitive router/mHC parameters as float32, including a BF16
        # source's exact represented values. Large linear matrices use BF16.
        if ".mlp.gate." in name or "hc_" in name or (kind == "F32" and len(shape) <= 1):
            dtype = np.float32
        raw = np.empty(expected, np.uint8)
        self.files[filename].readinto(data_start + start, raw)
        if kind == "BF16":
            import ml_dtypes

            array = raw.view(ml_dtypes.bfloat16)
        else:
            array = raw.view("<f4" if kind == "F32" else "<f2")
        self.bytes_read += expected
        return array.reshape(shape).astype(dtype, copy=False)
