"""Cross-check the packed row decoders against gguf-py's reference codecs."""

from __future__ import annotations

import numpy as np
import pytest
from glm53_offload.gguf_quant import block_layout, dequantize_rows

gguf = pytest.importorskip("gguf")
gguf_quants = pytest.importorskip("gguf.quants")


_QTYPE_NAMES = ("Q4_K", "Q5_K", "Q6_K", "Q8_0", "F16", "BF16", "F32")


def _raw_blocks(qtype: str, seed: int = 42) -> tuple[np.ndarray, int, int]:
    rng = np.random.default_rng(seed)
    values_per_block, bytes_per_block = block_layout(qtype)
    nblocks = 2
    leading_shape = (2, 3)

    if qtype == "F16":
        values = rng.uniform(-4, 4, leading_shape + (nblocks,)).astype("<f2")
        raw = values.view(np.uint8).reshape(leading_shape + (nblocks, bytes_per_block))
    elif qtype == "F32":
        values = rng.uniform(-4, 4, leading_shape + (nblocks,)).astype("<f4")
        raw = values.view(np.uint8).reshape(leading_shape + (nblocks, bytes_per_block))
    elif qtype == "BF16":
        values = rng.uniform(-4, 4, leading_shape + (nblocks,)).astype("<f4")
        bits = values.view(np.uint32)
        hi = (bits >> np.uint32(16)).astype(np.uint16)
        raw = np.stack(
            ((hi & np.uint16(0xFF)).astype(np.uint8), (hi >> 8).astype(np.uint8)),
            axis=-1,
        )
    else:
        raw = rng.integers(0, 256, leading_shape + (nblocks, bytes_per_block), dtype=np.uint8)
        # Set the half-precision block scales to finite, nonzero values. The
        # remaining packed fields are random, including signed sub-scales.
        scale_offsets = {
            "Q4_K": (0, 2),
            "Q5_K": (0, 2),
            "Q6_K": (208,),
            "Q8_0": (0,),
        }[qtype]
        for offset in scale_offsets:
            scales = rng.uniform(0.02, 0.3, leading_shape + (nblocks,)).astype("<f2")
            raw[..., offset : offset + 2] = scales.view(np.uint8).reshape(
                leading_shape + (nblocks, 2)
            )

    return raw, nblocks, values_per_block


def _reference(raw: np.ndarray, qtype: str, nblocks: int, values_per_block: int) -> np.ndarray:
    _, bytes_per_block = block_layout(qtype)
    flat_blocks = raw.reshape(-1, bytes_per_block)
    decoded = gguf_quants.dequantize(flat_blocks, gguf.GGMLQuantizationType[qtype])
    leading_shape = raw.shape[:-2]
    return np.asarray(decoded).reshape(leading_shape + (nblocks * values_per_block,))


@pytest.mark.parametrize("qtype", _QTYPE_NAMES)
def test_numpy_rows_match_gguf_py(qtype: str) -> None:
    raw, nblocks, values_per_block = _raw_blocks(qtype)
    width = nblocks * values_per_block
    want = _reference(raw, qtype, nblocks, values_per_block)

    got = dequantize_rows(raw, qtype, width)

    assert got.shape == raw.shape[:-2] + (width,)
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-7, equal_nan=True)


@pytest.mark.parametrize("qtype", _QTYPE_NAMES)
def test_jax_jit_rows_match_gguf_py(qtype: str) -> None:
    jax = pytest.importorskip("jax")
    raw, nblocks, values_per_block = _raw_blocks(qtype, seed=43)
    width = nblocks * values_per_block
    want = _reference(raw, qtype, nblocks, values_per_block)

    decode = jax.jit(lambda packed: dequantize_rows(packed, qtype, width, backend="jax"))
    got = np.asarray(decode(raw))

    assert got.shape == raw.shape[:-2] + (width,)
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-7, equal_nan=True)


def test_partial_final_block_is_cropped_to_row_width() -> None:
    raw, _nblocks, _values_per_block = _raw_blocks("Q4_K", seed=44)
    width = 256 + 17

    got = dequantize_rows(raw, "Q4_K", width)

    assert got.shape == raw.shape[:-2] + (width,)


@pytest.mark.parametrize(
    ("qtype", "expected"),
    [
        ("Q4_K", (256, 144)),
        ("Q5_K", (256, 176)),
        ("Q6_K", (256, 210)),
        ("Q8_0", (32, 34)),
        ("F16", (1, 2)),
        ("BF16", (1, 2)),
        ("F32", (1, 4)),
    ],
)
def test_block_layout(qtype: str, expected: tuple[int, int]) -> None:
    assert block_layout(qtype) == expected


def test_rejects_wrong_block_count_and_unknown_type() -> None:
    raw, _nblocks, _values_per_block = _raw_blocks("Q8_0")
    with pytest.raises(ValueError, match="requires 2 Q8_0 blocks"):
        dequantize_rows(raw[..., :1, :], "Q8_0", 64)
    with pytest.raises(ValueError, match="unsupported GGUF"):
        block_layout("Q4_0")
