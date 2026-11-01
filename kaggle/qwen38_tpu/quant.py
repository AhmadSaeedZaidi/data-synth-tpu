"""GGUF row decoding; packed parameters remain packed in HBM and host RAM."""

import numpy as np
from glm53_offload.gguf_quant import block_layout as common_layout
from glm53_offload.gguf_quant import dequantize_rows as common_decode

IQ4_VALUES = (-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113)
EXTRA = {"Q4_0": (32, 18), "Q4_1": (32, 20), "Q5_0": (32, 22), "Q5_1": (32, 24), "IQ4_NL": (32, 18)}


def block_layout(qtype):
    return EXTRA[qtype] if qtype in EXTRA else common_layout(qtype)


def dequantize_rows(raw, qtype, width, *, backend="numpy", dtype=None):
    if qtype not in EXTRA:
        return common_decode(raw, qtype, width, backend=backend, dtype=dtype)
    if backend == "numpy":
        xp, lax = np, None
    elif backend == "jax":
        import jax.numpy as xp
        from jax import lax
    else:
        raise ValueError("Unknown backend")
    raw = xp.asarray(raw)
    values, size = block_layout(qtype)
    blocks = (width + values - 1) // values
    if raw.dtype != xp.uint8 or raw.shape[-2:] != (blocks, size):
        raise ValueError("Invalid packed row layout")

    def fp16(x):
        bits = x[..., 0].astype(xp.uint16) | (x[..., 1].astype(xp.uint16) << 8)
        return (
            bits.view("<f2") if lax is None else lax.bitcast_convert_type(bits, xp.float16)
        ).astype(xp.float32)

    d = fp16(raw[..., :2])
    affine = qtype in ("Q4_1", "Q5_1")
    base = 4 if affine else 2
    high = 4 if qtype.startswith("Q5") else 0
    low = raw[..., base + high :]
    q = xp.concatenate((low & 15, low >> 4), axis=-1).astype(xp.int32)
    if high:
        field = raw[..., base : base + 4].astype(xp.uint32)
        bits = field[..., 0] | (field[..., 1] << 8) | (field[..., 2] << 16) | (field[..., 3] << 24)
        q = q | (((bits[..., None] >> xp.arange(32, dtype=xp.uint32)) & 1).astype(xp.int32) << 4)
    if qtype == "IQ4_NL":
        q = xp.asarray(IQ4_VALUES, dtype=xp.float32)[q]
    elif not affine:
        q = q - (16 if high else 8)
    out = d[..., None] * q
    if affine:
        out = out + fp16(raw[..., 2:4])[..., None]
    out = out.reshape(raw.shape[:-2] + (blocks * values,))[..., :width]
    return out.astype(xp.float32 if dtype is None else dtype)
