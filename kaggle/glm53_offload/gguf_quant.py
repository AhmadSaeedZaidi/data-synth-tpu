"""Small CPU and JAX decoders for the GGUF weight formats used by GLM.

The input to :func:`dequantize_rows` is the raw, little-endian GGUF payload
viewed as ``uint8`` with shape ``(..., blocks_per_row, bytes_per_block)``.
The block axis is the last matrix axis; callers can transpose the decoded
matrix afterwards to match their compute layout. This module only decodes
stored values. It never quantizes or re-quantizes weights.

For JIT use, keep ``qtype``, ``row_width``, ``dtype`` and ``backend`` static,
for example ``jax.jit(lambda x: dequantize_rows(x, "Q4_K", 256,
backend="jax"))``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# (values per block, bytes per block), matching GGMLQuantizationType sizes.
_BLOCK_LAYOUTS: dict[str, tuple[int, int]] = {
    "Q4_K": (256, 144),
    "Q5_K": (256, 176),
    "Q6_K": (256, 210),
    "Q8_0": (32, 34),
    "F16": (1, 2),
    "BF16": (1, 2),
    "F32": (1, 4),
}

_GGML_TYPE_NAMES = {
    0: "F32",
    1: "F16",
    8: "Q8_0",
    12: "Q4_K",
    13: "Q5_K",
    14: "Q6_K",
    30: "BF16",
}


def _normalize_qtype(qtype: Any) -> str:
    if hasattr(qtype, "name"):
        name = str(qtype.name).upper()
    elif isinstance(qtype, str):
        name = qtype.rsplit(".", 1)[-1].upper()
    else:
        try:
            name = _GGML_TYPE_NAMES[int(qtype)]
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError(f"unsupported GGUF quantization type: {qtype!r}") from exc
    if name not in _BLOCK_LAYOUTS:
        raise ValueError(f"unsupported GGUF quantization type: {qtype!r}")
    return name


def block_layout(qtype: Any) -> tuple[int, int]:
    """Return ``(values_per_block, bytes_per_block)`` for a supported type."""

    return _BLOCK_LAYOUTS[_normalize_qtype(qtype)]


def _decode(raw: Any, qtype: str, row_width: int, xp: Any, lax: Any | None) -> Any:
    """Backend-generic decoder. Shapes are statically known under JAX jit."""

    values_per_block, bytes_per_block = _BLOCK_LAYOUTS[qtype]
    if raw.ndim < 2:
        raise ValueError("raw must have shape (..., blocks_per_row, bytes_per_block)")
    if raw.shape[-1] != bytes_per_block:
        raise ValueError(f"{qtype} blocks contain {bytes_per_block} bytes; got {raw.shape[-1]}")
    if row_width <= 0:
        raise ValueError("row_width must be a positive integer")
    nblocks = (row_width + values_per_block - 1) // values_per_block
    if raw.shape[-2] != nblocks:
        raise ValueError(
            f"row_width {row_width} requires {nblocks} {qtype} blocks; got {raw.shape[-2]}"
        )

    leading_shape = raw.shape[:-2]

    def f16(field: Any) -> Any:
        bits = field[..., 0].astype(xp.uint16) | (field[..., 1].astype(xp.uint16) << xp.uint16(8))
        if lax is None:
            return bits.view("<f2").astype(xp.float32)
        return lax.bitcast_convert_type(bits, xp.float16).astype(xp.float32)

    def signed_i8(field: Any) -> Any:
        values = field.astype(xp.int16)
        return xp.where(values >= 128, values - 256, values).astype(xp.int8)

    if qtype == "F16":
        decoded = f16(raw).reshape(leading_shape + (nblocks,))
    elif qtype == "F32":
        bits = (
            raw[..., 0].astype(xp.uint32)
            | (raw[..., 1].astype(xp.uint32) << xp.uint32(8))
            | (raw[..., 2].astype(xp.uint32) << xp.uint32(16))
            | (raw[..., 3].astype(xp.uint32) << xp.uint32(24))
        )
        if lax is None:
            decoded = bits.view("<f4").reshape(leading_shape + (nblocks,))
        else:
            decoded = lax.bitcast_convert_type(bits, xp.float32).reshape(leading_shape + (nblocks,))
    elif qtype == "BF16":
        hi = raw[..., 0].astype(xp.uint32) | (raw[..., 1].astype(xp.uint32) << xp.uint32(8))
        bits = hi << xp.uint32(16)
        if lax is None:
            decoded = bits.view("<f4").reshape(leading_shape + (nblocks,))
        else:
            decoded = lax.bitcast_convert_type(bits, xp.float32).reshape(leading_shape + (nblocks,))
    elif qtype == "Q8_0":
        d = f16(raw[..., :2])
        qs = signed_i8(raw[..., 2:]).astype(xp.float32)
        decoded = (d[..., None] * qs).reshape(leading_shape + (nblocks * 32,))
    elif qtype in ("Q4_K", "Q5_K"):
        d = f16(raw[..., :2])
        dmin = f16(raw[..., 2:4])
        scales_raw = raw[..., 4:16].reshape(leading_shape + (nblocks, 3, 4))
        scale_bytes = scales_raw[..., 0, :]
        min_bytes = scales_raw[..., 1, :]
        mixed_bytes = scales_raw[..., 2, :]
        scales = xp.concatenate(
            (
                scale_bytes & xp.uint8(0x3F),
                (mixed_bytes & xp.uint8(0x0F)) | ((scale_bytes >> xp.uint8(2)) & xp.uint8(0x30)),
            ),
            axis=-1,
        )
        mins = xp.concatenate(
            (
                min_bytes & xp.uint8(0x3F),
                (mixed_bytes >> xp.uint8(4)) | ((min_bytes >> xp.uint8(2)) & xp.uint8(0x30)),
            ),
            axis=-1,
        )

        if qtype == "Q4_K":
            packed = raw[..., 16:].reshape(leading_shape + (nblocks, 4, 32))
            q = xp.stack((packed & xp.uint8(0x0F), packed >> xp.uint8(4)), axis=-2).reshape(
                leading_shape + (nblocks, 8, 32)
            )
        else:
            ql = raw[..., 48:176].reshape(leading_shape + (nblocks, 4, 32))
            ql = xp.stack((ql & xp.uint8(0x0F), ql >> xp.uint8(4)), axis=-2)
            ql = ql.reshape(leading_shape + (nblocks, 8, 32))
            qh = raw[..., 16:48]
            shifts = xp.arange(8, dtype=xp.uint8).reshape((8, 1))
            qh = ((qh[..., None, :] >> shifts) & xp.uint8(1)).reshape(
                leading_shape + (nblocks, 8, 32)
            )
            q = ql | (qh << xp.uint8(4))

        q = q.astype(xp.float32)
        decoded = (
            d[..., None, None] * scales[..., None].astype(xp.float32) * q
            - dmin[..., None, None] * mins[..., None].astype(xp.float32)
        ).reshape(leading_shape + (nblocks * 256,))
    elif qtype == "Q6_K":
        ql = raw[..., :128].reshape(leading_shape + (nblocks, 2, 64))
        ql = xp.stack((ql & xp.uint8(0x0F), ql >> xp.uint8(4)), axis=-2)
        ql = ql.reshape(leading_shape + (nblocks, 8, 32))
        qh = raw[..., 128:192].reshape(leading_shape + (nblocks, 2, 32))
        shifts = xp.arange(4, dtype=xp.uint8).reshape((4, 1)) * xp.uint8(2)
        qh = ((qh[..., None, :] >> shifts) & xp.uint8(0x03)).reshape(
            leading_shape + (nblocks, 8, 32)
        )
        q = (ql | (qh << xp.uint8(4))).astype(xp.int16) - 32
        q = q.reshape(leading_shape + (nblocks, 16, 16)).astype(xp.float32)
        scales = signed_i8(raw[..., 192:208]).astype(xp.float32)
        d = f16(raw[..., 208:210])
        decoded = (d[..., None, None] * scales[..., None] * q).reshape(
            leading_shape + (nblocks * 256,)
        )
    else:  # pragma: no cover - guarded by block_layout / _normalize_qtype
        raise AssertionError(f"unhandled quantization type {qtype}")

    return decoded[..., :row_width]


def dequantize_rows(
    raw: Any,
    qtype: Any,
    row_width: int,
    dtype: Any | None = None,
    backend: str = "numpy",
) -> Any:
    """Decode packed rows into ``(..., row_width)`` float values.

    ``raw`` must be a uint8 array whose final axes are blocks and bytes within
    each block. Quantized rows may end in a partial logical block; in that
    case the decoded padding is cropped to ``row_width``. ``backend`` selects
    NumPy or JAX. For JAX, all shape-defining arguments should be static when
    this function is called from ``jax.jit``.
    """

    normalized = _normalize_qtype(qtype)
    if not isinstance(row_width, (int, np.integer)):
        raise TypeError("row_width must be an integer")
    if backend == "numpy":
        array = np.asarray(raw)
        if array.dtype != np.uint8:
            raise TypeError(f"raw must have dtype uint8; got {array.dtype}")
        decoded = _decode(array, normalized, int(row_width), np, None)
        return decoded.astype(np.float32 if dtype is None else dtype, copy=False)
    if backend == "jax":
        try:
            import jax.numpy as jnp
            from jax import lax
        except ImportError as exc:  # pragma: no cover - depends on optional JAX
            raise ImportError("backend='jax' requires JAX to be installed") from exc
        array = jnp.asarray(raw)
        if array.dtype != jnp.uint8:
            raise TypeError(f"raw must have dtype uint8; got {array.dtype}")
        decoded = _decode(array, normalized, int(row_width), jnp, lax)
        return decoded.astype(jnp.float32 if dtype is None else dtype)
    raise ValueError("backend must be either 'numpy' or 'jax'")


__all__ = ["block_layout", "dequantize_rows"]
