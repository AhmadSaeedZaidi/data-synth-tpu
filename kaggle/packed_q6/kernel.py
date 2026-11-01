"""Experimental fused packed-Q6 projection; no full decoded weight tensor.

Uses the tiled FP32 accumulation / VMEM BlockSpec pattern described in JAX's
Pallas matmul guide and vLLM TPU kernels. Q6 byte handling is specific to this
module's lossless host layout. Hardware performance remains unvalidated:
callers must opt in explicitly; this is not part of model inference.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from .layout import OUTPUT_TILE, QK, SCALE_FETCH_BYTES, SCALE_TILE_BYTES, TILE_BYTES


def _rounded_bf16(x):
    # Preserve the actual BF16 weight boundary even when the compiler fuses
    # FP32 consumers. See the application's independent numerics regressions.
    bits = jax.lax.bitcast_convert_type(x.astype(jnp.float32), jnp.uint32)
    bits = (bits + jnp.uint32(0x7FFF) + ((bits >> 16) & 1)) & jnp.uint32(0xFFFF0000)
    return jax.lax.bitcast_convert_type(bits, jnp.float32).astype(jnp.bfloat16)


def decode_fp16_scale(bits):
    """Finite IEEE FP16 storage -> exact FP32 using only 32-bit operations.

    No FP16 arithmetic is required of the TPU kernel. FP16 subnormals are
    normal FP32 values; zero and its sign are preserved as well. Nonfinite
    source scales are rejected by the host packer.
    """
    bits = bits.astype(jnp.uint32)
    exponent = (bits >> 10) & 31
    mantissa = bits & 1023
    normal_bits = ((exponent + 112) << 23) | (mantissa << 13)
    normal = jax.lax.bitcast_convert_type(normal_bits, jnp.float32)
    magnitude = jnp.where(exponent == 0, mantissa.astype(jnp.float32) * (2**-24), normal)
    magnitude_bits = jax.lax.bitcast_convert_type(magnitude, jnp.uint32)
    return jax.lax.bitcast_convert_type(magnitude_bits | ((bits & 32768) << 16), jnp.float32)


def decode_tile(tile, scale_bytes):
    """Decode one SoA byte tile inside VMEM; [64, 256] BF16 operands."""
    low = tile[:8192].reshape(OUTPUT_TILE, 2, 64)
    high = tile[8192:12288].reshape(OUTPUT_TILE, 2, 32)
    scale = tile[12288:13312].reshape(OUTPUT_TILE, 16).astype(jnp.int32)
    scale = jnp.where(scale >= 128, scale - 256, scale).astype(jnp.float32)
    db = scale_bytes.reshape(OUTPUT_TILE, 2).astype(jnp.uint32)
    bits = db[:, 0] | (db[:, 1] << 8)
    d = decode_fp16_scale(bits)
    parts = []
    for half in range(2):
        for quarter in range(4):
            lo = low[:, half, (quarter % 2) * 32 : (quarter % 2 + 1) * 32]
            lo = (lo.astype(jnp.int32) >> (4 * (quarter // 2))) & 15
            hi = (high[:, half, :].astype(jnp.int32) >> (2 * quarter)) & 3
            parts.append((lo | (hi << 4)) - 32)
    codes = jnp.concatenate(parts, axis=1).astype(jnp.float32)
    scales = jnp.repeat(scale, 16, axis=1)
    return _rounded_bf16((d[:, None] * scales) * codes)


def _projection(x_ref, packed_ref, scale_ref, out_ref, accumulator, *, input_blocks):
    k = pl.program_id(2)

    @pl.when(k == 0)
    def initialize():
        accumulator[...] = jnp.zeros_like(accumulator)

    tile_index = pl.program_id(1) * input_blocks + k
    # Mosaic does not lower dynamic_slice on a vector register. Select among
    # static slices of this small aligned DMA window instead. Consecutive K
    # tiles reuse the same 512-byte input window through the Pallas pipeline.
    scale_window = scale_ref[...].astype(jnp.uint32)
    quarter = tile_index % 4
    scale_bytes = scale_window[3 * SCALE_TILE_BYTES :]
    for i in range(2, -1, -1):
        scale_bytes = jnp.where(
            quarter == i,
            scale_window[i * SCALE_TILE_BYTES : (i + 1) * SCALE_TILE_BYTES],
            scale_bytes,
        )
    weights = decode_tile(packed_ref[...], scale_bytes)
    accumulator[...] += jax.lax.dot_general(
        x_ref[...],
        weights,
        (((1,), (1,)), ((), ())),
        preferred_element_type=jnp.float32,
    )

    @pl.when(k == input_blocks - 1)
    def store():
        out_ref[...] = accumulator[...]


@partial(jax.jit, static_argnames=("output_features", "batch_tile", "interpret"))
def q6_matmul(x, packed, scales, *, output_features, batch_tile=8, interpret=False):
    """FP32 result of BF16 x @ decode(Q6_K).T with local, tiled decoding.

    x: [M, K] BF16. packed/scales: flat uint8 planes from pack_q6. Partial M/N/K tiles are
    zero-padded/cropped. Repacking and device transfer belong outside the
    measured decode loop. interpret=True is for correctness, never speed.
    """
    if x.ndim != 2 or min(x.shape) < 1 or x.dtype != jnp.bfloat16:
        raise ValueError("x must be a nonempty [M,K] BF16 array")
    if type(output_features) is not int or output_features < 1:
        raise ValueError("output_features must be positive")
    if type(batch_tile) is not int or batch_tile < 8 or batch_tile % 8:
        raise ValueError("batch_tile must be a positive multiple of 8")
    m, inputs = x.shape
    blocks = (inputs + QK - 1) // QK
    output_tiles = (output_features + OUTPUT_TILE - 1) // OUTPUT_TILE
    if packed.dtype != jnp.uint8 or packed.shape != (output_tiles * blocks * TILE_BYTES,):
        raise ValueError("Packed Q6 dtype/shape does not match projection dimensions")
    scale_size = (
        pl.cdiv(output_tiles * blocks * SCALE_TILE_BYTES, SCALE_FETCH_BYTES) * SCALE_FETCH_BYTES
    )
    if scales.dtype != jnp.uint8 or scales.shape != (scale_size,):
        raise ValueError("Packed Q6 scale dtype/shape does not match projection dimensions")
    padded_m = pl.cdiv(m, batch_tile) * batch_tile
    x = jnp.pad(x, ((0, padded_m - m), (0, blocks * QK - inputs)))
    call = pl.pallas_call(
        partial(_projection, input_blocks=blocks),
        # A 64-column window is legal when it is the complete trailing axis.
        # Keep output tiles separate until the small result is assembled;
        # this avoids padding resident weights from 320 columns to 384.
        out_shape=jax.ShapeDtypeStruct((output_tiles, padded_m, OUTPUT_TILE), jnp.float32),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=(
                pl.BlockSpec((batch_tile, QK), lambda m, n, k: (m, k)),
                pl.BlockSpec((TILE_BYTES,), lambda m, n, k: ((n * blocks + k),)),
                pl.BlockSpec((SCALE_FETCH_BYTES,), lambda m, n, k: ((n * blocks + k) // 4,)),
            ),
            out_specs=pl.BlockSpec((None, batch_tile, OUTPUT_TILE), lambda m, n, k: (n, m, 0)),
            scratch_shapes=(pltpu.VMEM((batch_tile, OUTPUT_TILE), jnp.float32),),
            grid=(padded_m // batch_tile, output_tiles, blocks),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary")
        ),
        interpret=interpret,
        name="packed_q6_k_projection",
    )
    result = call(x, packed, scales).transpose(1, 0, 2).reshape(padded_m, -1)
    return result[:m, :output_features]
