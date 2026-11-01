"""Explicit BF16 precision boundaries that cannot retain an FP32 scale."""

import jax
import jax.numpy as jnp


def rounded_bf16_float32(x):
    """Round to BF16 with ties to even, keeping the rounded value in FP32.

    Integer rounding protects required precision boundaries when an FP32
    consumer follows a BF16 conversion inside the same TPU fusion.
    """
    bits = jax.lax.bitcast_convert_type(x.astype(jnp.float32), jnp.uint32)
    rounded = (bits + jnp.uint32(0x7FFF) + ((bits >> 16) & 1)) & jnp.uint32(0xFFFF0000)
    nan = (bits & jnp.uint32(0x7FFFFFFF)) > jnp.uint32(0x7F800000)
    rounded = jnp.where(nan, bits | jnp.uint32(0x00400000), rounded)
    return jax.lax.bitcast_convert_type(rounded, jnp.float32)


def compute_cast(x, dtype):
    if dtype == jnp.bfloat16:
        x = rounded_bf16_float32(x)
    return x.astype(dtype)
