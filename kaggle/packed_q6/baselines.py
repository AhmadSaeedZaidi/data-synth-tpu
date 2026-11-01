"""Whole-projection XLA baselines, independent of the Pallas tile decoder.

The byte operations mirror the application's GGUF Q6_K decoder. Weight
arguments are flat bytes, matching packed residency in the model runtime.
No assumption is made about XLA materializing the decoded matrix in HBM.
"""

from functools import partial

import jax
import jax.numpy as jnp


def decode_rows(raw, inputs):
    outputs, blocks, _ = raw.shape
    lo = raw[..., :128].reshape(outputs, blocks, 2, 64)
    lo = jnp.stack((lo & jnp.uint8(15), lo >> jnp.uint8(4)), axis=-2)
    lo = lo.reshape(outputs, blocks, 8, 32)
    hi = raw[..., 128:192].reshape(outputs, blocks, 2, 32)
    shifts = jnp.arange(4, dtype=jnp.uint8).reshape(4, 1) * jnp.uint8(2)
    hi = ((hi[..., None, :] >> shifts) & jnp.uint8(3)).reshape(outputs, blocks, 8, 32)
    codes = ((lo | (hi << jnp.uint8(4))).astype(jnp.int16) - 32).astype(jnp.float32)
    codes = codes.reshape(outputs, blocks, 16, 16)
    scales = raw[..., 192:208].astype(jnp.int16)
    scales = jnp.where(scales >= 128, scales - 256, scales).astype(jnp.float32)
    db = raw[..., 208:210].astype(jnp.uint16)
    d = jax.lax.bitcast_convert_type(db[..., 0] | (db[..., 1] << 8), jnp.float16)
    return ((d.astype(jnp.float32)[..., None, None] * scales[..., None]) * codes).reshape(
        outputs, blocks * 256
    )[:, :inputs]


@partial(jax.jit, static_argnames=("output_features",))
def xla_q6_matmul(x, raw_bytes, *, output_features):
    raw = raw_bytes.reshape(output_features, (x.shape[1] + 255) // 256, 210)
    # Match the existing decode -> BF16 -> native dot path. Independent
    # hardware correctness gates detect any compiler precision changes.
    weights = decode_rows(raw, x.shape[1]).astype(jnp.bfloat16)
    return bf16_matmul(x, weights)


@jax.jit
def bf16_matmul(x, weights):
    return jax.lax.dot_general(
        x, weights, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32
    )
