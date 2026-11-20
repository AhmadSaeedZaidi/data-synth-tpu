"""Compile JAX subgraphs into TorchTPU operators without a second TPU runtime.

TorchTPU traces these typed functions with abstract arrays. Do not call
jax.devices(), device_put, DLPack, or a separately initialized JAX TPU client
inside a TorchTPU worker. All real buffers and execution stay with TorchTPU.
"""


def register_projection(name, qtype, width):
    import jax
    import jax.numpy as jnp
    import torch
    from qwen38_tpu.quant import dequantize_rows

    if qtype not in {"Q6_K", "Q8_0"}:
        raise ValueError("Unsupported packed projection type")

    def projection(x: jax.Array, raw: jax.Array) -> jax.Array:
        weight = dequantize_rows(raw, qtype, width, backend="jax", dtype=x.dtype)
        return jnp.matmul(x, weight.T, preferred_element_type=jnp.float32)

    # jax_op uses annotations to create the custom operator schema and exports
    # StableHLO; it is not a tensor-to-host conversion or a JAX execution call.
    return torch.tpu.pallas.jax_op(name, projection)


def register_cache_read(name, heads, width, group=32):
    import jax
    import jax.numpy as jnp
    import torch
    from qwen38_tpu.precision import compute_cast

    if heads < 1 or width < 1 or group < 1 or width % group:
        raise ValueError("Cache head width must divide into positive groups")

    def read(codes: jax.Array, scales: jax.Array, indices: jax.Array) -> jax.Array:
        batch, selected = indices.shape
        groups = heads * (width // group)
        addresses = jnp.broadcast_to(indices[:, None, :], (batch, groups, selected))
        scale = jnp.take_along_axis(scales, addresses.reshape(batch * groups, selected), axis=1)
        scale = scale.reshape(batch, groups, selected).swapaxes(1, 2).astype(jnp.float32)
        values = codes[jnp.arange(batch)[:, None], indices].reshape(batch, selected, groups, group)
        return compute_cast(
            (values.astype(jnp.float32) * scale[..., None]).reshape(batch, selected, heads, width),
            jnp.bfloat16,
        )

    return torch.tpu.pallas.jax_op(name, read)
