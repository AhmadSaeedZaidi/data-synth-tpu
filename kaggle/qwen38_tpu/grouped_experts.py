"""Bounded grouped FFNs with optional intermediate-channel tensor parallelism.

Uses batched native dots, not a stock MegaBlocks kernel: GGUF's mixed Q6_K
and Q8_0 block scales must survive intact. At most group_size experts are
expanded at once. Hot experts use additional row tiles without route drops.
"""

from functools import lru_cache

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from .compute import q8_matmul
from .math import silu
from .parallel import packed_spec
from .precision import compute_cast
from .quant import block_layout, dequantize_rows


@lru_cache(maxsize=64)
def grouped_expert_parallel(
    mesh,
    *,
    gate_type,
    up_type,
    down_type,
    hidden,
    intermediate,
    compute="bf16",
    row_tile=128,
    group_size=4,
):
    if compute not in {"bf16", "int8_q8"}:
        raise ValueError("Unknown expert compute policy")
    if type(group_size) is not int or group_size < 1:
        raise ValueError("Expert group size must be a positive integer")
    if type(row_tile) is not int or row_tile < 1:
        raise ValueError("Expert row tile must be a positive integer")
    tp = mesh.shape.get("tensor", 1)
    local_width = intermediate // tp
    if intermediate % tp or local_width % block_layout(down_type)[0]:
        raise ValueError("Tensor slices must contain whole down quantization blocks")

    def local(gate, up, down, x, ids, weights):
        count, batch = gate.shape[0], x.shape[0]
        first = jax.lax.axis_index("expert") * count
        routes = jnp.sum(
            jnp.where(
                ids[None, :, :] == (first + jnp.arange(count))[:, None, None],
                weights[None, :, :],
                0,
            ),
            axis=-1,
        )
        counts = jnp.count_nonzero(routes, axis=-1)
        active = jnp.count_nonzero(counts)
        groups = (count + group_size - 1) // group_size
        selected = jnp.nonzero(counts, size=groups * group_size, fill_value=0)[0]
        tile_size = min(row_tile, batch)

        def group(index, output):
            offset = index * group_size
            experts = jax.lax.dynamic_slice_in_dim(selected, offset, group_size)
            valid = offset + jnp.arange(group_size) < active
            routed = jnp.where(valid[:, None], routes[experts], 0)
            nrows = jnp.where(valid, counts[experts], 0)

            def unpack(table, qtype, out, width):
                block, size = block_layout(qtype)
                # A single expert is one contiguous packed row. Express that
                # directly instead of a vector gather over a one-element index.
                # Larger groups retain the measured v12 gather implementation.
                packed = (
                    jax.lax.dynamic_index_in_dim(table, experts[0], axis=0, keepdims=True)
                    if group_size == 1
                    else table[experts]
                )
                raw = packed.reshape(group_size, out, width // block, size)
                if compute == "int8_q8" and qtype == "Q8_0":
                    return raw
                return dequantize_rows(raw, qtype, width, backend="jax", dtype=x.dtype)

            with jax.named_scope("unpack_expert_weights"):
                gate_w = unpack(gate, gate_type, local_width, hidden)
                up_w = unpack(up, up_type, local_width, hidden)
                down_w = unpack(down, down_type, hidden, local_width)

            def project(tokens, matrix, qtype, *, partial=False):
                if compute == "int8_q8" and qtype == "Q8_0":
                    y = jax.vmap(lambda a, b: q8_matmul(a, b, output_dtype=jnp.float32))(
                        tokens, matrix
                    )
                else:
                    precision = jax.lax.Precision.HIGHEST if x.dtype == jnp.float32 else None
                    y = jnp.matmul(
                        tokens,
                        matrix.swapaxes(-1, -2),
                        preferred_element_type=jnp.float32,
                        precision=precision,
                    )
                return y if partial else compute_cast(y, x.dtype)

            def contribution(tokens, route):
                if gate_type == up_type:
                    # One wider dot shares the input read and dispatch.
                    joined = project(tokens, jnp.concatenate((gate_w, up_w), axis=1), gate_type)
                    g, u = jnp.split(joined, 2, axis=-1)
                else:
                    g, u = project(tokens, gate_w, gate_type), project(tokens, up_w, up_type)
                activation = compute_cast(silu(g) * u, x.dtype)
                y = project(activation, down_w, down_type, partial=True)
                if tp > 1:
                    # Sum partial down projections BEFORE the BF16 rounding
                    # and routing weight, matching the unpartitioned expert.
                    with jax.named_scope("tensor_partial_sum"):
                        y = jax.lax.psum(y, "tensor")
                y = compute_cast(y, x.dtype).astype(jnp.float32)
                route = route[..., None].astype(jnp.float32)
                # Padded expert slots and finished rows must contribute exactly
                # zero, including when unused weights contain nonfinite values.
                return jnp.where(route != 0, y * route, 0)

            if batch <= row_tile:
                values = contribution(jnp.broadcast_to(x, (group_size, *x.shape)), routed)
                # Fixed small loop retains expert accumulation order.
                return jax.lax.fori_loop(0, group_size, lambda g, out: out + values[g], output)

            size = ((batch + tile_size - 1) // tile_size) * tile_size
            rows = jax.vmap(lambda r: jnp.nonzero(r, size=size, fill_value=batch)[0])(routed)

            def tile(i, value):
                indices = jax.lax.dynamic_slice_in_dim(rows, i * tile_size, tile_size, axis=1)
                tokens = jnp.take(x, indices, axis=0, mode="fill", fill_value=0)
                route = jax.vmap(lambda r, ix: jnp.take(r, ix, mode="fill", fill_value=0))(
                    routed, indices
                )
                values = contribution(tokens, route)
                return value.at[indices.reshape(-1)].add(values.reshape(-1, hidden), mode="drop")

            return jax.lax.fori_loop(0, (jnp.max(nrows) + tile_size - 1) // tile_size, tile, output)

        output = jax.lax.fori_loop(
            0,
            (active + group_size - 1) // group_size,
            group,
            jnp.zeros((batch, hidden), jnp.float32),
        )
        # TP partners already hold identical complete expert contributions.
        # Reducing over tensor again would incorrectly multiply by TP size.
        with jax.named_scope("expert_output_sum"):
            return jax.lax.psum(output, "expert").astype(x.dtype)

    spec = packed_spec(mesh)
    return jax.jit(
        jax.shard_map(
            local,
            mesh=mesh,
            in_specs=(spec, spec, spec, P(), P(), P()),
            out_specs=P(),
            check_vma=False,
        )
    )
