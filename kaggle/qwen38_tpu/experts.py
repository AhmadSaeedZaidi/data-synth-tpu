"""Resident expert parallelism: only owner chips decode their packed expert rows."""

from functools import lru_cache

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from .compute import q8_matmul
from .math import linear, silu
from .precision import compute_cast
from .quant import block_layout, dequantize_rows

EXPERT_ROW_TILE = 128


@lru_cache(maxsize=32)
def expert_parallel(
    mesh,
    *,
    gate_type,
    up_type,
    down_type,
    hidden,
    intermediate,
    compute="bf16",
    row_tile=EXPERT_ROW_TILE,
    group_size=1,
    tensor_parallel=1,
):
    """Return a compiled routed FFN. E is partitioned, requests are replicated.

    Only active local experts are visited. Prompt rows are compacted into
    bounded tiles so each expert does not multiply the entire B * T prompt.
    Hot experts use as many tiles as necessary; no capacity drops routes.
    One small output all-reduce joins the owners' contributions.
    Packing stays byte-for-byte identical to GGUF, including 32-value down
    blocks where intermediate=640 cannot hold 256-value K blocks.
    """

    if group_size != 1 or tensor_parallel != 1:
        from .grouped_experts import grouped_expert_parallel
        from .parallel import expert_mesh

        return grouped_expert_parallel(
            expert_mesh(mesh, tensor_parallel),
            gate_type=gate_type,
            up_type=up_type,
            down_type=down_type,
            hidden=hidden,
            intermediate=intermediate,
            compute=compute,
            row_tile=row_tile,
            group_size=group_size,
        )
    if compute not in {"bf16", "int8_q8"}:
        raise ValueError("Unknown expert compute policy")
    if row_tile is not None and (type(row_tile) is not int or row_tile < 1):
        raise ValueError("Expert row tile must be a positive integer or None")

    def local(gate, up, down, x, ids, weights):
        local_count = gate.shape[0]
        first = jax.lax.axis_index("expert") * local_count
        batch, _ = ids.shape
        out = jnp.zeros((batch, hidden), jnp.float32)
        route_weights = jnp.sum(
            jnp.where(
                ids[None, :, :] == (first + jnp.arange(local_count))[:, None, None],
                weights[None, :, :],
                0,
            ),
            axis=-1,
        )
        counts = jnp.count_nonzero(route_weights, axis=-1)
        selected_experts = jnp.nonzero(counts, size=local_count, fill_value=0)[0]
        active = jnp.count_nonzero(counts)

        def request(index, result):
            selected = selected_experts[index]
            routed = route_weights[selected]

            def matrix(table, qtype, out, width):
                block, size = block_layout(qtype)
                packed = table[selected].reshape(out, width // block, size)
                if compute == "int8_q8" and qtype == "Q8_0":
                    return packed
                return dequantize_rows(packed, qtype, width, backend="jax", dtype=x.dtype)

            # Unpack once per active expert, outside its row-tile loop. Packed
            # parameter tensors remain owner-sharded and are never all-gathered.
            with jax.named_scope("unpack_expert_weights"):
                gate_w = matrix(gate, gate_type, intermediate, hidden)
                up_w = matrix(up, up_type, intermediate, hidden)
                down_w = matrix(down, down_type, hidden, intermediate)

            def project(token, weight, qtype):
                return (
                    q8_matmul(token, weight)
                    if compute == "int8_q8" and qtype == "Q8_0"
                    else linear(token, weight)
                )

            def contribution(token, weight):
                activation = compute_cast(
                    silu(project(token, gate_w, gate_type)) * project(token, up_w, up_type),
                    x.dtype,
                )
                return project(activation, down_w, down_type).astype(jnp.float32) * weight[
                    :, None
                ].astype(jnp.float32)

            if row_tile is None or batch <= row_tile:
                return result + contribution(x, routed)

            # Padding addresses are B, hence gather-as-zero / scatter-drop.
            # Pad the index buffer itself so dynamic_slice never clamps the
            # final partial tile back onto already processed request rows.
            size = ((batch + row_tile - 1) // row_tile) * row_tile
            rows = jnp.nonzero(routed, size=size, fill_value=batch)[0]

            def tile(tile_index, value):
                indices = jax.lax.dynamic_slice_in_dim(rows, tile_index * row_tile, row_tile)
                token = jnp.take(x, indices, axis=0, mode="fill", fill_value=0)
                weight = jnp.take(routed, indices, mode="fill", fill_value=0)
                return value.at[indices].add(contribution(token, weight), mode="drop")

            return jax.lax.fori_loop(0, (counts[selected] + row_tile - 1) // row_tile, tile, result)

        out = jax.lax.fori_loop(0, active, request, out)
        with jax.named_scope("expert_output_sum"):
            return jax.lax.psum(out, "expert").astype(x.dtype)

    distributed = jax.shard_map(
        local,
        mesh=mesh,
        in_specs=(P("expert"), P("expert"), P("expert"), P(), P(), P()),
        out_specs=P(),
        check_vma=False,
    )
    return jax.jit(distributed)
