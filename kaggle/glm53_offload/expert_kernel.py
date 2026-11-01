"""Tensor-parallel packed expert math, shared by inference and the TPU probe."""

import jax.numpy as jnp
from glm53 import model as M
from jax import lax

from .gguf_quant import block_layout, dequantize_rows


def tiled_experts(
    x, packed, idx, weights, cfg, qtypes, tile, token_rows=None, tile_capacities=None
):
    """Return the local FP32 expert sum; the caller performs the chip reduction.

    ``token_rows`` limits expert matmuls to routed tokens. ``None`` retains the
    original all-token kernel solely for numerical and hardware comparisons.
    Packed arguments are local [1, Ep, rows, bytes] shard_map slices.
    Optional tile capacities select fixed-size branches inside one scan, so a
    hot expert does not inflate every other tile or create per-layer shapes.
    """
    direct = len(packed) == 7
    if len(packed) not in (3, 7):
        raise ValueError("expected compact tables or a direct resident/miss view")
    tables = tuple(
        a[0].reshape(a.shape[1], a.shape[2], -1, block_layout(q)[1])
        for a, q in zip(packed[:3], qtypes, strict=True)
    )
    if direct:
        fresh_tables = tuple(
            a[0].reshape(a.shape[1], a.shape[2], -1, block_layout(q)[1])
            for a, q in zip(packed[3:6], qtypes, strict=True)
        )
        selectors = packed[6]
    x = x.astype(cfg.dtype)
    idx = idx.reshape(len(x), -1).astype(jnp.int32)
    weights = weights.reshape(idx.shape).astype(jnp.float32)

    def active_step(acc, tile_index):
        first = tile_index * tile
        ids = first + jnp.arange(tile, dtype=jnp.int32)
        if direct:
            slots = lax.dynamic_slice_in_dim(selectors, first, tile, axis=0)
            valid = (slots[:, 0] >= 0) | (slots[:, 1] >= 0)
            raw = tuple(
                jnp.where(
                    valid[:, None, None, None],
                    jnp.where(
                        (slots[:, 0] >= 0)[:, None, None, None],
                        resident[jnp.maximum(slots[:, 0], 0)],
                        fresh[jnp.maximum(slots[:, 1], 0)],
                    ),
                    0,
                )
                for resident, fresh in zip(tables, fresh_tables, strict=True)
            )
        else:
            raw = tuple(lax.dynamic_slice_in_dim(a, first, tile, axis=0) for a in tables)
        gate, up, down = (
            dequantize_rows(a, q, width, dtype=cfg.dtype, backend="jax")
            for a, q, width in zip(
                raw, qtypes, (cfg.hidden, cfg.hidden, cfg.moe_inter), strict=True
            )
        )

        def multiply(acc, capacity):
            if token_rows is None:
                inputs = x
                equation = "nd,emd->nem"
            else:
                rows = lax.dynamic_slice_in_dim(token_rows, first, tile, axis=0)[:, :capacity]
                safe_rows = jnp.maximum(rows, 0)
                inputs = x[safe_rows]
                equation = "erd,emd->erm"
            gate_proj, up_proj = (
                jnp.einsum(
                    equation,
                    inputs,
                    a,
                    precision=lax.Precision.HIGHEST,
                    preferred_element_type=jnp.float32,
                ).astype(cfg.dtype)
                for a in (gate, up)
            )
            hidden = M.swiglu_clamped(gate_proj, up_proj, cfg.swiglu_limit)
            partial = jnp.einsum(
                "nem,edm->ned" if token_rows is None else "erm,edm->erd",
                hidden,
                down,
                precision=lax.Precision.HIGHEST,
                preferred_element_type=jnp.float32,
            )
            if token_rows is None:
                match = idx[:, :, None] == ids[None, None, :]
                route_weight = jnp.sum(jnp.where(match, weights[:, :, None], 0.0), axis=1)
                return acc + jnp.sum(partial * route_weight[..., None], axis=1)

            # Read route weights only for this tile's selected token rows.
            match = idx[safe_rows] == ids[:, None, None]
            selected_weight = jnp.sum(jnp.where(match, weights[safe_rows], 0.0), axis=-1)
            valid = rows >= 0
            weighted = jnp.where(valid[..., None], partial * selected_weight[..., None], 0.0)
            destinations = jnp.where(valid, rows, len(x))
            return acc.at[destinations].add(weighted, mode="drop")

        if tile_capacities is None:
            return multiply(acc, None if token_rows is None else token_rows.shape[1])
        maximum = token_rows.shape[1]
        capacities = tuple(1 << i for i in range(maximum.bit_length()) if (1 << i) < maximum)
        capacities += (maximum,)
        choice = jnp.sum(tile_capacities[tile_index] > jnp.array(capacities, jnp.int32))
        branches = tuple(lambda a, c=c: multiply(a, c) for c in capacities)
        return lax.switch(choice, branches, acc)

    def step(acc, tile_index):
        if tile_capacities is None:
            return active_step(acc, tile_index), None
        return lax.cond(
            tile_capacities[tile_index] > 0,
            lambda a: active_step(a, tile_index),
            lambda a: a,
            acc,
        ), None

    result, _ = lax.scan(
        step,
        jnp.zeros((len(x), cfg.hidden), jnp.float32),
        jnp.arange((selectors.shape[0] if direct else tables[0].shape[0]) // tile, dtype=jnp.int32),
    )
    return result
