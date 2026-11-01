"""Block-scaled INT8 KV and Qwen's four-token sparse indexer.

Cache values are quantized once on insertion. Only selected tokens are gathered
and dequantized for attention. Indexer pooling and recurrent state are not INT8.
"""

import jax
import jax.numpy as jnp

from . import numerics
from .math import choose, linear, rms, rope, sigmoid
from .precision import compute_cast, rounded_bf16_float32


def quantize(x, group):
    grouped = x.astype(jnp.float32).reshape(*x.shape[:-1], -1, group)
    rounded = rounded_bf16_float32(jnp.max(jnp.abs(grouped), -1) / 127)
    scale = rounded.astype(jnp.bfloat16)
    divisor = jnp.where(rounded == 0, 1, rounded)
    codes = jnp.clip(jnp.rint(numerics.divide(grouped, divisor[..., None])), -127, 127).astype(
        jnp.int8
    )
    return codes.reshape(x.shape), scale


def write_quantized(cache, key, value, indices, active, c):
    rows = jnp.arange(key.shape[0])
    groups_per_row = c.kv_heads * (c.head_dim // c.cache_group)
    scale_rows = jnp.arange(key.shape[0] * groups_per_row)
    scale_columns = jnp.repeat(indices, groups_per_row)
    result = dict(cache)
    for name, token in (("k", key), ("v", value)):
        codes, scale = quantize(token, c.cache_group)
        codes = codes.reshape(token.shape[0], -1)
        result[name] = choose(active, cache[name].at[rows, indices].set(codes), cache[name])
        old = cache[name + "_scale"]
        updated = old.at[scale_rows, scale_columns].set(scale.flatten())
        # Scale rows flatten [batch, kv_heads, groups] so the long time axis
        # stays contiguous and tiny scale-group columns do not require padding.
        result[name + "_scale"] = jnp.where(
            jnp.repeat(active, groups_per_row)[:, None], updated, old
        )
    return result


def gather_scale_plane(plane, indices, groups, *, method="indexed"):
    """Read stored [batch * groups, time] BF16 scales without requantization.

    The rowwise candidate keeps time as the gather's contiguous minor axis,
    then arranges [batch, selected, groups] for dequantization. It avoids the
    baseline's simultaneous row/time advanced indexing into a tiny group
    output axis. Target speed and final compiler layouts remain unverified.
    """
    if indices.ndim != 2 or plane.ndim != 2 or plane.shape[0] != indices.shape[0] * groups:
        raise ValueError("Scale plane must be [batch * groups, time] for [batch, selected] indices")
    batch, selected = indices.shape
    if method == "indexed":
        scale_rows = jnp.arange(batch)[:, None] * groups + jnp.arange(groups)[None, :]
        return plane[scale_rows[:, None, :], indices[:, :, None]]
    if method == "rowwise":
        time_indices = jnp.broadcast_to(indices[:, None, :], (batch, groups, selected))
        gathered = jnp.take_along_axis(
            plane, time_indices.reshape(batch * groups, selected), axis=1, mode="clip"
        )
        # clip preserves the baseline indexing behavior, including negative
        # indices. Actual selected-token addresses are already in bounds.
        return gathered.reshape(batch, groups, selected).swapaxes(1, 2)
    raise ValueError("Scale gather method must be indexed or rowwise")


def read_quantized(cache, name, indices, c, dtype, *, scale_gather=None):
    scale_gather = c.kv_scale_gather if scale_gather is None else scale_gather
    rows = jnp.arange(indices.shape[0])
    codes = cache[name][rows[:, None], indices]
    groups = c.kv_heads * (c.head_dim // c.cache_group)
    scale = gather_scale_plane(cache[name + "_scale"], indices, groups, method=scale_gather)
    scale = scale.reshape(*indices.shape, c.kv_heads, -1).astype(jnp.float32)
    codes = codes.reshape(*scale.shape, c.cache_group).astype(jnp.float32)
    return compute_cast(
        (codes * scale[..., None]).reshape(*indices.shape, c.kv_heads, c.head_dim), dtype
    )


def select_tokens(x, w, cache, lengths, positions, active, c):
    """Mean raw keys, then normalize/rotate at block start; always retain tail."""
    rows = jnp.arange(x.shape[0])
    ratio = c.indexer_compress
    slot = lengths % ratio
    raw = linear(x, w["index_k"])
    tail = choose(active, cache["index_tail"].at[rows, slot].set(raw), cache["index_tail"])
    starts = choose(active & (slot == 0), positions, cache["index_start"])
    pooled = compute_cast(jnp.mean(tail.astype(jnp.float32), axis=1), x.dtype)
    block = rope(rms(pooled, w["index_k_norm"], c.eps)[:, None, :], starts, c)[:, 0]
    complete = active & (slot == ratio - 1)
    keys = choose(
        complete, cache["index_keys"].at[rows, lengths // ratio].set(block), cache["index_keys"]
    )
    query = linear(x, w["index_q"]).reshape(-1, c.indexer_heads, c.indexer_dim)
    query = rope(rms(query, w["index_q_norm"], c.eps), positions, c)
    scores = (
        jnp.sum(
            jax.nn.relu(
                jnp.einsum(
                    "bhd,btd->bth",
                    query.astype(jnp.float32),
                    keys.astype(jnp.float32),
                    precision=jax.lax.Precision.HIGHEST,
                )
            ),
            axis=-1,
        )
        / c.indexer_dim**0.5
    )
    count = (lengths + 1) // ratio
    scores = jnp.where(jnp.arange(keys.shape[1])[None, :] < count[:, None], scores, -jnp.inf)
    _, selected = jax.lax.top_k(scores, min(c.qsa_budget // ratio, keys.shape[1]))
    selected_tokens = (selected[..., None] * ratio + jnp.arange(ratio)).reshape(x.shape[0], -1)
    selected_valid = jnp.repeat(selected < count[:, None], ratio, axis=1)
    tail_indices = count[:, None] * ratio + jnp.arange(ratio - 1)
    tail_valid = jnp.arange(ratio - 1)[None, :] < ((lengths + 1) % ratio)[:, None]
    indices = jnp.concatenate((selected_tokens, tail_indices), axis=1)
    valid = jnp.concatenate((selected_valid, tail_valid), axis=1)
    # Masked out-of-capacity tail slots still need safe gather addresses.
    indices = jnp.where(valid, indices, 0)
    return indices, valid, {**cache, "index_keys": keys, "index_tail": tail, "index_start": starts}


def step(x, w, cache, lengths, positions, active, c, *, scale_gather=None):
    legacy = isinstance(cache, tuple)
    cache = dict(k=cache[0], v=cache[1]) if legacy else cache
    qgate = linear(x, w["q"]).reshape(-1, c.query_heads, 2 * c.head_dim)
    q, gate = jnp.split(qgate, 2, axis=-1)
    k = linear(x, w["k"]).reshape(-1, c.kv_heads, c.head_dim)
    v = linear(x, w["v"]).reshape(k.shape)
    q = rope(rms(q, w["q_norm"], c.eps), positions, c)
    k = rope(rms(k, w["k_norm"], c.eps), positions, c)
    rows = jnp.arange(x.shape[0])
    lengths = jnp.clip(lengths, 0, c.max_context - 1)
    if c.kv_cache_dtype == "q8":
        cache = write_quantized(cache, k, v, lengths, active, c)
    else:
        cache = {
            **cache,
            "k": choose(active, cache["k"].at[rows, lengths].set(k), cache["k"]),
            "v": choose(active, cache["v"].at[rows, lengths].set(v), cache["v"]),
        }
    if c.sparse_qsa:
        indices, mask, cache = select_tokens(x, w, cache, lengths, positions, active, c)
    else:
        indices = jnp.broadcast_to(jnp.arange(c.max_context), (x.shape[0], c.max_context))
        mask = indices <= lengths[:, None]
    if c.kv_cache_dtype == "q8":
        keys = read_quantized(cache, "k", indices, c, x.dtype, scale_gather=scale_gather)
        values = read_quantized(cache, "v", indices, c, x.dtype, scale_gather=scale_gather)
    elif c.sparse_qsa:
        keys, values = (cache[name][rows[:, None], indices] for name in ("k", "v"))
    else:
        keys, values = cache["k"], cache["v"]
    q = q.reshape(x.shape[0], c.kv_heads, c.query_heads // c.kv_heads, c.head_dim)
    raw_score = compute_cast(
        jnp.einsum("bkrd,btkd->bkrt", q, keys, preferred_element_type=jnp.float32), x.dtype
    )
    score = compute_cast(raw_score.astype(jnp.float32) / c.head_dim**0.5, x.dtype).astype(
        jnp.float32
    )
    prob = numerics.softmax(jnp.where(mask[:, None, None, :], score, -jnp.inf), axis=-1)
    out = jnp.einsum(
        "bkrt,btkd->bkrd", compute_cast(prob, x.dtype), values, preferred_element_type=jnp.float32
    ).reshape(gate.shape)
    out = compute_cast(compute_cast(out, x.dtype) * sigmoid(gate), x.dtype)
    result = linear(out.reshape(x.shape[0], -1), w["out"])
    return result, (cache["k"], cache["v"]) if legacy else cache
