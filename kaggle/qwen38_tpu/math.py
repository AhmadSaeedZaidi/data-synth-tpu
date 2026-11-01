"""JAX equations matching Qwen4Exp. Weights use effective (already folded) RMS gammas."""

import jax
import jax.numpy as jnp

from . import numerics
from .precision import compute_cast


def linear(x, weight):
    # Native BF16 operands with explicit FP32 accumulation on TPU. FP32
    # reference/routers retain their original precision.
    precision = jax.lax.Precision.HIGHEST if x.dtype == weight.dtype == jnp.float32 else None
    return compute_cast(
        jnp.matmul(x, weight.T, preferred_element_type=jnp.float32, precision=precision), x.dtype
    )


def silu(x):
    return compute_cast(x.astype(jnp.float32) * numerics.sigmoid(x), x.dtype)


def sigmoid(x):
    return compute_cast(numerics.sigmoid(x), x.dtype)


def rms(x, gamma, eps=1e-6):
    xf = x.astype(jnp.float32)
    return compute_cast(
        xf * jax.lax.rsqrt(jnp.mean(xf**2, axis=-1, keepdims=True) + eps) * gamma, x.dtype
    )


def grouped_rms(x, gamma, streams, eps):
    shape = x.shape
    return rms(x.reshape(*shape[:-1], streams, -1), gamma.reshape(streams, -1), eps).reshape(shape)


def residual_read(x, w, c):
    normed = grouped_rms(x, w["norm"], c.streams, c.eps)
    lo = silu(linear(normed, w["down"]) / c.streams)
    gates = sigmoid(linear(lo, w["up"]))
    gated = compute_cast(gates * normed, x.dtype)
    mixed = compute_cast(
        jnp.mean(gated.reshape(*x.shape[:-1], c.streams, c.hidden), axis=-2), x.dtype
    )
    inject = None if "inject" not in w else 2 * sigmoid(linear(normed, w["inject"]) / c.streams)
    return mixed, inject


def residual_write(x, y, inject, c):
    contribution = compute_cast(inject[..., None] * y[..., None, :], x.dtype)
    return compute_cast(
        (x.reshape(*x.shape[:-1], c.streams, c.hidden) + contribution).reshape(x.shape), x.dtype
    )


def choose(active, updated, old):
    return jnp.where(active.reshape((active.shape[0],) + (1,) * (updated.ndim - 1)), updated, old)


def delta_step(q, k, v, decay, beta, state):
    q, k, v = (z.astype(jnp.float32) for z in (q, k, v))
    q *= jax.lax.rsqrt(jnp.sum(q**2, -1, keepdims=True) + 1e-6) / q.shape[-1] ** 0.5
    k *= jax.lax.rsqrt(jnp.sum(k**2, -1, keepdims=True) + 1e-6)
    repeat = v.shape[-2] // k.shape[-2]
    q, k = jnp.repeat(q, repeat, axis=-2), jnp.repeat(k, repeat, axis=-2)
    state = state * numerics.exp(decay)[..., None, None]
    remembered = jnp.einsum("bhkv,bhk->bhv", state, k, precision=jax.lax.Precision.HIGHEST)
    delta = (v - remembered) * beta[..., None]
    state = state + k[..., :, None] * delta[..., None, :]
    return jnp.einsum("bhkv,bhk->bhv", state, q, precision=jax.lax.Precision.HIGHEST), state


def gdn_step(x, w, cache, active, c):
    previous_conv, previous_state = cache
    raw = linear(x, w["qkv"])
    conv = jnp.concatenate((previous_conv, raw[:, None, :]), axis=1)
    convolved = compute_cast(
        jnp.sum(conv.astype(jnp.float32) * w["conv"].T[None, :, :].astype(jnp.float32), axis=1),
        x.dtype,
    )
    projected = silu(convolved)
    cut = c.key_heads * c.state_dim
    q = projected[:, :cut].reshape(-1, c.key_heads, c.state_dim)
    k = projected[:, cut : 2 * cut].reshape(q.shape)
    v = projected[:, 2 * cut :].reshape(-1, c.value_heads, c.state_dim)
    a = linear(x, w["alpha"]).astype(jnp.float32)
    decay = w["a"][None, :] * numerics.softplus(a + w["dt"])  # GGUF stores -exp(A_log)
    beta = sigmoid(linear(x, w["beta"])).astype(jnp.float32)
    out, state = delta_step(q, k, v, decay, beta, previous_state)
    z = linear(x, w["gate"]).reshape(v.shape)
    xf = compute_cast(out, x.dtype).astype(jnp.float32)
    normalized = compute_cast(
        xf * jax.lax.rsqrt(jnp.mean(xf**2, -1, keepdims=True) + c.eps), x.dtype
    )
    out = compute_cast(normalized.astype(jnp.float32) * w["norm"] * numerics.sigmoid(z), x.dtype)
    result = linear(out.reshape(x.shape[0], -1), w["out"])
    return result, (
        choose(active, conv[:, 1:, :], previous_conv),
        choose(active, state, previous_state),
    )


GDN_CHUNK = 64


def unit_lower_solve(lower, rhs):
    """Solve (I + strictly_lower) X = rhs in FP32 by forward substitution.

    Nilpotent powers can grow combinatorially for correlated keys even when
    the true inverse is bounded. Building that inverse by repeated squaring
    loses catastrophic cancellation in FP32. Solve the RHS directly instead.
    """
    size = lower.shape[-1]
    if lower.shape[-2:] != (size, size) or rhs.shape[-2] != size:
        raise ValueError("Triangular solve dimensions differ")
    lower, rhs = lower.astype(jnp.float32), rhs.astype(jnp.float32)

    def row(index, solved):
        coefficients = jax.lax.dynamic_index_in_dim(lower, index, axis=-2, keepdims=False)
        coefficients = jnp.where(jnp.arange(size) < index, coefficients, 0)
        known = jnp.einsum(
            "...k,...kv->...v", coefficients, solved, precision=jax.lax.Precision.HIGHEST
        )
        value = jax.lax.dynamic_index_in_dim(rhs, index, axis=-2, keepdims=False) - known
        return jax.lax.dynamic_update_slice_in_dim(solved, value[..., None, :], index, axis=-2)

    return jax.lax.fori_loop(0, size, row, jnp.zeros_like(rhs))


def gdn_chunk(x, w, cache, active, c):
    """Chunk-parallel GDN prefill over [batch, steps, hidden].

    Experimental alternative to the serial ``gdn_step`` scan (and the reference
    ``torch_chunk_gated_delta_rule``): projections run on every token at once,
    the intra-chunk triangular system is solved by forward substitution, then a scan
    over 64-token chunks carries the recurrent state. Padded rows are frozen by
    zeroing their log decays and betas, which reproduces the serial
    per-token state skip algebraically. Numerical equivalence on the full
    checkpoint must be established before selecting this path for a trial.
    """
    previous_conv, previous_state = cache
    batch, steps, width = x.shape
    flat = x.reshape(batch * steps, width)
    raw = linear(flat, w["qkv"]).reshape(batch, steps, -1)
    buf = jnp.concatenate((previous_conv, raw), axis=1)
    taps = (
        buf[:, j : j + steps, :].astype(jnp.float32)
        * w["conv"].T[j].astype(jnp.float32)[None, None, :]
        for j in range(c.conv_kernel)
    )
    convolved = compute_cast(sum(taps), x.dtype)
    projected = silu(convolved)
    cut = c.key_heads * c.state_dim
    q = projected[..., :cut].reshape(batch, steps, c.key_heads, c.state_dim)
    k = projected[..., cut : 2 * cut].reshape(q.shape)
    v = projected[..., 2 * cut :].reshape(batch, steps, c.value_heads, c.state_dim)
    a = linear(flat, w["alpha"]).reshape(batch, steps, -1).astype(jnp.float32)
    decay = w["a"][None, None, :] * numerics.softplus(a + w["dt"][None, None, :])
    beta = sigmoid(linear(flat, w["beta"]).reshape(batch, steps, -1)).astype(jnp.float32)
    mask = active.astype(jnp.float32)[..., None]
    decay, beta = decay * mask, beta * mask
    q, k, v = (z.astype(jnp.float32) for z in (q, k, v))
    q = q * jax.lax.rsqrt(jnp.sum(q**2, -1, keepdims=True) + 1e-6) * c.state_dim**-0.5
    k = k * jax.lax.rsqrt(jnp.sum(k**2, -1, keepdims=True) + 1e-6)
    repeat = c.value_heads // c.key_heads
    q, k = jnp.repeat(q, repeat, axis=2), jnp.repeat(k, repeat, axis=2)
    pad = -steps % GDN_CHUNK
    total = steps + pad
    if pad:
        padded = [
            jnp.pad(z, ((0, 0), (0, pad)) + ((0, 0),) * (z.ndim - 2))
            for z in (q, k, v, decay[..., None], beta[..., None])
        ]
        q, k, v, decay, beta = padded
        decay, beta = decay[..., 0], beta[..., 0]
    chunks = total // GDN_CHUNK
    hidden_dim = c.state_dim
    shape = (batch, c.value_heads, chunks, GDN_CHUNK, hidden_dim)
    q = jnp.swapaxes(q, 1, 2).reshape(shape)
    k = jnp.swapaxes(k, 1, 2).reshape(shape)
    v = jnp.swapaxes(v, 1, 2).reshape(shape)
    steps_shape = (batch, c.value_heads, chunks, GDN_CHUNK)
    decay = jnp.swapaxes(decay, 1, 2).reshape(steps_shape)
    beta = jnp.swapaxes(beta, 1, 2).reshape(steps_shape)
    cum = jnp.cumsum(decay, axis=-1)
    k_beta = k * beta[..., None]
    v_beta = v * beta[..., None]
    lower = jnp.arange(GDN_CHUNK)[:, None] >= jnp.arange(GDN_CHUNK)[None, :]
    pairwise = numerics.exp(jnp.where(lower, cum[..., :, None] - cum[..., None, :], -jnp.inf))
    highest = jax.lax.Precision.HIGHEST
    ut = jnp.tril(jnp.matmul(k_beta, jnp.swapaxes(k, -1, -2), precision=highest) * pairwise, -1)
    intra = jnp.matmul(q, jnp.swapaxes(k, -1, -2), precision=highest) * pairwise
    cumulative_decay = numerics.exp(cum)
    decayed_k_beta = k_beta * cumulative_decay[..., None]
    solved = unit_lower_solve(ut, jnp.concatenate((v_beta, decayed_k_beta), axis=-1))
    new_values, k_cumdecay = jnp.split(solved, 2, axis=-1)
    q = q * cumulative_decay[..., None]
    k = k * numerics.exp(cum[..., -1:] - cum)[..., None]
    chunk_decay = cumulative_decay[..., -1, None, None]
    state = previous_state.astype(jnp.float32)
    outputs = []
    for index in range(chunks):
        v_new = new_values[:, :, index] - jnp.matmul(
            k_cumdecay[:, :, index], state, precision=highest
        )
        outputs.append(
            jnp.matmul(q[:, :, index], state, precision=highest)
            + jnp.matmul(intra[:, :, index], v_new, precision=highest)
        )
        state = state * chunk_decay[:, :, index] + jnp.matmul(
            jnp.swapaxes(k[:, :, index], -1, -2), v_new, precision=highest
        )
    core = jnp.concatenate(outputs, axis=2)[:, :, :steps]
    core = jnp.swapaxes(core, 1, 2)
    z = linear(flat, w["gate"]).reshape(batch, steps, c.value_heads, c.state_dim)
    xf = compute_cast(core, x.dtype).astype(jnp.float32)
    normalized = compute_cast(
        xf * jax.lax.rsqrt(jnp.mean(xf**2, -1, keepdims=True) + c.eps), x.dtype
    )
    out = compute_cast(normalized.astype(jnp.float32) * w["norm"] * numerics.sigmoid(z), x.dtype)
    result = linear(out.reshape(batch * steps, -1), w["out"]).reshape(batch, steps, -1)
    counts = jnp.sum(active, axis=1)
    picks = (jnp.maximum(counts - 1, 0) + 1)[:, None] + jnp.arange(c.conv_kernel - 1)[None, :]
    picked = jnp.take_along_axis(buf, picks[:, :, None], axis=1)
    new_conv = jnp.where((counts > 0)[:, None, None], picked, previous_conv)
    return result, (new_conv.astype(previous_conv.dtype), state.astype(previous_state.dtype))


def rope(x, positions, c):
    half = c.rotary_dim // 2
    inv = c.rope_theta ** (-jnp.arange(half, dtype=jnp.float32) * 2 / c.rotary_dim)
    axis = jnp.zeros(half, jnp.int32)
    for dim, offset in ((1, 1), (2, 2)):
        indices = jnp.arange(offset, c.mrope_sections[dim] * 3, 3)
        axis = axis.at[indices].set(dim)
    angles = jnp.take(positions, axis, axis=-1) * inv
    cs = compute_cast(jnp.cos(angles), x.dtype).astype(jnp.float32)[:, None, :]
    sn = compute_cast(jnp.sin(angles), x.dtype).astype(jnp.float32)[:, None, :]
    a, b = x[..., :half].astype(jnp.float32), x[..., half : c.rotary_dim].astype(jnp.float32)
    rotated = compute_cast(jnp.concatenate((a * cs - b * sn, b * cs + a * sn), -1), x.dtype)
    return jnp.concatenate((rotated, x[..., c.rotary_dim :]), -1)


def qsa_step(x, w, cache, lengths, positions, active, c):
    from .attention_cache import step

    return step(x, w, cache, lengths, positions, active, c)


def ple_step(x, features, w, history, active, c):
    key = grouped_rms(linear(features, w["key"]), w["key_norm"], c.streams, c.eps).reshape(
        -1, c.streams, c.hidden
    )
    query = grouped_rms(x, w["query_norm"], c.streams, c.eps).reshape(key.shape)
    product = compute_cast(key * query, x.dtype)
    score = compute_cast(jnp.sum(product.astype(jnp.float32), -1), x.dtype)
    score = compute_cast(score.astype(jnp.float32) / c.hidden**0.5, x.dtype)
    rooted = compute_cast(jnp.sqrt(jnp.maximum(jnp.abs(score.astype(jnp.float32)), 1e-6)), x.dtype)
    gate = sigmoid(jnp.sign(score) * rooted)
    value = linear(features, w["value"])
    gated = compute_cast(gate[..., None] * value[:, None, :], x.dtype).reshape(x.shape)
    normalized = grouped_rms(gated, w["conv_norm"], c.streams, c.eps)
    full = jnp.concatenate((history, normalized[:, None, :]), axis=1)
    taps = full[:, :: c.ngram_size, :]
    convolved = compute_cast(
        jnp.sum(taps.astype(jnp.float32) * w["conv"].T[None, :, :].astype(jnp.float32), axis=1),
        x.dtype,
    )
    output = compute_cast(compute_cast(x + gated, x.dtype) + silu(convolved), x.dtype)
    return choose(active, output.astype(x.dtype), x), choose(
        active, full[:, 1:, :].astype(history.dtype), history
    )


def routes_and_shared(x, w, active, c):
    probs = numerics.softmax(linear(x.astype(jnp.float32), w["router"].astype(jnp.float32)), -1)
    weights, ids = jax.lax.top_k(probs, c.top_k)
    weights = numerics.divide(weights, jnp.sum(weights, -1, keepdims=True))
    weights = compute_cast(jnp.where(active[:, None], weights, 0), x.dtype)
    hidden = silu(linear(x, w["shared_gate"])) * linear(x, w["shared_up"])
    shared = linear(hidden, w["shared_down"]) * sigmoid(linear(x, w["shared_router"])[:, None])
    return ids, weights, shared.astype(x.dtype)
