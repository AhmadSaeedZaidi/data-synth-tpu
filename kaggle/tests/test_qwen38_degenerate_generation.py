"""Catch the v9 correlated-key failure and stop invalid logits before argmax."""

from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from qwen38_tpu.config import ModelConfig
from qwen38_tpu.math import gdn_chunk, gdn_step, unit_lower_solve
from qwen38_tpu.run import InvalidLogitsError, checked_argmax, generate


@pytest.mark.parametrize("beta", [0.5, 0.9, 0.99])
def test_correlated_keys_do_not_destabilize_the_triangular_transform(beta):
    rng = np.random.default_rng(193)
    lower = np.tril(np.full((2, 64, 64), beta, np.float32), -1)
    distance = np.maximum(np.arange(64)[:, None] - np.arange(64)[None, :], 0)
    lower[1] *= np.exp(-0.01 * distance).astype(np.float32)
    rhs = rng.normal(size=(2, 64, 32)).astype(np.float32)
    expected = np.linalg.solve(lower.astype(np.float64) + np.eye(64), rhs.astype(np.float64))
    actual = np.asarray(jax.jit(unit_lower_solve)(lower, rhs))
    np.testing.assert_allclose(actual, expected, atol=8e-6, rtol=8e-6)


def correlated_gdn_fixture():
    """Repeated patch-like keys with weak decay; tiny CPU reproduction."""
    c = replace(ModelConfig(), hidden=32, key_heads=1, value_heads=1, state_dim=16)
    qkv = np.zeros((c.conv_channels, c.hidden), np.float32)
    qkv[[0, 16, 32], 0] = 1
    conv = np.zeros((c.conv_channels, c.conv_kernel), np.float32)
    conv[:, -1] = 1
    beta = np.zeros((1, c.hidden), np.float32)
    beta[0, 0] = 2.2
    weights = dict(
        qkv=qkv,
        conv=conv,
        alpha=np.zeros((1, c.hidden)),
        beta=beta,
        gate=np.zeros((16, c.hidden)),
        out=np.eye(c.hidden, 16),
    )
    weights = {k: jnp.asarray(v, jnp.bfloat16) for k, v in weights.items()}
    weights.update(a=jnp.array([-1e-5]), dt=jnp.zeros(1), norm=jnp.ones(16))
    x = jnp.ones((1, 512, c.hidden), jnp.bfloat16)
    active = jnp.ones(x.shape[:2], bool)
    cache = (
        jnp.zeros((1, c.conv_kernel - 1, c.conv_channels), jnp.bfloat16),
        jnp.zeros((1, c.value_heads, c.state_dim, c.state_dim), jnp.float32),
    )
    return c, weights, x, active, cache


def test_repeated_keys_remain_finite_across_eight_gdn_chunks():
    c, w, x, active, cache = correlated_gdn_fixture()

    def serial(x, w, cache, active):
        def step(carry, inputs):
            value, carry = gdn_step(inputs[0], w, carry, inputs[1], c)
            return carry, value

        state, y = jax.lax.scan(step, cache, (x.swapaxes(0, 1), active.T))
        return y.swapaxes(0, 1), state

    expected, state = jax.jit(serial)(x, w, cache, active)
    actual, updated = jax.jit(lambda x, w, cache, mask: gdn_chunk(x, w, cache, mask, c))(
        x, w, cache, active
    )
    assert np.isfinite(np.asarray(actual, np.float32)).all()
    np.testing.assert_allclose(
        np.asarray(actual, np.float32), np.asarray(expected, np.float32), atol=1e-4, rtol=1e-4
    )
    np.testing.assert_allclose(np.asarray(updated[1]), np.asarray(state[1]), atol=2e-5, rtol=2e-5)


def test_checked_argmax_reports_nan_and_infinity_instead_of_silent_token_zero():
    logits = jnp.array([[1, 4, 2], [jnp.nan, jnp.nan, jnp.nan], [1, jnp.inf, -jnp.inf]])
    selected = np.asarray(checked_argmax(logits))
    np.testing.assert_array_equal(selected[:, 1], [0, 3, 2])
    assert selected[0, 0] == 1


def test_generation_rejects_bad_prefill_before_lookup_or_decode(monkeypatch):
    from qwen38_tpu import run

    monkeypatch.setattr(run.tracing, "span", lambda *a, **kw: nullcontext({}))
    engine = SimpleNamespace(config=SimpleNamespace(generation_eos=(99,)))
    prompts = [dict(case=dict(name="vision-canary"))]
    with pytest.raises(InvalidLogitsError) as error:
        generate(
            engine,
            prompts,
            jnp.full((1, 16), jnp.nan),
            None,
            None,
            None,
            None,
            None,
            2048,
            lambda **kw: None,
        )
    assert error.value.diagnostics == dict(
        stage="prefill",
        decode_step=0,
        rows=[dict(row=0, case="vision-canary", nonfinite_logits=16)],
    )
