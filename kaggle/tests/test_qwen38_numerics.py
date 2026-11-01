"""V5 regression: high matmul precision is not high nonlinear precision."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh
from qwen38_tpu import numerics
from qwen38_tpu.numerics_validation import validate_numerics


def test_compiled_nonlinear_gate_against_independent_float64():
    result = validate_numerics(Mesh(np.asarray(jax.devices()), ("expert",)))
    assert result["status"] == "PASS"
    assert {item["operation"] for item in result["operations"]} == {
        "exp",
        "sigmoid",
        "softplus",
        "divide",
        "softmax",
    }


def test_residual_correction_recovers_from_v5e_sized_reciprocal_errors(monkeypatch):
    reciprocal = jax.lax.reciprocal
    # Larger than the documented worst-case default v5e reciprocal error.
    monkeypatch.setattr(jax.lax, "reciprocal", lambda x: reciprocal(x) * (1 + 2**-15))
    rng = np.random.default_rng(519)
    denominator = np.exp2(rng.uniform(-16, 16, 4096)).astype(np.float32)
    numerator = (rng.uniform(-127, 127, 4096) * denominator).astype(np.float32)
    expected = numerator.astype(np.float64) / denominator.astype(np.float64)
    actual = np.asarray(jax.jit(numerics.divide)(numerator, denominator))
    np.testing.assert_allclose(actual, expected, atol=1e-30, rtol=3e-7)


def test_division_preserves_zeros_and_special_values():
    x = np.array([0.0, -0.0, 1.0, -1.0, np.inf, np.nan, 0.0, 1.0, 1.0], np.float32)
    y = np.array([2.0, 2.0, 0.0, 0.0, 2.0, 2.0, 0.0, np.inf, -np.inf], np.float32)
    with np.errstate(all="ignore"):
        expected = x / y
    actual = np.asarray(jax.jit(numerics.divide)(x, y))
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(
        np.signbit(actual[~np.isnan(expected)]), np.signbit(expected[~np.isnan(expected)])
    )


def test_sigmoid_and_softplus_extremes_and_masked_attention():
    x = np.array([-np.inf, -80, 0, 80, np.inf, np.nan], np.float32)
    np.testing.assert_allclose(
        np.asarray(jax.jit(numerics.sigmoid)(x)), [0, np.exp(-80), 0.5, 1, 1, np.nan], rtol=5e-7
    )
    np.testing.assert_allclose(
        np.asarray(jax.jit(numerics.softplus)(x)),
        [0, np.exp(-80), np.log(2), 80, np.inf, np.nan],
        rtol=5e-7,
    )
    result = np.asarray(jax.jit(numerics.softmax)(jnp.array([[0.0, -jnp.inf, 0.0]])))
    np.testing.assert_array_equal(result, [[0.5, 0, 0.5]])


def test_elementary_gate_rejects_small_systematic_drift(monkeypatch):
    original = numerics.sigmoid
    monkeypatch.setattr(numerics, "sigmoid", lambda x: original(x) * (1 + 1e-5))
    diagnostics = {}
    with pytest.raises(AssertionError, match="sigmoid"):
        validate_numerics(Mesh(np.asarray(jax.devices()[:1]), ("expert",)), diagnostics)
    assert diagnostics["operations"][-1]["operation"] == "sigmoid"


def test_exp_accuracy_is_explicit_in_compiler_input():
    graph = jax.make_jaxpr(numerics.sigmoid)(jnp.zeros(8))
    exps = [equation for equation in graph.jaxpr.eqns if equation.primitive.name == "exp"]
    assert len(exps) == 1
    assert exps[0].params["accuracy"] == jax.lax.AccuracyMode.HIGHEST
