"""Independent FP64 host oracle for the v5e elementary arithmetic contract."""

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from . import numerics


def validate_numerics(mesh, diagnostics=None):
    rng = np.random.default_rng(506)
    x = np.linspace(-30, 30, 2049, dtype=np.float32)
    e = np.exp(-np.abs(x.astype(np.float64)))
    positive_exp = np.exp(x.astype(np.float64))
    sigmoid = np.where(x >= 0, 1, e) / (1 + e)
    softplus = np.maximum(x, 0) + np.log1p(e)
    denominator = np.exp2(rng.uniform(-16, 16, 2049)).astype(np.float32)
    numerator = (rng.uniform(-128, 128, 2049) * denominator).astype(np.float32)
    quotient = numerator.astype(np.float64) / denominator.astype(np.float64)
    scores = rng.uniform(-12, 12, (8, 32)).astype(np.float32)
    scores[:, :3] = -np.inf
    # The subtraction is an ordinary FP32 model operation; evaluate the
    # nonlinear part in FP64 from those exact rounded subtraction results.
    shifted = scores - np.max(scores, -1, keepdims=True)
    probabilities = np.exp(shifted.astype(np.float64))
    probabilities /= np.sum(probabilities, -1, keepdims=True)
    checks = (
        ("exp", numerics.exp, (x,), positive_exp, jnp.exp),
        ("sigmoid", numerics.sigmoid, (x,), sigmoid, jax.nn.sigmoid),
        ("softplus", numerics.softplus, (x,), softplus, jax.nn.softplus),
        ("divide", numerics.divide, (numerator, denominator), quotient, jnp.divide),
        ("softmax", numerics.softmax, (scores,), probabilities, jax.nn.softmax),
    )
    results = []
    replica = NamedSharding(mesh, P())
    for name, function, inputs, expected, default in checks:
        inputs = tuple(jax.device_put(value, replica) for value in inputs)
        actual = np.asarray(jax.jit(function)(*inputs)).astype(np.float64)
        baseline = np.asarray(jax.jit(default)(*inputs)).astype(np.float64)
        error, original_error = np.abs(actual - expected), np.abs(baseline - expected)
        # A per-element relative bound, including tiny exp/sigmoid tails.
        # The additive floor is only for exactly zero reference elements.
        bound = 5e-7 * np.abs(expected) + 1e-30
        result = dict(
            operation=name,
            elements=int(expected.size),
            max_absolute_error=float(np.max(error)),
            max_relative_error=float(np.max(error / np.maximum(np.abs(expected), 1e-30))),
            default_max_relative_error=float(
                np.max(original_error / np.maximum(np.abs(expected), 1e-30))
            ),
            relative_budget=5e-7,
        )
        results.append(result)
        if diagnostics is not None:
            diagnostics["operations"] = results
        if not np.all(np.isfinite(actual)) or np.any(error > bound):
            worst = np.unravel_index(np.argmax(error / bound), expected.shape)
            result["worst"] = dict(
                index=[int(i) for i in worst],
                actual=float(actual[worst]),
                expected=float(expected[worst]),
            )
            raise AssertionError(f"FP32 {name} exceeds independent 5e-7 relative budget")
    return dict(
        status="PASS",
        operations=results,
        independent_reference="numpy_float64",
        exp_accuracy="HIGHEST",
        division="one_residual_correction",
        softplus="bounded_series",
    )
