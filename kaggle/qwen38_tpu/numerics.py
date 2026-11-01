"""FP32 nonlinear arithmetic for quantization-sensitive TPU inference.

Matmul precision does not control TPU exp/logistic/reciprocal approximations.
Use accurate exp and correct the reciprocal-based quotient once. Softplus
uses a bounded log1p series instead of v5e's approximate log instruction.
No matrix operands or accumulator dtypes are changed here.
"""

import jax
import jax.numpy as jnp


def exp(x):
    return jax.lax.exp(x.astype(jnp.float32), accuracy=jax.lax.AccuracyMode.HIGHEST)


def divide(numerator, denominator):
    """One residual correction of TPU's approximate FP32 quotient.

    Barriers keep XLA from cancelling the correction back into a division.
    Model callers use finite, normal, nonzero denominators. Preserve ordinary
    division behavior for zeros, infinities, and NaNs as well.
    """
    numerator, denominator = (
        jnp.asarray(numerator, jnp.float32),
        jnp.asarray(denominator, jnp.float32),
    )
    inverse = jax.lax.optimization_barrier(jax.lax.reciprocal(denominator))
    estimate = jax.lax.optimization_barrier(numerator * inverse)
    residual = numerator - estimate * denominator
    corrected = estimate + residual * inverse
    valid = (
        jnp.isfinite(estimate)
        & jnp.isfinite(inverse)
        & jnp.isfinite(denominator)
        & (denominator != 0)
        & (numerator != 0)
    )
    return jnp.where(valid, corrected, numerator / denominator)


def sigmoid(x):
    x = x.astype(jnp.float32)
    e = exp(-jnp.abs(x))
    return divide(jnp.where(x >= 0, 1.0, e), 1.0 + e)


def softmax(x, axis=-1):
    x = x.astype(jnp.float32)
    e = exp(x - jax.lax.stop_gradient(jnp.max(x, axis=axis, keepdims=True)))
    return divide(e, jnp.sum(e, axis=axis, keepdims=True))


def softplus(x):
    """max(x,0) + log1p(exp(-abs(x))), with a uniformly bounded series.

    For u=exp(-abs(x)), z=u/(2+u) is in [0,1/3]. The eight-term
    2*atanh(z) expansion has truncation error below 1.1e-9, smaller than
    FP32 rounding near its maximum. This also avoids log(1+u) cancellation.
    """
    x = x.astype(jnp.float32)
    u = exp(-jnp.abs(x))
    z = divide(u, 2.0 + u)
    z2 = z * z
    polynomial = jnp.full_like(z, 1.0 / 15)
    for denominator in (13, 11, 9, 7, 5, 3, 1):
        polynomial = polynomial * z2 + 1.0 / denominator
    return jnp.maximum(x, 0) + 2 * z * polynomial
