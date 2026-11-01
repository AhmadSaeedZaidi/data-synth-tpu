"""Opt-in rounding primitive experiment; production keeps precision.py.

The v14 B4 trace gathers multiple uint32 rounding intermediates separately.
Expressing the precision boundary as one elementwise HLO operation lets the
partitioner communicate its result rather than separate branches of the integer
expression. This is a hypothesis until optimized TPU HLO and timing confirm it.

Custom partitioning was rejected locally: it requires one uniform mesh across
the fused program, unlike our attention mesh plus EP4/TP2 expert shard_map.
reduce_precision has no such custom-partition mesh requirement.
"""

import jax
import jax.numpy as jnp


def primitive_bf16(x):
    """Explicit 8-exponent/7-mantissa-bit rounding followed by BF16 storage.

    This is not an ordinary FP32 -> BF16 -> FP32 cast, which previously lost a
    required precision boundary during TPU fusion. Hardware admission must test
    both the resulting bits and consumers that convert the output back to FP32.
    """
    rounded = jax.lax.reduce_precision(x.astype(jnp.float32), exponent_bits=8, mantissa_bits=7)
    return rounded.astype(jnp.bfloat16)
