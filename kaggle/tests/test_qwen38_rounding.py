"""Independent rounding and distributed-consumer checks for the opt-in probe."""

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from qwen38_tpu.precision import compute_cast
from qwen38_tpu.rounding_candidate import primitive_bf16


def assert_bf16_bits(actual, expected):
    actual = np.asarray(actual)
    assert actual.dtype == expected.dtype == np.dtype(ml_dtypes.bfloat16)
    nan = np.isnan(expected.astype(np.float32))
    np.testing.assert_array_equal(np.isnan(actual.astype(np.float32)), nan)
    np.testing.assert_array_equal(actual[~nan].view(np.uint16), expected[~nan].view(np.uint16))


def test_every_bf16_bin_with_values_below_at_and_above_halfway():
    high = np.arange(65536, dtype=np.uint32)[:, None] << 16
    low = np.array([0, 0x7FFF, 0x8000, 0x8001], np.uint32)
    source = (high | low).reshape(-1).view(np.float32)
    with np.errstate(invalid="ignore", over="ignore"):
        expected = source.astype(ml_dtypes.bfloat16)
    assert_bf16_bits(jax.jit(primitive_bf16)(source), expected)


@pytest.mark.parametrize("batch", [1, 2, 4, 8])
@pytest.mark.parametrize("replicate_output", [False, True])
def test_ep4_tp2_sharding_rounding_and_replicated_consumers(batch, replicate_output):
    mesh = Mesh(np.asarray(jax.devices()).reshape(4, 2), ("expert", "tensor"))
    shard = NamedSharding(mesh, P(None, None, ("expert", "tensor")))
    replica = NamedSharding(mesh, P())
    source = np.random.default_rng(19).normal(size=(batch, 4, 2560)).astype(np.float32)
    source[:, 0, :4] = [1.00390625, -1.00390625, 1.01171875, -1.01171875]
    value = jax.device_put(source, shard)
    fn = jax.jit(
        primitive_bf16, in_shardings=shard, out_shardings=replica if replicate_output else shard
    )
    output = fn(value)
    assert output.sharding == (replica if replicate_output else shard)
    assert_bf16_bits(output, source.astype(ml_dtypes.bfloat16))


def test_fp32_consumer_cannot_erase_bf16_rounding():
    source = np.array([1.00390625, 1.01171875, -1.00390625, -1.01171875], np.float32)
    actual = np.asarray(jax.jit(lambda x: primitive_bf16(x).astype(jnp.float32) * 127)(source))
    expected = source.astype(ml_dtypes.bfloat16).astype(np.float32) * 127
    np.testing.assert_array_equal(actual, expected)
    assert not np.array_equal(actual, source * 127)


def test_rounding_then_stream_reduction_matches_existing_numerics():
    mesh = Mesh(np.asarray(jax.devices()).reshape(4, 2), ("expert", "tensor"))
    shard = NamedSharding(mesh, P(None, None, ("expert", "tensor")))
    replica = NamedSharding(mesh, P())
    source = jax.device_put(np.random.default_rng(5).normal(size=(4, 4, 2560)), shard)
    baseline = jax.jit(
        lambda x: jnp.mean(compute_cast(x, jnp.bfloat16).astype(jnp.float32), axis=1),
        in_shardings=shard,
        out_shardings=replica,
    )
    candidate = jax.jit(
        lambda x: jnp.mean(primitive_bf16(x).astype(jnp.float32), axis=1),
        in_shardings=shard,
        out_shardings=replica,
    )
    np.testing.assert_array_equal(np.asarray(candidate(source)), np.asarray(baseline(source)))


def test_model_policy_is_unchanged():
    # Keep the production call graph independent of this unmeasured experiment.
    graph = str(jax.make_jaxpr(lambda x: compute_cast(x, jnp.bfloat16))(jnp.ones(8)))
    assert "reduce_precision" not in graph


def test_compiler_input_has_an_explicit_rounding_primitive():
    graph = str(jax.make_jaxpr(primitive_bf16)(jnp.ones(8)))
    assert "reduce_precision[exponent_bits=8 mantissa_bits=7]" in graph


def test_candidate_in_full_bf16_vision_q8_model_against_independent_reference(monkeypatch):
    from qwen38_tpu import attention_cache, compute, experts, grouped_experts, math, vision
    from qwen38_tpu.validation import validate_long_context

    traces = []

    def cast(x, dtype):
        if dtype == jnp.bfloat16:
            traces.append(x.shape)
            return primitive_bf16(x)
        return x.astype(dtype)

    # Substitute only while constructing/tracing a fresh tiny engine in this
    # test. No process-global runtime policy or production default is changed.
    for module in (attention_cache, compute, experts, grouped_experts, math, vision):
        monkeypatch.setattr(module, "compute_cast", cast)
    result = validate_long_context(
        Mesh(np.asarray(jax.devices()), ("expert",)),
        "q8",
        dtype=jnp.bfloat16,
        expert_group_size=4,
        expert_tensor_parallel=2,
    )
    assert traces  # Prevent a cached baseline graph from silently passing.
    assert result["max_rms_error"] <= result["bf16_rms_budget"] == 0.010
    assert result["max_logit_error"] <= result["logit_tolerance"] == 0.025
