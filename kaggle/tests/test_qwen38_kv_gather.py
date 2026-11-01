"""Independent gather values, time partitions, Q8 boundaries and attention state."""

from functools import partial

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from qwen38_tpu.attention_cache import (
    gather_scale_plane,
    read_quantized,
    step,
    write_quantized,
)
from qwen38_tpu.config import ModelConfig
from qwen38_tpu.validation import tiny_model


def scale_oracle(plane, indices, groups):
    result = np.empty((*indices.shape, groups), plane.dtype)
    for batch in range(len(indices)):
        for slot, time in enumerate(indices[batch]):
            time = int(time)
            if time < 0:
                time += plane.shape[1]
            time = min(max(time, 0), plane.shape[1] - 1)
            for group in range(groups):
                result[batch, slot, group] = plane[batch * groups + group, time]
    return result


@pytest.mark.parametrize("method", ["indexed", "rowwise"])
@pytest.mark.parametrize("batch,groups,time", [(1, 16, 8), (2, 4, 32), (8, 16, 64)])
def test_gather_matches_independent_index_oracle_and_preserves_bf16_bits(
    method, batch, groups, time
):
    rng = np.random.default_rng(629)
    plane = rng.normal(size=(batch * groups, time)).astype(ml_dtypes.bfloat16)
    plane[:, 0] = 0
    plane[:, 1] = -0.0
    indices = np.tile(np.array([0, 0, 1, time // 8, time - 1, -1, -time - 2, time + 3]), (batch, 1))
    mesh = Mesh(np.asarray(jax.devices()[:8]), ("time",))
    source = jax.device_put(plane, NamedSharding(mesh, P(None, "time")))
    actual = np.asarray(
        jax.jit(partial(gather_scale_plane, groups=groups, method=method))(
            source, jnp.asarray(indices)
        )
    )
    expected = scale_oracle(plane, indices, groups)
    np.testing.assert_array_equal(actual, expected)
    # A distributed gather may sum masked values and normalize -0 to +0 in
    # BOTH paths. Require candidate/baseline bit agreement, and independently
    # check every numerical value instead of imposing a new signed-zero rule.
    baseline = np.asarray(
        jax.jit(partial(gather_scale_plane, groups=groups, method="indexed"))(
            source, jnp.asarray(indices)
        )
    )
    np.testing.assert_array_equal(actual.view(np.uint16), baseline.view(np.uint16))
    np.testing.assert_array_equal(np.asarray(source).view(np.uint16), plane.view(np.uint16))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_complete_cache_read_matches_independent_q8_oracle_with_inactive_rows(dtype):
    c = ModelConfig(max_context=32, kv_cache_dtype="q8")
    rng = np.random.default_rng(710)
    batch, groups = 2, 16
    source = rng.normal(size=(batch, 2, 256)).astype(ml_dtypes.bfloat16)
    source[0, 0, :32] = 0
    cache = {name: jnp.zeros((batch, 32, 512), jnp.int8) for name in ("k", "v")}
    cache.update(
        {name: jnp.zeros((batch * groups, 32), jnp.bfloat16) for name in ("k_scale", "v_scale")}
    )
    updated = jax.jit(
        lambda cache: write_quantized(
            cache,
            jnp.asarray(source),
            jnp.asarray(source),
            jnp.array([31, 8]),
            jnp.array([True, False]),
            c,
        )
    )(cache)
    indices = jnp.array([[31, 0, 31, 1], [8, 0, 8, 31]])
    grouped = source[0].astype(np.float32).reshape(2, 8, 32)
    scale = (np.max(np.abs(grouped), -1) / 127).astype(ml_dtypes.bfloat16).astype(np.float32)
    codes = np.clip(np.rint(grouped / np.where(scale == 0, 1, scale)[..., None]), -127, 127)
    reconstructed = (codes * scale[..., None]).reshape(2, 256)
    expected = np.zeros((batch, 4, 2, 256), np.float32)
    expected[0, [0, 2]] = reconstructed
    expected = expected.astype(np.dtype(dtype))
    for method in ("indexed", "rowwise"):
        actual = np.asarray(
            jax.jit(
                lambda cache: read_quantized(cache, "k", indices, c, dtype, scale_gather=method)
            )(updated)
        )
        np.testing.assert_array_equal(actual, expected)
    for name in ("k", "v", "k_scale", "v_scale"):
        np.testing.assert_array_equal(np.asarray(updated[name]).reshape(batch, -1)[1], 0)


def test_attention_candidate_preserves_outputs_and_all_cache_state_across_steps():
    mesh = Mesh(np.asarray(jax.devices()[:8]), ("expert",))
    engine, _, _ = tiny_model(mesh, "long-reference", kv_cache_dtype="q8", dtype=jnp.bfloat16)
    engine.reset(2)
    c = engine.config
    weights = engine.layers[3]["attn"]
    states = {method: engine.state["layers"][3] for method in ("indexed", "rowwise")}
    functions = {
        method: jax.jit(
            lambda x, state, lengths, pos, active, method=method: step(
                x, weights, state, lengths, pos, active, c, scale_gather=method
            )
        )
        for method in states
    }
    rng = np.random.default_rng(38)
    for time in range(9):
        x = jnp.asarray(rng.normal(size=(2, c.hidden)), jnp.bfloat16)
        active = jnp.array([True, time < 5])
        lengths = jnp.array([time, min(time, 4)])
        positions = jnp.repeat(lengths[:, None], 3, axis=1)
        outputs = {}
        for method in states:
            outputs[method], states[method] = functions[method](
                x, states[method], lengths, positions, active
            )
        np.testing.assert_array_equal(
            np.asarray(outputs["rowwise"]), np.asarray(outputs["indexed"])
        )
        for actual, expected in zip(
            jax.tree.leaves(states["rowwise"]), jax.tree.leaves(states["indexed"]), strict=True
        ):
            np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def test_gather_rejects_invalid_contract():
    with pytest.raises(ValueError, match="Scale plane"):
        gather_scale_plane(jnp.zeros((16, 8)), jnp.zeros((2, 3), jnp.int32), 16)
    with pytest.raises(ValueError, match="method"):
        gather_scale_plane(jnp.zeros((16, 8)), jnp.zeros((1, 3), jnp.int32), 16, method="other")
