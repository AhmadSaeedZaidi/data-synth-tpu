"""Independent arithmetic, sparse selection and gate regression tests."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh
from qwen38_tpu.attention_cache import quantize, read_quantized, select_tokens, write_quantized
from qwen38_tpu.compute import native_dot_proof, numpy_q8_matmul, pack_q8, q8_matmul
from qwen38_tpu.config import ModelConfig
from qwen38_tpu.validation import tiny_model, validate_long_context


@pytest.mark.parametrize("policy", ["bf16", "int8_q8"])
def test_sparse_q8_official_full_model(policy):
    result = validate_long_context(
        Mesh(np.asarray(jax.devices()[:1]), ("expert",)), "q8", expert_compute=policy
    )
    assert result["prompt_tokens"] > result["token_budget"]
    assert result["max_logit_error"] < result["logit_tolerance"]


def test_actual_bf16_full_model_and_native_dot_types():
    result = validate_long_context(
        Mesh(np.asarray(jax.devices()[:1]), ("expert",)), "q8", dtype=jnp.bfloat16
    )
    assert result["max_rms_error"] <= result["bf16_rms_budget"] == 0.010
    assert result["logit_tolerance"] == 0.025
    assert native_dot_proof()["int8_dots"] == [dict(inputs=["int8", "int8"], output="int32")]


def test_single_row_compute_calibration_exercises_nonzero_activations(monkeypatch):
    from qwen38_tpu import compute

    seen = []
    original = compute.numpy_q8_matmul

    def oracle(x, packed):
        seen.append(np.asarray(x).copy())
        return original(x, packed)

    monkeypatch.setattr(compute, "numpy_q8_matmul", oracle)
    result = compute.choose_policy([1], repeats=1, shapes=((32, 32),))
    assert result["status"] == "PASS"
    assert len(seen) == 1 and np.count_nonzero(seen[0]) > 0


def test_full_gate_catches_silent_bf16_fallback_in_integer_policy(monkeypatch):
    from qwen38_tpu import validation

    original = validation.expert_parallel

    def wrong(mesh, **kwargs):
        kwargs["compute"] = "bf16"
        return original(mesh, **kwargs)

    monkeypatch.setattr(validation, "expert_parallel", wrong)
    with pytest.raises(AssertionError, match="prefill token"):
        validation.validate_long_context(
            Mesh(np.asarray(jax.devices()[:1]), ("expert",)), "q8", expert_compute="int8_q8"
        )


def test_golden_checksum_gate_rejects_changed_bytes(monkeypatch):
    from pathlib import Path

    original = Path.read_bytes

    def changed(path):
        result = original(path)
        return result + b"changed" if path.name == "long-reference.npz" else result

    monkeypatch.setattr(Path, "read_bytes", changed)
    with pytest.raises(ValueError, match="checksum"):
        tiny_model(Mesh(np.asarray(jax.devices()[:1]), ("expert",)), "long-reference")


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_integer_dot_independent_oracle_outliers_zero_and_overflow(dtype):
    rng = np.random.default_rng(837)
    x = rng.normal(size=(8, 256)).astype(np.float32)
    x[0] = 0
    x[1, ::32] *= 20
    x[2] = 127
    raw = pack_q8(rng.normal(size=(64, 256)).astype(np.float32))
    token = jnp.asarray(x, dtype)
    expected = numpy_q8_matmul(np.asarray(token).astype(np.float32), raw)
    actual = np.asarray(jax.jit(q8_matmul)(token, jnp.asarray(raw))).astype(np.float32)
    expected = np.asarray(jnp.asarray(expected, dtype)).astype(np.float32)
    np.testing.assert_allclose(
        actual,
        expected,
        atol=0.0005 if dtype == jnp.float32 else 0.5,
        rtol=2e-6 if dtype == jnp.float32 else 0.004,
    )
    np.testing.assert_array_equal(actual[0], 0)


def test_real_head_q8_cache_independent_oracle_and_inactive_scales():
    c = ModelConfig(max_context=8, kv_cache_dtype="q8")
    rng = np.random.default_rng(81)
    source = jnp.asarray(rng.normal(size=(2, 2, 256)), jnp.bfloat16)
    cache = {name: jnp.zeros((2, 8, 512), jnp.int8) for name in ("k", "v")}
    cache.update({name: jnp.zeros((32, 8), jnp.bfloat16) for name in ("k_scale", "v_scale")})
    updated = write_quantized(cache, source, source, jnp.array([3, 7]), jnp.array([True, False]), c)
    actual = np.asarray(read_quantized(updated, "k", jnp.array([[3], [7]]), c, jnp.float32))[:, 0]
    grouped = np.asarray(source[0]).astype(np.float32).reshape(2, 8, 32)
    scale = np.asarray(jnp.asarray(np.max(abs(grouped), -1) / 127, jnp.bfloat16)).astype(np.float32)
    codes = np.clip(np.rint(grouped / np.where(scale == 0, 1, scale)[..., None]), -127, 127)
    np.testing.assert_array_equal(actual[0], (codes * scale[..., None]).reshape(2, 256))
    np.testing.assert_array_equal(actual[1], 0)
    for name in ("k_scale", "v_scale"):
        np.testing.assert_array_equal(np.asarray(updated[name])[16:], 0)
    zero, scale = quantize(jnp.zeros((2, 2, 256)), 32)
    np.testing.assert_array_equal(zero, 0)
    np.testing.assert_array_equal(scale, 0)


def test_sparse_selection_matches_independent_pooling_and_mrope_including_ties():
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("expert",))
    engine, _, _ = tiny_model(mesh, "long-reference")
    c, w = engine.config, engine.layers[3]["attn"]
    engine.reset(2)
    state = engine.state["layers"][3]
    rng = np.random.default_rng(33)
    inputs = rng.normal(size=(2, 67, c.hidden)).astype(np.float32)
    raw = inputs @ np.asarray(w["index_k"]).T

    def rms(x, gamma):
        return x / np.sqrt(np.mean(x * x, -1, keepdims=True) + c.eps) * np.asarray(gamma)

    def rope(x, pos):
        half = c.rotary_dim // 2
        axis = np.array([0, 1, 2, 0])
        angle = pos[:, axis] * c.rope_theta ** (-np.arange(half) * 2 / c.rotary_dim)
        cs, sn = np.cos(angle)[:, None], np.sin(angle)[:, None]
        a, b = x[..., :half], x[..., half : 2 * half]
        return np.concatenate((a * cs - b * sn, b * cs + a * sn), -1)

    positions = np.stack((np.arange(67), np.arange(67) // 2, np.arange(67) % 7), -1)
    function = jax.jit(
        lambda x, cache, n, p: select_tokens(x, w, cache, n, p, jnp.array([True, True]), c)
    )
    for step in range(67):
        pos = np.broadcast_to(positions[step], (2, 3))
        indices, valid, state = function(inputs[:, step], state, jnp.full((2,), step), pos)
        n = (step + 1) // 4
        pooled = raw[:, : n * 4].reshape(2, n, 4, 8).mean(2)
        keys = rms(pooled, w["index_k_norm"])
        starts = np.broadcast_to(positions[: n * 4 : 4], (2, n, 3))
        if n:
            keys = rope(keys.reshape(2 * n, 1, 8), starts.reshape(2 * n, 3)).reshape(2, n, 8)
        query = rms(
            (inputs[:, step] @ np.asarray(w["index_q"]).T).reshape(2, 4, 8), w["index_q_norm"]
        )
        query = rope(query, pos)
        scores = np.maximum(np.einsum("bhd,btd->bth", query, keys), 0).sum(-1) / np.sqrt(8)
        for row in range(2):
            selected = np.asarray(indices)[row, np.asarray(valid)[row]]
            assert len(selected) == min(n, 8) * 4 + (step + 1) % 4
            assert len(np.unique(selected)) == len(selected)
            assert set(range(n * 4, step + 1)).issubset(set(selected))
            chosen = selected[selected < n * 4] // 4
            if n > 8:
                assert np.min(scores[row, chosen]) >= np.partition(scores[row], n - 8)[n - 8] - 2e-5
            assert np.max(selected) <= step
    # Degenerate ties permit any top blocks, but no future tokens or duplicates.
    tied = {**w, "index_q": jnp.zeros_like(w["index_q"])}
    indices, valid, _ = select_tokens(
        inputs[:, -1],
        tied,
        state,
        jnp.full((2,), 66),
        np.broadcast_to(positions[-1], (2, 3)),
        jnp.array([True, True]),
        c,
    )
    for row in range(2):
        selected = np.asarray(indices)[row, np.asarray(valid)[row]]
        assert len(np.unique(selected)) == 35 and np.max(selected) == 66
    engine.release_state()


def test_invalid_cache_configuration_and_long_memory_plan():
    from pathlib import Path

    from qwen38_tpu.checkpoint import Checkpoint

    with pytest.raises(ValueError, match="dtype"):
        ModelConfig(kv_cache_dtype="fp8")
    with pytest.raises(ValueError, match="budget"):
        ModelConfig(qsa_budget=2049)
    c = ModelConfig(max_context=8192, kv_cache_dtype="q8")
    assert c.state_bytes(8) < replace(c, kv_cache_dtype="bf16").state_bytes(8)
    cp = Checkpoint(Path(__file__).parents[1] / "qwen38_tpu/checkpoints/q6.json")
    assert cp.validate(c) == 1558
    assert cp.memory_plan(c, 8, hbm_bytes=8 * 16909332480)["fits"]


def test_eight_device_q8_integer_and_bf16_reference():
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    code = """import json,jax,numpy as np
from jax.sharding import Mesh
from qwen38_tpu.validation import validate_long_context,validate_q8_codec
m=Mesh(np.asarray(jax.devices()),('expert',))
codec=validate_q8_codec(m)
results=[validate_long_context(m,'q8'),validate_long_context(m,'q8',expert_compute='int8_q8'),
         validate_long_context(m,'q8',dtype=jax.numpy.bfloat16)]
print(json.dumps(dict(devices=m.size,results=results,codec=codec)))
"""
    env = {
        **os.environ,
        "XLA_FLAGS": "--xla_force_host_platform_device_count=8",
        "JAX_PLATFORMS": "cpu",
        "OPENBLAS_NUM_THREADS": "2",
        "OMP_NUM_THREADS": "2",
        "PYTHONPATH": str(root / "kaggle"),
    }
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    import json

    assert json.loads(result.stdout.strip().splitlines()[-1])["devices"] == 8
