"""Catch the v3 TPU rounding failure and preserve every independent gate."""

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh
from qwen38_tpu.attention_cache import rounded_bf16_float32
from qwen38_tpu.gates import NumericalGateError, run_gates
from qwen38_tpu.validation import validate_long_context, validate_q8_codec


def test_explicit_rounding_matches_independent_bf16_halfway_ties_and_special_values():
    rng = np.random.default_rng(49)
    random = rng.normal(size=1024).astype(np.float32)
    ties = np.array(
        [0.0, -0.0, 1.00390625, -1.00390625, 1.01171875, -1.01171875, np.inf, -np.inf, np.nan],
        np.float32,
    )
    source = np.concatenate((random, ties))
    expected = source.astype(np.dtype(jnp.bfloat16)).astype(np.float32)
    actual = np.asarray(jax.jit(rounded_bf16_float32)(source))
    np.testing.assert_array_equal(
        actual[~np.isnan(source)].view(np.uint32), expected[~np.isnan(source)].view(np.uint32)
    )
    assert np.isnan(actual[np.isnan(source)]).all()


def test_compiled_q8_codes_use_stored_scale_at_tiny_and_real_head_widths():
    result = validate_q8_codec(Mesh(np.asarray(jax.devices()[:1]), ("expert",)))
    assert len(result["codec_shapes"]) == 4
    assert all(x["code_mismatches"] == x["scale_mismatches"] == 0 for x in result["codec_shapes"])
    assert all(x["unrounded_divisor_changes"] > 0 for x in result["codec_shapes"])


def unrounded_divisor(x, group):
    grouped = x.astype(jnp.float32).reshape(*x.shape[:-1], -1, group)
    full = jnp.max(jnp.abs(grouped), -1) / 127
    codes = jnp.clip(jnp.rint(grouped / jnp.where(full == 0, 1, full)[..., None]), -127, 127)
    return codes.astype(jnp.int8).reshape(x.shape), full.astype(jnp.bfloat16)


def test_codec_rejects_v3_unrounded_divisor(monkeypatch):
    from qwen38_tpu import validation

    monkeypatch.setattr(validation, "quantize", unrounded_divisor)
    diagnostics = {}
    with pytest.raises(AssertionError):
        validate_q8_codec(Mesh(np.asarray(jax.devices()[:1]), ("expert",)), diagnostics)
    assert diagnostics["codec_shapes"][0]["code_mismatches"] > 0


def test_full_model_rejects_v3_fingerprint_and_records_cache_codes(monkeypatch):
    from qwen38_tpu import attention_cache

    monkeypatch.setattr(attention_cache, "quantize", unrounded_divisor)
    diagnostics = {}
    with pytest.raises(AssertionError, match="prefill token 0"):
        validate_long_context(
            Mesh(np.asarray(jax.devices()[:1]), ("expert",)), "q8", diagnostics=diagnostics
        )
    comparison = diagnostics["last_comparison"]
    assert comparison["absolute_tolerance"] == 4e-5
    assert 0.00060 < comparison["max_absolute_error"] < 0.00063
    observed = np.array(
        [0.07165709883, -0.48046886921, -0.31754577160, 0.23749953508, 0.05955734104]
    )
    np.testing.assert_allclose(
        diagnostics["first_token"]["actual_logits"][0][:5], observed, atol=3e-6, rtol=0
    )
    cache = diagnostics["first_token"]["q8_cache"]["3"]
    assert {"codes", "scales", "reconstructed"} == set(cache["v"])
    failed = diagnostics["failure_q8_cache"]["3"]
    assert failed["token_start"] == 0 and failed["token_stop"] == 1
    assert failed["v"]["reference_mismatches"] > 0
    assert diagnostics["comparisons"][0]["label"] == "q8 serial prefill token 0"
    json.dumps(diagnostics)  # Failure evidence must survive report serialization.


def test_gate_failures_preserve_evidence_and_run_remaining_independent_checks(tmp_path):
    seen = []

    def failed(metrics):
        metrics["first_token"] = {"codes": [1, 2], "scales": [0.5]}
        seen.append("failed")
        raise AssertionError("wrong cache")

    def passed(metrics):
        seen.append("passed")
        return dict(status="PASS")

    report = dict(status="RUNNING")
    with pytest.raises(NumericalGateError, match="failed"):
        run_gates([("failed", failed), ("passed", passed)], report, tmp_path, lambda **kwargs: None)
    assert seen == ["failed", "passed"]
    stored = json.loads((tmp_path / "report.json").read_text())
    assert stored["gate_failures"] == ["failed"]
    assert stored["failed"]["first_token"]["codes"] == [1, 2]
    assert stored["passed"]["status"] == "PASS"


def test_gate_missing_pass_is_rejected(tmp_path):
    with pytest.raises(NumericalGateError):
        run_gates([("missing", lambda _: {})], {}, tmp_path, lambda **kw: None)
