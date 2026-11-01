"""Incomplete probes cannot select a kernel; selected kernels pass model goldens."""

import copy
import json
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh
from qwen38_tpu.config import ModelConfig
from qwen38_tpu.experiments import run_probe, select_gather
from qwen38_tpu.validation import validate_long_context


def measured_report():
    return dict(
        status="passed",
        cases=[
            dict(
                batch=b,
                capacity=8192,
                selected=2051,
                scope=scope,
                pattern=pattern,
                methods={
                    m: dict(
                        correctness="exact",
                        latency_ms=dict(median=10 if m == "indexed" else 6),
                        compiler_memory=dict(temp_size_in_bytes=1024),
                    )
                    for m in ("indexed", "rowwise")
                },
            )
            for b in (1, 2, 4, 8)
            for scope in ("scales", "complete_read")
            for pattern in ("uniform", "short_prefix")
        ],
    )


def test_gather_requires_full_read_win_and_complete_exact_coverage():
    report = measured_report()
    assert select_gather(report)["selected"] == "rowwise"
    for missing in range(len(report["cases"])):
        bad = copy.deepcopy(report)
        bad["cases"].pop(missing)
        assert select_gather(bad)["selected"] == "indexed"
    duplicate = copy.deepcopy(report)
    duplicate["cases"].append(duplicate["cases"][0])
    assert select_gather(duplicate)["selected"] == "indexed"
    for row in report["cases"]:
        if row["scope"] == "complete_read":
            row["methods"]["rowwise"]["latency_ms"]["median"] = 12
    assert select_gather(report)["selected"] == "indexed"


@pytest.mark.parametrize("fault", ["correctness", "nan", "memory", "regression", "failed"])
def test_gather_rejects_bad_measurements(fault):
    report = measured_report()
    row = next(r for r in report["cases"] if r["scope"] == "complete_read")
    if fault == "failed":
        report["status"] = "failed"
    elif fault == "correctness":
        row["methods"]["rowwise"]["correctness"] = "inexact"
    elif fault == "nan":
        row["methods"]["rowwise"]["latency_ms"]["median"] = float("nan")
    elif fault == "memory":
        row["methods"]["rowwise"]["compiler_memory"]["temp_size_in_bytes"] = 128 << 20
    else:
        row["methods"]["rowwise"]["latency_ms"]["median"] = 30
    assert select_gather(report)["selected"] == "indexed"


def test_probe_timeout_cannot_become_success(tmp_path):
    output = tmp_path / "probe.json"
    result = run_probe(
        [sys.executable, "-c", "import time; time.sleep(10)"], output, tmp_path, timeout=0.1
    )
    assert result["status"] == "timed_out"
    assert json.loads(output.read_text())["status"] == "timed_out"


def test_probe_zero_exit_without_report_is_failure(tmp_path):
    result = run_probe(
        [sys.executable, "-c", "pass"], tmp_path / "probe.json", tmp_path, timeout=10
    )
    assert result["status"] == "failed"


@pytest.mark.parametrize("status", ["passed", "failed", "timed_out"])
def test_bundled_rounding_preserves_failure_without_changing_model_policy(
    tmp_path, monkeypatch, status
):
    from qwen38_tpu import experiments

    config = dict(quality_first=True, expert_compute="bf16", rounding_probe=True)
    before = dict(config)

    def probe(command, output, directory, *, timeout, monitor):
        assert command[command.index("--timeout-seconds") + 1] == "600"
        assert timeout == 630
        return dict(status=status, cases=[])

    monkeypatch.setattr(experiments, "run_probe", probe)
    result = experiments.run_rounding_probe(config, tmp_path, None, remaining_seconds=20000)
    assert result["status"] == status and result["production_enabled"] is False
    assert config == before
    assert json.loads((tmp_path / "rounding-probe.json").read_text()) == result


def test_bundled_probe_reserves_time_for_the_model(tmp_path, monkeypatch):
    from qwen38_tpu import experiments

    def unexpected(*args, **kwargs):
        pytest.fail("No probe should launch with only four hours remaining")

    monkeypatch.setattr(experiments, "run_probe", unexpected)
    result = experiments.run_rounding_probe({}, tmp_path, None, remaining_seconds=14400)
    assert result["status"] == "skipped" and result["reason"] == "reserved_full_model_budget"


def test_bundled_probe_shortens_to_include_cleanup_and_model_budget(tmp_path, monkeypatch):
    from qwen38_tpu import experiments

    def probe(command, output, directory, *, timeout, monitor):
        assert command[command.index("--timeout-seconds") + 1] == "190"
        assert timeout + 20 + 60 + 4 * 3600 == 14700
        return dict(status="passed")

    monkeypatch.setattr(experiments, "run_probe", probe)
    result = experiments.run_rounding_probe({}, tmp_path, None, remaining_seconds=14700)
    assert result["status"] == "passed"


def test_unclean_device_shutdown_still_prevents_model_launch(tmp_path, monkeypatch):
    from qwen38_tpu import experiments

    def unsafe(*args, **kwargs):
        raise RuntimeError("Probe descendants still own the TPU")

    monkeypatch.setattr(experiments, "run_probe", unsafe)
    with pytest.raises(RuntimeError, match="still own the TPU"):
        experiments.run_rounding_probe({}, tmp_path, None, remaining_seconds=20000)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_configured_rowwise_kernel_passes_independent_model_goldens(dtype):
    mesh = Mesh(np.asarray(jax.devices()[:8]), ("expert",))
    result = validate_long_context(mesh, "q8", dtype=dtype, kv_scale_gather="rowwise")
    assert result["status"] == "PASS" and result["kv_scale_gather"] == "rowwise"
    with pytest.raises(ValueError, match="KV scale gather"):
        ModelConfig(kv_scale_gather="unknown")
