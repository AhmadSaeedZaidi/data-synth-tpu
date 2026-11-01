"""Reject invalid probes and misleading speed results before TPU work."""

import json
import subprocess
import sys

import numpy as np
import pytest
from qwen38_tpu import rounding_benchmark as probe


@pytest.mark.parametrize("bad", [np.nan, np.inf, 0.01, -0.0])
def test_value_gate_rejects_nonfinite_drift_and_signed_zero(bad):
    with pytest.raises((ValueError, AssertionError)):
        probe.check_values(np.array([bad], np.float32), np.array([0], np.float32))


def test_collective_inventory_counts_definitions_not_uses():
    text = (
        "%all-gather.1 = f32[8] all-gather(%x), channel_id=1\n"
        "%add = f32[8] add(%all-gather.1, %x)\n"
        "%all-reduce-start.2 = (f32[8],f32[8]) all-reduce-start(%x)\n"
        "%all-reduce-done.2 = f32[8] all-reduce-done(%all-reduce-start.2)"
    )
    assert len(probe.collective_lines(text)) == 2


@pytest.mark.parametrize(
    "argv",
    [
        ["--batches", "128"],
        ["--repetitions", "1000"],
        ["--timeout-seconds", "901"],
        ["--correctness-only", "--capture-trace"],
    ],
)
def test_probe_bounds(argv):
    with pytest.raises(SystemExit):
        probe.parse_args(argv)


def test_cpu_cannot_be_reported_as_a_tpu_speed_result(tmp_path):
    args = probe.parse_args(["--output", str(tmp_path / "report.json")])
    assert probe.worker(args) == 1
    result = json.loads(args.output.read_text())
    assert result["status"] == "failed"
    assert "TPU devices" in result["error"]
    assert not result["cases"]


def test_watchdog_timeout_cannot_reuse_a_previous_success(monkeypatch, tmp_path):
    output = tmp_path / "report.json"
    output.write_text('{"status":"passed","cases":[{"old":true}]}')
    monkeypatch.setattr(sys, "argv", ["probe", "--output", str(output)])

    def timeout(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", timeout)
    assert probe.main() == 124
    result = json.loads(output.read_text())
    assert result["status"] == "timed_out" and not result["cases"]
