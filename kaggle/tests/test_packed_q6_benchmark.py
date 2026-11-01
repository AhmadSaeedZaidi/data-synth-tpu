"""A failed correctness gate or hung child must never become a speed result."""

import json
import sys

import numpy as np
import pytest
from packed_q6.benchmark import check_result, parse_args, save, supervise


@pytest.mark.parametrize("bad", [np.nan, np.inf, 1.0])
def test_gate_rejects_nonfinite_and_biased_outputs(bad):
    expected = np.zeros((1, 2), np.float32)
    actual = np.full_like(expected, bad)
    with pytest.raises((ValueError, AssertionError)):
        check_result(actual, expected)


def test_supervisor_kills_hung_native_work_and_preserves_last_progress(tmp_path):
    path = tmp_path / "result.json"
    save(path, {"status": "running", "phase": "compiling", "cases": [{"status": "passed"}]})
    assert supervise([sys.executable, "-c", "import time; time.sleep(5)"], path, 0.05) == 124
    report = json.loads(path.read_text())
    assert report["status"] == "timed_out"
    assert report["phase"] == "compiling"
    assert report["cases"] == [{"status": "passed"}]


@pytest.mark.parametrize("return_code", [0, 7])
def test_supervisor_rejects_crash_or_missing_report(tmp_path, return_code):
    path = tmp_path / "result.json"
    code = supervise([sys.executable, "-c", f"raise SystemExit({return_code})"], path, 5)
    assert code != 0
    assert json.loads(path.read_text())["status"] == "failed"


def test_benchmark_cannot_silently_expand_to_unbounded_work():
    with pytest.raises(SystemExit):
        parse_args(["--timeout-seconds", "901"])
    with pytest.raises(SystemExit):
        parse_args(["--batches", "1024"])
    with pytest.raises(SystemExit):
        parse_args(["--repetitions", "100000"])
