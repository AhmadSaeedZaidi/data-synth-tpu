"""Execute staged notebook startup and quota cleanup without a TPU or network."""

import importlib.util
import json
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from glm53_offload import diagnostics, monitoring, supervisor, tracking_check


@pytest.fixture
def staged(tmp_path):
    source = Path(__file__).resolve().parents[2] / "scripts/build_qwen38_notebook.py"
    spec = importlib.util.spec_from_file_location("qwen38_notebook_builder", source)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    root = builder.build(tmp_path / "staging")
    notebook = json.loads((root / "notebook/qwen38-v5e8.ipynb").read_text())
    inputs, working = tmp_path / "input", tmp_path / "working"
    inputs.mkdir()
    working.mkdir()
    shutil.copyfile(root / "dataset/qwen38-runtime.bundle", inputs / "qwen38-runtime.bundle")
    cells = [
        "".join(cell["source"])
        .replace("/kaggle/input", str(inputs))
        .replace("/kaggle/working", str(working))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]
    return cells, working


@pytest.fixture
def started(staged, monkeypatch):
    cells, working = staged
    env = {}
    # Bootstrap modifies sys.path just as Kaggle does; isolate that change.
    monkeypatch.setattr(sys, "path", list(sys.path))
    exec(cells[0], env)
    finished = []
    monitor = SimpleNamespace(finish=lambda code: finished.append(code))
    monkeypatch.setattr(tracking_check, "load_kaggle_key", lambda: {"status": "PASS"})
    monkeypatch.setattr(tracking_check, "require_live_tracking", lambda *args: {"status": "PASS"})
    monkeypatch.setattr(monitoring, "RunMonitor", lambda *args, **kwargs: monitor)
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0))
    # Exercise the notebook's timeout arguments without creating background SDK threads.
    monkeypatch.setattr(diagnostics, "bounded_call", lambda function, timeout: function())
    return cells, working, env, finished


def test_bootstrap_roundtrip_and_checksum_gate(staged, monkeypatch):
    cells, working = staged
    monkeypatch.setattr(sys, "path", list(sys.path))
    env = {}
    exec(cells[0], env)
    assert (working / "qwen38-runtime/qwen38_tpu/engine.py").is_file()
    assert env["CONFIG"]["quant"] == "q6"
    assert env["CONFIG"]["batch_sizes"] == [1, 2, 4, 8]
    bundle = env["_assets"]
    bundle.write_bytes(bundle.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="checksum differs"):
        exec(cells[0], {})


def test_full_evaluation_bundle_reaches_worker_configuration(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    bundle = root / "output/current/inputs/qwen38-evaluation.bundle"
    if not bundle.exists():
        pytest.skip("private evaluation bundle is staged locally")
    spec = importlib.util.spec_from_file_location(
        "full_evaluation_builder", root / "scripts/build_qwen38_notebook.py"
    )
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    stage = builder.build(
        tmp_path / "stage", evaluation_bundle=bundle, batch_sizes=[8, 16, 32], requests=64
    )
    document = json.loads((stage / "notebook/qwen38-v5e8.ipynb").read_text())
    working = tmp_path / "working"
    working.mkdir()
    code = "".join(next(c for c in document["cells"] if c["cell_type"] == "code")["source"])
    code = code.replace("/kaggle/input", str(stage / "dataset")).replace(
        "/kaggle/working", str(working)
    )
    monkeypatch.setattr(sys, "path", list(sys.path))
    env = {}
    exec(code, env)
    config = env["CONFIG"]
    assert Path(config["evaluation_bundle"]).read_bytes() == bundle.read_bytes()
    assert config["evaluation_expected_cases"] == 49 and config["evaluation_batch_size"] == 8
    assert config["batch_sizes"] == [8, 16, 32] and config["requests"] == 64
    assert config["expert_parallel"] == "auto" and config["fuse_layers"] is True
    assert config["expert_compute"] == "bf16" and config["gdn_prefill"] == "serial"
    assert config["stop_on_evaluation_collapse"] is True
    assert config["profile_decode"] is True


def test_private_authentication_rejects_public_or_tracked_notebooks(tmp_path, monkeypatch):
    import stat

    source = Path(__file__).resolve().parents[2] / "scripts/build_qwen38_notebook.py"
    spec = importlib.util.spec_from_file_location("qwen38_auth_builder", source)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    key_file = tmp_path / "task-key"
    key_file.write_text("task-test-key")
    notebook_path = tmp_path / "output/notebook/trial.ipynb"
    notebook_path.parent.mkdir(parents=True)
    notebook_path.touch()
    metadata_path = notebook_path.with_name("kernel-metadata.json")
    metadata_path.write_text(json.dumps({"is_private": False}))
    document = {"cells": [], "metadata": {}}
    with pytest.raises(ValueError, match="private notebook"):
        builder.embed_tracking_key(document, notebook_path, key_file)
    metadata_path.write_text(json.dumps({"is_private": True}))
    builder.embed_tracking_key(document, notebook_path, key_file)
    assert stat.S_IMODE(notebook_path.stat().st_mode) == 0o600
    assert "task-test-key" in notebook_path.read_text()
    with pytest.raises(ValueError, match="ignored output"):
        builder.embed_tracking_key(document, tmp_path / "tracked.ipynb", key_file)
    assert not (tmp_path / "tracked.ipynb").exists()


def test_secret_failure_stops_before_install_and_preserves_evidence(started, monkeypatch):
    cells, working, env, finished = started
    installed = []
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: installed.append(True))

    def unavailable():
        raise tracking_check.TrackingCheckError("secret unavailable")

    monkeypatch.setattr(tracking_check, "load_kaggle_key", unavailable)
    with pytest.raises(tracking_check.TrackingCheckError):
        exec(cells[1], env)
    assert not installed and not finished
    report = json.loads((working / "qwen38-trial/report.json").read_text())
    assert report == {"status": "ERROR", "phase": "startup", "error_type": "TrackingCheckError"}
    with zipfile.ZipFile(working / "qwen38-evidence.zip") as archive:
        assert "report.json" in archive.namelist()
        assert "config.json" in archive.namelist()


def test_tracking_failure_closes_monitor_and_preserves_evidence(started, monkeypatch):
    cells, working, env, finished = started

    def missing(*args):
        raise tracking_check.TrackingCheckError("remote records missing")

    monkeypatch.setattr(tracking_check, "require_live_tracking", missing)
    with pytest.raises(tracking_check.TrackingCheckError):
        exec(cells[1], env)
    assert finished == [1]
    with zipfile.ZipFile(working / "qwen38-evidence.zip") as archive:
        assert "dependency-setup.log" in archive.namelist()


def test_expired_setup_never_spawns_worker(started, monkeypatch):
    cells, working, env, finished = started
    exec(cells[1], env)
    env["_started"] = time.monotonic() - env["CONFIG"]["deadline_seconds"]
    spawned = []
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: spawned.append(True))
    with pytest.raises(TimeoutError, match="budget exhausted"):
        exec(cells[2], env)
    assert not spawned and finished == [1]
    assert (working / "qwen38-evidence.zip").is_file()


def test_missing_worker_report_is_failure_even_with_zero_exit(started, monkeypatch):
    cells, working, env, finished = started
    exec(cells[1], env)
    process = SimpleNamespace(returncode=0, poll=lambda: 0)
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(supervisor, "supervise", lambda *args, **kwargs: {"status": "EXITED"})
    with pytest.raises(RuntimeError, match="trial failed"):
        exec(cells[2], env)
    assert finished == [1]
    report = json.loads((working / "qwen38-trial/report.json").read_text())
    assert report["error_type"] == "MissingWorkerReport"
    with zipfile.ZipFile(working / "qwen38-evidence.zip") as archive:
        assert "worker.log" in archive.namelist()


def test_probe_selection_reaches_worker_before_launch(started, monkeypatch):
    from qwen38_tpu import experiments

    cells, working, env, finished = started
    exec(cells[1], env)
    env["CONFIG"]["optimization_probes"] = True
    calls = []

    def probe(config, directory, monitor, *, remaining_seconds):
        assert 0 < remaining_seconds <= 21600
        config["kv_scale_gather"] = "rowwise"
        calls.append(True)
        return {"kv_gather": {"selected": "rowwise"}}

    def launch(*args, **kwargs):
        assert calls == [True]
        config = json.loads((working / "qwen38-trial/config.json").read_text())
        assert config["kv_scale_gather"] == "rowwise"
        (working / "qwen38-trial/report.json").write_text(json.dumps({"status": "PASS"}))
        return SimpleNamespace(returncode=0, poll=lambda: 0)

    monkeypatch.setattr(experiments, "run_optimization_probes", probe)
    monkeypatch.setattr(subprocess, "Popen", launch)
    monkeypatch.setattr(supervisor, "supervise", lambda *args, **kwargs: {"status": "EXITED"})
    exec(cells[2], env)
    assert finished == [0]


@pytest.mark.parametrize("status", ["passed", "failed", "timed_out"])
def test_bundled_rounding_failure_still_runs_full_model_and_keeps_evidence(
    started, monkeypatch, status
):
    from qwen38_tpu import experiments

    cells, working, env, finished = started
    exec(cells[1], env)
    env["CONFIG"]["rounding_probe"] = True
    calls = []

    def probe(config, directory, monitor, *, remaining_seconds):
        assert remaining_seconds > 14400
        result = dict(status=status, production_enabled=False)
        (directory / "rounding-probe.json").write_text(json.dumps(result))
        calls.append("probe")
        return result

    def launch(*args, **kwargs):
        assert calls == ["probe"]
        calls.append("full_model")
        (working / "qwen38-trial/report.json").write_text(json.dumps({"status": "PASS"}))
        return SimpleNamespace(returncode=0, poll=lambda: 0)

    monkeypatch.setattr(experiments, "run_rounding_probe", probe)
    monkeypatch.setattr(subprocess, "Popen", launch)
    monkeypatch.setattr(supervisor, "supervise", lambda *args, **kwargs: {"status": "EXITED"})
    exec(cells[2], env)
    assert calls == ["probe", "full_model"] and finished == [0]
    report = json.loads((working / "qwen38-trial/report.json").read_text())
    assert report["status"] == "PASS" and report["rounding_preflight"]["status"] == status
    with zipfile.ZipFile(working / "qwen38-evidence.zip") as archive:
        assert "rounding-probe.json" in archive.namelist()


def test_watchdog_status_and_tracking_timeout_preserve_worker_results(started, monkeypatch):
    cells, working, env, finished = started
    exec(cells[1], env)
    report_path = working / "qwen38-trial/report.json"
    report_path.write_text(json.dumps({"status": "RUNNING", "batch_results": [{"batch": 1}]}))
    process = SimpleNamespace(returncode=-15, poll=lambda: -15)
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: process)
    budgets = []

    def timed_out(*args, **kwargs):
        budgets.append(kwargs["deadline_seconds"])
        return {"status": "TIMEOUT"}

    def hung(function, timeout):
        assert timeout == 25
        raise TimeoutError("SDK blocked")

    monkeypatch.setattr(supervisor, "supervise", timed_out)
    # This is the bounded function captured by the generated cell.
    env["bounded_call"] = hung
    with pytest.raises(RuntimeError, match="trial failed"):
        exec(cells[2], env)
    assert 0 < budgets[0] <= 21600 - 60
    report = json.loads(report_path.read_text())
    assert report["status"] == "TIMEOUT" and report["batch_results"] == [{"batch": 1}]
    assert not finished
    with zipfile.ZipFile(working / "qwen38-evidence.zip") as archive:
        cleanup = json.loads(archive.read("tracking-cleanup.json"))
        assert cleanup == {"status": "ERROR", "error_type": "TimeoutError"}
