"""Submission must gate TPU work on live tracking and preserve failure evidence."""

import importlib.util
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from qwen38_tpu import rounding_session as session


@pytest.fixture
def isolated(monkeypatch):
    events = []
    monkeypatch.setattr(os, "environ", dict(os.environ))
    monkeypatch.setattr(session, "load_kaggle_key", lambda: events.append("key"))
    monkeypatch.setattr(session.importlib.metadata, "version", lambda _: "0.11.2")
    monkeypatch.setattr(session.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0))
    monitor = SimpleNamespace(
        run=SimpleNamespace(summary={}), finish=lambda code: events.append(code)
    )
    monkeypatch.setattr(session, "RunMonitor", lambda *a, **k: monitor)
    monkeypatch.setattr(session, "bounded_call", lambda fn, timeout: fn())
    monkeypatch.setattr(session.tracing, "configure", lambda *a: events.append("weave"))
    monkeypatch.setattr(session.tracing, "finish", lambda *a: events.append("finished"))

    @contextmanager
    def span(*args, **kwargs):
        yield {}

    monkeypatch.setattr(session.tracing, "span", span)

    def proof(*args):
        events.append("proof")
        return {"status": "PASS"}

    def probe(command, output, directory, *, timeout, monitor):
        assert "proof" in events
        assert int(command[command.index("--timeout-seconds") + 1]) <= 600
        assert timeout <= 630 and "--capture-trace" in command
        assert output.name == "report.json" and directory.name == "qwen38-trial"
        events.append("probe")
        (directory / "report.log").write_text("compiled and verified\n")
        return {"status": "passed", "cases": []}

    monkeypatch.setattr(session, "require_live_tracking", proof)
    monkeypatch.setattr(session, "run_probe", probe)
    return events


def test_live_readback_precedes_worker_and_reports_are_downloadable(tmp_path, isolated):
    directory = tmp_path / "qwen38-trial"
    report = session.run(directory, "digest")
    assert report["status"] == "passed" and report["tracking_readback"] == "PASS"
    assert json.loads((directory / "report.json").read_text()) == report
    assert not (directory / "worker.log").is_symlink()
    assert (directory / "worker.log").read_text() == "compiled and verified\n"
    assert json.loads((directory / "supervisor.json").read_text())["worker_cap_seconds"] == 600
    assert isolated.index("proof") < isolated.index("probe") and isolated[-1] == 0


def test_failed_tracking_never_starts_tpu_probe(tmp_path, isolated, monkeypatch):
    def fail(*args):
        raise RuntimeError("unacknowledged tracking")

    monkeypatch.setattr(session, "require_live_tracking", fail)
    result = session.run(tmp_path / "qwen38-trial", "digest")
    assert result["status"] == "failed" and result["phase"] == "tracking_startup"
    assert "probe" not in isolated and isolated[-1] == 1


def test_dependency_failure_is_preserved_before_monitor_and_worker(tmp_path, isolated, monkeypatch):
    monkeypatch.setattr(session.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1))
    result = session.run(tmp_path / "qwen38-trial", "digest")
    assert result["status"] == "failed" and result["phase"] == "dependencies"
    assert "proof" not in isolated and "probe" not in isolated


def test_tracked_notebook_bootstrap_executes_the_tracked_runner(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "rounding_builder", root / "scripts/build_qwen38_rounding_experiment.py"
    )
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    result = builder.stage(tmp_path / "staged", tracking=True)
    nb = json.loads((tmp_path / "staged/bf16-rounding-probe.ipynb").read_text())
    working = tmp_path / "working"
    working.mkdir()
    monkeypatch.setattr(sys, "path", list(sys.path))
    calls = []
    monkeypatch.setattr(
        session,
        "run",
        lambda directory, digest: (
            calls.append((directory, digest)) or {"status": "passed", "cases": []}
        ),
    )
    env = {}
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            exec("".join(cell["source"]).replace("/kaggle/working", str(working)), env)
    assert calls == [(working / "qwen38-trial", result["source_sha256"])]
    assert (env["source"] / "glm53_offload/tracking_check.py").is_file()
