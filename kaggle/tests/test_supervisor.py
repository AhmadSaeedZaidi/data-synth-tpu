import json
import os
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from glm53_offload.diagnostics import (
    bounded_call,
    optional_device_trace,
    package_evidence,
    request_stack_dump,
)
from glm53_offload.supervisor import finalize_report, stop_worker, supervise


def test_optional_startup_call_cannot_block_launch_forever():
    entered, release = threading.Event(), threading.Event()

    def blocked():
        entered.set()
        release.wait(5)

    try:
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            bounded_call(blocked, timeout=0.05)
        assert entered.is_set()
        assert time.monotonic() - started < 1
    finally:
        release.set()


def test_finished_worker_is_not_misclassified_as_stalled(tmp_path):
    (tmp_path / "progress.json").write_text(json.dumps({"updated_at": 1}))
    process = SimpleNamespace(pid=123, poll=lambda: 0, returncode=0)
    result = supervise(
        process,
        tmp_path,
        SimpleNamespace(tick=lambda pid: None),
        deadline_seconds=1,
        idle_seconds=1,
    )
    assert result["status"] == "EXITED"
    assert result["returncode"] == 0


def test_delayed_tpu_shutdown_preserves_deadline_and_finalizes_partial_report(
    tmp_path, monkeypatch
):
    from glm53_offload import supervisor

    waits = []
    signals = []

    def pending(timeout):
        waits.append(timeout)
        raise subprocess.TimeoutExpired("worker", timeout)

    process = SimpleNamespace(
        pid=123,
        poll=lambda: None,
        returncode=None,
        wait=pending,
        terminate=lambda: signals.append("term"),
        kill=lambda: signals.append("kill"),
    )
    monkeypatch.setattr(supervisor, "request_stack_dump", lambda *args: False)
    result = supervise(
        process,
        tmp_path,
        SimpleNamespace(tick=lambda pid: None),
        deadline_seconds=0,
        idle_seconds=0,
    )
    assert result["status"] == "TIMEOUT" and result["termination"]["pending"] is True
    assert waits == [15, 10] and signals == ["term", "kill"]
    (tmp_path / "report.json").write_text(json.dumps(dict(status="RUNNING", evaluation=["saved"])))
    # Also recover the durable reason when the caller received no result.
    report = finalize_report(tmp_path, None, -15)
    assert report["status"] == "TIMEOUT" and report["evaluation"] == ["saved"]
    assert report["worker_returncode"] == -15


def test_slow_term_can_finish_in_grace_without_kill():
    process = SimpleNamespace(pid=123, poll=lambda: None, returncode=None)

    def finish(timeout):
        assert timeout == 15
        process.returncode = -15

    process.wait = finish
    process.terminate = lambda: None
    result = stop_worker(process)
    assert result == dict(terminate_sent=True, kill_sent=False, pending=False, returncode=-15)


def test_tracking_hang_does_not_disable_deadline(tmp_path):
    entered, release = threading.Event(), threading.Event()

    def blocked(pid):
        entered.set()
        release.wait(5)

    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        started = time.monotonic()
        result = supervise(
            process, tmp_path, SimpleNamespace(tick=blocked), deadline_seconds=0.15, idle_seconds=0
        )
        assert entered.is_set()
        assert result["status"] == "TIMEOUT"
        assert time.monotonic() - started < 3
        assert process.poll() is not None
        assert result["monitor_thread_stopped"] is False
        assert json.loads((tmp_path / "supervisor.json").read_text())["status"] == "TIMEOUT"
    finally:
        release.set()
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)


@pytest.mark.skipif(os.name != "posix", reason="POSIX signal handler")
def test_registered_worker_produces_stack_dump_before_stop(tmp_path):
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).parents[1])}
    script = (
        "import sys,time; from glm53_offload.diagnostics import WorkerDiagnostics; "
        "d=WorkerDiagnostics(sys.argv[1]); time.sleep(30)"
    )
    process = subprocess.Popen([sys.executable, "-c", script, str(tmp_path)], env=env)
    try:
        expires = time.monotonic() + 5
        while not (tmp_path / "worker-diagnostics.json").exists() and time.monotonic() < expires:
            time.sleep(0.01)
        assert request_stack_dump(process, tmp_path)
        path = tmp_path / "worker-stacks.log"
        while path.stat().st_size == 0 and time.monotonic() < expires:
            time.sleep(0.01)
        assert "<module>" in path.read_text()
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_evidence_is_bounded_and_excludes_model_files(tmp_path):
    (tmp_path / "worker.log").write_bytes(b"x" * (3 << 20))
    (tmp_path / "report.json").write_text('{"status":"RUNNING"}')
    (tmp_path / "model.gguf").write_text("model weights must never be archived")
    (tmp_path / "secret.key").write_text("secret must never be archived")
    output = tmp_path / "evidence.zip"
    manifest = package_evidence(tmp_path, output, limit_bytes=4096)
    assert sum(record.get("captured_bytes", 0) for record in manifest["files"]) <= 4096
    with zipfile.ZipFile(output) as archive:
        assert "report.json" in archive.namelist()
        assert "model.gguf" not in archive.namelist()
        assert "secret.key" not in archive.namelist()
        assert json.loads(archive.read("EVIDENCE_MANIFEST.json")) == manifest
    assert next(record for record in manifest["files"] if record["path"] == "worker.log")[
        "truncated"
    ]


def test_evidence_preserves_batch_profile_labels_and_repeated_filenames(tmp_path):
    for batch in (1, 2, 4, 8):
        folder = tmp_path / "profiles" / f"batch-{batch}"
        trace = folder / "device-trace" / "plugins" / "profile" / "timestamp"
        trace.mkdir(parents=True)
        (folder / "profile-context.json").write_text(json.dumps({"batch": batch}))
        (folder / "device-trace-status.json").write_text('{"status":"CAPTURED"}')
        (trace / "host.xplane.pb").write_bytes(bytes([batch]))
    archive_path = tmp_path / "evidence.zip"
    manifest = package_evidence(tmp_path, archive_path)
    assert len(manifest["files"]) == 12
    with zipfile.ZipFile(archive_path) as archive:
        for batch in (1, 2, 4, 8):
            root = f"profiles/batch-{batch}"
            assert json.loads(archive.read(f"{root}/profile-context.json")) == {"batch": batch}
            assert archive.read(
                f"{root}/device-trace/plugins/profile/timestamp/host.xplane.pb"
            ) == bytes([batch])


def test_native_profiler_failure_does_not_repeat_or_hide_inference(monkeypatch, tmp_path):
    import jax.profiler

    class FailedTrace:
        def __enter__(self):
            raise RuntimeError("backend profiler unavailable")

    monkeypatch.setattr(jax.profiler, "trace", lambda *args, **kwargs: FailedTrace())
    calls = []
    with optional_device_trace(tmp_path / "device-trace"):
        calls.append("inference")
    assert calls == ["inference"]
    assert (
        json.loads((tmp_path / "device-trace-status.json").read_text())["status"] == "UNAVAILABLE"
    )
    with pytest.raises(ValueError, match="model failure"):
        with optional_device_trace(tmp_path / "device-trace"):
            raise ValueError("model failure")
