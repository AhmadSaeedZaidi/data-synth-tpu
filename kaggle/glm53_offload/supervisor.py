"""Parent watchdog whose deadlines do not depend on the tracking service."""

import json
import queue
import subprocess
import threading
import time
from pathlib import Path

from .diagnostics import atomic_json, request_stack_dump


class MonitorPump:
    """One daemon thread, at most one queued tick; a stuck SDK cannot grow work."""

    def __init__(self, monitor, pid):
        self.monitor, self.pid = monitor, pid
        self.queue = queue.Queue(maxsize=1)
        self.stopping = threading.Event()
        self.errors = []
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self.stopping.is_set():
            try:
                self.queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self.monitor.tick(self.pid)
            except Exception as error:
                self.errors.append(type(error).__name__)
                self.errors[:] = self.errors[-8:]

    def tick(self):
        try:
            self.queue.put_nowait(True)
        except queue.Full:
            pass

    def close(self):
        self.stopping.set()
        self.thread.join(timeout=0.5)
        return not self.thread.is_alive()


def stop_worker(process, *, grace_seconds=15, kill_seconds=10):
    """Bound TPU teardown without losing the original stop reason.

    TPU signal handling can outlast a two-second grace period. Even SIGKILL
    may not be reaped immediately; report that explicitly instead of throwing
    a second TimeoutExpired that hides the session deadline.
    """
    result = dict(terminate_sent=False, kill_sent=False, pending=False)
    if process.poll() is None:
        try:
            process.terminate()
            result["terminate_sent"] = True
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            try:
                process.kill()
                result["kill_sent"] = True
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=kill_seconds)
            except subprocess.TimeoutExpired:
                result["pending"] = True
    result["returncode"] = process.returncode
    return result


def finalize_report(directory, result, returncode):
    """Reconcile a partial worker report with durable supervisor evidence."""
    directory = Path(directory)
    supervisor = directory / "supervisor.json"
    state = result or (json.loads(supervisor.read_text()) if supervisor.exists() else {})
    path = directory / "report.json"
    report = (
        json.loads(path.read_text())
        if path.exists()
        else dict(status="ERROR", error_type="MissingWorkerReport")
    )
    report.update(worker_returncode=returncode, supervisor_status=state.get("status"))
    if state.get("status") in {"TIMEOUT", "STALLED", "INTERRUPTED"}:
        report["status"] = state["status"]
    elif report.get("status") == "RUNNING":
        report.update(status="ERROR", error_type="IncompleteWorkerReport")
    if "termination" in state:
        report["termination"] = state["termination"]
    atomic_json(path, report)
    return report


def supervise(process, directory, monitor, *, deadline_seconds, idle_seconds):
    """Persist terminal supervisor state, including kills with a RUNNING report."""
    directory = Path(directory)
    started = time.monotonic()
    pump = MonitorPump(monitor, process.pid)
    state = {
        "status": "RUNNING",
        "pid": process.pid,
        "started_at": time.time(),
        "deadline_seconds": deadline_seconds,
    }
    atomic_json(directory / "supervisor.json", state)
    try:
        while process.poll() is None:
            elapsed = time.monotonic() - started
            remaining = deadline_seconds - elapsed
            try:
                snapshot = json.loads((directory / "progress.json").read_text())
            except (OSError, ValueError):
                snapshot = {}
            age = (
                max(0, time.time() - snapshot["updated_at"])
                if snapshot.get("updated_at")
                else elapsed
            )
            state.update(
                stage=snapshot.get("stage"),
                detail=snapshot.get("detail"),
                progress_age_seconds=age,
            )
            if remaining <= 0 or (idle_seconds > 0 and age > idle_seconds):
                state["status"] = "TIMEOUT" if remaining <= 0 else "STALLED"
                atomic_json(directory / "supervisor.json", state)
                state["stack_dump_requested"] = request_stack_dump(process, directory)
                if state["stack_dump_requested"]:
                    try:
                        process.wait(timeout=0.1)
                    except subprocess.TimeoutExpired:
                        pass
                if process.poll() is not None:
                    break
                state["termination"] = stop_worker(process)
                break
            pump.tick()
            try:
                process.wait(timeout=min(2, remaining))
            except subprocess.TimeoutExpired:
                pass
        if state["status"] == "RUNNING":
            state["status"] = "EXITED"
    except BaseException as error:
        if state["status"] not in {"TIMEOUT", "STALLED"}:
            state["status"] = "INTERRUPTED"
        state["supervisor_error"] = type(error).__name__
        if process.poll() is None:
            request_stack_dump(process, directory)
            state["termination"] = stop_worker(process, grace_seconds=0)
        raise
    finally:
        state.update(
            returncode=process.returncode,
            elapsed_seconds=time.monotonic() - started,
            finished_at=time.time(),
            monitor_errors=list(pump.errors),
            monitor_thread_stopped=pump.close(),
        )
        atomic_json(directory / "supervisor.json", state)
    return state
