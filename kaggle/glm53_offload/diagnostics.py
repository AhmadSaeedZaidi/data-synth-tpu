"""Small crash diagnostics and a bounded, downloadable evidence archive."""

import faulthandler
import hashlib
import json
import os
import queue
import signal
import sys
import threading
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path


def atomic_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, default=str) + "\n")
    temporary.replace(path)


def bounded_call(function, timeout):
    """Bound an optional external call without waiting for its SDK to clean up."""
    result = queue.Queue(maxsize=1)

    def invoke():
        try:
            result.put((True, function()))
        except Exception as error:
            result.put((False, error))

    threading.Thread(target=invoke, daemon=True).start()
    try:
        success, value = result.get(timeout=timeout)
    except queue.Empty:
        raise TimeoutError("optional startup call exceeded its time budget") from None
    if not success:
        raise value
    return value


class WorkerDiagnostics:
    def __init__(self, directory):
        directory = Path(directory)
        self.file = (directory / "worker-stacks.log").open("a")
        self.was_enabled = faulthandler.is_enabled()
        self.stack_signal = None
        faulthandler.enable(file=self.file, all_threads=True)
        candidate = getattr(signal, "SIGUSR2", None)
        if candidate is not None and signal.getsignal(candidate) == signal.SIG_DFL:
            faulthandler.register(candidate, file=self.file, all_threads=True)
            self.stack_signal = int(candidate)
        atomic_json(
            directory / "worker-diagnostics.json",
            {
                "pid": os.getpid(),
                "stack_signal": self.stack_signal,
                "started_at": time.time(),
                "scope": "Python stack frames; no locals or environment",
            },
        )

    def close(self):
        if self.stack_signal is not None:
            faulthandler.unregister(self.stack_signal)
        if self.was_enabled:
            faulthandler.enable(file=sys.__stderr__, all_threads=True)
        else:
            faulthandler.disable()
        self.file.close()


def request_stack_dump(process, directory):
    """Signal only a live worker that explicitly registered this handler."""
    try:
        info = json.loads((Path(directory) / "worker-diagnostics.json").read_text())
        if info.get("pid") == process.pid and info.get("stack_signal") and process.poll() is None:
            process.send_signal(info["stack_signal"])
            return True
    except (OSError, ValueError, ProcessLookupError):
        pass
    return False


@contextmanager
def optional_device_trace(directory):
    """Capture one blocked decode call; profiler failures do not repeat inference."""
    context = None
    entered = False
    note = {"status": "STARTING"}
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    try:
        import jax.profiler

        context = jax.profiler.trace(str(directory), create_perfetto_trace=True)
        context.__enter__()
        entered = True
    except Exception as error:
        note.update(status="UNAVAILABLE", error_type=type(error).__name__)
    try:
        yield
    finally:
        if entered:
            try:
                context.__exit__(None, None, None)
                note["status"] = "CAPTURED"
            except Exception as error:
                note.update(status="EXPORT_FAILED", error_type=type(error).__name__)
        atomic_json(directory.parent / "device-trace-status.json", note)


def package_evidence(directory, destination, *, extra_files=(), limit_bytes=24 << 20):
    """Archive known diagnostics, taking tails of logs; never model/cache files.

    The manifest marks any truncation and records the exact captured hashes.
    This function works even when the child did not produce a report.
    """
    directory, destination = Path(directory), Path(destination)
    names = (
        "supervisor.json",
        "report.json",
        "config.json",
        "progress.json",
        "monitor.json",
        "tracking-readback.json",
        "worker-diagnostics.json",
        "worker-traceback.txt",
        "tracing-status.json",
        "device-trace-status.json",
        "events.jsonl",
        "metrics.jsonl",
        "traces.jsonl",
        "routing-replay.jsonl",
        "execution-timing.jsonl",
        "worker-stacks.log",
        "worker.log",
    )
    sources = [directory / name for name in names] + [Path(p) for p in extra_files]
    sources += sorted((directory / "vision-fixtures").glob("*"))
    sources += sorted((directory / "device-trace").rglob("*.gz"))[:4]
    sources += sorted((directory / "device-trace").rglob("*.pb"))[:4]
    # Each admitted Qwen batch has its own trace. Preserve the label in the
    # archive so repeated XPlane/status filenames cannot collide.
    for profile in sorted((directory / "profiles").glob("*"))[:8]:
        sources += [profile / "profile-context.json", profile / "device-trace-status.json"]
        sources += sorted((profile / "device-trace").rglob("*.gz"))[:2]
        sources += sorted((profile / "device-trace").rglob("*.pb"))[:1]
    captured, records, remaining = {}, [], limit_bytes
    for path in sources:
        if not path.is_file() or path.is_symlink():
            continue
        name = (
            path.relative_to(directory).as_posix()
            if path.is_relative_to(directory / "profiles")
            else path.name
        )
        size = path.stat().st_size
        if path.suffix in (".gz", ".pb") and size > (2 << 20):
            records.append({"path": name, "source_bytes": size, "omitted": True})
            continue
        maximum = min(remaining, (8 << 20) if path.name == "report.json" else (2 << 20))
        if path.suffix in (".gz", ".pb") and size > maximum:
            records.append({"path": name, "source_bytes": size, "omitted": True})
            continue
        if maximum <= 0:
            records.append({"path": name, "source_bytes": size, "omitted": True})
            continue
        with path.open("rb") as stream:
            if size > maximum:
                stream.seek(size - maximum)
            data = stream.read(maximum)
        if name in captured:
            raise ValueError(f"duplicate evidence filename: {name}")
        captured[name] = data
        remaining -= len(data)
        records.append(
            {
                "path": name,
                "source_bytes": size,
                "captured_bytes": len(data),
                "truncated": size > len(data),
                "capture": "tail" if size > len(data) else "full",
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    manifest = {"format": 1, "created_at": time.time(), "files": records}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in captured.items():
            archive.writestr(name, data)
        archive.writestr("EVIDENCE_MANIFEST.json", json.dumps(manifest, indent=2) + "\n")
    temporary.replace(destination)
    return manifest
