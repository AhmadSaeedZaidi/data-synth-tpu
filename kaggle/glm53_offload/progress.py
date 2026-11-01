"""Small atomic progress snapshots readable without calling into JAX."""

import json
import threading
import time
from pathlib import Path

_lock = threading.Lock()
_path = None
_state = {}
_last_write = 0.0


def configure(path):
    global _path, _state, _last_write
    with _lock:
        _path = Path(path) if path is not None else None
        _state = {"download_bytes": 0, "download_retries": 0}
        _last_write = 0.0


def update(*, downloaded=0, retried=0, force=False, **fields):
    """Count successful range bytes, including concurrent readers; throttle disk writes."""
    global _last_write
    with _lock:
        if _path is None:
            return
        stage_changed = "stage" in fields and fields["stage"] != _state.get("stage")
        if stage_changed:
            _state["stage_started_at"] = time.time()
            _state.pop("detail", None)
        _state.update(fields)
        _state["download_bytes"] += downloaded
        _state["download_retries"] += retried
        _state["updated_at"] = time.time()
        now = time.monotonic()
        if not (force or stage_changed) and now - _last_write < 5:
            return
        temporary = _path.with_suffix(".tmp")
        temporary.write_text(json.dumps(_state) + "\n")
        temporary.replace(_path)
        _last_write = now
