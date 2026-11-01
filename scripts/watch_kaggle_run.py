#!/usr/bin/env python3
"""Poll one Kaggle notebook version and notify once when its run ends."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import tempfile
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any, Callable, Iterator
from urllib.error import URLError
from urllib.request import urlopen

LOG = logging.getLogger("watch_kaggle_run")
STATE_FORMAT = 1
DEFAULT_INTERVAL_SECONDS = 180
DEFAULT_MAX_HOURS = 48
MAX_MAX_HOURS = 48
MAX_OUTPUT_PAGES = 5
MAX_OUTPUT_PAGE_SIZE = 100
MAX_ARTIFACT_ATTEMPTS = 3
ARTIFACT_RETRY_SECONDS = 5
MAX_CALLBACK_OUTPUT_BYTES = 8_192
ARTIFACT_LIMITS = {
    # Full vision/SFT trials retain reasoning and per-case grading for hundreds
    # of bounded requests. Keep retrieval bounded without rejecting that report.
    "report.json": 8_000_000,
    "worker.log": 4_000_000,
    "worker-traceback.txt": 1_000_000,
    "supervisor.json": 64_000,
    "tracking-readback.json": 64_000,
    "progress.json": 200_000,
    "monitor.json": 200_000,
}
TERMINAL_STATUSES = {
    "COMPLETE",
    "ERROR",
    "FAILED",
    "FAILURE",
    "CANCELED",
    "CANCELLED",
    "CANCEL_ACKNOWLEDGED",
}


class WatchError(RuntimeError):
    """A permanent watcher configuration or API consistency error."""


class VersionMismatch(WatchError):
    """The notebook's current saved version differs from the requested run."""


class VersionNotReady(WatchError):
    """Kaggle metadata has not caught up to the requested notebook version yet."""


class WatchTimeout(WatchError):
    """The bounded watch duration elapsed before the notebook finished."""


class ArtifactTooLarge(WatchError):
    """A selected report/log artifact exceeded its per-file size limit."""


class AmbiguousArtifacts(WatchError):
    """More than one per-run output folder contained report/log artifacts."""


def parse_notebook_ref(value: str) -> tuple[str, str]:
    """Return a validated ``owner/slug`` reference; version belongs separately."""

    parts = value.split("/")
    if len(parts) != 2 or any(not part for part in parts):
        raise argparse.ArgumentTypeError("notebook ref must be owner/slug (without a version)")
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", part) for part in parts):
        raise argparse.ArgumentTypeError(
            "notebook owner and slug may contain letters, digits, _ and -"
        )
    return parts[0], parts[1]


def normalize_status(status: Any) -> str:
    """Normalize SDK enum or string status values to uppercase names."""

    name = getattr(status, "name", None)
    raw = name if isinstance(name, str) else str(status)
    if "." in raw:
        raw = raw.rsplit(".", 1)[-1]
    return raw.strip().upper().replace(" ", "_")


def is_terminal_status(status: Any) -> bool:
    return normalize_status(status) in TERMINAL_STATUSES


def decoded_session_status(response: Any, payload: Any) -> Any:
    """Do not turn the SDK's default zero enum into evidence of a queue entry.

    Protobuf JSON can omit QUEUED because its enum value is zero. An empty
    response is therefore ambiguous: preserve the SDK interpretation as
    supporting information, while requiring explicit evidence for reporting.
    """
    if not isinstance(payload, dict):
        raise WatchError("Kaggle status response was not a JSON object")
    explicit = payload.get("status") is not None
    return SimpleNamespace(
        status=normalize_status(response.status) if explicit else "UNCONFIRMED",
        failure_message=getattr(response, "failure_message", ""),
        sdk_status=normalize_status(response.status),
        status_explicit=explicit,
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temp_name = temporary.name
            json.dump(value, temporary, indent=2, sort_keys=True)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temp_name, path)
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)


def _load_state(path: Path, notebook_ref: str, expected_version: int) -> dict[str, Any]:
    if not path.exists():
        now = time.time()
        return {
            "format": STATE_FORMAT,
            "notebook_ref": notebook_ref,
            "expected_version": expected_version,
            "started_at": now,
            "deadline_at": None,
            "created_at": _now_iso(),
            "last_status": None,
            "last_error": None,
            "terminal_status": None,
            "artifacts": {},
            "artifact_error": None,
            "failure_message": None,
            "callback_attempted": False,
            "callback_state": "NOT_CONFIGURED",
        }

    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WatchError(f"cannot read watcher state ({type(exc).__name__})") from exc
    if not isinstance(state, dict) or state.get("format") != STATE_FORMAT:
        raise WatchError("watcher state has an unsupported format")
    if state.get("notebook_ref") != notebook_ref:
        raise WatchError("watcher state belongs to a different notebook ref")
    if state.get("expected_version") != expected_version:
        raise WatchError("watcher state belongs to a different expected version")
    return state


def _artifact_parts(remote_name: str) -> tuple[str, str] | None:
    """Return a run-folder key and basename for a known runtime artifact."""

    if not isinstance(remote_name, str) or "\\" in remote_name or "\x00" in remote_name:
        return None
    path = PurePosixPath(remote_name)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        return None
    basename = path.name
    if basename not in ARTIFACT_LIMITS:
        return None
    parts = path.parts
    if parts == (basename,):
        return "legacy", basename
    if parts == ("experiment-run", basename):
        return "legacy", basename
    if (
        len(parts) >= 2
        and parts[-2] == "qwen38-trial"
        and parts[:-2] in {(), ("working",), ("kaggle", "working")}
    ):
        return "qwen38", basename
    if len(parts) < 3 or parts[-3] != "experiment-run":
        return None
    run_id = parts[-2]
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", run_id):
        return None
    prefixes = {
        (),
        ("glm53-offload-runtime",),
        ("working", "glm53-offload-runtime"),
        ("kaggle", "working", "glm53-offload-runtime"),
    }
    if parts[:-3] not in prefixes:
        return None
    return run_id, basename


def _select_artifact_urls(files: list[Any]) -> dict[str, str]:
    """Select one URL per allowlisted artifact, ignoring all other output files."""

    by_run: dict[str, dict[str, str]] = {}
    for item in files:
        artifact = _artifact_parts(getattr(item, "file_name", ""))
        url = getattr(item, "url", "")
        if artifact and isinstance(url, str) and url:
            run_id, basename = artifact
            by_run.setdefault(run_id, {}).setdefault(basename, url)
    if len(by_run) > 1:
        raise AmbiguousArtifacts("multiple experiment-run folders contain report/log candidates")
    return next(iter(by_run.values()), {})


def _bounded_download(url: str, limit: int, timeout: float = 30.0) -> bytes:
    if not isinstance(url, str) or not url.startswith("https://"):
        raise WatchError("artifact URL was not HTTPS")
    with urlopen(url, timeout=timeout) as response:
        size = response.headers.get("Content-Length")
        if size is not None and int(size) > limit:
            raise ArtifactTooLarge(f"artifact exceeds {limit} byte limit")
        data = response.read(limit + 1)
    if len(data) > limit:
        raise ArtifactTooLarge(f"artifact exceeds {limit} byte limit")
    return data


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temp_name = temporary.name
            temporary.write(data)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temp_name, path)
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)


def _drain_callback_pipe(pipe: Any, target: bytearray) -> None:
    while True:
        chunk = pipe.read(4_096)
        if not chunk:
            break
        remaining = MAX_CALLBACK_OUTPUT_BYTES - len(target)
        if remaining > 0:
            target.extend(chunk[:remaining])


def _invoke_callback(argv: list[str], prompt: str, timeout: float) -> dict[str, Any]:
    """Run a callback without a shell and retain only bounded stdout/stderr."""

    result: dict[str, Any] = {
        "returncode": None,
        "timed_out": False,
        "error_type": None,
        "stdout": b"",
        "stderr": b"",
    }
    try:
        process = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
        )
    except Exception as exc:
        result["error_type"] = type(exc).__name__
        return result

    stdout = bytearray()
    stderr = bytearray()
    readers = [
        threading.Thread(target=_drain_callback_pipe, args=(process.stdout, stdout), daemon=True),
        threading.Thread(target=_drain_callback_pipe, args=(process.stderr, stderr), daemon=True),
    ]
    for reader in readers:
        reader.start()
    try:
        try:
            if process.stdin is not None:
                process.stdin.write(prompt.encode("utf-8"))
                process.stdin.close()
        except BrokenPipeError:
            pass
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            result["timed_out"] = True
            process.kill()
            process.wait()
    except Exception as exc:
        result["error_type"] = type(exc).__name__
        if process.poll() is None:
            process.kill()
            process.wait()
    finally:
        if process.stdin is not None and not process.stdin.closed:
            process.stdin.close()
        for reader in readers:
            reader.join(timeout=5)
        result["returncode"] = process.returncode
        result["stdout"] = bytes(stdout)
        result["stderr"] = bytes(stderr)
    if result["timed_out"] and result["error_type"] is None:
        result["error_type"] = "TimeoutExpired"
    return result


class KaggleApiAdapter:
    """Narrow wrapper over read-only Kaggle kernel/session API endpoints."""

    def __init__(self, api: Any, downloader: Callable[[str, int], bytes] = _bounded_download):
        self.api = api
        self.downloader = downloader

    @contextmanager
    def _client(self) -> Iterator[Any]:
        """Use finite HTTP timeouts around the SDK's otherwise unbounded requests."""

        with self.api.build_kaggle_client() as client:
            http_client = client.http_client()
            http_client._init_session()
            session = http_client._session
            original_send = session.send

            def send_with_timeout(request: Any, **kwargs: Any) -> Any:
                kwargs.setdefault("timeout", (10, 30))
                return original_send(request, **kwargs)

            session.send = send_with_timeout
            yield client

    def _get_kernel_metadata(self, notebook_ref: str) -> Any:
        owner, slug = parse_notebook_ref(notebook_ref)
        from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest

        request = ApiGetKernelRequest()
        request.user_name = owner
        request.kernel_slug = slug
        with self._client() as client:
            response = client.kernels.kernels_api_client.get_kernel(request)
        return response.metadata

    def current_version(self, notebook_ref: str) -> int:
        metadata = self._get_kernel_metadata(notebook_ref)
        version = getattr(metadata, "current_version_number", None)
        if not isinstance(version, int) or version < 1:
            raise WatchError("Kaggle did not return a valid current notebook version")
        return version

    def session_status(self, notebook_ref: str, expected_version: int) -> Any:
        owner, slug = parse_notebook_ref(notebook_ref)
        from kagglesdk.kernels.types.kernels_api_service import (
            ApiGetKernelSessionStatusRequest,
        )

        request = ApiGetKernelSessionStatusRequest()
        request.user_name = owner
        request.kernel_slug = slug
        request.version_label = f"v{expected_version}"
        payload = None
        with self._client() as client:
            session = client.http_client()._session
            original_send = session.send

            def capture_status(request: Any, **kwargs: Any) -> Any:
                nonlocal payload
                http_response = original_send(request, **kwargs)
                if http_response.status_code == 200:
                    payload = http_response.json()
                return http_response

            session.send = capture_status
            response = client.kernels.kernels_api_client.get_kernel_session_status(request)
        current_version = self.current_version(notebook_ref)
        if current_version < expected_version:
            raise VersionNotReady(
                f"Kaggle reports notebook version {current_version}; expected {expected_version}"
            )
        if current_version > expected_version:
            raise VersionMismatch(
                f"notebook current version is {current_version}; expected {expected_version}"
            )
        return decoded_session_status(response, payload)

    def download_terminal_artifacts(
        self,
        notebook_ref: str,
        expected_version: int,
        output_dir: Path,
    ) -> dict[str, dict[str, Any]]:
        """List output metadata and download only the small report/log allowlist."""

        owner, slug = parse_notebook_ref(notebook_ref)
        from kagglesdk.kernels.types.kernels_api_service import (
            ApiListKernelSessionOutputRequest,
        )

        before = self.current_version(notebook_ref)
        if before < expected_version:
            raise VersionNotReady(
                f"Kaggle reports notebook version {before}; expected {expected_version}"
            )
        if before > expected_version:
            raise VersionMismatch(
                f"notebook current version is {before}; expected {expected_version}"
            )
        selected: dict[str, str] = {}
        page_token: str | None = None
        for _ in range(MAX_OUTPUT_PAGES):
            request = ApiListKernelSessionOutputRequest()
            request.user_name = owner
            request.kernel_slug = slug
            request.version_label = f"v{expected_version}"
            request.page_size = MAX_OUTPUT_PAGE_SIZE
            if page_token:
                request.page_token = page_token
            with self._client() as client:
                response = client.kernels.kernels_api_client.list_kernel_session_output(request)
            for basename, url in _select_artifact_urls(response.files or []).items():
                selected.setdefault(basename, url)
            page_token = getattr(response, "next_page_token", None)
            if not page_token:
                break
        if page_token:
            raise AmbiguousArtifacts(
                "output listing exceeded the bounded page limit; cannot establish a unique run"
            )

        after = self.current_version(notebook_ref)
        if after < expected_version:
            raise VersionNotReady(
                f"Kaggle reports notebook version {after}; expected {expected_version}"
            )
        if after > expected_version:
            raise VersionMismatch(
                f"notebook current version is {after}; expected {expected_version}"
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        downloaded: dict[str, dict[str, Any]] = {}
        for basename, url in selected.items():
            data = self.downloader(url, ARTIFACT_LIMITS[basename])
            if len(data) > ARTIFACT_LIMITS[basename]:
                raise ArtifactTooLarge(f"artifact exceeds {ARTIFACT_LIMITS[basename]} byte limit")
            local_path = output_dir / basename
            _atomic_write_bytes(local_path, data)
            downloaded[basename] = {
                "path": str(local_path.resolve()),
                "bytes": len(data),
            }
        return downloaded


class KaggleRunWatcher:
    """Resumable watcher core with injectable API, clock, and callback."""

    def __init__(
        self,
        *,
        notebook_ref: str,
        expected_version: int,
        output_dir: Path,
        state_path: Path,
        api: Any,
        max_hours: float = DEFAULT_MAX_HOURS,
        callback_argv: list[str] | None = None,
        callback_timeout: float = 300.0,
        logger: logging.Logger = LOG,
        clock: Callable[[], float] = time.time,
        sleeper: Callable[[float], None] = time.sleep,
    ):
        parse_notebook_ref(notebook_ref)
        if not isinstance(expected_version, int) or expected_version < 1:
            raise WatchError("expected version must be a positive integer")
        if not 0 < max_hours <= MAX_MAX_HOURS:
            raise WatchError(f"max-hours must be greater than 0 and at most {MAX_MAX_HOURS}")
        if callback_timeout <= 0 or callback_timeout > 300:
            raise WatchError("callback timeout must be greater than 0 and at most 300 seconds")
        if callback_argv is not None and (
            not callback_argv or any(not isinstance(arg, str) or not arg for arg in callback_argv)
        ):
            raise WatchError("callback argv must contain one or more non-empty strings")

        self.notebook_ref = notebook_ref
        self.expected_version = expected_version
        self.output_dir = output_dir
        self.state_path = state_path
        self.api = api
        self.callback_argv = callback_argv
        self.callback_timeout = callback_timeout
        self.logger = logger
        self.clock = clock
        self.sleeper = sleeper
        self.state = _load_state(state_path, notebook_ref, expected_version)
        if not self.state.get("callback_attempted"):
            self.state["callback_state"] = (
                "PENDING" if callback_argv is not None else "NOT_CONFIGURED"
            )
        requested_deadline = self.state["started_at"] + max_hours * 3600
        prior_deadline = self.state.get("deadline_at")
        self.state["deadline_at"] = min(
            requested_deadline,
            float(prior_deadline) if prior_deadline is not None else requested_deadline,
        )
        self.save()

    def save(self) -> None:
        _atomic_write_json(self.state_path, self.state)

    def _log_error_once(self, error_key: str, label: str) -> None:
        if self.state.get("last_error") != error_key:
            self.logger.error(label)
        self.state["last_error"] = error_key
        self.save()

    def _notify_once(self, status: str) -> None:
        if self.callback_argv is None or self.state.get("callback_attempted"):
            return
        event_id = f"{self.notebook_ref}@{self.expected_version}:{status}"
        artifact_names = sorted(self.state.get("artifacts", {}))
        artifact_summary = ", ".join(artifact_names) if artifact_names else "none downloaded"
        artifact_error = self.state.get("artifact_error")
        prompt = (
            f"Kaggle notebook {self.notebook_ref} version {self.expected_version} "
            f"reached terminal status {status}.\n"
            f"Watcher state and any downloaded artifacts are under {self.output_dir.resolve()}.\n"
            f"Downloaded artifacts: {artifact_summary}.\n"
            + (f"Artifact retrieval error: {artifact_error}.\n" if artifact_error else "")
            + "Please inspect available artifacts and provide the user a short run summary. "
            "Do not submit, cancel, or rerun the notebook.\n"
        )
        self.state["callback_attempted"] = True
        self.state["callback_event_id"] = event_id
        self.state["callback_state"] = "ATTEMPTED"
        self.state["callback_attempted_at"] = _now_iso()
        self.save()
        result = _invoke_callback(self.callback_argv, prompt, self.callback_timeout)
        returncode = result["returncode"]
        acknowledged = returncode == 0 and not result["timed_out"] and not result["error_type"]
        self.state["callback_state"] = "ACKNOWLEDGED" if acknowledged else "FAILED"
        self.state["callback_returncode"] = returncode
        if result["error_type"]:
            self.state["callback_error_type"] = result["error_type"]
        self.state["callback_log"] = str((self.output_dir / "callback.log").resolve())
        diagnostic = (
            f"event={event_id}\n"
            f"state={self.state['callback_state']}\n"
            f"returncode={returncode}\n"
            f"timed_out={result['timed_out']}\n"
            f"error_type={result['error_type']}\n"
            "stdout:\n"
            + result["stdout"].decode("utf-8", errors="replace")
            + "\nstderr:\n"
            + result["stderr"].decode("utf-8", errors="replace")
        )
        _atomic_write_bytes(
            self.output_dir / "callback.log",
            diagnostic.encode("utf-8")[: 2 * MAX_CALLBACK_OUTPUT_BYTES + 1_024],
        )
        if not acknowledged:
            self.logger.error("completion callback failed (%s)", result["error_type"] or returncode)
        self.save()

    def _finish_terminal(self, status: str) -> str:
        self.state["terminal_status"] = status
        attempts = int(self.state.get("artifact_attempts", 0))
        if not self.state.get("artifacts_attempted"):
            while attempts < MAX_ARTIFACT_ATTEMPTS:
                attempts += 1
                self.state["artifact_attempts"] = attempts
                self.save()
                artifact_error = None
                try:
                    found = self.api.download_terminal_artifacts(
                        self.notebook_ref,
                        self.expected_version,
                        self.output_dir,
                    )
                    self.state.setdefault("artifacts", {}).update(found)
                except VersionMismatch as exc:
                    self.state["artifact_error"] = type(exc).__name__
                    self.save()
                    self.logger.error("notebook version changed during artifact snapshot; stopping")
                    raise
                except VersionNotReady:
                    self.state["artifact_error"] = "VersionNotReady"
                    self.state["artifact_attempts"] = 0
                    self.save()
                    raise
                except Exception as exc:
                    artifact_error = type(exc).__name__
                if artifact_error:
                    if self.state.get("artifact_error") != artifact_error:
                        self.logger.error("bounded output retrieval failed (%s)", artifact_error)
                    self.state["artifact_error"] = artifact_error
                elif "report.json" in self.state.get("artifacts", {}):
                    self.state["artifact_error"] = None
                    break
                elif attempts < MAX_ARTIFACT_ATTEMPTS:
                    self.state["artifact_error"] = "report.json not listed yet"
                self.save()
                if attempts < MAX_ARTIFACT_ATTEMPTS:
                    self.sleeper(ARTIFACT_RETRY_SECONDS)
            if "report.json" not in self.state.get("artifacts", {}):
                self.state["artifact_error"] = (
                    f"report.json unavailable after {MAX_ARTIFACT_ATTEMPTS} bounded output attempts"
                )
                self.logger.error("report.json unavailable after bounded output retries")
            self.state["artifacts_attempted"] = True
        self.save()
        self._notify_once(status)
        return status

    def poll_once(self) -> str | None:
        terminal = self.state.get("terminal_status")
        if terminal:
            return self._finish_terminal(str(terminal))

        try:
            current_version = self.api.current_version(self.notebook_ref)
            if current_version < self.expected_version:
                raise VersionNotReady(
                    f"Kaggle reports notebook version {current_version}; "
                    f"expected {self.expected_version}"
                )
            if current_version != self.expected_version:
                raise VersionMismatch(
                    "notebook current version is "
                    f"{current_version}; expected {self.expected_version}"
                )
            response = self.api.session_status(self.notebook_ref, self.expected_version)
        except VersionMismatch as exc:
            error_key = f"VersionMismatch:{exc}"
            self._log_error_once(
                error_key,
                "watch stopped: current notebook version differs from expected "
                f"version {self.expected_version}",
            )
            raise
        except VersionNotReady:
            self._log_error_once(
                "VersionNotReady",
                f"Kaggle version metadata has not reached expected version {self.expected_version}",
            )
            return None
        except Exception as exc:
            self._log_error_once(type(exc).__name__, f"Kaggle poll error ({type(exc).__name__})")
            return None

        if self.state.get("last_error") is not None:
            self.logger.info("Kaggle polling recovered")
            self.state["last_error"] = None
        status = normalize_status(getattr(response, "status", response))
        failure_message = getattr(response, "failure_message", "")
        if is_terminal_status(status) and isinstance(failure_message, str) and failure_message:
            self.state["failure_message"] = failure_message[:4_096]
        if status != self.state.get("last_status"):
            prior = self.state.get("last_status") or "UNKNOWN"
            self.logger.info("Kaggle status: %s -> %s", prior, status)
            self.state["last_status"] = status
            self.state["last_status_at"] = _now_iso()
        self.save()
        if is_terminal_status(status):
            return self._finish_terminal(status)
        return None

    def run(self, interval_seconds: float = DEFAULT_INTERVAL_SECONDS) -> str:
        if interval_seconds <= 0:
            raise WatchError("poll interval must be positive")
        while True:
            if self.clock() >= float(self.state["deadline_at"]):
                raise WatchTimeout(
                    "maximum watch duration elapsed before a terminal notebook status"
                )
            try:
                terminal = self.poll_once()
            except VersionNotReady:
                remaining = float(self.state["deadline_at"]) - self.clock()
                if remaining <= 0:
                    raise WatchTimeout(
                        "maximum watch duration elapsed before expected version metadata settled"
                    )
                self.sleeper(min(interval_seconds, remaining))
                continue
            if terminal:
                return terminal
            remaining = float(self.state["deadline_at"]) - self.clock()
            if remaining <= 0:
                raise WatchTimeout(
                    "maximum watch duration elapsed before a terminal notebook status"
                )
            self.sleeper(min(interval_seconds, remaining))


def _parse_callback(value: str | None) -> list[str] | None:
    if value is None:
        return None
    try:
        argv = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError("callback must be a JSON array of argv strings") from exc
    if (
        not isinstance(argv, list)
        or not argv
        or any(not isinstance(arg, str) or not arg for arg in argv)
    ):
        raise argparse.ArgumentTypeError(
            "callback must be a non-empty JSON array of non-empty strings"
        )
    return argv


def _create_kaggle_api(token_file: Path, isolated_config_dir: Path) -> KaggleApiAdapter:
    if not token_file.is_file():
        raise WatchError(f"Kaggle access token file not found: {token_file}")
    isolated_config_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    if (isolated_config_dir / "kaggle.json").exists():
        raise WatchError("isolated Kaggle config directory must not contain kaggle.json")
    os.environ["KAGGLE_CONFIG_DIR"] = str(isolated_config_dir.resolve())
    os.environ["KAGGLE_API_TOKEN"] = str(token_file.expanduser().resolve())
    os.environ.pop("KAGGLE_USERNAME", None)
    os.environ.pop("KAGGLE_KEY", None)
    os.environ.pop("KAGGLE_API_V1_TOKEN", None)

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise WatchError("Kaggle API package is unavailable; check PYTHONPATH") from exc
    api = KaggleApi()
    try:
        api.authenticate()
    except SystemExit as exc:
        raise WatchError("Kaggle authentication failed") from exc
    return KaggleApiAdapter(api)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--notebook-ref",
        required=True,
        type=parse_notebook_ref,
        metavar="OWNER/SLUG",
    )
    parser.add_argument("--expected-version", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--interval-seconds", type=float, default=DEFAULT_INTERVAL_SECONDS)
    parser.add_argument("--max-hours", type=float, default=DEFAULT_MAX_HOURS)
    parser.add_argument("--state-file", type=Path)
    parser.add_argument(
        "--once",
        action="store_true",
        help="check status once and exit instead of polling until terminal",
    )
    parser.add_argument("--token-file", type=Path, default=Path.home() / ".kaggle/access_token")
    parser.add_argument("--callback-argv-json", type=_parse_callback)
    parser.add_argument("--callback-timeout-seconds", type=float, default=300.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    notebook_ref = "/".join(args.notebook_ref)
    if args.expected_version < 1:
        parser.error("--expected-version must be a positive integer")
    if args.interval_seconds <= 0:
        parser.error("--interval-seconds must be positive")
    if args.max_hours <= 0 or args.max_hours > MAX_MAX_HOURS:
        parser.error(f"--max-hours must be greater than 0 and at most {MAX_MAX_HOURS}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.state_file or (args.output_dir / "watch-state.json")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        with tempfile.TemporaryDirectory(prefix="kaggle-watch-auth-") as config_dir:
            adapter = _create_kaggle_api(args.token_file, Path(config_dir))
            watcher = KaggleRunWatcher(
                notebook_ref=notebook_ref,
                expected_version=args.expected_version,
                output_dir=args.output_dir,
                state_path=state_path,
                api=adapter,
                max_hours=args.max_hours,
                callback_argv=args.callback_argv_json,
                callback_timeout=args.callback_timeout_seconds,
            )
            if args.once:
                watcher.poll_once()
            else:
                watcher.run(interval_seconds=args.interval_seconds)
    except (WatchError, URLError) as exc:
        LOG.error("watch stopped (%s)", type(exc).__name__)
        return 2
    except Exception as exc:
        LOG.error("watch stopped unexpectedly (%s)", type(exc).__name__)
        return 2
    except KeyboardInterrupt:
        LOG.error("watch interrupted; saved state can be resumed")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
