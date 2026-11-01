from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

WATCHER_PATH = Path(__file__).resolve().parents[2] / "scripts/watch_kaggle_run.py"
SPEC = importlib.util.spec_from_file_location("watch_kaggle_run", WATCHER_PATH)
assert SPEC is not None and SPEC.loader is not None
watcher = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = watcher
SPEC.loader.exec_module(watcher)


class FakeApi:
    def __init__(self, *, version: int = 4, statuses: list[object] | None = None):
        self.version = version
        self.statuses = list(statuses or ["RUNNING"])
        self.status_calls = 0
        self.artifact_calls = 0

    def current_version(self, notebook_ref: str) -> int:
        assert notebook_ref == "user/notebook"
        return self.version

    def session_status(self, notebook_ref: str, expected_version: int) -> object:
        assert notebook_ref == "user/notebook"
        assert expected_version == 4
        self.status_calls += 1
        result = self.statuses.pop(0)
        if isinstance(result, Exception):
            raise result
        if hasattr(result, "status"):
            return result
        return SimpleNamespace(status=result)

    def download_terminal_artifacts(
        self,
        notebook_ref: str,
        expected_version: int,
        output_dir: Path,
    ) -> dict[str, dict[str, object]]:
        assert notebook_ref == "user/notebook"
        assert expected_version == 4
        self.artifact_calls += 1
        return {"report.json": {"path": str(output_dir / "report.json"), "bytes": 14}}


def make_watcher(
    tmp_path: Path,
    api: FakeApi,
    **kwargs: object,
) -> watcher.KaggleRunWatcher:
    output_dir = tmp_path / "output"
    return watcher.KaggleRunWatcher(
        notebook_ref="user/notebook",
        expected_version=4,
        output_dir=output_dir,
        state_path=output_dir / "watch-state.json",
        api=api,
        **kwargs,
    )


@pytest.mark.parametrize(
    "raw,sdk,expected",
    [
        ({}, "QUEUED", "UNCONFIRMED"),
        ({"failureMessage": ""}, "QUEUED", "UNCONFIRMED"),
        ({"status": "QUEUED"}, "QUEUED", "QUEUED"),
        ({"status": "RUNNING"}, "RUNNING", "RUNNING"),
        ({"status": "COMPLETE"}, "COMPLETE", "COMPLETE"),
        ({"status": "CANCEL_REQUESTED"}, "CANCEL_REQUESTED", "CANCEL_REQUESTED"),
        ({"status": "CANCEL_ACKNOWLEDGED"}, "CANCEL_ACKNOWLEDGED", "CANCEL_ACKNOWLEDGED"),
    ],
)
def test_missing_wire_status_cannot_confirm_a_queue(raw, sdk, expected):
    result = watcher.decoded_session_status(SimpleNamespace(status=sdk, failure_message=""), raw)
    assert result.status == expected
    assert result.sdk_status == sdk
    assert result.status_explicit == ("status" in raw)


def test_unconfirmed_queue_is_not_terminal(tmp_path):
    api = FakeApi(statuses=[watcher.decoded_session_status(SimpleNamespace(status="QUEUED"), {})])
    instance = make_watcher(tmp_path, api)
    assert instance.poll_once() is None
    assert instance.state["last_status"] == "UNCONFIRMED"
    assert instance.state["terminal_status"] is None
    assert api.artifact_calls == 0


@pytest.mark.parametrize(
    "payload,decoded,expected",
    [
        ({}, "QUEUED", "UNCONFIRMED"),
        ({"status": "CANCEL_ACKNOWLEDGED"}, "CANCEL_ACKNOWLEDGED", "CANCEL_ACKNOWLEDGED"),
    ],
)
def test_adapter_reads_wire_status_for_exact_version(monkeypatch, payload, decoded, expected):
    from contextlib import contextmanager

    module_name = "kagglesdk.kernels.types.kernels_api_service"
    monkeypatch.setitem(
        sys.modules, module_name, SimpleNamespace(ApiGetKernelSessionStatusRequest=SimpleNamespace)
    )
    session = SimpleNamespace(
        send=lambda *args, **kwargs: SimpleNamespace(
            status_code=200,
            json=lambda: payload,
        )
    )

    def status(request):
        assert request.version_label == "v7"
        assert request.user_name == "user" and request.kernel_slug == "notebook"
        session.send(object())
        return SimpleNamespace(status=decoded, failure_message="")

    client = SimpleNamespace(
        http_client=lambda: SimpleNamespace(_session=session),
        kernels=SimpleNamespace(
            kernels_api_client=SimpleNamespace(get_kernel_session_status=status)
        ),
    )

    @contextmanager
    def get_client():
        yield client

    adapter = watcher.KaggleApiAdapter(None)
    monkeypatch.setattr(adapter, "_client", get_client)
    monkeypatch.setattr(adapter, "current_version", lambda _: 7)
    result = adapter.session_status("user/notebook", 7)
    assert result.status == expected


def test_stops_if_current_notebook_version_changed(tmp_path: Path) -> None:
    api = FakeApi(version=5)
    instance = make_watcher(tmp_path, api)

    with pytest.raises(watcher.VersionMismatch):
        instance.poll_once()

    assert api.status_calls == 0
    assert instance.state["terminal_status"] is None
    assert "VersionMismatch" in instance.state["last_error"]


def test_terminal_status_callback_is_at_most_once_across_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = FakeApi(statuses=["COMPLETE"])
    callback_calls: list[dict[str, object]] = []

    def fake_invoke(argv: list[str], prompt: str, timeout: float) -> dict[str, object]:
        callback_calls.append({"argv": argv, "prompt": prompt, "timeout": timeout})
        return {
            "returncode": 0,
            "timed_out": False,
            "error_type": None,
            "stdout": b"wake accepted",
            "stderr": b"",
        }

    monkeypatch.setattr(watcher, "_invoke_callback", fake_invoke)
    first = make_watcher(tmp_path, api, callback_argv=["wake-tool", "--thread", "abc"])

    assert first.poll_once() == "COMPLETE"
    assert len(callback_calls) == 1
    call = callback_calls[0]
    assert call["argv"] == ["wake-tool", "--thread", "abc"]
    assert "status COMPLETE" in call["prompt"]
    assert api.artifact_calls == 1
    assert (
        (first.output_dir / "callback.log")
        .read_text(encoding="utf-8")
        .endswith("wake accepted\nstderr:\n")
    )

    resumed = make_watcher(tmp_path, api, callback_argv=["wake-tool", "--thread", "abc"])
    assert resumed.poll_once() == "COMPLETE"
    assert len(callback_calls) == 1
    assert api.status_calls == 1
    assert api.artifact_calls == 1
    saved = json.loads(resumed.state_path.read_text(encoding="utf-8"))
    assert saved["callback_state"] == "ACKNOWLEDGED"


def test_ambiguous_callback_failure_is_not_retried_after_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = FakeApi(statuses=["ERROR"])
    callback_calls: list[list[str]] = []

    def ambiguous_invoke(argv: list[str], _prompt: str, _timeout: float) -> dict[str, object]:
        callback_calls.append(argv)
        return {
            "returncode": None,
            "timed_out": True,
            "error_type": "TimeoutExpired",
            "stdout": b"",
            "stderr": b"ack may have been lost",
        }

    monkeypatch.setattr(watcher, "_invoke_callback", ambiguous_invoke)
    first = make_watcher(tmp_path, api, callback_argv=["wake-tool"])
    assert first.poll_once() == "ERROR"
    assert first.state["callback_state"] == "FAILED"

    resumed = make_watcher(tmp_path, api, callback_argv=["wake-tool"])
    assert resumed.poll_once() == "ERROR"
    assert callback_calls == [["wake-tool"]]
    assert resumed.state["callback_attempted"] is True


def test_transient_poll_error_is_saved_and_recovers(tmp_path: Path) -> None:
    api = FakeApi(statuses=[OSError("temporary network interruption"), "RUNNING"])
    instance = make_watcher(tmp_path, api)

    assert instance.poll_once() is None
    assert instance.state["last_error"] == "OSError"
    assert instance.poll_once() is None
    assert instance.state["last_status"] == "RUNNING"
    assert instance.state["last_error"] is None


def test_lower_current_version_retries_until_expected_version_is_visible(
    tmp_path: Path,
) -> None:
    api = FakeApi(version=3)
    instance = make_watcher(tmp_path, api)

    assert instance.poll_once() is None
    assert instance.state["last_error"] == "VersionNotReady"
    assert api.status_calls == 0

    api.version = 4
    assert instance.poll_once() is None
    assert instance.state["last_status"] == "RUNNING"
    assert instance.state["last_error"] is None


def test_terminal_output_publication_gets_three_bounded_attempts(tmp_path: Path) -> None:
    class DelayedArtifactsApi(FakeApi):
        def download_terminal_artifacts(
            self,
            notebook_ref: str,
            expected_version: int,
            output_dir: Path,
        ) -> dict[str, dict[str, object]]:
            self.artifact_calls += 1
            if self.artifact_calls < 3:
                return {}
            return {"report.json": {"path": str(output_dir / "report.json"), "bytes": 14}}

    delays: list[float] = []
    api = DelayedArtifactsApi(statuses=["COMPLETE"])
    instance = make_watcher(tmp_path, api, sleeper=delays.append)

    assert instance.poll_once() == "COMPLETE"
    assert api.artifact_calls == 3
    assert delays == [watcher.ARTIFACT_RETRY_SECONDS] * 2
    assert instance.state["artifacts_attempted"] is True
    assert "report.json" in instance.state["artifacts"]


def test_platform_failure_message_is_bounded_and_persisted(tmp_path: Path) -> None:
    api = FakeApi(statuses=[SimpleNamespace(status="ERROR", failure_message="x" * 6_000)])
    instance = make_watcher(tmp_path, api)

    assert instance.poll_once() == "ERROR"
    assert len(instance.state["failure_message"]) == 4_096


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("COMPLETE", True),
        ("ERROR", True),
        ("FAILED", True),
        ("CANCELED", True),
        ("CANCEL_ACKNOWLEDGED", True),
        ("QUEUED", False),
        ("RUNNING", False),
        ("CANCEL_REQUESTED", False),
    ],
)
def test_kaggle_terminal_status_classification(status: str, expected: bool) -> None:
    assert watcher.is_terminal_status(status) is expected


def test_output_allowlist_rejects_model_files_and_traversal() -> None:
    files = [
        SimpleNamespace(
            file_name="glm53-offload-runtime/experiment-run/q6-abc123/report.json",
            url="https://storage.example/report",
        ),
        SimpleNamespace(
            file_name="glm53-offload-runtime/experiment-run/q6-abc123/worker.log",
            url="https://storage.example/log",
        ),
        SimpleNamespace(
            file_name="model/cache/weights.safetensors",
            url="https://storage.example/model",
        ),
        SimpleNamespace(
            file_name="model/cache/report.json",
            url="https://storage.example/fake-report",
        ),
        SimpleNamespace(
            file_name="../report.json",
            url="https://storage.example/traversal",
        ),
        SimpleNamespace(
            file_name="glm53-offload-runtime/experiment-run/q6-abc123/worker-traceback.txt",
            url="https://storage.example/traceback",
        ),
    ]

    selected = watcher._select_artifact_urls(files)

    assert selected == {
        "report.json": "https://storage.example/report",
        "worker.log": "https://storage.example/log",
        "worker-traceback.txt": "https://storage.example/traceback",
    }


def test_output_allowlist_rejects_multiple_run_folders() -> None:
    files = [
        SimpleNamespace(
            file_name="glm53-offload-runtime/experiment-run/q6-run-a/report.json",
            url="https://storage.example/a",
        ),
        SimpleNamespace(
            file_name="glm53-offload-runtime/experiment-run/q6-run-b/report.json",
            url="https://storage.example/b",
        ),
    ]

    with pytest.raises(watcher.AmbiguousArtifacts):
        watcher._select_artifact_urls(files)


@pytest.mark.parametrize("prefix", ["", "working/", "kaggle/working/"])
def test_qwen_output_collects_report_and_tracking_without_model_files(prefix: str) -> None:
    files = [
        SimpleNamespace(
            file_name=prefix + "qwen38-trial/" + name,
            url="https://storage.example/" + name,
        )
        for name in ("report.json", "supervisor.json", "tracking-readback.json", "worker.log")
    ]
    files.extend(
        SimpleNamespace(file_name=name, url="https://storage.example/rejected")
        for name in (
            prefix + "qwen38-trial/weights.gguf",
            prefix + "qwen38-runtime/report.json",
            prefix + "qwen38-trial/../report.json",
        )
    )
    assert watcher._select_artifact_urls(files) == {
        name: "https://storage.example/" + name
        for name in ("report.json", "supervisor.json", "tracking-readback.json", "worker.log")
    }


def test_download_limit_is_enforced_without_reading_unbounded_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeResponse:
        headers = {"Content-Length": "1000000"}

        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self, amount: int) -> bytes:
            raise AssertionError("known oversized files must be rejected before reading")

    monkeypatch.setattr(watcher, "urlopen", lambda *_args, **_kwargs: FakeResponse())
    with pytest.raises(watcher.ArtifactTooLarge):
        watcher._bounded_download("https://storage.example/huge", 64)


def test_callback_uses_argv_without_shell_and_caps_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeProcess:
        def __init__(self):
            self.stdin = io.BytesIO()
            self.stdout = io.BytesIO(b"ok" * (watcher.MAX_CALLBACK_OUTPUT_BYTES + 10))
            self.stderr = io.BytesIO(b"diagnostic")
            self.returncode = 0

        def wait(self, timeout: float) -> int:
            captured["timeout"] = timeout
            return 0

        def poll(self) -> int:
            return self.returncode

        def kill(self) -> None:
            self.returncode = -9

    def fake_popen(argv: list[str], **kwargs: object) -> FakeProcess:
        captured["argv"] = argv
        captured.update(kwargs)
        return FakeProcess()

    monkeypatch.setattr(watcher.subprocess, "Popen", fake_popen)
    result = watcher._invoke_callback(["wake", "--thread", "abc"], "prompt", 12)

    assert captured["argv"] == ["wake", "--thread", "abc"]
    assert captured["shell"] is False
    assert captured["timeout"] == 12
    assert len(result["stdout"]) == watcher.MAX_CALLBACK_OUTPUT_BYTES
    assert result["stderr"] == b"diagnostic"
