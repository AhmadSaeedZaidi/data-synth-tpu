"""Check that tracking gates reject missing credentials or missing remote records."""

import json
import sys
from types import SimpleNamespace

import pytest
from glm53_offload import tracking_check


def test_embedded_environment_key_skips_kaggle_secret_lookup(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "task-test-value")
    monkeypatch.setitem(sys.modules, "kaggle_secrets", None)
    assert tracking_check.load_kaggle_key() == {"source": "environment", "status": "PASS"}


def test_failed_secret_lookup_classifies_error_without_copying_response(monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    secret_value = "never-publish-this-test-value"

    class BackendError(Exception):
        pass

    def retrieve(label):
        raise BackendError("No user secrets exist; response contains " + secret_value)

    monkeypatch.setitem(
        sys.modules,
        "kaggle_secrets",
        SimpleNamespace(UserSecretsClient=lambda: SimpleNamespace(get_secret=retrieve)),
    )
    with pytest.raises(tracking_check.TrackingCheckError) as caught:
        tracking_check.load_kaggle_key()
    assert "secret_not_attached" in str(caught.value)
    assert secret_value not in str(caught.value)


@pytest.mark.parametrize("mode,weave", [("offline", True), ("online", False)])
def test_missing_online_tracking_rejected_before_weave_configuration(tmp_path, mode, weave):
    monitor = SimpleNamespace(state={"mode": mode}, run=SimpleNamespace())
    with pytest.raises(tracking_check.TrackingCheckError):
        tracking_check.require_live_tracking(monitor, {"weave_enabled": weave}, tmp_path)
    assert not (tmp_path / "tracking-readback.json").exists()


def test_successful_gate_requires_open_remote_span_and_matching_wandb_marker(tmp_path, monkeypatch):
    from glm53_offload import tracing

    class Summary(dict):
        pass

    client = SimpleNamespace(entity="test", flush=lambda: None)
    root = SimpleNamespace(trace_id="trace-test")
    configure = []
    finished = []

    def init(config, directory):
        configure.append(config)
        monkeypatch.setattr(tracing, "_client", client)
        monkeypatch.setattr(tracing, "_root", root)

    monkeypatch.setattr(tracing, "configure", init)
    monkeypatch.setattr(tracing, "finish", lambda status: finished.append(status))
    # Use a real context manager without publishing through a fake client.
    from contextlib import contextmanager

    @contextmanager
    def span(*args, **kwargs):
        yield {}

    monkeypatch.setattr(tracing, "span", span)
    monitor = SimpleNamespace(
        state={"mode": "online"},
        run=SimpleNamespace(
            summary=Summary(),
            log=lambda _: None,
            id="run-test",
            url="https://wandb.ai/test/run-test",
        ),
    )
    monkeypatch.setattr(
        tracking_check,
        "read_weave_calls",
        lambda *args: [
            {
                "op_name": "weave:///test/project/op/glm53.tracking_ready:digest",
                "ended_at": None,
            }
        ],
    )
    monkeypatch.setattr(
        tracking_check, "read_wandb_summary", lambda *args: {"summary": dict(monitor.run.summary)}
    )
    proof = tracking_check.require_live_tracking(
        monitor, {"weave_enabled": True, "wandb_project": "project"}, tmp_path
    )
    assert proof["open_weave_span_observed"] and proof["wandb_heartbeat_observed"]
    assert isinstance(monitor.run.summary["tracking/readback_marker"], str)
    assert configure[0]["allow_tracking_fallback"] is False
    assert finished == ["TRACKING_CHECK_COMPLETE"]
    assert json.loads((tmp_path / "tracking-readback.json").read_text())["status"] == "PASS"


def test_wait_for_remote_never_accepts_an_unacknowledged_record(monkeypatch):
    tick = iter([0, 0, 1, 2, 3])
    monkeypatch.setattr(tracking_check.time, "monotonic", lambda: next(tick))
    monkeypatch.setattr(tracking_check.time, "sleep", lambda _: None)
    with pytest.raises(tracking_check.TrackingCheckError, match="not acknowledged"):
        tracking_check.wait_for_remote(lambda: [], bool, timeout=2)
