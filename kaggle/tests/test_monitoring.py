import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from glm53_offload import progress, tracing
from glm53_offload.monitoring import RunMonitor


@pytest.fixture(autouse=True)
def reset_progress():
    yield
    progress.configure(None)
    tracing.finish("TEST_COMPLETE")


def test_parallel_download_counts_are_not_lost(tmp_path):
    path = tmp_path / "progress.json"
    progress.configure(path)
    progress.update(stage="HOST_EXPERT_LOAD", force=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: progress.update(downloaded=1024), range(128)))
    progress.update(retried=2, force=True)
    snapshot = json.loads(path.read_text())
    assert snapshot["download_bytes"] == 128 * 1024
    assert snapshot["download_retries"] == 2
    assert snapshot["stage"] == "HOST_EXPERT_LOAD"
    assert not path.with_suffix(".tmp").exists()


def test_monitor_reports_stale_progress_without_refreshing_it(tmp_path, monkeypatch):
    from glm53_offload import monitoring

    now = [100.0]
    monkeypatch.setattr(monitoring.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(monitoring.time, "time", lambda: now[0])
    monitor = RunMonitor(tmp_path, {"inputs": [{"text": "private prompt"}]})
    path = tmp_path / "progress.json"
    path.write_text(
        json.dumps({"stage": "HOST_EXPERT_LOAD", "updated_at": 90, "download_bytes": 100})
    )
    monitor.tick(os.getpid())
    assert monitor.metrics["progress/age_seconds"] == 10
    now[0] = 130
    monitor.tick(os.getpid())
    assert monitor.metrics["progress/age_seconds"] == 40
    assert monitor.metrics["download/bytes_per_second"] == 0
    assert monitor.metrics["worker/VmRSS_bytes"] > 0
    assert "private prompt" not in (tmp_path / "metrics.jsonl").read_text()
    assert json.loads(path.read_text())["updated_at"] == 90


def test_worker_log_is_relayed_incrementally(tmp_path, capsys):
    monitor = RunMonitor(tmp_path, {})
    path = tmp_path / "worker.log"
    path.write_text("layer 1\n")
    monitor.relay()
    monitor.relay()
    with path.open("a") as f:
        f.write("layer 2\n")
    monitor.relay()
    assert capsys.readouterr().out == "layer 1\nlayer 2\n"


def test_qwen_evaluation_groups_are_visible_without_fake_event_accuracy(tmp_path):
    report = dict(
        status="RUNNING",
        evaluation=dict(
            expected_cases=49,
            completed_cases=8,
            event_accuracy=None,
            groups={
                "sft_formatting": dict(
                    completed=4, asserted=4, passed=2, syntax_and_schema_passed=3, content_passed=2
                ),
                "beamng_pipeline_review": dict(
                    completed=4,
                    asserted=4,
                    passed=3,
                    syntax_and_schema_passed=3,
                    content_passed=None,
                ),
            },
        ),
    )
    (tmp_path / "report.json").write_text(json.dumps(report))
    monitor = RunMonitor(tmp_path, {})
    monitor.tick(force=True)
    assert monitor.metrics["evaluation/completed_cases"] == 8
    assert monitor.metrics["evaluation/sft_formatting/syntax_and_schema_passed"] == 3
    assert monitor.metrics["evaluation/beamng_pipeline_review/passed"] == 3
    assert "evaluation/event_accuracy" not in monitor.metrics


def test_live_execution_timings_survive_incomplete_tail_without_report(tmp_path):
    from glm53_offload.monitoring import read_execution_samples

    path = tmp_path / "execution-timing.jsonl"
    sample = {"phase": "prefill", "tokens": 256, "sample_number": 1, "status": "START"}
    path.write_text("x" * (70 << 10) + "\n" + json.dumps(sample) + '\n{"partial":')
    assert read_execution_samples(tmp_path) == [sample]
    monitor = RunMonitor(tmp_path, {})
    monitor.tick()
    prefix = "execution/prefill/tokens_256/sample_1/"
    assert monitor.metrics[prefix + "completed"] == 0
    sample.update(status="DONE", staging_cache_wait_seconds=2.5, packed_bytes_submitted=4096)
    path.write_text(json.dumps(sample) + "\n")
    monitor.tick(force=True)
    assert monitor.metrics[prefix + "completed"] == 1
    assert monitor.metrics[prefix + "staging_cache_wait_seconds"] == 2.5
    assert monitor.metrics[prefix + "packed_bytes_submitted"] == 4096


def test_live_timing_purposes_and_layers_do_not_overwrite_each_other(tmp_path):
    samples = [
        {
            "purpose": purpose,
            "phase": "decode",
            "tokens": 8,
            "layer": layer,
            "sample_number": 1,
            "status": "DONE",
            "elapsed_seconds": value,
        }
        for purpose, layer, value in (
            ("startup", 0, 4),
            ("matched_images", 0, 1),
            ("matched_images", 21, 2),
        )
    ]
    (tmp_path / "execution-timing.jsonl").write_text(
        "\n".join(json.dumps(sample) for sample in samples) + "\n"
    )
    monitor = RunMonitor(tmp_path, {})
    monitor.tick(force=True)
    for sample in samples:
        prefix = f"execution/{sample['purpose']}/layer_{sample['layer']}/decode/tokens_8/sample_1"
        assert monitor.metrics[prefix + "/elapsed_seconds"] == sample["elapsed_seconds"]


def test_online_tracking_requires_credentials_before_worker_launch(tmp_path, monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="Kaggle secret"):
        RunMonitor(tmp_path, {}, mode="online")


def test_tracking_startup_failure_keeps_local_metrics(tmp_path, monkeypatch):
    def fail(**kwargs):
        raise OSError("network unavailable")

    monkeypatch.setenv("WANDB_API_KEY", "secret-test")
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=fail, Settings=lambda **kw: kw))
    monitor = RunMonitor(tmp_path, {"allow_tracking_fallback": True}, mode="online")
    monitor.tick()
    assert monitor.state["mode"] == "disabled"
    assert monitor.state["requested_mode"] == "online"
    assert (tmp_path / "metrics.jsonl").is_file()
    assert "secret-test" not in (tmp_path / "monitor.json").read_text()


def test_weave_startup_failure_keeps_local_nested_traces(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("network unavailable")

    monkeypatch.setitem(sys.modules, "weave", SimpleNamespace(init=fail))
    tracing.configure(
        {
            "weave_enabled": True,
            "wandb_mode": "online",
            "allow_tracking_fallback": True,
        },
        tmp_path,
    )
    with tracing.span("outer", prompt_tokens=32):
        with tracing.span("inner", layer=1):
            pass
    records = [json.loads(line) for line in (tmp_path / "traces.jsonl").read_text().splitlines()]
    assert [r["operation"] for r in records] == ["inner", "outer"]
    assert json.loads((tmp_path / "tracing-status.json").read_text())["mode"] == "local"


def test_wandb_receives_only_metrics_and_explicit_config(tmp_path, monkeypatch):
    calls = []
    run = SimpleNamespace(
        summary={},
        log=lambda data: calls.append(data),
        finish=lambda **kw: calls.append(kw),
        url="https://wandb.ai/test/project/runs/123",
        id="123",
    )
    init = []
    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(
            init=lambda **kw: init.append(kw) or run,
            Settings=lambda **kw: kw,
        ),
    )
    monkeypatch.setenv("WANDB_API_KEY", "secret-test-value")
    monitor = RunMonitor(
        tmp_path,
        {
            "quant": "Q6_K_XL",
            "inputs": "private",
            "api_key": "secret",
            "direct_expert_cache": True,
            "integer_dot_probe": True,
            "integer_projection_probe": True,
            "matched_replay_steps": 16,
            "profile_matched_decode": True,
            "vision_resolution_control_tokens": 1024,
        },
        mode="online",
    )
    assert init[0]["config"] == {
        "quant": "Q6_K_XL",
        "direct_expert_cache": True,
        "integer_dot_probe": True,
        "integer_projection_probe": True,
        "matched_replay_steps": 16,
        "profile_matched_decode": True,
        "vision_resolution_control_tokens": 1024,
    }
    assert init[0]["settings"]["console"] == "off"
    report = {
        "stage": "COMPLETE",
        "status": "PASS",
        "validation": {"vision": "PASS"},
        "benchmarks": [{"batch_size": 8, "tokens_per_second": 12.0, "answers": ["private"]}],
        "matched_replay_benchmarks": [
            {
                "batch_size": 8,
                "tokens_per_second": 9.0,
                "forced_token_ids": ["private"],
                "host_gather_seconds": 1.5,
            }
        ],
        "integer_dot_probe": {
            "cases": [
                {
                    "dtype": "int4",
                    "status": "PASS",
                    "shape_per_chip": [8, 1024, 256],
                    "median_seconds": 0.002,
                    "stablehlo_dot": ["private"],
                }
            ]
        },
        "quality": {"asserted_cases": 12, "failed_cases": []},
        "matched_decode_diagnostic": {
            "status": "MEASURED",
            "case_names": ["private"],
            "replay": {"host_gather_seconds": 2.0},
        },
        "vision_resolution_controls": [
            {
                "case": "road_text",
                "status": "FAIL",
                "answer": "private",
                "image_max_tokens": 1024,
                "answer_match": False,
            }
        ],
    }
    (tmp_path / "report.json").write_text(json.dumps(report))
    monitor.finish(0)
    assert calls[0]["batch_8/tokens_per_second"] == 12
    assert calls[0]["validation/vision_pass"] == 1
    assert calls[0]["matched_replay_benchmarks/8/tokens_per_second"] == 9.0
    assert calls[0]["integer_dot_probe/int4/rows_8/measured"] == 1
    assert calls[0]["integer_dot_probe/int4/rows_8/median_seconds"] == 0.002
    assert calls[0]["matched_decode_diagnostic/host_gather_seconds"] == 2.0
    assert calls[0]["matched_decode_diagnostic/completed"] == 1
    assert calls[0]["vision_resolution_controls/road_text/image_max_tokens"] == 1024
    assert calls[0]["vision_resolution_controls/road_text/completed"] == 1
    assert calls[0]["vision_resolution_controls/road_text/answer_match"] == 0
    assert calls[-1] == {"exit_code": 0}
    assert "answers" not in str(calls)
    assert "private" not in str(calls)
    assert "secret-test-value" not in (tmp_path / "monitor.json").read_text()


def test_logging_failure_does_not_abort_worker_monitor(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("network unavailable")

    run = SimpleNamespace(summary={}, log=fail, finish=fail)
    monkeypatch.setitem(
        sys.modules, "wandb", SimpleNamespace(init=lambda **kw: run, Settings=lambda **kw: kw)
    )
    monitor = RunMonitor(tmp_path, {}, mode="offline")
    monitor.tick()
    monitor.finish(1)
    state = json.loads((tmp_path / "monitor.json").read_text())
    assert state["status"] == "ERROR"
    assert state["tracking_error"] == "OSError"
    assert (tmp_path / "metrics.jsonl").is_file()


def test_weave_nested_spans_errors_and_scalar_outputs(tmp_path, monkeypatch):
    stack, calls, completed = [], [], []

    def create_call(**kw):
        call = {**kw, "parent": stack[-1]["op"] if stack else None}
        stack.append(call)
        calls.append(call)
        return call

    def finish_call(call, **kw):
        assert stack.pop() is call
        completed.append((call, kw))

    client = SimpleNamespace(
        entity="test", create_call=create_call, finish_call=finish_call, flush=lambda: None
    )
    settings = []
    monkeypatch.setitem(
        sys.modules,
        "weave",
        SimpleNamespace(init=lambda project, **kw: settings.append(kw) or client),
    )
    tracing.configure({"weave_enabled": True, "wandb_mode": "online", "quant": "Q6_K_XL"}, tmp_path)

    @tracing.operation("generate")
    def generate(secret_argument):
        with tracing.span("prefill", prompt_tokens=32):
            pass
        return {"generated_tokens": 8, "answers": [secret_argument]}

    assert generate("private prompt")["generated_tokens"] == 8
    with pytest.raises(ValueError, match="original error"):
        with tracing.span("failure"):
            raise ValueError("original error")
    tracing.finish("ERROR")
    assert not stack
    assert calls[1]["parent"] == "glm53.experiment"
    assert calls[2]["parent"] == "glm53.generate"
    assert "private prompt" not in (tmp_path / "traces.jsonl").read_text()
    failure = next(kw for call, kw in completed if call["op"] == "glm53.failure")
    assert str(failure["exception"]) == "ValueError"
    assert settings[0]["settings"]["use_calls_complete"] is False


def test_weave_requires_online_mode(tmp_path):
    with pytest.raises(ValueError, match="online"):
        tracing.configure({"weave_enabled": True, "wandb_mode": "offline"}, tmp_path)
