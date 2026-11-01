"""Quota accounting, process cleanup and failure evidence for the quant comparison."""

import json
from types import SimpleNamespace

import pytest
from glm53_offload.comparison import CORE_CASES, run_comparison


def invoke(tmp_path, launch, *, clock, config=None, started=0):
    settings = {
        "quant": "Q6_K_XL",
        "run_model": True,
        "deadline_seconds": 21600,
        "reasoning_effort": "max",
        "thinking_budget": None,
        "max_new_tokens": 256,
        "batch_sizes": [1, 4, 8],
        **(config or {}),
    }
    return run_comparison(
        settings,
        runtime_root=tmp_path,
        runtime_dir=tmp_path / "runtime",
        bundle_sha256="fixture-sha",
        session_started_at=started,
        launcher_source="fixture-launcher",
        launch_trial=launch,
        clock=lambda: clock[0],
    )


def record_worker(environment, status="PASS", supervisor="EXITED"):
    environment.update(
        report={
            "status": status,
            "quality": {"scope": "comparison", "cases": [{"name": "count_circles_7"}]},
            "elapsed_seconds": 10,
        },
        supervision={"status": supervisor},
        process=SimpleNamespace(poll=lambda: 0),
    )


def test_one_budget_includes_setup_and_carries_unused_time_to_second_quant(tmp_path):
    clock = [600]  # Dependency setup already used ten minutes.
    environments = []

    def launch(source, environment):
        environments.append(environment)
        record_worker(environment)
        clock[0] += 3600

    report, path = invoke(tmp_path, launch, clock=clock)
    assert report["status"] == "PASS"
    assert [e["CONFIG"]["quant"] for e in environments] == ["Q4_K_XL", "Q6_K_XL"]
    assert environments[0] is not environments[1]
    assert [e["CONFIG"]["deadline_seconds"] for e in environments] == [10440, 17280]
    for env in environments:
        assert env["CONFIG"]["quality_case_names"] == CORE_CASES
        assert env["CONFIG"]["experiment_mode"] == "full"
        assert env["CONFIG"]["thinking_budget"] is None
        assert env["CONFIG"]["reasoning_effort"] == "max"
        assert not env["CONFIG"]["long_image_benchmarks"]
    assert json.loads(path.read_text())["session_elapsed_seconds"] == 7800
    assert list(tmp_path.glob("quant-comparison-*.zip"))


@pytest.mark.parametrize("failure", ["FAIL", "TIMEOUT", "ERROR"])
def test_first_failure_preserves_partial_results_and_does_not_cancel_second(tmp_path, failure):
    clock = [0]
    calls = []

    def launch(source, environment):
        calls.append(environment["CONFIG"]["quant"])
        if len(calls) == 1:
            record_worker(
                environment,
                status="RUNNING" if failure == "TIMEOUT" else failure,
                supervisor="TIMEOUT" if failure == "TIMEOUT" else "EXITED",
            )
            clock[0] += environment["CONFIG"]["deadline_seconds"] + 25
            raise RuntimeError("first worker failed after saving evidence")
        record_worker(environment)

    report, path = invoke(tmp_path, launch, clock=clock)
    assert calls == ["Q4_K_XL", "Q6_K_XL"]
    assert report["trials"][0]["quality"]["cases"]
    assert report["trials"][0]["status"] == failure
    assert report["trials"][1]["status"] == "PASS"
    assert report["status"] == ("FAIL" if failure == "FAIL" else "INCOMPLETE")
    assert json.loads(path.read_text())["status"] == report["status"]


def test_no_second_model_is_started_if_first_process_has_not_exited(tmp_path):
    calls = []

    def launch(source, environment):
        calls.append(environment["CONFIG"]["quant"])
        record_worker(environment, status="ERROR")
        environment["process"] = SimpleNamespace(poll=lambda: None)
        raise RuntimeError("worker has not been reaped")

    report, _ = invoke(tmp_path, launch, clock=[0])
    assert calls == ["Q4_K_XL"]
    assert report["status"] == "INCOMPLETE"
    assert report["trials"][1]["status"] == "SKIPPED_WORKER_NOT_EXITED"


def test_insufficient_load_budget_is_recorded_without_starting_any_model(tmp_path):
    def unexpected(*args):
        raise AssertionError("must not start a model without its minimum load budget")

    report, _ = invoke(tmp_path, unexpected, clock=[21000])
    assert report["status"] == "INCOMPLETE"
    assert all(row["status"] == "SKIPPED_BUDGET" for row in report["trials"])


def test_interrupt_stops_the_comparison_and_preserves_its_index(tmp_path):
    def interrupted(*args):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        invoke(tmp_path, interrupted, clock=[0])
    path = next(tmp_path.rglob("comparison-report.json"))
    assert json.loads(path.read_text())["status"] == "INTERRUPTED"


def test_no_model_comparison_does_not_claim_model_quality(tmp_path):
    def launch(source, environment):
        record_worker(environment, status="PREFLIGHT_ONLY")

    report, _ = invoke(tmp_path, launch, clock=[0], config={"run_model": False})
    assert report["status"] == "PREFLIGHT_ONLY"


def test_configured_real_scene_cases_and_larger_cache_reach_both_workers(tmp_path):
    configurations = []
    cases = [f"beamng_case_{i}" for i in range(8)]

    def launch(source, environment):
        configurations.append(environment["CONFIG"])
        record_worker(environment)

    report, _ = invoke(
        tmp_path,
        launch,
        clock=[0],
        config={
            "quality_case_names": cases,
            "expert_cache_mib_per_device": 7168,
            "beamng_fixture_bundle": "fixtures.bundle",
            "beamng_fixture_sha256": "fixture-sha",
        },
    )
    assert report["status"] == "PASS"
    assert len(configurations) == 2
    for config in configurations:
        assert config["quality_case_names"] == cases
        assert config["expert_cache_mib_per_device"] == 7168
        assert config["beamng_fixture_sha256"] == "fixture-sha"


@pytest.mark.parametrize("quants", [[], ["Q4_K_XL"] * 2, ["bad"], [{}]])
def test_invalid_quantizations_fail_before_worker_start(tmp_path, quants):
    def unexpected(*args):
        raise AssertionError("invalid configuration must not invoke worker")

    with pytest.raises(ValueError, match="quantizations"):
        invoke(tmp_path, unexpected, clock=[0], config={"quantizations": quants})
