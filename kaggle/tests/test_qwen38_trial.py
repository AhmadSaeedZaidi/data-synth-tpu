"""Protect full batch occupancy, distinct quality coverage and per-chip limits."""

import json
from types import SimpleNamespace

import pytest
from qwen38_tpu.trial import MemoryTracker, replay_workload, update_summaries, validate_batches


def test_fixed_replay_workload_really_fills_32_and_preserves_cohort():
    prompts = [{"case": {"name": f"case-{i}"}, "ids": [i]} for i in range(8)]
    workload = replay_workload(prompts, 64, [8, 16, 32])
    assert len({p["request_id"] for p in workload}) == 64
    for batch in (8, 16, 32):
        chunks = [workload[first : first + batch] for first in range(0, 64, batch)]
        assert all(len(chunk) == batch for chunk in chunks)
        assert all(len({p["case"]["name"] for p in chunk}) == 8 for chunk in chunks)
    with pytest.raises(ValueError, match="fill every"):
        replay_workload(prompts, None, [8, 16, 32])
    with pytest.raises(ValueError, match="complete copies"):
        replay_workload(prompts, 36, [4, 12])


@pytest.mark.parametrize(
    "batches,requests", [([16, 8], 64), ([8, 8], 64), ([0], 64), ([8, 32], 48), ([8], 512)]
)
def test_invalid_trial_rejected_before_loading(batches, requests):
    with pytest.raises(ValueError):
        validate_batches(batches, requests)


def test_replays_do_not_inflate_quality_and_speed_uses_actual_rows():
    rows = [
        dict(name="ocr", batch=16, passed=i != 1, answer="7" if i != 1 else "?", generated_tokens=4)
        for i in range(16)
    ]
    report = {
        "batch_results": [
            dict(
                requested_batch=16,
                batch=16,
                cases=rows,
                decode_seconds=2,
                prefill_seconds=2,
                forward_tokens=48,
                allocated_row_steps=64,
            )
        ]
    }
    update_summaries(report)
    assert report["quality"]["asserted_cases"] == 1
    assert report["quality"]["failed_cases"] == ["ocr@batch16"]
    assert report["quality"]["passed_request_checks"] == 15
    assert report["quality"]["replay_answer_variations"] == ["ocr@batch16"]
    metrics = report["benchmarks"][0]
    assert metrics["actual_batches"] == [16]
    assert metrics["requests"] == 16
    assert metrics["aggregate_tokens_per_second"] == 24
    assert metrics["prepared_requests_per_second"] == 4
    assert metrics["active_row_utilization"] == 0.75


def test_memory_gate_checks_each_chip_and_observed_growth(tmp_path):
    gib = 1 << 30
    stats = [{"bytes_limit": 16 * gib, "bytes_in_use": 13 * gib} for _ in range(8)]
    devices = [SimpleNamespace(memory_stats=lambda i=i: stats[i]) for i in range(8)]
    tracker = MemoryTracker(devices, tmp_path / "memory.jsonl")
    plan = {"fits": True, "state_bytes": 2 * gib}
    assert tracker.admit(8, plan)["admitted"]
    baseline = len(tracker.records) - 1
    for s in stats:
        s["bytes_in_use"] += gib
    tracker.sample("decode", 8)
    tracker.complete(8, baseline)
    for s in stats:
        s["bytes_in_use"] -= gib
    assert tracker.admit(16, plan)["admitted"]
    assert not tracker.admit(32, plan)["admitted"]
    # A single full chip denies a shape even when total free HBM looks large.
    stats[0]["bytes_in_use"] = int(15.8 * gib)
    assert not tracker.admit(8, plan)["admitted"]
    assert tracker.summary()["sampled_peak_total_bytes_in_use"] is not None
    records = [json.loads(line) for line in tracker.path.read_text().splitlines()]
    assert records[-1]["minimum_free_bytes"] < tracker.min_free_bytes


def test_unavailable_counters_and_oversized_plan_are_skipped(tmp_path):
    tracker = MemoryTracker([SimpleNamespace(memory_stats=lambda: None)], tmp_path / "memory.jsonl")
    assert (
        tracker.admit(16, {"fits": True, "state_bytes": 1})["reason"]
        == "allocator_counters_unavailable"
    )
    assert tracker.admit(32, {"fits": False})["reason"] == "logical_memory_plan_exceeds_hbm"


@pytest.mark.parametrize("skip_32", [False, True])
@pytest.mark.parametrize("full_evaluation", [False, True])
@pytest.mark.parametrize("parallel_mode", ["serial", "auto"])
@pytest.mark.parametrize(
    "failed_gate", [None, "elementary_numeric_gate", "q8_codec_gate", "expert_routing_gate"]
)
def test_worker_runs_full_batches_and_preserves_smaller_results(
    tmp_path,
    monkeypatch,
    skip_32,
    full_evaluation,
    failed_gate,
    parallel_mode,
    failure_mode=None,
):
    import sys

    import numpy as np
    from qwen38_tpu import run

    config = dict(
        quant="q6",
        batch_sizes=[8, 16, 32],
        requests=64,
        max_context=2048,
        max_new_tokens=512,
        reasoning_effort="xhigh",
        expert_parallel=parallel_mode,
        profile_decode=True,
        output_dir=str(tmp_path),
        beamng_fixture_bundle="fixture",
        beamng_fixture_sha256="verified",
    )
    config_path = tmp_path / "config.json"
    if full_evaluation:
        config.update(evaluation_bundle="full-fixture", evaluation_sha256="verified")
        evaluation_cases = [
            dict(name="format-test", group="sft_formatting", asserted=True),
            dict(name="pair-test", group="beamng_pipeline_review", asserted=True),
            dict(name="no_image_control", group="control", asserted=False),
        ]
        if failure_mode == "evaluation_collapse":
            evaluation_cases += [dict(name=f"canary-{i}", asserted=True) for i in range(2)]
        monkeypatch.setattr(
            run, "load_bundle", lambda *args: ({}, evaluation_cases, {"coverage": {}}, "verified")
        )
    config_path.write_text(json.dumps(config))
    chosen_group = 1 if failure_mode in {"small_group", "selected_gate"} else 4
    monkeypatch.setattr(sys, "argv", ["run", "--config", str(config_path)])
    devices = [
        SimpleNamespace(
            platform="tpu",
            device_kind="TPU v5 lite",
            memory_stats=lambda: dict(bytes_limit=1 << 40, bytes_in_use=1000),
        )
        for _ in range(8)
    ]
    monkeypatch.setattr(run.jax, "devices", lambda: devices)
    monkeypatch.setattr(run, "ordered_devices", lambda devices: devices)
    monkeypatch.setattr(run.jax.sharding, "Mesh", lambda *args: object())
    monkeypatch.setattr(run, "validate", lambda *args, **kw: dict(status="PASS"))
    monkeypatch.setattr(run, "validate_numerics", lambda *args, **kw: dict(status="PASS"))
    monkeypatch.setattr(run, "validate_q8_codec", lambda *args, **kw: dict(status="PASS"))
    long_gate_calls = []

    def long_gate(*args, **kw):
        long_gate_calls.append(kw)
        if failure_mode == "selected_gate" and "prefill_expert_group_size" in kw:
            raise AssertionError("injected selected-policy mismatch")
        return dict(status="PASS")

    monkeypatch.setattr(run, "validate_long_context", long_gate)
    monkeypatch.setattr(run, "native_dot_proof", lambda: dict(status="PASS"))
    monkeypatch.setattr(run, "validate_expert_routing", lambda *args, **kw: dict(status="PASS"))
    calibration_calls = []

    def calibrate(mesh, config, *, compute, rows, decode_rows, emit):
        assert not failed_gate, "Failed gates must prevent performance selection"
        calibration_calls.append(rows)
        assert sorted(set(decode_rows)) == (
            [len(evaluation_cases), 8, 16, 32] if full_evaluation else [8, 16, 32]
        )
        if failure_mode == "calibration":
            return dict(
                status="FAIL",
                selected=None,
                selection_error="injected memory rejection",
                measurements=[dict(temporary_bytes=505 << 20)],
            )
        return dict(
            status="PASS",
            selected=dict(
                name="grouped_ep4_tp2",
                group_size=chosen_group,
                tensor_parallel=2,
                prefill_group_size=1 if failure_mode == "mixed_phases" else chosen_group,
            ),
        )

    monkeypatch.setattr(run, "calibrate_experts", calibrate)
    if failed_gate:

        def failed(*args, diagnostics=None, **kwargs):
            diagnostics["code_mismatches"] = 1
            raise AssertionError("injected cache mismatch")

        monkeypatch.setattr(
            run,
            {
                "elementary_numeric_gate": "validate_numerics",
                "q8_codec_gate": "validate_q8_codec",
                "expert_routing_gate": "validate_expert_routing",
            }[failed_gate],
            failed,
        )
    monkeypatch.setattr(run, "WorkerDiagnostics", lambda _: SimpleNamespace(close=lambda: None))
    checkpoint = SimpleNamespace(
        validate=lambda c: 1510,
        memory_plan=lambda c, batch, **kw: dict(
            fits=not (skip_32 and batch == 32),
            state_bytes=400,
            host_ngram_bytes=0,
            host_token_embedding_bytes=0,
        ),
        host_ngram=lambda c: (None, None),
    )

    def get_checkpoint(*args, **kwargs):
        assert not failed_gate, "Failed gates must prevent access to full weights"
        assert failure_mode != "calibration", "Rejected calibration must prevent full weights"
        assert failure_mode != "selected_gate", "Rejected selected policy must prevent full weights"
        return checkpoint

    monkeypatch.setattr(run, "Checkpoint", get_checkpoint)
    from glm53_offload import weights

    monkeypatch.setattr(weights, "available_ram", lambda: 128 << 30)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kw: None)),
    )
    monkeypatch.setattr(run, "load_vision", lambda *args: None)
    monkeypatch.setattr(run, "VisionEncoder", lambda *args: None)
    monkeypatch.setattr(run, "token_table", lambda *args: None)
    real_cases = [dict(name=f"beamng-{i}") for i in range(4)]
    monkeypatch.setattr(run, "load_scene_bundle", lambda *args: ({}, real_cases, {}, "verified"))
    monkeypatch.setattr(
        run,
        "prepare_prompt",
        lambda _, case, *args, **kw: dict(
            case=case,
            ids=np.arange(16),
            positions=np.zeros((16, 3), int),
            next_position=16,
            max_new_tokens=512,
            image_grids=[],
        ),
    )
    engine = SimpleNamespace(
        layers=[], experts=[], head=np.zeros(1), final_weights={}, release_state=lambda: None
    )

    def load_engine(*args, **kw):
        assert kw["expert_tensor_parallel"] == (2 if parallel_mode == "auto" else 1)
        assert kw["expert_group_size"] == (chosen_group if parallel_mode == "auto" else 1)
        assert kw["prefill_expert_group_size"] == (
            chosen_group if parallel_mode == "auto" and failure_mode != "mixed_phases" else 1
        )
        assert kw["gdn_prefill"] == "serial"
        assert kw["fuse_layers"] is True
        return engine

    monkeypatch.setattr(run.Engine, "from_checkpoint", load_engine)
    measured = []
    profiles = []
    monkeypatch.setattr(run, "prefill", lambda engine, chunk, *args: (None, None, 1.0))

    def generate(engine, chunk, *args, record_cases=True, profile_directory=None):
        if profile_directory:
            profiles.append((record_cases, str(profile_directory)))
        if record_cases:
            measured.append(len(chunk))
        rows = [
            dict(
                name=p["case"]["name"],
                batch=len(chunk),
                request_id=p["request_id"],
                replay_index=p["replay_index"],
                generated_tokens=4,
                answer="7",
                answer_phase="THINKING" if failure_mode == "evaluation_collapse" else "FINAL",
                truncated=failure_mode == "evaluation_collapse",
                passed=failure_mode != "evaluation_collapse"
                and p["case"]["name"] != "read_text_stop_7",
                group=p["case"].get("group", "unknown"),
                asserted=p["case"].get("asserted", True),
            )
            for p in chunk
        ]
        return rows, dict(
            decode_seconds=1, forward_tokens=3 * len(chunk), allocated_row_steps=4 * len(chunk)
        )

    monkeypatch.setattr(run, "generate", generate)
    if failed_gate:
        from qwen38_tpu.gates import NumericalGateError

        with pytest.raises(NumericalGateError):
            run.main()
        report = json.loads((tmp_path / "report.json").read_text())
        assert report["status"] == "ERROR"
        expected_failures = [failed_gate]
        if failed_gate == "expert_routing_gate" and parallel_mode == "auto":
            expected_failures.extend(["grouped_tp1_routing_gate", "grouped_tp2_routing_gate"])
        assert report["gate_failures"] == expected_failures
        assert report[failed_gate]["code_mismatches"] == 1
        assert report["bf16_numeric_gate"]["status"] == "PASS"
        assert report["native_compute_proof"]["status"] == "PASS"
        assert not report["batch_results"] and not report["evaluation_results"]
        assert "NumericalGateError" in (tmp_path / "worker-traceback.txt").read_text()
        return
    if failure_mode == "calibration":
        with pytest.raises(RuntimeError, match="Expert calibration rejected"):
            run.main()
        report = json.loads((tmp_path / "report.json").read_text())
        assert report["expert_calibration"]["measurements"] == [dict(temporary_bytes=505 << 20)]
        assert not measured and not report["batch_results"]
        return
    if failure_mode == "selected_gate":
        from qwen38_tpu.gates import NumericalGateError

        with pytest.raises(NumericalGateError):
            run.main()
        report = json.loads((tmp_path / "report.json").read_text())
        assert report["status"] == "ERROR"
        assert report["gate_failures"] == [
            "selected_phase_float32_gate",
            "selected_phase_bfloat16_gate",
        ]
        assert not measured and not report["batch_results"]
        return
    run.main()
    report = json.loads((tmp_path / "report.json").read_text())
    if failure_mode == "evaluation_collapse":
        assert report["status"] == "QUALITY_FAILED"
        assert report["evaluation"]["completed_cases"] == 5
        assert report["benchmark_coverage"] == "SKIPPED"
        assert report["evaluation_sanity"]["status"] == "FAIL"
        assert not report["batch_results"] and not report["batch_trials"]
        assert measured == [5] and profiles == [
            (True, str(tmp_path / "profiles" / "evaluation-first" / "device-trace"))
        ]
        return
    if parallel_mode == "auto":
        assert report["expert_calibration"]["selected"]["tensor_parallel"] == 2
        expected_rows = (
            [3, 8, 16, 32, 384, 1024, 2048] if full_evaluation else [8, 16, 32, 1024, 2048]
        )
        assert calibration_calls == [expected_rows]
    if parallel_mode == "auto":
        assert report[f"selected_group{chosen_group}_routing_gate"]["status"] == "PASS"
        for label in ("float32", "bfloat16"):
            assert report[f"selected_phase_{label}_gate"]["status"] == "PASS"
        selected = [kw for kw in long_gate_calls if "prefill_expert_group_size" in kw]
        assert len(selected) == 2
        assert all(
            kw["prefill_expert_group_size"]
            == (1 if failure_mode == "mixed_phases" else chosen_group)
            for kw in selected
        )
    assert report["status"] == "QUALITY_FAILED"
    assert report["benchmark_coverage"] == ("PARTIAL" if skip_32 else "COMPLETE")
    assert measured == ([3] if full_evaluation else []) + [8] * 8 + [16] * 4 + (
        [] if skip_32 else [32] * 2
    )
    assert all(r["batch"] == r["requested_batch"] for r in report["batch_results"])
    assert report["quality"]["unique_cases"] == 8 + (2 if full_evaluation else 0)
    assert report["quality"]["asserted_cases"] == (16 if skip_32 else 24) + (
        2 if full_evaluation else 0
    )
    assert report["quality"]["request_checks"] == (128 if skip_32 else 192) + (
        2 if full_evaluation else 0
    )
    if full_evaluation:
        assert report["evaluation"]["coverage"] == "COMPLETE"
        assert report["evaluation"]["event_accuracy"] is None
        assert report["quality"]["unasserted_controls"] == 1
        assert all(b["requests"] == 64 for b in report["benchmarks"])
    assert report["batch_trials"][-1]["status"] == ("SKIPPED" if skip_32 else "COMPLETE")
    assert report["workload"]["kind"] == "labeled_cohort_replay"
    assert profiles == (
        [(True, str(tmp_path / "profiles" / "evaluation-first" / "device-trace"))]
        if full_evaluation
        else []
    ) + [
        (False, str(tmp_path / "profiles" / f"batch-{batch}" / "device-trace"))
        for batch in ([8, 16] if skip_32 else [8, 16, 32])
    ]
    progress = json.loads((tmp_path / "progress.json").read_text())
    assert progress["hbm_minimum_free_bytes_per_chip"] > 0


@pytest.mark.parametrize(
    "failure_mode",
    ["calibration", "evaluation_collapse", "mixed_phases", "small_group", "selected_gate"],
)
def test_worker_persists_new_guard_failures_and_skips_remaining_work(
    tmp_path, monkeypatch, failure_mode
):
    test_worker_runs_full_batches_and_preserves_smaller_results(
        tmp_path,
        monkeypatch,
        skip_32=True,
        full_evaluation=True,
        failed_gate=None,
        parallel_mode="auto",
        failure_mode=failure_mode,
    )
