"""Sequential quantization trials with one shared session budget and saved evidence."""

import tempfile
import time
from pathlib import Path

from .diagnostics import atomic_json, package_evidence

# Eight independent prompts for actual decode batches 1/4/8, covering colour,
# counting, OCR and spatial reasoning. The standalone full trial keeps all 13.
CORE_CASES = [
    "colour_red",
    "colour_green",
    "colour_blue",
    "count_circles_7",
    "count_circles_3",
    "read_text_stop_7",
    "read_text_go_3",
    "red_on_left",
]


def launch(source, environment):
    # The existing launcher starts and joins a fresh OS worker. No JAX model
    # arrays enter the notebook parent; process exit releases RAM and TPU buffers.
    exec(compile(source, "quantization-trial", "exec"), environment)


def run_comparison(
    config,
    *,
    runtime_root,
    runtime_dir,
    bundle_sha256,
    session_started_at,
    launcher_source,
    launch_trial=launch,
    clock=time.monotonic,
):
    quants = config.get("quantizations", ["Q4_K_XL", "Q6_K_XL"])
    if (
        not isinstance(quants, list)
        or not quants
        or any(not isinstance(q, str) for q in quants)
        or len(set(quants)) != len(quants)
        or any(q not in ("Q4_K_XL", "Q6_K_XL") for q in quants)
    ):
        raise ValueError("quantizations must contain distinct Q4_K_XL/Q6_K_XL trials")
    runtime_root, runtime_dir = Path(runtime_root), Path(runtime_dir)
    root = runtime_root / "experiment-run"
    root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="quant-comparison-", dir=root))
    path = directory / "comparison-report.json"
    report = {
        "status": "RUNNING",
        "quantizations": quants,
        "deadline_seconds": config["deadline_seconds"],
        "batch_sizes": config.get("batch_sizes", [1, 4, 8]),
        "runtime_bundle_sha256": bundle_sha256,
        "scope": (
            "matched eight single-image checks and configured decode batches; "
            "not the full 13-case battery"
        ),
        "trials": [],
    }

    def save():
        report["session_elapsed_seconds"] = max(0, clock() - session_started_at)
        atomic_json(path, report)

    def remaining():
        return config["deadline_seconds"] - (clock() - session_started_at)

    save()
    print("Comparison report:", path, flush=True)
    try:
        for index, quant in enumerate(quants):
            # Retain two minutes for parent cleanup/evidence. Setup time and all
            # previous trials count against the same six hours. An early finish
            # leaves its unused time for the following trial.
            allocation = max(0, int((remaining() - 120) / (len(quants) - index)))
            minimum = 5400 if config["run_model"] else 1
            row = {"quant": quant, "allocated_seconds": allocation, "status": "STARTING"}
            report["trials"].append(row)
            if allocation < minimum:
                row.update(status="SKIPPED_BUDGET", remaining_seconds=max(0, remaining()))
                save()
                continue
            trial = dict(config)
            trial.update(
                quant=quant,
                experiment_mode="full",
                deadline_seconds=allocation,
                quality_case_names=list(config.get("quality_case_names", CORE_CASES)),
                long_image_benchmarks=False,
                comparison_id=directory.name,
            )
            trial.pop("worker_budget_seconds", None)
            trial.pop("work_dir", None)
            environment = {
                "CONFIG": trial,
                "Path": Path,
                "tempfile": tempfile,
                "_runtime_root": runtime_root,
                "_runtime_dir": runtime_dir,
                "_EXPECTED_ARCHIVE_SHA256": bundle_sha256,
                "_experiment_started_at": clock(),
            }
            save()
            print(
                f"Starting {quant}; shared-session allocation {allocation / 60:.1f} min", flush=True
            )
            try:
                launch_trial(launcher_source, environment)
            except Exception as error:
                # A failed accuracy check or worker must not prevent the other
                # quantization from using its reserved budget.
                row["launcher_error"] = type(error).__name__
            except BaseException:
                row["status"] = "INTERRUPTED"
                report["status"] = "INTERRUPTED"
                save()
                raise
            worker = environment.get("report", {})
            supervisor = environment.get("supervision", {})
            row.update(
                status=str(worker.get("status", "ERROR")).upper(),
                supervisor_status=supervisor.get("status"),
                report_path=str(environment.get("REPORT_PATH", "")),
                evidence_path=str(environment.get("EVIDENCE_PATH", "")),
                quality=worker.get("quality"),
                benchmarks=worker.get("benchmarks"),
                early_decode_probe=worker.get("early_decode_probe"),
                matched_replay_benchmarks=worker.get("matched_replay_benchmarks"),
                matched_decode_diagnostic=worker.get("matched_decode_diagnostic"),
                vision_resolution_controls=worker.get("vision_resolution_controls"),
                integer_dot_probe=worker.get("integer_dot_probe"),
                integer_projection_probe=worker.get("integer_projection_probe"),
                elapsed_seconds=worker.get("elapsed_seconds"),
                weights=worker.get("weights"),
                hbm_plan=worker.get("hbm_plan"),
                expert_cache=worker.get("expert_cache"),
                scene_fixtures=worker.get("scene_fixtures"),
                reasoning_effort=trial.get("reasoning_effort"),
                generation_token_limit=trial.get("max_new_tokens"),
            )
            if supervisor.get("status") in ("TIMEOUT", "STALLED", "INTERRUPTED", "SPAWN_FAILED"):
                row["status"] = supervisor["status"]
            if row.get("launcher_error") and row["status"] in ("PASS", "PREFLIGHT_ONLY"):
                row["status"] = "ERROR"
            save()
            print(f"Saved {quant}: {row['status']}", flush=True)
            process = environment.get("process")
            if process is not None and process.poll() is None:
                row["status"] = "WORKER_NOT_EXITED"
                report["trials"].extend(
                    {"quant": q, "status": "SKIPPED_WORKER_NOT_EXITED"} for q in quants[index + 1 :]
                )
                save()
                break
        statuses = [row["status"] for row in report["trials"]]
        if not config["run_model"] and all(s == "PREFLIGHT_ONLY" for s in statuses):
            report["status"] = "PREFLIGHT_ONLY"
        elif all(s == "PASS" for s in statuses):
            report["status"] = "PASS"
        elif all(s in ("PASS", "FAIL") for s in statuses):
            report["status"] = "FAIL"
        else:
            report["status"] = "INCOMPLETE"
    finally:
        save()
        package_evidence(directory, runtime_root / (directory.name + ".zip"), extra_files=[path])
    return report, path
