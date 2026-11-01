"""Small Kaggle CPU check using the production monitoring and tracing modules."""

import importlib.metadata
import json
import os
import time
from pathlib import Path

from glm53_offload import tracing
from glm53_offload.diagnostics import atomic_json, bounded_call
from glm53_offload.monitoring import RunMonitor
from glm53_offload.tracking_check import (
    load_kaggle_key,
    operation_name,
    read_wandb_summary,
    read_weave_calls,
    require_live_tracking,
    wait_for_remote,
)


def run(directory, config):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    proof = {"status": "STARTING", "probe_id": config["comparison_id"], "model_loaded": False}
    monitor = None
    exit_code = 1
    try:
        proof["credentials"] = load_kaggle_key()
        proof["versions"] = {name: importlib.metadata.version(name) for name in ("wandb", "weave")}
        monitor = RunMonitor(
            directory,
            config,
            mode="online",
            project=config["wandb_project"],
            entity=config["wandb_entity"],
        )
        proof["startup"] = require_live_tracking(monitor, config, directory)
        tracing.configure(config, directory)
        client, root = tracing._client, tracing._root
        project_id = f"{client.entity}/{config['wandb_project']}"
        proof["trace_id"] = root.trace_id
        with tracing.span("cpu_probe_wait", probe_id=proof["probe_id"]):
            for heartbeat in range(3):
                atomic_json(
                    directory / "progress.json",
                    {
                        "stage": "CPU_TRACKING_PROBE",
                        "updated_at": time.time(),
                        "detail": "heartbeat",
                    },
                )
                monitor.run.summary["cpu_probe/heartbeat"] = heartbeat
                monitor.tick(os.getpid(), force=True)
                time.sleep(1)
            bounded_call(client.flush, timeout=15)
            wait_for_remote(
                lambda: read_weave_calls(project_id, root.trace_id),
                lambda rows: any(
                    operation_name(c) == "glm53.cpu_probe_wait" and not c.get("ended_at")
                    for c in rows
                ),
            )
            proof["in_progress_span_visible"] = True
            wait_for_remote(
                lambda: read_wandb_summary(client.entity, config["wandb_project"], monitor.run.id),
                lambda data: data["summary"].get("cpu_probe/heartbeat") == 2,
            )
            proof["heartbeat_visible_while_running"] = True
            with tracing.span("cpu_probe_nested", value=7) as result:
                result["value"] = 49
        try:
            with tracing.span("cpu_probe_expected_failure"):
                raise ValueError("expected CPU probe failure")
        except ValueError:
            pass
        trace_id, run_id = root.trace_id, monitor.run.id
        tracing.finish("PASS")

        def complete(rows):
            by_name = {operation_name(c): c for c in rows}
            names = (
                "glm53.experiment",
                "glm53.cpu_probe_wait",
                "glm53.cpu_probe_nested",
                "glm53.cpu_probe_expected_failure",
            )
            return all(by_name.get(n, {}).get("ended_at") for n in names) and bool(
                by_name[names[-1]].get("exception")
            )

        calls = wait_for_remote(lambda: read_weave_calls(project_id, trace_id), complete)
        by_name = {operation_name(c): c for c in calls}
        assert (
            by_name["glm53.cpu_probe_nested"]["parent_id"] == by_name["glm53.cpu_probe_wait"]["id"]
        )
        proof.update(
            status="PASS",
            call_count=len(calls),
            nested_calls_verified=True,
            expected_error_verified=True,
            completed_calls_verified=True,
            wandb_run_id=run_id,
            wandb_url=monitor.run.url,
            weave_url=f"https://wandb.ai/{project_id}/weave/calls",
        )
        atomic_json(directory / "report.json", {"status": "PASS", "stage": "COMPLETE"})
        monitor.finish(0)
        final = wait_for_remote(
            lambda: read_wandb_summary(client.entity, config["wandb_project"], run_id),
            lambda data: data["state"] == "finished",
        )
        proof["finished_wandb_run_verified"] = final["state"] == "finished"
        exit_code = 0
    except Exception as error:
        proof.update(status="FAIL", error_type=type(error).__name__)
        # Only our controlled error messages are suitable for published output.
        if type(error).__name__ == "TrackingCheckError":
            proof["reason"] = str(error)
    finally:
        tracing.finish(proof["status"])
        if monitor is not None and exit_code:
            monitor.finish(exit_code)
        atomic_json(directory / "tracking-proof.json", proof)
        print(json.dumps(proof, indent=2), flush=True)
    if exit_code:
        raise RuntimeError("CPU tracking probe failed; see tracking-proof.json") from None
    return proof
