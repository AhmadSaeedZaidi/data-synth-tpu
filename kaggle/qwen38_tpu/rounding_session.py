"""Tracked notebook parent for the bounded rounding probe; never imports JAX."""

import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from glm53_offload import tracing
from glm53_offload.diagnostics import atomic_json, bounded_call
from glm53_offload.monitoring import RunMonitor
from glm53_offload.tracking_check import load_kaggle_key, require_live_tracking

from .experiments import run_probe


def run(directory, source_sha256):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    monitor = None
    result = {"status": "starting", "phase": "dependencies", "cases": []}
    output = directory / "report.json"
    atomic_json(output, result)
    config = dict(
        experiment_mode="bf16_rounding_probe",
        batch_sizes=[1, 2, 4, 8],
        run_model=False,
        deadline_seconds=1200,
        wandb_mode="online",
        wandb_project="qwen38-kaggle-tpu",
        wandb_entity="your-wandb-entity",
        weave_enabled=True,
        trace_prefix="qwen38_rounding",
        local_traces=True,
        require_live_tracking=True,
        allow_tracking_fallback=False,
        runtime_bundle_sha256=source_sha256,
    )
    atomic_json(directory / "config.json", config)
    try:
        load_kaggle_key()
        with (directory / "dependency-setup.log").open("w") as log:
            dependencies = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--disable-pip-version-check",
                    "jax[tpu]==0.11.2",
                    "numpy==2.5.3",
                    "ml-dtypes==0.6.0",
                    "wandb==0.30.0",
                    "weave==0.53.11",
                    "pydantic==2.13.5",
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=300,
            )
        if dependencies.returncode:
            raise RuntimeError("Pinned dependency installation failed")
        for name in ("jax", "jaxlib"):
            if importlib.metadata.version(name) != "0.11.2":
                raise RuntimeError("Pinned JAX version unavailable")
        result["phase"] = "tracking_startup"
        atomic_json(output, result)
        monitor = RunMonitor(
            directory,
            config,
            mode="online",
            project=config["wandb_project"],
            entity=config["wandb_entity"],
        )
        proof = bounded_call(lambda: require_live_tracking(monitor, config, directory), timeout=150)
        print("W&B/Weave live readback:", proof["status"], flush=True)
        bounded_call(lambda: tracing.configure(config, directory), timeout=30)
        cap = min(600, int(1200 - (time.monotonic() - started) - 90))
        if cap < 30:
            raise TimeoutError("Insufficient session budget for the rounding probe")
        result["phase"] = "benchmark_startup"
        atomic_json(output, result)
        os.environ.update(JAX_PLATFORMS="tpu", OPENBLAS_NUM_THREADS="2", OMP_NUM_THREADS="2")
        # run_probe writes report.log; expose its live tail through RunMonitor.
        worker_log = directory / "worker.log"
        if not worker_log.exists() and not worker_log.is_symlink():
            worker_log.symlink_to("report.log")
        with tracing.span("benchmark", batches=config["batch_sizes"], timeout_seconds=cap) as span:
            result = run_probe(
                [
                    sys.executable,
                    "-m",
                    "qwen38_tpu.rounding_benchmark",
                    "--output",
                    str(output),
                    "--timeout-seconds",
                    str(cap),
                    "--capture-trace",
                ],
                output,
                directory,
                timeout=cap + 30,
                monitor=monitor,
            )
            span.update(probe_status=result.get("status"), cases=len(result.get("cases", [])))
        for case in result.get("cases", []):
            if case.get("canary"):
                continue
            # These are result records, not claims about when computation ran.
            with tracing.span("case_result", batch=case["batch"], scope=case["scope"]) as span:
                ratio = case.get("baseline_over_candidate_latency")
                span.update(speedup=ratio, methods=case.get("methods", {}))
                if ratio is not None and monitor.run is not None:
                    monitor.run.summary[f"rounding/B{case['batch']}/{case['scope']}"] = ratio
        result["tracking_readback"] = proof["status"]
        result["runtime_bundle_sha256"] = source_sha256
    except BaseException as error:
        if output.exists():
            result = json.loads(output.read_text())
        result.update(status="failed", session_error_type=type(error).__name__)
    finally:
        # Kaggle output export need not preserve symlinks. Leave a regular log
        # for the completion checker's existing allowlist after live relay ends.
        source_log, worker_log = directory / "report.log", directory / "worker.log"
        if source_log.is_file():
            if worker_log.is_symlink():
                worker_log.unlink()
            shutil.copyfile(source_log, worker_log)
        atomic_json(output, result)
        atomic_json(
            directory / "supervisor.json",
            dict(
                status=result.get("status"),
                elapsed_seconds=time.monotonic() - started,
                worker_cap_seconds=600,
                session_cap_seconds=1200,
            ),
        )
        for name, cleanup in (
            ("weave", lambda: tracing.finish(result.get("status", "failed"))),
            (
                "wandb",
                lambda: (
                    monitor.finish(0 if result.get("status") == "passed" else 1)
                    if monitor is not None
                    else None
                ),
            ),
        ):
            try:
                bounded_call(cleanup, timeout=25)
            except Exception as error:
                atomic_json(
                    directory / f"{name}-cleanup.json", dict(error_type=type(error).__name__)
                )
    return result
