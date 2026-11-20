"""Bounded optional backend investigation inside a full model session.

The notebook parent imports no accelerator framework. Setup and workers are
isolated process groups; independent baseline work survives ordinary failures.
"""

import sys
import tempfile
import time
from pathlib import Path


def run_investigation(config, directory, monitor, *, remaining_seconds):
    from glm53_offload import tracing
    from glm53_offload.diagnostics import atomic_json, bounded_call
    from qwen38_tpu.experiments import run_probe

    directory = Path(directory)
    out = directory / "torchtpu"
    out.mkdir(exist_ok=True)
    destination = directory / "torchtpu-comparison.json"
    budget = min(2700, int(remaining_seconds) - 4 * 3600 - 110)
    report = dict(
        status="running",
        production_enabled=False,
        full_model_backend="jax",
        tasks={},
        budget_seconds=max(0, budget),
    )
    if budget < 120:
        report.update(status="skipped", reason="reserved_full_model_budget")
        atomic_json(destination, report)
        return report
    started = time.monotonic()
    # Reuse the already authenticated parent tracking connection. Experimental
    # telemetry failure is recorded without sacrificing independent model work.
    try:
        bounded_call(lambda: tracing.configure(config, directory), timeout=30)
    except Exception as error:
        report["tracing_error_type"] = type(error).__name__
    runtime = Path(__file__).resolve().parents[1]
    # Kaggle exports everything under /kaggle/working. Keep the installed
    # environment outside outputs: thousands of wheel files obscured v15's
    # diagnostic files and exhausted the completion checker's listing bound.
    environment = Path(tempfile.mkdtemp(prefix="qwen38-torchtpu-", dir="/tmp")) / "environment"
    python = environment / "bin/python"

    def task(name, command, cap):
        left = budget - (time.monotonic() - started) - 20
        if left < 30:
            value = dict(status="skipped", reason="shared_investigation_deadline")
        else:
            atomic_json(out / "progress.json", dict(phase=name))
            with tracing.span("torchtpu_task", task=name, timeout_seconds=min(cap, left)) as span:
                value = run_probe(
                    command,
                    out / f"{name}.json",
                    directory,
                    timeout=min(cap, left),
                    monitor=monitor,
                )
                span.update(
                    task_status=value.get("status"),
                    cases=len(value.get("cases", [])),
                    error_type=value.get("error_type"),
                    error=value.get("error"),
                )
        report["tasks"][name] = value
        atomic_json(destination, report)
        if getattr(monitor, "run", None) is not None:
            monitor.run.summary["torchtpu/" + name] = value.get("status", "failed")
            if value.get("error"):
                monitor.run.summary["torchtpu_errors/" + name] = value["error"][:1000]
        return value

    # Each setup subprocess has its own session. Preserve the baseline libtpu
    # and wheels; only the isolated candidate receives its pinned dependencies.
    setup_code = """import json,subprocess,sys
from pathlib import Path
directory,requirements,environment=map(Path,sys.argv[1:4])
subprocess.run([sys.executable,'-m','venv',str(environment)],check=True,timeout=60)
python=environment/'bin/python'
subprocess.run([str(python),'-m','pip','install','--disable-pip-version-check','--no-input','--timeout','30','--retries','1','-r',str(requirements)],check=True,timeout=850)
(directory/'setup.json').write_text(json.dumps({'status':'passed'}))
"""
    setup = task(
        "setup",
        [
            sys.executable,
            "-c",
            setup_code,
            str(out),
            str(runtime / "requirements-torchtpu.txt"),
            str(environment),
        ],
        930,
    )
    if setup.get("status") == "passed":
        for method in ("builtin_x6", "explicit_x6"):
            task(
                "precision-" + method,
                [
                    str(python),
                    "-m",
                    "qwen38_torch.precision_check",
                    "--method",
                    method,
                    "--output",
                    str(out / f"precision-{method}.json"),
                ],
                240,
            )
    # Baseline JAX still runs even if candidate installation is unavailable.
    for method in ("jax", "torch_eager", "torch_compile", "hybrid"):
        if method != "jax" and setup.get("status") != "passed":
            report["tasks"][method] = dict(status="skipped", reason="candidate_install_failed")
            continue
        interpreter = sys.executable if method == "jax" else str(python)
        task(
            method,
            [
                interpreter,
                "-m",
                "qwen38_torch.comparison",
                "--method",
                method,
                "--output",
                str(out / f"{method}.json"),
                "--capture-trace",
            ],
            240,
        )
    if setup.get("status") == "passed":
        task(
            "vision",
            [
                str(python),
                "-m",
                "qwen38_torch.vision_check",
                "--device",
                "tpu",
                "--output",
                str(out / "vision.json"),
            ],
            180,
        )
        for tp in (1, 2):
            name = f"ep{8 // tp}-tp{tp}"
            task(
                name,
                [
                    str(python),
                    "-m",
                    "qwen38_torch.distributed_check",
                    "--device",
                    "tpu",
                    "--tp",
                    str(tp),
                    "--output",
                    str(out / f"{name}.json"),
                ],
                240,
            )
    report.update(
        status="passed"
        if report["tasks"] and all(v.get("status") == "passed" for v in report["tasks"].values())
        else "incomplete",
        elapsed_seconds=time.monotonic() - started,
    )
    try:
        bounded_call(lambda: tracing.finish(report["status"]), timeout=25)
    except Exception as error:
        report["tracing_cleanup_error_type"] = type(error).__name__
    atomic_json(destination, report)
    return report
