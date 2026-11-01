"""Isolated TPU probes before the full model owns the device; no JAX in parent."""

import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def select_gather(report, *, minimum_speedup=1.10, maximum_regression=1.15):
    """Require complete, exact, memory-safe full-read wins at every tested shape."""
    result = dict(selected="indexed", reason="incomplete_or_failed_probe")
    if report.get("status") != "passed":
        return result
    expected = {
        (b, scope, pattern)
        for b in (1, 2, 4, 8)
        for scope in ("scales", "complete_read")
        for pattern in ("uniform", "short_prefix")
    }
    rows = [
        r
        for r in report.get("cases", [])
        if r.get("capacity") == 8192 and r.get("selected") == 2051
    ]
    if len(rows) != len(expected):
        return result
    cases = {(r["batch"], r["scope"], r["pattern"]): r for r in rows}
    if cases.keys() != expected:
        return result
    ratios = []
    for key, row in cases.items():
        methods = row.get("methods", {})
        if any(methods.get(m, {}).get("correctness") != "exact" for m in ("indexed", "rowwise")):
            return dict(result, reason="correctness_rejected")
        if key[1] != "complete_read":
            continue
        timings = [methods[m].get("latency_ms", {}).get("median") for m in ("indexed", "rowwise")]
        if any(not isinstance(t, (int, float)) or not math.isfinite(t) or t <= 0 for t in timings):
            return dict(result, reason="invalid_timings")
        memory = [
            methods[m].get("compiler_memory", {}).get("temp_size_in_bytes")
            for m in ("indexed", "rowwise")
        ]
        if any(type(x) is not int or x < 0 for x in memory) or memory[1] > memory[0] + (64 << 20):
            return dict(result, reason="temporary_memory_rejected")
        ratios.append(timings[0] / timings[1])
    speedup = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
    worst = max(1 / r for r in ratios)
    accepted = speedup >= minimum_speedup and worst <= maximum_regression
    return dict(
        selected="rowwise" if accepted else "indexed",
        reason="measured_win" if accepted else "speed_rejected",
        geometric_speedup=speedup,
        worst_latency_ratio=worst,
        minimum_speedup=minimum_speedup,
        maximum_regression=maximum_regression,
    )


def _live_group_members(group):
    """Linux TPU workers must release their device before the model starts."""
    members = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            state = (entry / "stat").read_text().rsplit(")", 1)[1].split()[0]
            if state not in {"Z", "X"} and os.getpgid(int(entry.name)) == group:
                members.append(int(entry.name))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return members


def run_probe(command, output, directory, *, timeout, monitor=None):
    """Bound a complete subprocess tree and publish readable progress while compiling."""
    from glm53_offload.diagnostics import atomic_json

    output.unlink(missing_ok=True)
    started = time.monotonic()
    process = None
    returncode = None
    status = None
    env = dict(
        os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]), PYTHONUNBUFFERED="1"
    )
    try:
        with output.with_suffix(".log").open("w") as log:
            process = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True
            )
            while True:
                returncode = process.poll()
                if returncode is not None:
                    break
                if time.monotonic() - started >= timeout:
                    status = "timed_out"
                    break
                try:
                    progress = json.loads(output.read_text()).get("phase", "starting")
                except (FileNotFoundError, json.JSONDecodeError):
                    progress = "starting"
                atomic_json(
                    directory / "progress.json",
                    dict(
                        stage="optimization_probe",
                        detail=f"{output.stem}: {progress}",
                        updated_at=time.time(),
                    ),
                )
                if monitor is not None:
                    monitor.tick(process.pid, force=True)
                time.sleep(2)
    finally:
        # Also kill descendants if the probe's supervisor crashed or a notebook
        # interrupt occurred while a native worker still owns the TPU.
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                raise RuntimeError(
                    "Probe process has not released the TPU; aborting model launch"
                ) from None
            deadline = time.monotonic() + 10
            while _live_group_members(process.pid):
                if time.monotonic() >= deadline:
                    raise RuntimeError("Probe descendants still own the TPU; aborting model launch")
                time.sleep(0.1)
    try:
        report = json.loads(output.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        report = {}
    report["status"] = status or (
        "passed" if returncode == 0 and report.get("status") == "passed" else "failed"
    )
    report["outer_supervisor"] = dict(
        returncode=returncode, elapsed_seconds=time.monotonic() - started, timeout_seconds=timeout
    )
    atomic_json(output, report)
    return report


def run_optimization_probes(config, directory, monitor, *, remaining_seconds):
    from glm53_offload.diagnostics import atomic_json

    directory = Path(directory)
    runtime = Path(__file__).resolve().parents[1]
    cap = min(900, int(remaining_seconds) - 120)
    if cap < 30:
        raise TimeoutError("Insufficient session budget for optimization preflight")
    gather = run_probe(
        [
            sys.executable,
            str(runtime / "experiments/benchmark_qwen38_kv_gather.py"),
            "--output",
            str(directory / "kv-gather.json"),
            "--timeout-seconds",
            str(max(1, cap - 20)),
        ],
        directory / "kv-gather.json",
        directory,
        timeout=cap,
        monitor=monitor,
    )
    selection = select_gather(gather)
    config["kv_scale_gather"] = selection["selected"]
    result = dict(kv_gather=selection, packed_q6="not_requested")
    if getattr(monitor, "run", None) is not None:
        monitor.run.summary["optimization/kv_gather"] = selection
        monitor.run.config.update({"kv_scale_gather": selection["selected"]}, allow_val_change=True)
    if config.get("packed_q6_probe", False):
        cap = min(900, int(remaining_seconds - gather["outer_supervisor"]["elapsed_seconds"]) - 120)
        if cap >= 30:
            packed = run_probe(
                [
                    sys.executable,
                    "-m",
                    "packed_q6.benchmark",
                    "--batches",
                    "1",
                    "8",
                    "--outputs",
                    "320",
                    "640",
                    "--output",
                    str(directory / "packed-q6.json"),
                    "--timeout-seconds",
                    str(max(1, cap - 20)),
                ],
                directory / "packed-q6.json",
                directory,
                timeout=cap,
                monitor=monitor,
            )
            result["packed_q6"] = packed.get("status", "failed")
            if getattr(monitor, "run", None) is not None:
                monitor.run.summary["optimization/packed_q6_probe"] = result["packed_q6"]
    atomic_json(directory / "optimization-preflight.json", result)
    return result


def run_rounding_probe(config, directory, monitor, *, remaining_seconds):
    """Optional measurement within a full trial, never a production promotion.

    Reserve four hours for setup/model work still ahead. A failed or timed-out
    compiler probe returns its evidence; run_probe raises if device-owning
    descendants cannot be retired, because model launch would then be unsafe.
    """
    from glm53_offload.diagnostics import atomic_json

    directory = Path(directory)
    output = directory / "rounding-probe.json"
    # Include the outer watchdog's 30 seconds, process-group cleanup's 20,
    # and the notebook's final 60-second evidence/termination allowance.
    cap = min(600, int(remaining_seconds) - 4 * 3600 - 110)
    if cap < 30:
        result = dict(
            status="skipped", reason="reserved_full_model_budget", production_enabled=False
        )
        atomic_json(output, result)
        return result
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
    # No microbenchmark alone can change the inference precision policy.
    result["production_enabled"] = False
    atomic_json(output, result)
    if getattr(monitor, "run", None) is not None:
        monitor.run.summary["optimization/rounding_probe"] = result.get("status", "failed")
    return result
