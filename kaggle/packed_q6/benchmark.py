"""Bounded, single-chip projection experiment; no model or credentials needed.

CPU interpretation verifies values only. TPU timing includes host dispatch
and synchronization and must not be translated into model tokens/second.
"""

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from functools import partial
from pathlib import Path


def save(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def check_result(actual, expected):
    import numpy as np

    if actual.shape != expected.shape or actual.dtype != np.float32:
        raise ValueError("Projection shape or FP32 output contract violated")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("Nonfinite projection rejected before timing")
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-4)
    error = actual.astype(np.float64) - expected
    return {
        "passed": True,
        "max_absolute_error": float(np.max(np.abs(error))),
        "rms_error": float(np.sqrt(np.mean(error**2))),
        "rtol": 2e-5,
        "atol": 2e-4,
    }


def memory_stats(executable):
    try:
        stats = executable.memory_analysis()
        return {
            key: getattr(stats, key, None)
            for key in (
                "argument_size_in_bytes",
                "output_size_in_bytes",
                "temp_size_in_bytes",
                "alias_size_in_bytes",
                "generated_code_size_in_bytes",
            )
        }
    except (AttributeError, NotImplementedError):
        return {"available": False}


def run_case(report, args, *, batch, outputs, inputs, canary=False):
    import jax
    import ml_dtypes
    import numpy as np

    from .baselines import bf16_matmul, xla_q6_matmul
    from .kernel import q6_matmul
    from .layout import pack_q6, unpack_q6
    from .reference import decode_reference, fixture, projection_reference

    row = {"batch": batch, "outputs": outputs, "inputs": inputs, "canary": canary, "methods": {}}
    report["cases"].append(row)
    report["phase"] = {"operation": "prepare", "shape": [batch, outputs, inputs]}
    save(args.output, report)
    raw = fixture(outputs, inputs)
    start = time.perf_counter()
    packed = pack_q6(raw, inputs)
    row["host_repack_seconds"] = time.perf_counter() - start
    np.testing.assert_array_equal(unpack_q6(packed), raw)
    row["storage"] = packed.storage()
    row["storage"]["predecoded_bf16_bytes"] = outputs * inputs * 2
    x = np.random.default_rng(721).normal(size=(batch, inputs)).astype(np.float32)
    if batch > 1:
        x[0] = 0
    expected = projection_reference(x, raw)
    weights = decode_reference(raw, inputs).astype(ml_dtypes.bfloat16)
    start = time.perf_counter()
    x, data, scales, raw_device, weights = jax.device_put(
        (x.astype(ml_dtypes.bfloat16), packed.data, packed.scales, raw.reshape(-1), weights)
    )
    jax.block_until_ready((x, data, scales, raw_device, weights))
    row["all_method_inputs_upload_seconds"] = time.perf_counter() - start
    row["device_array_bytes_reported"] = {
        "packed": data.on_device_size_in_bytes() + scales.on_device_size_in_bytes(),
        "raw": raw_device.on_device_size_in_bytes(),
        "bf16": weights.on_device_size_in_bytes(),
    }
    functions = {
        "pallas_q6": (
            jax.jit(partial(q6_matmul, output_features=outputs, interpret=args.interpret)),
            (x, data, scales),
        ),
        "xla_q6": (jax.jit(partial(xla_q6_matmul, output_features=outputs)), (x, raw_device)),
        "predecoded_bf16": (bf16_matmul, (x, weights)),
    }
    compiled = {}
    for name, (function, arguments) in functions.items():
        report["phase"]["operation"] = f"compile:{name}"
        save(args.output, report)
        print(f"Compiling {name}: M={batch} N={outputs} K={inputs}", flush=True)
        start = time.perf_counter()
        executable = function.lower(*arguments).compile()
        result = {"compile_seconds": time.perf_counter() - start}
        row["methods"][name] = result
        result["compiler_memory"] = memory_stats(executable)
        report["phase"]["operation"] = f"verify:{name}"
        save(args.output, report)
        start = time.perf_counter()
        actual = executable(*arguments).block_until_ready()
        result["first_execution_seconds"] = time.perf_counter() - start
        result["correctness"] = check_result(np.asarray(actual), expected)
        compiled[name] = (executable, arguments)
        save(args.output, report)
    if not args.interpret:
        report["phase"]["operation"] = "timing"
        save(args.output, report)
        samples = {name: [] for name in compiled}
        for executable, arguments in compiled.values():
            for _ in range(2):
                executable(*arguments).block_until_ready()
        rng = np.random.default_rng(42)
        # Interleave methods to reduce order/thermal bias. Packed weights are
        # runtime arguments and each invocation performs its own decoding.
        for _ in range(args.repetitions):
            for name in rng.permutation(list(compiled)):
                executable, arguments = compiled[name]
                start = time.perf_counter()
                executable(*arguments).block_until_ready()
                samples[name].append((time.perf_counter() - start) * 1000)
        for name, values in samples.items():
            row["methods"][name]["latency_ms"] = {
                "samples": values,
                "median": float(np.median(values)),
                "min": min(values),
                "max": max(values),
            }
        medians = {name: row["methods"][name]["latency_ms"]["median"] for name in compiled}
        row["xla_over_pallas_latency_ratio"] = medians["xla_q6"] / medians["pallas_q6"]
        row["pallas_over_predecoded_latency_ratio"] = (
            medians["pallas_q6"] / medians["predecoded_bf16"]
        )
    row["status"] = "passed"
    save(args.output, report)
    print(f"Passed M={batch} N={outputs} K={inputs}; timing={not args.interpret}", flush=True)


def worker(args):
    report = {
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "cpu_interpret_correctness_only" if args.interpret else "tpu_projection_benchmark",
        "timing_scope": "host dispatch + single-chip execution + synchronization; no model tok/s",
        "versions": {name: importlib.metadata.version(name) for name in ("jax", "jaxlib", "numpy")},
        "python": platform.python_version(),
        "cases": [],
    }
    save(args.output, report)
    try:
        import jax

        devices = jax.devices()
        selected = devices[0]
        report["devices"] = [str(d) for d in devices]
        report["device_kind"] = selected.device_kind
        report["selected_device"] = str(selected)
        report["available_device_count"] = len(devices)
        report["chips_used"] = 1
        if args.interpret and selected.platform != "cpu":
            raise ValueError("Use JAX_PLATFORMS=cpu for the interpreter correctness check")
        if not args.interpret and selected.platform != "tpu":
            raise ValueError(
                "Speed measurements require a TPU; use --interpret for CPU correctness"
            )
        with jax.default_device(selected):
            run_case(report, args, batch=1, outputs=64, inputs=256, canary=True)
            for outputs in args.outputs:
                for batch in args.batches:
                    run_case(report, args, batch=batch, outputs=outputs, inputs=2560)
        report["status"] = "passed"
        report["phase"] = {"operation": "complete"}
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    save(args.output, report)
    return 0 if report["status"] == "passed" else 1


def supervise(command, output, timeout_seconds):
    """Kill the child even if it is stuck in a native compiler call."""
    started = time.monotonic()
    process = subprocess.Popen(command, env=os.environ.copy())
    status = None
    try:
        while True:
            remaining = timeout_seconds - (time.monotonic() - started)
            if remaining <= 0:
                raise subprocess.TimeoutExpired(command, timeout_seconds)
            try:
                code = process.wait(timeout=min(15, remaining))
                break
            except subprocess.TimeoutExpired:
                if time.monotonic() - started >= timeout_seconds:
                    raise
                try:
                    progress = json.loads(Path(output).read_text()).get("phase", "starting")
                except (FileNotFoundError, json.JSONDecodeError):
                    progress = "starting"
                print(f"Watchdog: {time.monotonic() - started:.0f}s; {progress}", flush=True)
    except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
        status = "timed_out" if isinstance(exc, subprocess.TimeoutExpired) else "interrupted"
        process.kill()
        process.wait()
        code = 124 if status == "timed_out" else 130
    try:
        report = json.loads(Path(output).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        report = {}
    if status:
        report["status"] = status
    elif code != 0 or report.get("status") != "passed":
        report["status"] = "failed"
        code = code or 1
    report["supervisor"] = {
        "return_code": code,
        "elapsed_seconds": time.monotonic() - started,
        "timeout_seconds": timeout_seconds,
    }
    save(output, report)
    return code


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("packed-q6-results.json"))
    parser.add_argument("--interpret", action="store_true")
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--outputs", type=int, nargs="+", default=[320, 640, 1280])
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not set(args.batches).issubset({1, 2, 4, 8, 16}):
        parser.error("Supported small expert-row batches: 1, 2, 4, 8, 16")
    if not set(args.outputs).issubset({320, 640, 1280}):
        parser.error("Supported expert projection widths: 320, 640, 1280")
    if not 3 <= args.repetitions <= 31 or not 1 <= args.timeout_seconds <= 900:
        parser.error("Require 3..31 repetitions and 1..900 seconds")
    args.batches = list(dict.fromkeys(args.batches))
    args.outputs = list(dict.fromkeys(args.outputs))
    return args


def main():
    args = parse_args()
    if args._worker:
        return worker(args)
    # Reset stale results before starting; an earlier success cannot mask a
    # child crash before its first write. A native compiler hang is bounded.
    save(args.output, {"status": "starting", "cases": []})
    command = [sys.executable, "-m", "packed_q6.benchmark", *sys.argv[1:], "--_worker"]
    return supervise(command, args.output, args.timeout_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
