#!/usr/bin/env python3
"""Bounded eight-device comparison of Qwen Q8 cache gather candidates."""

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def worker(args):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kaggle"))
    report = {"status": "running", "correctness_only": args.correctness_only, "cases": []}
    save(args.output, report)
    try:
        import functools

        import jax
        import jax.numpy as jnp
        import ml_dtypes
        import numpy as np
        from jax.sharding import Mesh, NamedSharding
        from jax.sharding import PartitionSpec as P
        from qwen38_tpu.attention_cache import gather_scale_plane, read_quantized
        from qwen38_tpu.config import ModelConfig

        devices = jax.devices()[:8]
        if len(devices) != 8:
            raise ValueError("Eight devices are required to preserve the time-sharded baseline")
        if not args.correctness_only and any(d.platform != "tpu" for d in devices):
            raise ValueError("Speed measurement requires eight TPU devices")
        report.update(
            jax_version=jax.__version__,
            devices=[str(d) for d in devices],
            device_kind=devices[0].device_kind,
        )
        mesh = Mesh(np.asarray(devices), ("expert",))
        plane_layout = NamedSharding(mesh, P(None, "expert"))
        code_layout = NamedSharding(mesh, P(None, "expert", None))
        replica = NamedSharding(mesh, P())
        lowering = args.output.parent / "lowering"
        lowering.mkdir(exist_ok=True)
        # A small canary precedes the real context/selected-token dimensions.
        shapes = [(1, 64, 17), *[(b, args.capacity, 2051) for b in args.batches]]
        for batch, capacity, selected in shapes:
            c = ModelConfig(max_context=capacity, kv_cache_dtype="q8")
            rng = np.random.default_rng(413 + batch)
            scale_host = rng.uniform(0, 0.03, (batch * 16, capacity)).astype(ml_dtypes.bfloat16)
            scale_host[:, 0] = 0
            code_host = rng.integers(-127, 128, (batch, capacity, 512), dtype=np.int8)
            plane = jax.device_put(scale_host, plane_layout)
            codes = jax.device_put(code_host, code_layout)
            jax.block_until_ready((plane, codes))
            compiled = {}
            for scope in ("scales", "complete_read"):
                for method in ("indexed", "rowwise"):
                    if scope == "scales":
                        function = functools.partial(gather_scale_plane, groups=16, method=method)
                        function = jax.jit(
                            function, in_shardings=(plane_layout, replica), out_shardings=replica
                        )
                    else:

                        def complete(code, scale, index, method=method):
                            return read_quantized(
                                {"k": code, "k_scale": scale},
                                "k",
                                index,
                                c,
                                jnp.bfloat16,
                                scale_gather=method,
                            ).reshape(batch, selected, 512)

                        function = jax.jit(
                            complete,
                            in_shardings=(code_layout, plane_layout, replica),
                            out_shardings=replica,
                        )
                    indices = jax.device_put(np.zeros((batch, selected), np.int32), replica)
                    arguments = (plane, indices) if scope == "scales" else (codes, plane, indices)
                    report["phase"] = f"compile {scope}/{method} B{batch} T{capacity} L{selected}"
                    save(args.output, report)
                    print(report["phase"], flush=True)
                    started = time.perf_counter()
                    lowered = function.lower(*arguments)
                    executable = lowered.compile()
                    seconds = time.perf_counter() - started
                    path = lowering / f"{scope}-{method}-b{batch}-t{capacity}.mlir"
                    path.write_text(str(lowered.compiler_ir(dialect="stablehlo")))
                    # StableHLO precedes partitioning. Preserve backend text too
                    # so reviewers can inspect the final collectives and layouts.
                    optimized = executable.as_text()
                    if optimized:
                        path.with_suffix(".optimized-hlo.txt").write_text(optimized)
                    memory = executable.memory_analysis()
                    compiled[scope, method] = (
                        executable,
                        seconds,
                        {
                            key: getattr(memory, key, None)
                            for key in (
                                "argument_size_in_bytes",
                                "temp_size_in_bytes",
                                "output_size_in_bytes",
                                "alias_size_in_bytes",
                            )
                        },
                    )
            for pattern in ("uniform", "short_prefix"):
                limit = capacity if pattern == "uniform" else min(capacity, 256)
                index_host = rng.integers(0, limit, (batch, selected), dtype=np.int32)
                index_host[:, :4] = [0, capacity - 1, 0, capacity // 8]
                if pattern == "short_prefix":
                    # Early sparse attention has many masked slots addressed
                    # to zero, not 2051 distinct meaningful history tokens.
                    index_host[:, min(257, selected) :] = 0
                index = jax.device_put(index_host, replica)
                expected_scale = np.stack(
                    [scale_host[b * 16 : (b + 1) * 16][:, index_host[b]].T for b in range(batch)]
                )
                selected_codes = np.stack([code_host[b, index_host[b]] for b in range(batch)])
                expected_read = (
                    (
                        selected_codes.reshape(batch, selected, 16, 32).astype(np.float32)
                        * expected_scale.astype(np.float32)[..., None]
                    )
                    .reshape(batch, selected, 512)
                    .astype(ml_dtypes.bfloat16)
                )
                for scope in ("scales", "complete_read"):
                    report["phase"] = f"verify {scope} B{batch} T{capacity} {pattern}"
                    save(args.output, report)
                    expected = expected_scale if scope == "scales" else expected_read
                    arguments = (plane, index) if scope == "scales" else (codes, plane, index)
                    row = {
                        "batch": batch,
                        "capacity": capacity,
                        "selected": selected,
                        "scope": scope,
                        "pattern": pattern,
                        "methods": {},
                    }
                    report["cases"].append(row)
                    for method in ("indexed", "rowwise"):
                        executable, seconds, memory = compiled[scope, method]
                        actual = np.asarray(executable(*arguments).block_until_ready())
                        if not np.isfinite(actual.astype(np.float32)).all():
                            raise ValueError("Nonfinite cache read")
                        np.testing.assert_array_equal(actual, expected)
                        row["methods"][method] = {
                            "correctness": "exact",
                            "compile_seconds": seconds,
                            "compiler_memory": memory,
                        }
                    if not args.correctness_only:
                        samples = {method: [] for method in ("indexed", "rowwise")}
                        for method in samples:
                            for _ in range(2):
                                compiled[scope, method][0](*arguments).block_until_ready()
                        for _ in range(args.repetitions):
                            for method in rng.permutation(list(samples)):
                                started = time.perf_counter()
                                compiled[scope, method][0](*arguments).block_until_ready()
                                samples[method].append((time.perf_counter() - started) * 1000)
                        for method, values in samples.items():
                            row["methods"][method]["latency_ms"] = {
                                "samples": values,
                                "median": float(np.median(values)),
                            }
                        row["baseline_over_candidate_latency"] = float(
                            np.median(samples["indexed"]) / np.median(samples["rowwise"])
                        )
                    if args.capture_trace and capacity == args.capacity:
                        trace = args.output.parent / "device-trace" / f"b{batch}-{scope}-{pattern}"
                        report["phase"] = f"profile {scope} B{batch} T{capacity} {pattern}"
                        save(args.output, report)
                        with jax.profiler.trace(str(trace), create_perfetto_trace=True):
                            for method in ("indexed", "rowwise"):
                                with jax.profiler.TraceAnnotation(
                                    "kv_gather",
                                    method=method,
                                    batch=batch,
                                    scope=scope,
                                    pattern=pattern,
                                    capacity=capacity,
                                    selected=selected,
                                ):
                                    for _ in range(3):
                                        compiled[scope, method][0](*arguments).block_until_ready()
                        row["device_trace_directory"] = str(trace)
                    save(args.output, report)
                    print(report["phase"] + ": passed", flush=True)
        report.update(
            status="passed",
            phase="complete",
            timing_scope="Warm synchronized calls include dispatch; no model tokens/s prediction.",
        )
    except Exception as error:
        report.update(
            status="failed",
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
        )
    save(args.output, report)
    return 0 if report["status"] == "passed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("kv-gather-results.json"))
    parser.add_argument("--correctness-only", action="store_true")
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--capacity", type=int, choices=(8192, 16384), default=8192)
    parser.add_argument("--capture-trace", action="store_true")
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (
        not set(args.batches).issubset({1, 2, 4, 8})
        or not 3 <= args.repetitions <= 31
        or not 1 <= args.timeout_seconds <= 900
    ):
        parser.error("Require B1/B2/B4/B8, 3..31 repetitions and 1..900 seconds")
    args.batches = list(dict.fromkeys(args.batches))
    if args._worker:
        return worker(args)
    environment = os.environ.copy()
    if args.correctness_only:
        environment["JAX_PLATFORMS"] = "cpu"
        environment.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")
    save(args.output, {"status": "starting", "cases": []})
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--_worker"]
    process = subprocess.Popen(command, env=environment)
    started = time.monotonic()
    status = None
    try:
        while True:
            remaining = args.timeout_seconds - (time.monotonic() - started)
            if remaining <= 0:
                raise subprocess.TimeoutExpired(command, args.timeout_seconds)
            try:
                code = process.wait(timeout=min(15, remaining))
                break
            except subprocess.TimeoutExpired:
                if time.monotonic() - started >= args.timeout_seconds:
                    raise
                report = json.loads(args.output.read_text())
                print(
                    f"Watchdog {time.monotonic() - started:.0f}s: "
                    f"{report.get('phase', 'starting')}",
                    flush=True,
                )
    except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
        status = "timed_out" if isinstance(error, subprocess.TimeoutExpired) else "interrupted"
        process.kill()
        process.wait()
        code = 124 if status == "timed_out" else 130
    report = json.loads(args.output.read_text())
    if status:
        report["status"] = status
    elif code != 0 or report.get("status") != "passed":
        report["status"] = "failed"
        code = code or 1
    report["supervisor"] = {
        "elapsed_seconds": time.monotonic() - started,
        "timeout_seconds": args.timeout_seconds,
        "return_code": code,
    }
    save(args.output, report)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
