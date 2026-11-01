"""Bounded eight-device BF16 rounding experiment, separate from model inference.

Run with PYTHONPATH=kaggle python -m qwen38_tpu.rounding_benchmark.
CPU mode checks values only. TPU mode records synchronized warm timings and
optimized HLO: fewer source operations alone do not prove fewer collectives.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path


def save(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def collective_lines(hlo):
    # Count instructions, not references to their names in later instructions.
    return [
        line.strip()
        for line in hlo.splitlines()
        if re.search(r"\b(?:all-gather|all-reduce|reduce-scatter|all-to-all)(?:-start)?\(", line)
    ]


def check_values(actual, expected):
    import numpy as np

    actual = np.asarray(actual)
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError("Rounding output shape/dtype changed")
    if not np.isfinite(actual.astype(np.float32)).all():
        raise ValueError("Nonfinite benchmark output")
    # Finite values, including the sign of zero, must agree exactly.
    np.testing.assert_array_equal(actual.view(np.uint8), expected.view(np.uint8))


def worker(args):
    report = dict(status="running", correctness_only=args.correctness_only, cases=[])
    save(args.output, report)
    try:
        import jax
        import jax.numpy as jnp
        import ml_dtypes
        import numpy as np
        from jax.sharding import Mesh, NamedSharding
        from jax.sharding import PartitionSpec as P

        from .precision import compute_cast
        from .rounding_candidate import primitive_bf16

        devices = jax.devices()
        if len(devices) != 8:
            raise ValueError("Require exactly eight devices for the EP4/TP2 comparison")
        if not args.correctness_only and any(d.platform != "tpu" for d in devices):
            raise ValueError("Speed measurement requires TPU devices")
        report.update(jax_version=jax.__version__, device_kind=devices[0].device_kind)
        mesh = Mesh(np.asarray(devices).reshape(4, 2), ("expert", "tensor"))
        replica = NamedSharding(mesh, P())
        sharded = NamedSharding(mesh, P(None, None, ("expert", "tensor")))
        weight_layout = NamedSharding(mesh, P(("expert", "tensor"), None))
        lowering = args.output.parent / "rounding-lowering"
        lowering.mkdir(exist_ok=True)
        rng = np.random.default_rng(739)
        # Small canary first. The 320 -> 4*2560 projection then models a real
        # low-rank residual projection, without loading the full checkpoint.
        shapes = [(1, 128, True), *[(b, 2560, False) for b in args.batches]]
        for batch, width, canary in shapes:
            for scope in (
                "round_sharded",
                "round_replicated",
                "round_reduce",
                "round_rescale",
                "projection",
            ):
                row = dict(batch=batch, width=width, canary=canary, scope=scope, methods={})
                report["cases"].append(row)
                shape = (batch, 4, width)
                host = rng.normal(size=shape).astype(np.float32)
                host[..., :4] = [1.00390625, -1.00390625, 1.01171875, -1.01171875]
                expected = host.astype(ml_dtypes.bfloat16)
                if scope == "projection":
                    # Dyadic inputs make the FP32 dot exactly representable;
                    # a host float64 oracle can then check rounding bitwise.
                    inputs = 32 if canary else 320
                    x = (rng.integers(-16, 17, (batch, inputs)) / 32).astype(ml_dtypes.bfloat16)
                    weight = (rng.integers(-16, 17, (4 * width, inputs)) / 32).astype(
                        ml_dtypes.bfloat16
                    )
                    expected = (
                        (x.astype(np.float64) @ weight.astype(np.float64).T)
                        .reshape(shape)
                        .astype(ml_dtypes.bfloat16)
                    )
                    arguments = (jax.device_put(x, replica), jax.device_put(weight, weight_layout))
                    in_layout = (replica, weight_layout)
                else:
                    arguments = (jax.device_put(host, sharded),)
                    in_layout = (sharded,)
                if scope == "round_reduce":
                    expected = np.mean(expected.astype(np.float32), axis=1)
                elif scope == "round_rescale":
                    # Catch precision-boundary elision inside an FP32 consumer,
                    # rather than only validating a materialized BF16 output.
                    expected = expected.astype(np.float32) * 127
                out_layout = sharded if scope == "round_sharded" else replica
                jax.block_until_ready(arguments)
                compiled = {}
                for method in ("baseline", "reduce_precision"):
                    cast = (
                        primitive_bf16
                        if method == "reduce_precision"
                        else (lambda x: compute_cast(x, jnp.bfloat16))
                    )
                    if scope == "projection":

                        def function(x, w, cast=cast):
                            dot = jnp.matmul(x, w.T, preferred_element_type=jnp.float32)
                            return cast(dot.reshape(batch, 4, width))

                    elif scope == "round_reduce":

                        def function(x, cast=cast):
                            return jnp.mean(cast(x).astype(jnp.float32), axis=1)

                    elif scope == "round_rescale":

                        def function(x, cast=cast):
                            return cast(x).astype(jnp.float32) * 127

                    else:
                        function = cast
                    report["phase"] = f"compile {scope}/{method} B{batch} width={width}"
                    save(args.output, report)
                    print(report["phase"], flush=True)
                    started = time.perf_counter()
                    lowered = jax.jit(
                        function, in_shardings=in_layout, out_shardings=out_layout
                    ).lower(*arguments)
                    executable = lowered.compile()
                    compile_seconds = time.perf_counter() - started
                    name = f"{scope}-{method}-b{batch}-w{width}"
                    (lowering / f"{name}.mlir").write_text(str(lowered.compiler_ir("stablehlo")))
                    hlo = executable.as_text() or ""
                    (lowering / f"{name}.optimized-hlo.txt").write_text(hlo)
                    memory = executable.memory_analysis()
                    row["methods"][method] = dict(
                        compile_seconds=compile_seconds,
                        collectives=collective_lines(hlo),
                        compiler_memory={
                            k: getattr(memory, k, None)
                            for k in (
                                "argument_size_in_bytes",
                                "output_size_in_bytes",
                                "temp_size_in_bytes",
                                "alias_size_in_bytes",
                            )
                        },
                    )
                    check_values(executable(*arguments).block_until_ready(), expected)
                    row["methods"][method]["correctness"] = "exact_independent_reference"
                    compiled[method] = executable
                    save(args.output, report)
                if not args.correctness_only:
                    samples = {method: [] for method in compiled}
                    for fn in compiled.values():
                        for _ in range(2):
                            fn(*arguments).block_until_ready()
                    for _ in range(args.repetitions):
                        for method in rng.permutation(list(compiled)):
                            started = time.perf_counter()
                            compiled[method](*arguments).block_until_ready()
                            samples[method].append((time.perf_counter() - started) * 1000)
                    for method, timings in samples.items():
                        row["methods"][method]["latency_ms"] = dict(
                            median=float(np.median(timings)), samples=timings
                        )
                    row["baseline_over_candidate_latency"] = float(
                        np.median(samples["baseline"]) / np.median(samples["reduce_precision"])
                    )
                if args.capture_trace and not canary:
                    trace = args.output.parent / "rounding-traces" / f"b{batch}-{scope}"
                    trace.mkdir(parents=True, exist_ok=True)
                    with jax.profiler.trace(str(trace), create_perfetto_trace=True):
                        for method, fn in compiled.items():
                            with jax.profiler.TraceAnnotation(method, batch=batch, scope=scope):
                                for _ in range(3):
                                    fn(*arguments).block_until_ready()
                    row["trace"] = str(trace)
                report["phase"] = f"verified {scope} B{batch} width={width}"
                save(args.output, report)
        report.update(
            status="passed",
            phase="complete",
            production_enabled=False,
            limitation="Isolated subgraphs; no full-model speed or quality claim. "
            "Timings exclude compile/transfers and include synchronized dispatch. "
            "CPU HLO cannot establish TPU collective count or latency.",
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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("rounding-results.json"))
    parser.add_argument("--correctness-only", action="store_true")
    parser.add_argument("--capture-trace", action="store_true")
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--repetitions", type=int, default=9)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if (
        not set(args.batches).issubset({1, 2, 4, 8})
        or not 3 <= args.repetitions <= 31
        or not 1 <= args.timeout_seconds <= 900
    ):
        parser.error("Require B1/B2/B4/B8, 3..31 repetitions and 1..900 seconds")
    if args.capture_trace and args.correctness_only:
        parser.error("Device trace capture requires TPU mode")
    args.batches = list(dict.fromkeys(args.batches))
    return args


def main():
    args = parse_args()
    if args._worker:
        return worker(args)
    environment = dict(os.environ)
    if args.correctness_only:
        environment["JAX_PLATFORMS"] = "cpu"
        environment.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")
    save(args.output, dict(status="starting", cases=[]))
    command = [sys.executable, "-m", "qwen38_tpu.rounding_benchmark", *sys.argv[1:], "--_worker"]
    started = time.monotonic()
    status = None
    try:
        code = subprocess.run(command, env=environment, timeout=args.timeout_seconds).returncode
    except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
        status = "timed_out" if isinstance(error, subprocess.TimeoutExpired) else "interrupted"
        code = 124 if status == "timed_out" else 130
    report = json.loads(args.output.read_text())
    if status or code != 0 or report.get("status") != "passed":
        report["status"] = status or "failed"
        code = code or 1
    report["supervisor"] = dict(
        return_code=code,
        elapsed_seconds=time.monotonic() - started,
        timeout_seconds=args.timeout_seconds,
    )
    save(args.output, report)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
