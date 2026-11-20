"""Matched packed projections and KV reads; CPU correctness or isolated TPU workers.

JAX and TorchTPU execution must use different OS processes. The hybrid worker
uses JAX only for abstract export through TorchTPU's documented custom-op API.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np


def save(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sample(function, arguments, sync, repeats):
    for _ in range(2):
        sync(function(*arguments))
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        sync(function(*arguments))
        timings.append((time.perf_counter() - start) * 1000)
    return dict(latency_ms=dict(median=statistics.median(timings), samples=timings))


def fixtures(batch, real_shapes):
    from packed_q6.reference import fixture, projection_reference

    rng = np.random.default_rng(502 + batch)
    hidden, local = (2560, 320) if real_shapes else (256, 32)
    for qtype, width, outputs in (("Q6_K", hidden, local), ("Q8_0", local, hidden)):
        x = rng.normal(0, 0.02, (batch, width)).astype(np.float32)
        import ml_dtypes

        x = x.astype(ml_dtypes.bfloat16)
        if qtype == "Q6_K":
            raw = fixture(outputs, width, seed=721)
            expected = projection_reference(x, raw)
        else:
            raw = rng.integers(0, 256, (outputs, width // 32, 34), dtype=np.uint8)
            raw[..., :2] = np.asarray([0.003], "<f2").view(np.uint8)
            weight = raw[..., 2:].view(np.int8).astype(np.float32) * np.float32(np.float16(0.003))
            weight = weight.reshape(outputs, width).astype(ml_dtypes.bfloat16).astype(np.float32)
            expected = (x.astype(np.float64) @ weight.astype(np.float64).T).astype(np.float32)
        yield (
            dict(
                name=f"{qtype}-b{batch}-{width}to{outputs}",
                kind="projection",
                qtype=qtype,
                width=width,
            ),
            (x, raw),
            expected,
        )
    heads, width, capacity, selected = (2, 256, 16384, 2051) if real_shapes else (2, 32, 17, 9)
    group = 32
    codes = rng.integers(-127, 128, (batch, capacity, heads * width), dtype=np.int8)
    scales = (rng.uniform(0.0001, 0.01, (batch * heads * width // group, capacity))).astype(
        ml_dtypes.bfloat16
    )
    # Mixed valid lengths, repeated addresses, and explicit safe zero addresses.
    lengths = np.linspace(capacity // 4, capacity, batch, dtype=int)
    indices = np.stack([rng.integers(0, n, selected, dtype=np.int32) for n in lengths])
    indices[:, :2] = 0
    groups = heads * width // group
    plane = scales.reshape(batch, groups, capacity).astype(np.float32)
    expected = (
        codes[np.arange(batch)[:, None], indices]
        .reshape(batch, selected, groups, group)
        .astype(np.float32)
    )
    expected *= np.take_along_axis(plane, indices[:, None, :], axis=2).transpose(0, 2, 1)[..., None]
    expected = (
        expected.reshape(batch, selected, heads, width)
        .astype(ml_dtypes.bfloat16)
        .astype(np.float32)
    )
    yield (
        dict(name=f"kv-b{batch}-t{capacity}-s{selected}", kind="cache", heads=heads, width=width),
        (codes, scales, indices),
        expected,
    )


def jax_backend(case, inputs, cpu):
    import jax
    import jax.numpy as jnp
    from qwen38_tpu.attention_cache import read_quantized
    from qwen38_tpu.config import ModelConfig
    from qwen38_tpu.quant import dequantize_rows

    devices = jax.devices()
    device = devices[0]
    if not cpu and device.platform != "tpu":
        raise RuntimeError("JAX speed measurement requires TPU; no CPU fallback")
    if case["kind"] == "projection":

        def fn(x, raw):
            w = dequantize_rows(raw, case["qtype"], case["width"], backend="jax", dtype=x.dtype)
            return jnp.matmul(x, w.T, preferred_element_type=jnp.float32)
    else:
        c = ModelConfig(
            kv_heads=case["heads"], head_dim=case["width"], rotary_dim=32, kv_cache_dtype="q8"
        )

        def fn(codes, scales, indices):
            return read_quantized(dict(k=codes, k_scale=scales), "k", indices, c, jnp.bfloat16)

    return (
        jax.jit(fn),
        tuple(jax.device_put(a, device) for a in inputs),
        jax.block_until_ready,
        lambda a: np.asarray(a).astype(np.float32),
        dict(
            device=device.device_kind,
            jax=jax.__version__,
            jaxlib=importlib.metadata.version("jaxlib"),
            libtpu=importlib.metadata.version("libtpu") if not cpu else None,
        ),
    )


def torch_backend(case, inputs, cpu, method):
    import torch

    from .ops import decode_rows, matmul_f32, read_cache
    from .runtime import configure_tpu

    info = dict(device="cpu", torch=torch.__version__)
    if not cpu:
        info = configure_tpu()
    device = "cpu" if cpu else "tpu"
    arguments = []
    for a in inputs:
        # NumPy's ml_dtypes BF16 is not a torch.from_numpy supported dtype.
        if a.dtype.name == "bfloat16":
            value = torch.from_numpy(a.astype(np.float32)).to(torch.bfloat16)
        else:
            value = torch.from_numpy(a.copy())
        arguments.append(value.to(device))
    if method == "hybrid":
        if cpu:
            raise ValueError("Hybrid execution requires TorchTPU; no fake CPU timing")
        from .hybrid import register_cache_read, register_projection

        name = "qwen38_trial::" + case["name"].replace("-", "_")
        fn = (
            register_projection(name, case["qtype"], case["width"])
            if case["kind"] == "projection"
            else register_cache_read(name, case["heads"], case["width"])
        )
    elif case["kind"] == "projection":

        def fn(x, raw):
            return matmul_f32(x, decode_rows(raw, case["qtype"], case["width"], dtype=x.dtype))
    else:

        def fn(codes, scales, indices):
            return read_cache(codes, scales, indices, case["heads"], case["width"])

    if method != "torch_eager":
        fn = torch.compile(fn, backend="eager" if cpu else "tpu", fullgraph=True, dynamic=False)

    def sync(value):
        if not cpu:
            torch.tpu.synchronize()

    return fn, tuple(arguments), sync, lambda a: a.detach().float().cpu().numpy(), info


def run(args):
    report = dict(
        status="RUNNING",
        method=args.method,
        correctness_only=args.correctness_only,
        production_enabled=False,
        scope="single_chip_subgraphs_not_full_model",
        cases=[],
    )
    save(args.output, report)
    try:
        for batch in args.batches:
            for case, inputs, reference in fixtures(batch, not args.correctness_only):
                report["phase"] = case["name"]
                save(args.output, report)
                print(report["phase"], flush=True)
                setup = jax_backend if args.method == "jax" else torch_backend
                options = () if args.method == "jax" else (args.method,)
                fn, arguments, sync, host, info = setup(
                    case, inputs, args.correctness_only, *options
                )
                report["environment"] = info
                # Correctness gate BEFORE any latency is admitted.
                started = time.perf_counter()
                output = fn(*arguments)
                sync(output)
                compile_and_first = time.perf_counter() - started
                actual = host(output)
                if actual.shape != reference.shape or not np.isfinite(actual).all():
                    raise ValueError("Wrong shape or nonfinite output")
                if case["kind"] == "cache":
                    np.testing.assert_array_equal(actual, reference)
                else:
                    np.testing.assert_allclose(actual, reference, atol=2e-5, rtol=2e-4)
                row = dict(
                    **case,
                    correctness="PASS",
                    max_error=float(np.max(np.abs(actual - reference))),
                    input_sha256=hashlib.sha256(
                        b"".join(np.ascontiguousarray(a).tobytes() for a in inputs)
                    ).hexdigest(),
                )
                if not args.correctness_only:
                    timing = sample(fn, arguments, sync, args.repeats)
                    row["compile_and_first_execute_seconds"] = compile_and_first
                    row.update(timing)
                    if getattr(args, "capture_trace", False) and batch == 4:
                        trace = args.output.parent / "backend-profiles" / args.method / case["name"]
                        trace.mkdir(parents=True, exist_ok=True)
                        try:
                            if args.method == "jax":
                                import jax

                                with jax.profiler.trace(str(trace), create_perfetto_trace=True):
                                    for _ in range(5):
                                        sync(fn(*arguments))
                            else:
                                import torch
                                from torch.tpu.profiler import TpuProfilerConfig

                                with torch.profiler.profile(
                                    activities=[
                                        torch.profiler.ProfilerActivity.CPU,
                                        torch.profiler.ProfilerActivity.TPU,
                                    ],
                                    experimental_config=TpuProfilerConfig(run_dir=trace),
                                    on_trace_ready=torch.profiler.tensorboard_trace_handler(
                                        str(trace)
                                    ),
                                ):
                                    for _ in range(5):
                                        sync(fn(*arguments))
                            row["profile"] = dict(status="CAPTURED", path=str(trace))
                        except Exception as error:
                            row["profile"] = dict(status="ERROR", error_type=type(error).__name__)
                report["cases"].append(row)
                save(args.output, report)
        report["status"] = "passed"
    except Exception as error:
        report.update(status="ERROR", error_type=type(error).__name__, error=str(error)[:2000])
    save(args.output, report)
    return 0 if report["status"] == "passed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method", choices=["jax", "torch_eager", "torch_compile", "hybrid"], required=True
    )
    parser.add_argument("--correctness-only", action="store_true")
    parser.add_argument("--capture-trace", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--repeats", type=int, default=11)
    args = parser.parse_args()
    if not set(args.batches).issubset({1, 2, 4, 8}) or not 3 <= args.repeats <= 31:
        parser.error("Require batches 1/2/4/8 and 3..31 repetitions")
    if args.correctness_only:
        os.environ["JAX_PLATFORMS"] = "cpu"
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
