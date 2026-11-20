"""Matched native-highest and explicit six-product BF16 matrix experiments.

Each method owns a separate worker process. References use CPU FP64 on the
same FP32 input bytes. CPU correctness mode never reports hardware timings.
This tests operators, not a full Torch model or an accumulated recurrent error.
"""

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np

from .comparison import sample, save


def fixtures(cpu=False):
    rng = np.random.default_rng(63016)
    for batch in (1, 2, 4, 8):
        # Router-like dense projection and a grouped recurrent-state read.
        width, outputs = (256, 32) if cpu else (2560, 256)
        yield (
            f"router-b{batch}",
            rng.normal(0, 0.1, (batch, width)).astype(np.float32),
            rng.normal(0, 0.1, (outputs, width)).astype(np.float32),
        )
        width = 32 if cpu else 256
        yield (
            f"state-read-b{batch}",
            rng.normal(0, 0.1, (batch * 4, 1, width)).astype(np.float32),
            rng.normal(0, 0.1, (batch * 4, width, width)).astype(np.float32),
        )
    # A near-cancelling dot and operands with a broad normal exponent range.
    # Keep the same fixtures on CPU and TPU so these gates are tested locally.
    x = rng.normal(0, 0.2, (4, 256)).astype(np.float32)
    w = rng.normal(0, 0.2, (8, 256)).astype(np.float32)
    x[:, 128:] = x[:, :128]
    w[:, 128:] = -w[:, :128] + np.float32(2**-20)
    yield "cancellation", x, w
    powers = np.linspace(-12, 12, 256).astype(np.float32)
    yield (
        "mixed-exponents",
        (rng.normal(0, 0.1, (4, 256)) * np.exp2(powers)).astype(np.float32),
        (rng.normal(0, 0.1, (8, 256)) * np.exp2(-powers)).astype(np.float32),
    )


def run(args):
    import torch

    from .ops import matmul_bf16_x6, matmul_f32

    report = dict(
        status="running",
        method=args.method,
        correctness_only=args.correctness_only,
        production_enabled=False,
        scope="fp32_matmul_operators_not_full_model_or_recurrent_rollout",
        retained_terms=["hh", "hm", "mh", "hl", "mm", "lh"],
        discarded_terms=["ml", "lm", "ll"],
        tolerance=dict(atol=1e-5, rtol=1e-5),
        cases=[],
    )
    save(args.output, report)
    try:
        if args.correctness_only:
            device = "cpu"
            report["environment"] = dict(device=device, torch=torch.__version__)
        else:
            from .runtime import configure_tpu

            report["environment"] = configure_tpu()
            device = "tpu"
        operator = matmul_bf16_x6 if args.method == "explicit_x6" else matmul_f32

        def sync(value):
            if device == "tpu":
                torch.tpu.synchronize()

        for name, x, w in fixtures(args.correctness_only):
            # Every shape is used once, with all warm samples taken together.
            # Retire Dynamo's prior shape guards rather than exceeding its
            # eight-specialization limit on this bounded ten-shape sweep.
            torch._dynamo.reset()
            fn = torch.compile(
                operator,
                backend="eager" if args.correctness_only else "tpu",
                fullgraph=True,
                dynamic=False,
            )
            report["phase"] = name
            save(args.output, report)
            reference = x.astype(np.float64) @ w.astype(np.float64).swapaxes(-1, -2)
            row = dict(
                name=name,
                shapes=[list(x.shape), list(w.shape)],
                input_sha256=hashlib.sha256(x.tobytes() + w.tobytes()).hexdigest(),
            )
            arguments = tuple(torch.from_numpy(a).to(device) for a in (x, w))
            started = time.perf_counter()
            value = fn(*arguments)
            sync(value)
            first_seconds = time.perf_counter() - started
            actual = value.detach().float().cpu().numpy()
            if actual.shape != reference.shape or not np.isfinite(actual).all():
                raise ValueError("Wrong shape or nonfinite six-product output")
            error = actual.astype(np.float64) - reference
            row.update(
                max_absolute_error=float(np.max(np.abs(error))),
                relative_l2_error=float(
                    np.linalg.norm(error) / max(np.linalg.norm(reference), 1e-30)
                ),
            )
            try:
                np.testing.assert_allclose(actual, reference, atol=1e-5, rtol=1e-5)
            except AssertionError:
                row["correctness"] = "FAIL"
            else:
                row["correctness"] = "PASS"
                if not args.correctness_only:
                    row["compile_and_first_execute_seconds"] = first_seconds
                    row.update(sample(fn, arguments, sync, 7))
            report["cases"].append(row)
            save(args.output, report)
        report["status"] = (
            "passed" if all(c["correctness"] == "PASS" for c in report["cases"]) else "failed"
        )
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error)[:2000])
    save(args.output, report)
    return 0 if report["status"] == "passed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["builtin_x6", "explicit_x6"], required=True)
    parser.add_argument("--correctness-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
