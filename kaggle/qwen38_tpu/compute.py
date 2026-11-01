"""Native BF16 matmul and gated W8A8 compute for GGUF Q8_0 experts.

INT8 is a candidate, not a synonym for packed storage. Per-32 weight scales
require separate integer partial products. The TPU gate must show that this
actually beats BF16 on both prompt and decode shapes before selecting it.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np

from . import numerics
from .math import linear
from .precision import compute_cast
from .quant import dequantize_rows


def q8_matmul(x, packed, *, output_dtype=None):
    """W8A8 dot, INT8 operands / INT32 accumulator / FP32 scaled reduction."""
    bits = packed[..., 0].astype(jnp.uint16) | (packed[..., 1].astype(jnp.uint16) << 8)
    weight_scale = jax.lax.bitcast_convert_type(bits, jnp.float16).astype(jnp.float32)
    weights = jax.lax.bitcast_convert_type(packed[..., 2:], jnp.int8)
    grouped = x.astype(jnp.float32).reshape(x.shape[0], -1, 32)
    scale = jnp.max(jnp.abs(grouped), -1) / 127
    codes = jnp.clip(
        jnp.rint(numerics.divide(grouped, jnp.where(scale == 0, 1, scale)[..., None])), -127, 127
    ).astype(jnp.int8)
    partial = jnp.einsum("bgk,ogk->bgo", codes, weights, preferred_element_type=jnp.int32)
    result = jnp.sum(
        partial.astype(jnp.float32) * scale[..., None] * weight_scale.T[None, :, :], axis=1
    )
    return compute_cast(result, x.dtype if output_dtype is None else output_dtype)


def bf16_q8_matmul(x, packed):
    return linear(x, dequantize_rows(packed, "Q8_0", x.shape[-1], backend="jax", dtype=x.dtype))


def pack_q8(weights):
    rows = np.asarray(weights, np.float32).reshape(weights.shape[0], -1, 32)
    scale = (np.max(np.abs(rows), -1) / 127).astype("<f2")
    codes = np.clip(np.rint(rows / np.where(scale == 0, 1, scale)[..., None]), -127, 127).astype(
        np.int8
    )
    return np.concatenate(
        (scale.view(np.uint8).reshape(*scale.shape, 2), codes.view(np.uint8)), axis=-1
    )


def numpy_q8_matmul(x, packed):
    """Independent scalar-block reference; no JAX/decoder implementation reuse."""
    scale = np.ascontiguousarray(packed[..., :2]).view("<f2").reshape(packed.shape[:2])
    weights = np.ascontiguousarray(packed[..., 2:]).view(np.int8).astype(np.int32)
    grouped = np.asarray(x, np.float32).reshape(x.shape[0], -1, 32)
    act_scale = np.max(np.abs(grouped), -1) / 127
    codes = np.clip(
        np.rint(grouped / np.where(act_scale == 0, 1, act_scale)[..., None]), -127, 127
    ).astype(np.int32)
    out = np.zeros((x.shape[0], packed.shape[0]), np.float32)
    for group in range(weights.shape[1]):
        partial = codes[:, group] @ weights[:, group].T
        out += partial.astype(np.float32) * act_scale[:, group, None] * scale[:, group][None, :]
    return out


def native_dot_proof():
    def dots(function, *args):
        graph = jax.make_jaxpr(function)(*args)
        return [
            dict(inputs=[str(v.aval.dtype) for v in e.invars], output=str(e.outvars[0].aval.dtype))
            for e in graph.jaxpr.eqns
            if e.primitive.name == "dot_general"
        ]

    x, w = jnp.ones((8, 32), jnp.bfloat16), jnp.ones((32, 32), jnp.bfloat16)
    bf16 = dots(linear, x, w)
    integer = dots(q8_matmul, x, jnp.asarray(pack_q8(np.ones((32, 32), np.float32))))
    if bf16 != [dict(inputs=["bfloat16", "bfloat16"], output="float32")]:
        raise RuntimeError("BF16 projection did not use BF16 operands / FP32 accumulation")
    if integer != [dict(inputs=["int8", "int8"], output="int32")]:
        raise RuntimeError("INT8 candidate did not use INT8 operands / INT32 accumulation")
    return dict(
        status="PASS",
        bf16_dots=bf16,
        int8_dots=integer,
        device_kinds=[d.device_kind for d in jax.devices()],
    )


def choose_policy(rows, emit=lambda **kw: None, *, repeats=5, shapes=((2560, 640), (640, 2560))):
    """Bounded on-device calibration before full weights; fail closed on bugs.

    Additional activation quantization is allowed only with <=1.2% relative
    L2 error, an independent integer oracle, and >=10% wins for every shape.
    Calibration is synthetic; full-model Q8/W8A8 golden and real tasks are
    separately required. A correct but slower candidate stays disabled.
    """
    rng = np.random.default_rng(723)
    integer = jax.jit(q8_matmul)
    floating = jax.jit(bf16_q8_matmul)
    results = []
    proof = native_dot_proof()
    for width, out in shapes:
        raw = pack_q8(rng.normal(0, 0.025, (out, width)).astype(np.float32))
        packed = jax.device_put(raw, jax.devices()[0])
        for batch in sorted(set(rows)):
            emit(stage="compute_calibration", detail=f"rows={batch} {width}->{out}")
            host = rng.normal(size=(batch, width)).astype(np.float32)
            if width == 640:
                host = (host / (1 + np.exp(-host))) * rng.normal(size=host.shape)
            # Include a zero row and bounded heavy tails in representative data.
            if batch > 1:
                host[0] = 0
            if batch > 1:
                host[1, ::32] *= 8
            x = jax.device_put(host, jax.devices()[0]).astype(jnp.bfloat16)
            actual = np.asarray(integer(x, packed)).astype(np.float32)
            expected = numpy_q8_matmul(np.asarray(x).astype(np.float32), raw)
            np.testing.assert_allclose(
                actual, expected.astype(jnp.bfloat16).astype(np.float32), atol=0.016, rtol=0.008
            )
            baseline = np.asarray(floating(x, packed)).astype(np.float32)
            error = float(np.linalg.norm(actual - baseline) / max(np.linalg.norm(baseline), 1e-12))

            def seconds(fn):
                jax.block_until_ready(fn(x, packed))
                values = []
                for _ in range(repeats):
                    started = time.monotonic()
                    jax.block_until_ready(fn(x, packed))
                    values.append(time.monotonic() - started)
                return float(np.median(values))

            bf16_s, int8_s = seconds(floating), seconds(integer)
            results.append(
                dict(
                    rows=batch,
                    width=width,
                    output=out,
                    relative_l2_error=error,
                    bf16_seconds=bf16_s,
                    int8_seconds=int8_s,
                    accuracy_pass=error <= 0.012,
                    speed_pass=int8_s <= bf16_s * 0.9,
                )
            )
    selected = "int8_q8" if all(r["accuracy_pass"] and r["speed_pass"] for r in results) else "bf16"
    return dict(
        status="PASS",
        selected=selected,
        shapes=results,
        int8_lowering="PASS",
        native_dot_proof=proof,
        activation_quantization="symmetric_signed_int8_per_32_fp32_scale",
        reason="Every shape passed accuracy and speed"
        if selected != "bf16"
        else "Candidate did not meet every accuracy and speed threshold",
    )
