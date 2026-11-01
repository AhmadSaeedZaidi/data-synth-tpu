"""Experimental integer contractions of unchanged GGUF Q4_K weight codes.

Production inference retains BF16 activations. These kernels quantify the
cost and error of activation quantization before enabling it in the decoder.
Both operands of a native INT4 dot are signed INT4, with INT32 accumulation.
No mixed-dtype promotion or CPU emulation is silently labelled native INT4.
"""

import hashlib
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from . import tracing


def q4_groups(raw):
    """Return signed codes, scales and offsets for each original 32-value group.

    GGUF stores unsigned q in [0,15], w = d*s*q - dmin*m. Recentring
    q to [-8,7] preserves the weight when offset = 8*d*s - dmin*m.
    The offset depends on output channel AND contraction group.
    """
    raw = jnp.asarray(raw)
    if raw.ndim < 2 or raw.shape[-1] != 144 or raw.dtype != jnp.uint8:
        raise ValueError("Q4_K requires uint8 blocks with 144 bytes")

    def half(field):
        bits = field[..., 0].astype(jnp.uint16) | (field[..., 1].astype(jnp.uint16) << 8)
        return lax.bitcast_convert_type(bits, jnp.float16).astype(jnp.float32)

    d, dmin = half(raw[..., :2]), half(raw[..., 2:4])
    a, b, c = (raw[..., start : start + 4] for start in (4, 8, 12))
    scales = jnp.concatenate((a & 63, (c & 15) | ((a >> 2) & 48)), axis=-1)
    mins = jnp.concatenate((b & 63, (c >> 4) | ((b >> 2) & 48)), axis=-1)
    shape = raw.shape[:-2] + (raw.shape[-2] * 8,)
    scale = (d[..., None] * scales.astype(jnp.float32)).reshape(shape)
    minimum = (dmin[..., None] * mins.astype(jnp.float32)).reshape(shape)
    packed = raw[..., 16:].reshape(raw.shape[:-1] + (4, 32))
    unsigned = jnp.stack((packed & 15, packed >> 4), axis=-2).reshape(shape + (32,))
    signed = (unsigned.astype(jnp.int8) - 8).astype(jnp.int8)
    return signed, scale, 8 * scale - minimum


def quantize_groups(x, bits):
    """Symmetric dynamic activation quantization per token and 32-value group."""
    if bits not in (4, 8) or x.shape[-1] % 32:
        raise ValueError("activation bits must be 4 or 8 and width a multiple of 32")
    grouped = x.astype(jnp.float32).reshape(x.shape[:-1] + (-1, 32))
    maximum = jnp.max(jnp.abs(grouped), axis=-1, keepdims=True)
    bound = (1 << (bits - 1)) - 1
    scale = jnp.where(maximum > 0, jnp.maximum(maximum / bound, jnp.finfo(jnp.float32).tiny), 1)
    codes = jnp.clip(jnp.rint(grouped / scale), -bound, bound).astype(jnp.int8)
    return codes, scale[..., 0]


def split_activation_nibbles(codes):
    """Exact signed-nibble representation of every INT8 value, including -128."""
    wide = codes.astype(jnp.int16)
    high = jnp.floor_divide(wide, 16).astype(jnp.int8)
    low = (wide - 16 * high.astype(jnp.int16) - 8).astype(jnp.int8)
    return high, low


def grouped_q4_matmul(x, raw, *, activation_bits=8, native_int4=False):
    """[expert,token,K] times GGUF [expert,output,K/256,144].

    W4A8 normally uses INT8 dots, keeping original four-bit weight codes.
    The native variant exactly splits the INT8 activation into two signed
    nibbles: a = 16*hi + lo + 8. Two INT4 dots plus a weight-sum correction
    retain the same quantized activation, rather than rounding it to INT4.
    W4A4 uses one INT4 dot, with greater activation error. All scale and offset
    corrections occur BEFORE summing contraction groups.
    """
    if x.ndim != 3 or raw.ndim != 4 or x.shape[0] != raw.shape[0]:
        raise ValueError("expected expert-major activations and packed weight matrices")
    if x.shape[-1] != raw.shape[-2] * 256:
        raise ValueError("Q4_K matmul requires complete 256-value input blocks")
    codes, scale = quantize_groups(x, activation_bits)
    weight, weight_scale, offset = q4_groups(raw)

    def dot(lhs, rhs, dtype):
        # Batch over expert and contraction group; contract only each group's
        # 32 codes. Group scales cannot be pulled outside the full K reduction.
        return lax.dot_general(
            lhs.transpose(0, 2, 1, 3).astype(dtype),
            rhs.transpose(0, 2, 3, 1).astype(dtype),
            (((3,), (2,)), ((0, 1), (0, 1))),
            preferred_element_type=jnp.int32,
            precision=lax.Precision.DEFAULT,
        ).transpose(0, 2, 3, 1)

    if native_int4 and activation_bits == 8:
        high, low = split_activation_nibbles(codes)
        product = 16 * dot(high, weight, jnp.int4) + dot(low, weight, jnp.int4)
        product += 8 * jnp.sum(weight.astype(jnp.int32), axis=-1)[:, None]
    else:
        product = dot(codes, weight, jnp.int4 if native_int4 else jnp.int8)
    scaled = product.astype(jnp.float32) * scale[:, :, None] * weight_scale[:, None]
    activation_sum = jnp.sum(codes.astype(jnp.int32), axis=-1).astype(jnp.float32) * scale
    scaled += activation_sum[:, :, None] * offset[:, None]
    return jnp.sum(scaled, axis=-1)


@tracing.operation("integer_dot_probe")
def dot_probe(deadline=lambda: None, publish=None, *, rows=(1, 8, 128), repeats=3):
    """Small bounded MXU eligibility/timing probe on every available chip.

    Timings are blocked, after compilation and warmup, on resident inputs.
    Optimized HLO evidence is retained; an s4 dot is NOT instruction-level
    proof of the advertised peak. CPU INT4 execution is explicitly skipped.
    """
    mesh = Mesh(np.array(jax.devices()), ("x",))
    sharding = NamedSharding(mesh, P("x"))
    n, k, out = mesh.size, 1024, 256
    rng = np.random.default_rng(731)
    weights = rng.integers(-8, 8, (n, k, out), dtype=np.int8)
    rhs = jax.device_put(weights, sharding)
    result = {
        "jax_version": jax.__version__,
        "platform": jax.devices()[0].platform,
        "device_kind": jax.devices()[0].device_kind,
        "devices": n,
        "scope": "resident integer-valued matrix inputs; excludes GGUF decoding and streaming",
        "cases": [],
    }
    for m in rows:
        inputs = rng.integers(-8, 8, (n, m, k), dtype=np.int8)
        lhs = jax.device_put(inputs, sharding)
        reference = inputs.astype(np.int32) @ weights.astype(np.int32)
        for name, dtype in (("bf16", jnp.bfloat16), ("int8", jnp.int8), ("int4", jnp.int4)):
            deadline()
            item = {"dtype": name, "shape_per_chip": [m, k, out], "status": "COMPILING"}
            result["cases"].append(item)
            if publish:
                publish(result)

            def local(a, b):
                y = lax.dot_general(
                    a[0].astype(dtype),
                    b[0].astype(dtype),
                    (((1,), (0,)), ((), ())),
                    preferred_element_type=jnp.float32 if name == "bf16" else jnp.int32,
                    precision=lax.Precision.DEFAULT,
                )
                return y[None]

            program = jax.jit(
                jax.shard_map(
                    local, mesh=mesh, in_specs=(P("x"), P("x")), out_specs=P("x"), check_vma=False
                )
            )
            try:
                started = time.perf_counter()
                lowered = program.lower(lhs, rhs)
                ir = str(lowered.compiler_ir())
                dots = [line.strip() for line in ir.splitlines() if "stablehlo.dot_general" in line]
                item["stablehlo_dot"] = dots
                item["stablehlo_sha256"] = hashlib.sha256(ir.encode()).hexdigest()
                if name == "int4" and result["platform"] != "tpu":
                    item.update(status="SKIPPED", reason="native INT4 execution requires TPU")
                    continue
                executable = lowered.compile()
                item["compile_seconds"] = time.perf_counter() - started
                optimized = executable.as_text()
                item["optimized_hlo_sha256"] = hashlib.sha256(optimized.encode()).hexdigest()
                item["optimized_hlo_contains_s4"] = "s4[" in optimized
                deadline()
                jax.block_until_ready(executable(lhs, rhs))
                timings = []
                for _ in range(repeats):
                    deadline()
                    started = time.perf_counter()
                    actual = jax.block_until_ready(executable(lhs, rhs))
                    timings.append(time.perf_counter() - started)
                np.testing.assert_array_equal(np.asarray(actual), reference)
                elapsed = float(np.median(timings))
                item.update(
                    status="PASS",
                    seconds=timings,
                    median_seconds=elapsed,
                    effective_tops_per_chip=2 * m * k * out / elapsed / 1e12,
                )
            except (NotImplementedError, RuntimeError, ValueError) as error:
                item.update(
                    status="UNAVAILABLE", error_type=type(error).__name__, error=str(error)[:600]
                )
            if publish:
                publish(result)
    result["status"] = "COMPLETE"
    result["int4_available"] = any(
        item["dtype"] == "int4" and item["status"] == "PASS" for item in result["cases"]
    )
    result["measured_cases"] = sum(item["status"] == "PASS" for item in result["cases"])
    if publish:
        publish(result)
    return result
