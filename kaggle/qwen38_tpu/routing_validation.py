"""Small independent routed-FFN gate, including prompt tiling and hot experts."""

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from .compute import numpy_q8_matmul, pack_q8
from .experts import expert_parallel
from .parallel import expert_mesh, put_expert_bytes
from .quant import dequantize_rows


def validate_expert_routing(mesh, diagnostics=None, *, group_size=1, tensor_parallel=1, batch=273):
    """Mixed Q6_K/Q8_0, >two tiles for one expert, duplicates and finished rows.

    The oracle routes each expert independently using NumPy, without the JAX
    compaction/tiling code. The full-model pinned gates remain separate.
    """
    rng = np.random.default_rng(738)
    # More than four local experts tests multiple groups AND a partial tail
    # even in EP8; the hot expert separately exercises multiple row tiles.
    hidden, intermediate, count = 256, 32 * tensor_parallel, 5 * mesh.size
    moe_mesh = expert_mesh(mesh, tensor_parallel)
    packed = {}
    for name in ("gate", "up"):
        raw = rng.integers(0, 256, (count, intermediate, 1, 210), dtype=np.uint8)
        raw[..., 192:208] = rng.integers(1, 9, raw[..., 192:208].shape, dtype=np.uint8)
        raw[..., 208:210] = np.array([1 / 2048], "<f2").view(np.uint8)
        packed[name] = raw
    packed["down"] = pack_q8(
        rng.normal(0, 0.08, (count * hidden, intermediate)).astype(np.float32)
    ).reshape(count, hidden, intermediate // 32, 34)
    device = [
        put_expert_bytes(
            raw, name, "Q8_0" if name == "down" else "Q6_K", hidden, intermediate, moe_mesh
        )
        for name, raw in packed.items()
    ]
    ids = np.stack((np.zeros(batch, np.int32), 1 + np.arange(batch) % (count - 1)), -1)
    ids[:8, 1] = 0  # duplicate routes combine, rather than overwrite
    weights = np.full(ids.shape, 0.5, np.float32)
    if batch > 1:
        weights[::37] = 0  # finished requests must contribute zero; keep B1 nonzero
    source = rng.normal(0, 0.3, (batch, hidden)).astype(np.float32)
    results = []
    for dtype in (np.float32, np.dtype(jnp.bfloat16)):
        x = source.astype(dtype)
        routing = weights.astype(dtype)
        for policy in ("bf16", "int8_q8"):
            expected = np.zeros((batch, hidden), np.float32)
            for expert in range(count):
                route = np.sum(np.where(ids == expert, routing, 0).astype(np.float32), -1)

                def project(value, name):
                    raw = packed[name][expert]
                    if policy == "int8_q8" and name == "down":
                        return numpy_q8_matmul(value.astype(np.float32), raw).astype(dtype)
                    qtype = "Q8_0" if name == "down" else "Q6_K"
                    width = intermediate if name == "down" else hidden
                    matrix = dequantize_rows(raw, qtype, width).astype(dtype)
                    return (value.astype(np.float32) @ matrix.astype(np.float32).T).astype(dtype)

                gate = project(x, "gate").astype(np.float32)
                activated = (gate / (1 + np.exp(-gate))).astype(dtype)
                product = (
                    activated.astype(np.float32) * project(x, "up").astype(np.float32)
                ).astype(dtype)
                expected += project(product, "down").astype(np.float32) * route[:, None]
            expected = expected.astype(dtype).astype(np.float32)
            function = expert_parallel(
                mesh,
                gate_type="Q6_K",
                up_type="Q6_K",
                down_type="Q8_0",
                hidden=hidden,
                intermediate=intermediate,
                compute=policy,
                group_size=group_size,
                tensor_parallel=tensor_parallel,
            )
            arguments = (
                *device,
                jax.device_put(x, NamedSharding(mesh, P())),
                jax.device_put(ids, NamedSharding(mesh, P())),
                jax.device_put(routing, NamedSharding(mesh, P())),
            )
            actual = np.asarray(function(*arguments)).astype(np.float32)
            tolerance = 3e-4 if dtype == np.float32 else 0.003
            item = dict(
                dtype=str(np.dtype(dtype)),
                policy=policy,
                expert_group_size=group_size,
                expert_parallel=moe_mesh.shape["expert"],
                tensor_parallel=tensor_parallel,
                batch=batch,
                row_tile=128,
                hot_expert_rows=int(np.count_nonzero(weights[:, 0])),
                max_abs_error=float(np.max(np.abs(actual - expected))),
                absolute_tolerance=tolerance,
            )
            results.append(item)
            if diagnostics is not None:
                diagnostics["shapes"] = results
            np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=tolerance)
            np.testing.assert_array_equal(actual[np.all(weights == 0, axis=-1)], 0)
            hlo = str(function.lower(*arguments).compiler_ir()).lower()
            if "all_gather" in hlo or "all_reduce" not in hlo:
                raise AssertionError("Expert weights must stay on their owner; only outputs reduce")
            zeros = function(*arguments[:-1], jnp.zeros_like(arguments[-1]))
            np.testing.assert_array_equal(np.asarray(zeros), 0)
    return dict(status="PASS", devices=mesh.size, shapes=results, expert_weights_all_gathered=False)
