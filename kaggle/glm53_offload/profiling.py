"""Bounded A/B probe of the actual expert kernels, without a full model load."""

import time

import jax
import numpy as np
from glm53.engine import AXIS, R, local_cfg, shard_map
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from . import progress, tracing
from .expert_kernel import tiled_experts
from .gguf_quant import dequantize_rows
from .integer_kernel import grouped_q4_matmul
from .routing import expert_tile_capacities, expert_token_rows
from .weights import BASE_REPO, BASE_REVISION, PackedExperts, fetch_json, hf_url, open_gguf


def compiler_estimates(executable):
    """Backend estimates are optional and are not measured peak HBM/FLOPs."""
    result = {}
    try:
        costs = executable.cost_analysis()
        result["cost"] = costs if isinstance(costs, dict) else {"unavailable": "no cost mapping"}
    except Exception as error:
        result["cost_unavailable"] = type(error).__name__
    try:
        memory = executable.memory_analysis()
        result["memory"] = (
            {key: getattr(memory, key) for key in dir(memory) if key.endswith("_size_in_bytes")}
            if memory is not None
            else {"unavailable": "no memory analysis"}
        )
    except Exception as error:
        result["memory_unavailable"] = type(error).__name__
    return result


@tracing.operation("integer_projection_profile")
def integer_projection_profile(store, mesh, deadline, publish=None):
    """Compare one unchanged real Q4_K expert projection with integer variants.

    Synthetic activations, all-chip resident inputs, no host streaming. Only
    arithmetic consistency is asserted; activation error is reported, not
    misrepresented as model quality. Native INT4 must execute on TPU.
    """
    selected = next(
        (
            (layer, number)
            for layer in store.tables
            for number, qtype in enumerate(store.qtypes(layer))
            if qtype == "Q4_K"
        ),
        None,
    )
    result = {
        "status": "RUNNING",
        "cases": [],
        "scope": (
            "one real GGUF Q4_K projection with synthetic BF16 activations; resident inputs; "
            "includes unpack, quantize, dot and affine correction; excludes streaming, routing, "
            "attention, vision and model quality; integer activation math is diagnostic only"
        ),
    }
    if selected is None:
        result.update(status="SKIPPED", reason="checkpoint has no Q4_K expert projection")
        if publish:
            publish(result)
        return result
    layer, number = selected
    raw = store.tables[layer][number][:, :1].copy()
    width = raw.shape[-2] * 256
    result.update(
        layer=layer,
        projection=("gate", "up", "down")[number],
        input_width=width,
        output_width=raw.shape[2],
    )
    sharding = NamedSharding(mesh, P(AXIS))
    packed = jax.device_put(raw, sharding)
    for tokens in (1, 8):
        x = np.random.default_rng(943).normal(size=(mesh.size, 1, tokens, width)).astype(np.float32)
        x = jax.device_put(x, sharding).astype(jax.numpy.bfloat16)
        outputs = {}
        for name, bits, native in (
            ("bf16", None, False),
            ("w4a8_int8", 8, False),
            ("w4a8_two_int4", 8, True),
            ("w4a4_int4", 4, True),
        ):
            deadline()
            item = {"tokens": tokens, "mode": name, "status": "COMPILING"}
            result["cases"].append(item)
            if publish:
                publish(result)
            if native and jax.devices()[0].platform != "tpu":
                item.update(status="SKIPPED", reason="native INT4 requires TPU")
                continue

            def local(a, b):
                if bits is None:
                    w = dequantize_rows(b[0], "Q4_K", width, dtype=a.dtype, backend="jax")
                    y = jax.numpy.einsum(
                        "erk,eok->ero",
                        a[0],
                        w,
                        preferred_element_type=jax.numpy.float32,
                        precision=jax.lax.Precision.HIGHEST,
                    )
                else:
                    y = grouped_q4_matmul(a[0], b[0], activation_bits=bits, native_int4=native)
                return y[None]

            program = jax.jit(
                shard_map(
                    local,
                    mesh=mesh,
                    in_specs=(P(AXIS), P(AXIS)),
                    out_specs=P(AXIS),
                    check_vma=False,
                )
            )
            try:
                started = time.perf_counter()
                lowered = program.lower(x, packed)
                item["stablehlo_dot"] = [
                    line.strip()
                    for line in str(lowered.compiler_ir()).splitlines()
                    if "stablehlo.dot_general" in line
                ]
                executable = lowered.compile()
                item["compile_seconds"] = time.perf_counter() - started
                item["compiler_estimates"] = compiler_estimates(executable)
                deadline()
                jax.block_until_ready(executable(x, packed))
                seconds = []
                for _ in range(3):
                    deadline()
                    started = time.perf_counter()
                    output = jax.block_until_ready(executable(x, packed))
                    seconds.append(time.perf_counter() - started)
                outputs[name] = np.asarray(output, dtype=np.float32)
                if not np.isfinite(outputs[name]).all():
                    raise ValueError("non-finite integer projection")
                baseline = outputs["bf16"]
                difference = outputs[name] - baseline
                item.update(
                    status="MEASURED",
                    seconds=seconds,
                    median_seconds=float(np.median(seconds)),
                    relative_l2_error=float(
                        np.linalg.norm(difference) / max(np.linalg.norm(baseline), 1e-20)
                    ),
                    max_abs_error=float(np.max(np.abs(difference))),
                    quality_gate="NOT_RUN",
                )
                if name == "w4a8_two_int4" and "w4a8_int8" in outputs:
                    np.testing.assert_allclose(
                        outputs[name], outputs["w4a8_int8"], rtol=1e-5, atol=1e-3
                    )
                    item["same_activation_precision_as_int8"] = True
            except (NotImplementedError, RuntimeError) as error:
                item.update(
                    status="UNAVAILABLE", error_type=type(error).__name__, error=str(error)[:600]
                )
            if publish:
                publish(result)
    result["status"] = "COMPLETE"
    result["native_int4_measurements"] = sum(
        "int4" in item["mode"] and item["status"] == "MEASURED" for item in result["cases"]
    )
    if publish:
        publish(result)
    return result


def profile_case(cfg, store, layer, mesh, tokens, pattern, repeats, deadline, publish=None):
    """Use blocked execution and separate compilation, gather and transfer times."""
    total = store.tables[layer][0].shape[1]
    idx = np.arange(tokens * cfg.topk, dtype=np.int32).reshape(tokens, cfg.topk) % total
    if pattern == "hot_expert":
        idx[:, 0] = 0
        idx[:, 1:] = 1 + np.arange(tokens * (cfg.topk - 1)).reshape(tokens, -1) % (total - 1)
    elif pattern != "spread":
        raise ValueError("unknown routing pattern")
    weights = np.full(idx.shape, 1.0 / cfg.topk, np.float32)
    x = np.random.default_rng(71).normal(size=(tokens, cfg.hidden)).astype(np.float32)
    started = time.perf_counter()
    packed, ids, ep = store.gather(layer, idx)
    gather_seconds = time.perf_counter() - started
    token_rows = expert_token_rows(ids, ep)
    tile_capacities = expert_tile_capacities(token_rows, store.tile)
    started = time.perf_counter()
    sharding = NamedSharding(mesh, P(AXIS))
    device_packed = tuple(jax.device_put(a.reshape(*a.shape[:3], -1), sharding) for a in packed)
    jax.block_until_ready(device_packed)
    transfer_seconds = time.perf_counter() - started
    replicated = NamedSharding(mesh, R)
    args = tuple(
        jax.device_put(a, replicated) for a in (x, ids, weights, token_rows, tile_capacities)
    )
    lcfg = local_cfg(cfg, mesh.size)
    results, outputs = {}, {}
    partial = {
        "name": f"tokens_{tokens}_{pattern}",
        "status": "RUNNING",
        "tokens": tokens,
        "experts": ep,
        "route_capacity": token_rows.shape[1],
        "measurements": results,
        "host_gather_seconds": gather_seconds,
        "blocked_transfer_seconds": transfer_seconds,
    }

    def checkpoint(phase):
        partial["phase"] = phase
        if publish is not None:
            publish(partial)

    for name, routed, tiled in (
        ("all_tokens", False, False),
        ("routed_tokens", True, False),
        ("tile_routed_tokens", True, True),
    ):
        deadline()
        checkpoint(f"compile {name}")
        progress.update(detail=f"profile {tokens} tokens {pattern}: compile {name}", force=True)

        def kernel(x, idx, weights, rows, capacities, packed):
            partial = tiled_experts(
                x,
                packed,
                idx,
                weights,
                lcfg,
                store.qtypes(layer),
                store.tile,
                rows if routed else None,
                capacities if tiled else None,
            )
            return lcfg.reduce(partial)

        program = jax.jit(
            shard_map(
                kernel,
                mesh=mesh,
                in_specs=(R, R, R, R, R, (P(AXIS),) * 3),
                out_specs=R,
                check_vma=False,
            )
        )
        started = time.perf_counter()
        executable = program.lower(*args, device_packed).compile()
        compile_seconds = time.perf_counter() - started
        results[name] = {
            "compile_seconds": compile_seconds,
            "seconds": [],
            "compiler_estimates": compiler_estimates(executable),
        }
        checkpoint(f"warmup {name}")
        deadline()
        jax.block_until_ready(executable(*args, device_packed))
        timings = []
        for repeat in range(repeats):
            deadline()
            progress.update(detail=f"profile {tokens} tokens {pattern}: {name} repeat {repeat + 1}")
            started = time.perf_counter()
            output = jax.block_until_ready(executable(*args, device_packed))
            timings.append(time.perf_counter() - started)
            results[name]["seconds"] = list(timings)
            checkpoint(f"{name} repeat {repeat + 1}")
        outputs[name] = np.asarray(output)
    for output in outputs.values():
        if not np.isfinite(output).all():
            raise ValueError("non-finite expert probe output")
    for name in ("routed_tokens", "tile_routed_tokens"):
        np.testing.assert_allclose(outputs[name], outputs["all_tokens"], rtol=5e-3, atol=5e-3)
    old_seconds = float(np.median(results["all_tokens"]["seconds"]))
    new_seconds = float(np.median(results["routed_tokens"]["seconds"]))
    tiled_seconds = float(np.median(results["tile_routed_tokens"]["seconds"]))
    return {
        "name": f"tokens_{tokens}_{pattern}",
        "status": "PASS",
        "tokens": tokens,
        "experts": ep,
        "route_capacity": token_rows.shape[1],
        "routed_token_expert_pairs": int(np.count_nonzero(token_rows >= 0)),
        "computed_token_expert_pairs": int(token_rows.size),
        "tile_computed_token_expert_pairs": int(tile_capacities.sum()) * store.tile,
        "all_token_expert_pairs": ep * tokens,
        "host_gather_seconds": gather_seconds,
        "blocked_transfer_seconds": transfer_seconds,
        "bytes_transferred": sum(a.nbytes for a in packed),
        "old_kernel_seconds": old_seconds,
        "routed_kernel_seconds": new_seconds,
        "tile_routed_kernel_seconds": tiled_seconds,
        "tile_kernel_speedup": new_seconds / tiled_seconds,
        "kernel_speedup": old_seconds / new_seconds,
        "max_output_error": float(np.max(np.abs(outputs["routed_tokens"] - outputs["all_tokens"]))),
        "tile_max_output_error": float(
            np.max(np.abs(outputs["tile_routed_tokens"] - outputs["all_tokens"]))
        ),
        "measurements": results,
    }


@tracing.operation("expert_profile")
def run_expert_profile(config, report, save, deadline):
    from .run import cfg_from_json, log

    deadline()
    cfg = cfg_from_json(fetch_json(hf_url(BASE_REPO, BASE_REVISION, "config.json")))
    layer = cfg.mlp_types.index("sparse")
    report["stage"] = "PROFILE_EXPERT_LOAD"
    save()
    source = open_gguf(config["quant"], config.get("gguf_dirs", []), log)
    try:
        store = PackedExperts.from_gguf(
            source,
            cfg,
            len(jax.devices()),
            tile=config["expert_tile"],
            reserve_bytes=2 << 30,
            log=log,
            layers=[layer],
            expert_limit=config["profile_experts"],
        )
    finally:
        for shard in source.shards.values():
            shard.pf.close()
    report["expert_profile"] = {
        "layer": layer,
        "experts_loaded": config["profile_experts"],
        "packed_bytes_loaded": store.nbytes,
        "cases": [],
        "scope": (
            "isolated real-checkpoint expert kernels with synthetic activations/routing; "
            "excludes attention, full vision, generation, host LRU and model tokens/s"
        ),
    }
    mesh = Mesh(np.array(jax.devices()), (AXIS,))
    if config.get("integer_projection_probe", True):

        def publish_integer(value):
            report["integer_projection_probe"] = value
            save()

        integer_projection_profile(store, mesh, deadline, publish_integer)
    report["stage"] = "EXPERT_KERNEL_PROFILE"
    save()
    # Decode sizes plus a representative prefill chunk. A hot-expert case also
    # exposes occupancy padding: it can legitimately eliminate the speedup.
    sizes = list(dict.fromkeys([*config["batch_sizes"], config["prefill_chunk"]]))
    cases = [(n, "spread") for n in sizes] + [(config["prefill_chunk"], "hot_expert")]
    for tokens, pattern in cases:
        deadline()
        active = {"name": f"tokens_{tokens}_{pattern}", "status": "RUNNING", "phase": "staging"}
        report["expert_profile"]["cases"].append(active)
        save()

        def publish(partial):
            active.update(partial)
            report["expert_stats"] = dict(store.stats)
            save()

        with tracing.span("expert_profile_case", tokens=tokens, routing=pattern):
            try:
                result = profile_case(
                    cfg,
                    store,
                    layer,
                    mesh,
                    tokens,
                    pattern,
                    config["profile_repeats"],
                    deadline,
                    publish=publish,
                )
            except Exception as error:
                active.update(status="ERROR", error_type=type(error).__name__)
                save()
                raise
        active.update(result)
        report["expert_stats"] = dict(store.stats)
        log(f"{result['name']}: kernel speedup {result['kernel_speedup']:.3f}x")
        save()
