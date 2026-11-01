"""Bounded, warmed EP/TP comparison before the full checkpoint is admitted."""

import gc
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from .compute import pack_q8
from .experts import expert_parallel
from .parallel import expert_mesh, packed_spec, split_expert_bytes

CANDIDATES = (
    dict(name="serial_ep8", group_size=1, tensor_parallel=1),
    dict(name="grouped_ep8", group_size=4, tensor_parallel=1),
    dict(name="grouped_ep4_tp2", group_size=4, tensor_parallel=2),
    dict(name="sliced_ep4_tp2", group_size=1, tensor_parallel=2),
    dict(name="paired_ep4_tp2", group_size=2, tensor_parallel=2),
)


def _summarize(measurements, minimum_speedup, maximum_regression, temporary_limit_bytes):
    baseline = [r for r in measurements if r["candidate"] == CANDIDATES[0]["name"]]

    def key(r):
        return r["quant"], r["rows"], r["routing"]

    def valid_time(r):
        return np.isfinite(r["seconds"]) and r["seconds"] > 0

    if (
        not baseline
        or len({key(r) for r in baseline}) != len(baseline)
        or any(not r["accuracy_pass"] or not valid_time(r) for r in baseline)
    ):
        raise ValueError("A complete correct serial baseline with valid timings is required")
    reference = {key(r): r for r in baseline}
    summaries = []
    for candidate in CANDIDATES:
        rows = [r for r in measurements if r["candidate"] == candidate["name"]]
        complete = len(rows) == len(reference) and {key(r) for r in rows} == set(reference)
        accurate = complete and all(r["accuracy_pass"] and valid_time(r) for r in rows)
        memory_pass = complete and all(
            r.get("temporary_bytes") is not None
            and 0 <= r["temporary_bytes"] <= temporary_limit_bytes
            for r in rows
        )
        ratios = [reference[key(r)]["seconds"] / r["seconds"] for r in rows] if accurate else []
        # The pinned Q6 model has 47 mixed Q6/Q8 layers and one all-Q8 layer.
        weights = [47 if r["quant"] == "mixed_q6_q8" else 1 for r in rows]
        speedup = float(np.exp(np.average(np.log(ratios), weights=weights))) if ratios else 0.0
        worst = float(max(1 / ratio for ratio in ratios)) if ratios else None
        accepted = (
            accurate and memory_pass and speedup >= minimum_speedup and worst <= maximum_regression
        )
        summaries.append(
            dict(
                **candidate,
                complete=complete,
                accuracy_pass=accurate,
                memory_pass=memory_pass,
                weighted_speedup=speedup,
                worst_latency_ratio=worst,
                accepted=accepted,
            )
        )
    return summaries


def _choose(summaries):
    eligible = [r for r in summaries if r["accepted"]]
    if eligible:
        return max(eligible, key=lambda r: r["weighted_speedup"])
    baseline = next((r for r in summaries if r["name"] == "serial_ep8"), None)
    if baseline and baseline["accuracy_pass"] and baseline["memory_pass"]:
        return baseline
    raise ValueError("No complete, accurate expert policy fits the temporary memory limit")


def select_candidate(
    measurements, *, minimum_speedup=1.10, maximum_regression=1.15, temporary_limit_bytes=256 << 20
):
    """Require correct coverage and memory safety, including the fallback."""
    summaries = _summarize(measurements, minimum_speedup, maximum_regression, temporary_limit_bytes)
    selected = _choose(summaries)
    return dict(
        selected={k: selected[k] for k in CANDIDATES[0]},
        candidates=summaries,
        minimum_speedup=minimum_speedup,
        maximum_regression=maximum_regression,
        temporary_limit_bytes=temporary_limit_bytes,
    )


def select_phase_candidates(
    measurements,
    decode_rows,
    *,
    minimum_speedup=1.10,
    maximum_regression=1.15,
    temporary_limit_bytes=256 << 20,
):
    """Select phase kernels with ONE physical EP/TP placement.

    EP8 may use grouped decode and serial prefill on the same packed arrays.
    EP4/TP2 must pass both phases; switching TP would require redistributing
    all expert weights. The score is an equal-phase geometric mean of the
    calibrated layer-weighted speedups, not a full-model throughput estimate.
    """
    decode_rows = set(decode_rows)
    if not decode_rows or not decode_rows <= {r["rows"] for r in measurements}:
        raise ValueError("Decode calibration coverage is missing")
    phases = {
        phase: _summarize(
            [r for r in measurements if (r["rows"] in decode_rows) == (phase == "decode")],
            minimum_speedup,
            maximum_regression,
            temporary_limit_bytes,
        )
        for phase in ("decode", "prefill")
    }
    for summaries in phases.values():
        for row in summaries:
            row["eligible_for_phase"] = bool(
                row["accuracy_pass"]
                and row["memory_pass"]
                and row["worst_latency_ratio"] <= maximum_regression
            )
    layouts = []
    for tp in sorted({c["tensor_parallel"] for c in CANDIDATES}):
        candidates = {
            phase: [c for c in summaries if c["tensor_parallel"] == tp and c["eligible_for_phase"]]
            for phase, summaries in phases.items()
        }
        if any(not rows for rows in candidates.values()):
            continue
        choices = {
            phase: max(rows, key=lambda r: r["weighted_speedup"])
            for phase, rows in candidates.items()
        }
        layouts.append(
            dict(
                tensor_parallel=tp,
                phases={phase: row["name"] for phase, row in choices.items()},
                group_size=choices["decode"]["group_size"],
                prefill_group_size=choices["prefill"]["group_size"],
                weighted_speedup=float(
                    np.sqrt(np.prod([r["weighted_speedup"] for r in choices.values()]))
                ),
            )
        )
    # Require the gain over the combined workload, not separately in both
    # phases. A neutral TP2 prefill must not veto a substantial decode gain.
    for layout in layouts:
        layout["accepted"] = layout["weighted_speedup"] >= minimum_speedup
    eligible = [r for r in layouts if r["accepted"]]
    if eligible:
        best = max(eligible, key=lambda r: r["weighted_speedup"])
    elif all(
        next(r for r in rows if r["name"] == "serial_ep8")["eligible_for_phase"]
        for rows in phases.values()
    ):
        best = dict(
            tensor_parallel=1,
            group_size=1,
            prefill_group_size=1,
            phases=dict(decode="serial_ep8", prefill="serial_ep8"),
        )
    else:
        raise ValueError("No safe expert placement covers both decode and prefill")
    return dict(
        selected=dict(
            name=f"decode:{best['phases']['decode']}/prefill:{best['phases']['prefill']}",
            group_size=best["group_size"],
            prefill_group_size=best["prefill_group_size"],
            tensor_parallel=best["tensor_parallel"],
        ),
        phases=phases,
        layouts=layouts,
        decode_rows=sorted(decode_rows),
        minimum_speedup=minimum_speedup,
        maximum_regression=maximum_regression,
        temporary_limit_bytes=temporary_limit_bytes,
        score="equal-phase geometric mean; fixed packed-weight placement",
    )


def _prototype(rng, rows, width, qtype):
    if qtype == "Q8_0":
        return pack_q8(rng.normal(0, 0.025, (rows, width)).astype(np.float32))[None]
    raw = rng.integers(0, 256, (1, rows, width // 256, 210), dtype=np.uint8)
    raw[..., 192:208] = rng.integers(1, 4, raw[..., 192:208].shape, dtype=np.uint8)
    raw[..., 208:210] = np.array([1 / 4096], "<f2").view(np.uint8)
    return raw


def _place(prototypes, types, mesh, c):
    """One layer of packed HBM, at most one chip's matrix staged on the host."""
    ep, tp = mesh.shape["expert"], mesh.shape.get("tensor", 1)
    count = c.experts // ep
    result = {}
    for name in ("gate", "up", "down"):
        pieces = split_expert_bytes(
            prototypes[name], name, types[name], c.hidden, c.intermediate, tp
        )
        arrays = []
        for rank, device in enumerate(mesh.devices.flat):
            # Repeated weight values, with real expert count/shape/routing.
            # Dynamic array arguments prevent constant-folding the computation.
            host = np.broadcast_to(pieces[rank % tp], (count, pieces.shape[-1]))
            array = jax.device_put(host, device)
            array.block_until_ready()
            arrays.append(array)
        result[name] = jax.make_array_from_single_device_arrays(
            (c.experts * tp, pieces.shape[-1]), NamedSharding(mesh, packed_spec(mesh)), arrays
        )
    return result


def calibrate_experts(
    mesh,
    c,
    *,
    compute="bf16",
    rows=(1, 8, 16, 32, 128, 256, 512),
    decode_rows=None,
    repeats=3,
    budget_seconds=300,
    emit=lambda **kw: None,
):
    """Measure synchronized full FFNs, not just isolated dot products.

    Uses full H/I/E/top-k, balanced and skewed routes, both real quant layouts,
    and BF16 inputs. Timing excludes compilation. This is a synthetic selector;
    full-model throughput and correctness gates are separately required.
    """
    if mesh.size != 8 or any(d.platform != "tpu" for d in mesh.devices.flat):
        raise ValueError("Performance selection must run on the eight target TPUs")
    if repeats < 3 or budget_seconds <= 0 or not rows or any(r < 1 for r in rows):
        raise ValueError("Invalid calibration bounds")
    rng = np.random.default_rng(3826)
    started = time.monotonic()
    measurements, reference = [], {}
    source = {}
    replica = NamedSharding(mesh, P())
    for batch in sorted(set(rows)):
        x = rng.normal(0, 0.3, (batch, c.hidden)).astype(np.dtype(jnp.bfloat16))
        for routing in ("spread", "hot"):
            ids = ((np.arange(batch)[:, None] * c.top_k + np.arange(c.top_k)) * 37 + 11) % c.experts
            if routing == "hot":
                ids[:] = np.arange(c.top_k)
            weights = np.full(ids.shape, 1 / c.top_k, np.dtype(jnp.bfloat16))
            # Keep B1 nonzero (the old dot calibrator accidentally zeroed it).
            if batch > 1:
                weights[-1] = 0
            source[batch, routing] = tuple(
                jax.device_put(a, replica) for a in (x, ids.astype(np.int32), weights)
            )
    stopped = False
    for quant, gate_type in (("mixed_q6_q8", "Q6_K"), ("q8", "Q8_0")):
        types = dict(gate=gate_type, up=gate_type, down="Q8_0")
        prototypes = {
            name: _prototype(
                rng,
                c.hidden if name == "down" else c.intermediate,
                c.intermediate if name == "down" else c.hidden,
                types[name],
            )
            for name in types
        }
        for candidate in CANDIDATES:
            is_reference = candidate["name"] == "serial_ep8"
            # Always finish both serial baselines; partial candidates cannot win.
            if not is_reference and time.monotonic() - started >= budget_seconds:
                stopped = True
                continue
            emit(stage="expert_calibration", detail=f"{quant} {candidate['name']} allocation")
            moe_mesh = expert_mesh(mesh, candidate["tensor_parallel"])
            packed = _place(prototypes, types, moe_mesh, c)
            fn = expert_parallel(
                mesh,
                gate_type=types["gate"],
                up_type=types["up"],
                down_type=types["down"],
                hidden=c.hidden,
                intermediate=c.intermediate,
                compute=compute,
                group_size=candidate["group_size"],
                tensor_parallel=candidate["tensor_parallel"],
            )
            try:
                for (batch, routing), inputs in source.items():
                    if not is_reference and time.monotonic() - started >= budget_seconds:
                        stopped = True
                        break
                    emit(
                        stage="expert_calibration",
                        detail=f"{quant} {candidate['name']} rows={batch} {routing}",
                    )
                    args = (packed["gate"], packed["up"], packed["down"], *inputs)
                    compile_started = time.monotonic()
                    compiled = fn.lower(*args).compile()
                    compile_seconds = time.monotonic() - compile_started
                    actual = np.asarray(jax.block_until_ready(compiled(*args))).astype(np.float32)
                    key = quant, batch, routing
                    if is_reference:
                        reference[key] = actual
                    expected = reference[key]
                    relative = float(
                        np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-12)
                    )
                    accuracy = bool(
                        np.isfinite(actual).all()
                        and relative <= 0.012
                        and np.allclose(actual, expected, atol=0.003, rtol=0.015)
                    )
                    values = []
                    for _ in range(repeats):
                        clock = time.perf_counter()
                        jax.block_until_ready(compiled(*args))
                        values.append(time.perf_counter() - clock)
                    stats = compiled.memory_analysis()
                    measurements.append(
                        dict(
                            candidate=candidate["name"],
                            reference_candidate="serial_ep8",
                            quant=quant,
                            rows=batch,
                            routing=routing,
                            seconds=float(np.median(values)),
                            samples_seconds=values,
                            compile_seconds=compile_seconds,
                            relative_l2_error=relative,
                            max_abs_error=float(np.max(np.abs(actual - expected))),
                            accuracy_pass=accuracy,
                            temporary_bytes=stats.temp_size_in_bytes if stats else None,
                        )
                    )
                    del compiled, args
            finally:
                jax.block_until_ready(packed)
                jax.tree.map(lambda a: a.delete(), packed)
                del packed
                gc.collect()
    expected = {(q, b, r) for q in ("mixed_q6_q8", "q8") for b, r in source}
    baseline = [r for r in measurements if r["candidate"] == "serial_ep8"]
    if (
        len(baseline) != len(expected)
        or {(r["quant"], r["rows"], r["routing"]) for r in baseline} != expected
    ):
        raise ValueError("Serial calibration did not cover the requested workload")
    try:
        decision = (
            select_candidate(measurements)
            if decode_rows is None
            else select_phase_candidates(measurements, decode_rows)
        )
        decision["status"] = "PASS"
    except ValueError as error:
        # Retain the measurements in both Weave and report.json when memory
        # admission rejects every policy; do not lose the failure evidence.
        decision = dict(status="FAIL", selected=None, selection_error=str(error))
    return dict(
        **decision,
        measurements=measurements,
        elapsed_seconds=time.monotonic() - started,
        budget_exhausted=stopped,
        budget_scope="checked between compiled calls; one compile cannot be interrupted",
        weight_values="one synthetic expert per projection repeated across real expert count",
        limitation="Synthetic selector; no full-model speedup is established",
    )
