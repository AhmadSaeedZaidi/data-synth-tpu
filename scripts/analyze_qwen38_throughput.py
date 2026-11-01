#!/usr/bin/env python3
"""Reproduce conditional decode rooflines from checkpoint and run evidence.

These are bandwidth ceilings, not benchmark predictions. Routing is modeled as
independent, uniformly selected distinct top-k experts per active row. No TPU
performance is inferred from a CPU execution of this script.
"""

import argparse
import json
import math
import statistics
from pathlib import Path


def dense_matrices(tensors):
    groups = {}
    for name, tensor in tensors.items():
        if (
            name.startswith(("v.", "mm."))
            or name in {"token_embd.weight", "per_layer_token_embd.weight"}
            or "_exps." in name
            or len(tensor["dims"]) != 2
            or "conv" in name
        ):
            continue
        if "hc_" in name or name.startswith("output_hc"):
            group = "hyperconnections"
        elif "shexp" in name:
            group = "shared_experts"
        elif "indexer." in name:
            group = "indexer"
        elif "gate_inp" in name:
            group = "router"
        elif name == "output.weight":
            group = "lm_head"
        elif ".ple_" in name:
            group = "ple"
        else:
            group = "attention_projections"
        groups[group] = groups.get(group, 0) + math.prod(tensor["dims"])
    return groups


def analyze(audit, v6, tracking, calibration):
    tensors, metadata = audit["tensors"], audit["metadata"]
    prefix = "qwen4exp."
    layers = metadata[prefix + "block_count"]
    experts = metadata[prefix + "expert_count"]
    top_k = metadata[prefix + "expert_used_count"]
    expert_tensors = {n: t for n, t in tensors.items() if "_exps." in n}
    expert_params = sum(math.prod(t["dims"]) for t in expert_tensors.values())
    packed_bytes = sum(t["nbytes"] for t in expert_tensors.values())
    q6_params = sum(math.prod(t["dims"]) for t in expert_tensors.values() if t["type"] == "Q6_K")
    q8_params = sum(math.prod(t["dims"]) for t in expert_tensors.values() if t["type"] == "Q8_0")
    dense = dense_matrices(tensors)
    dense_bytes = v6["hbm_plan"]["language_dense_bytes"]
    qsa_layers = layers // metadata[prefix + "full_attention_interval"]
    gdn_layers = layers - qsa_layers
    state_dim = metadata[prefix + "ssm.state_size"]
    value_heads = metadata[prefix + "ssm.time_step_rank"]
    kv_heads = metadata[prefix + "attention.head_count_kv"]
    query_heads = metadata[prefix + "attention.head_count"]
    head_dim = metadata[prefix + "attention.key_length"]
    qsa_budget = metadata[prefix + "attention.indexer.top_k"]
    # Current sparse attention additionally reserves three tail positions.
    selected_tokens = qsa_budget + 3
    recurrent_rw_per_row = 2 * gdn_layers * value_heads * state_dim**2 * 4
    kv_read_per_row = qsa_layers * 2 * selected_tokens * kv_heads * head_dim * (1 + 2 / 32)
    active_expert_flops = 2 * expert_params * top_k / experts
    dense_flops = 2 * sum(dense.values())
    # Two state-vector contractions (four operations/cell), plus state decay
    # and rank-one update (three operations/cell); exclude small activations.
    recurrent_flops = 7 * gdn_layers * value_heads * state_dim**2
    qsa_flops = 4 * qsa_layers * query_heads * selected_tokens * head_dim
    approximate_flops = active_expert_flops + dense_flops + recurrent_flops + qsa_flops
    bandwidths = dict(jax_hardware_reference=8 * 820e9, cloud_v5e=8 * 800 * 2**30)
    roofs = []
    for batch in (1, 8, 16, 32):
        unique = experts * (1 - (1 - top_k / experts) ** batch)
        packed_read = packed_bytes * unique / experts
        state_read_write = batch * (recurrent_rw_per_row + kv_read_per_row)
        minimum = dense_bytes + packed_read + state_read_write
        # Conditional scenarios: each expanded value is written and then read
        # in HBM (two bytes each). XLA may instead keep it in VMEM/fuse it.
        q6_materialized = minimum + 4 * q6_params * unique / experts
        all_materialized = minimum + 4 * expert_params * unique / experts
        traffic = dict(
            ideal_fused_packed=minimum,
            q6_expanded_hbm_q8_native=q6_materialized,
            all_experts_expanded_hbm=all_materialized,
        )
        roofs.append(
            dict(
                active_rows=batch,
                expected_unique_experts_per_layer=unique,
                routes_per_selected_expert=batch * top_k / unique,
                packed_expert_read_bytes=packed_read,
                state_minimum_read_write_bytes=state_read_write,
                scenarios={
                    name: dict(
                        logical_bytes_per_step=amount,
                        optimistic_step_ms={n: amount / bw * 1000 for n, bw in bandwidths.items()},
                        aggregate_tokens_per_second={
                            n: batch * bw / amount for n, bw in bandwidths.items()
                        },
                    )
                    for name, amount in traffic.items()
                },
            )
        )
    old = []
    for benchmark in v6["benchmarks"]:
        batch = benchmark["batch_size"]
        chunks = [r for r in v6["batch_results"] if r["requested_batch"] == batch]
        steps = sum(r["allocated_row_steps"] / r["batch"] for r in chunks)
        old.append(
            dict(
                batch=batch,
                steps=steps,
                mean_wall_step_ms=benchmark["decode_seconds"] / steps * 1000,
                mean_model_step_ms=benchmark["timing_breakdown"]["model_step_seconds"]
                / steps
                * 1000,
                decode_tokens_per_second=benchmark["aggregate_tokens_per_second"],
                active_row_utilization=benchmark["active_row_utilization"],
                constant_step_full_occupancy_tokens_per_second=batch
                * steps
                / benchmark["decode_seconds"],
            )
        )
    current = {}
    for series in ("first", "latest"):
        calls = [
            c["output"]
            for c in tracking[series]
            if "/op/qwen38.decode_step:" in c["op_name"] and c.get("ended_at")
        ]
        median = statistics.median(c["model_step_seconds"] for c in calls)
        warm = [c for c in calls if c["model_step_seconds"] < 2 * median]
        current[series] = dict(
            samples=len(calls),
            warm_samples=len(warm),
            excluded_model_seconds=[c["model_step_seconds"] for c in calls if c not in warm],
            median_model_step_ms=median * 1000,
            warm_tokens_per_second=sum(c["generated_tokens"] for c in warm)
            / sum(c["seconds"] for c in warm),
            mean_generated_rows=sum(c["generated_tokens"] for c in warm) / len(warm),
            host_lookup_fraction=sum(c["host_lookup_seconds"] for c in warm)
            / sum(c["seconds"] for c in warm),
        )
    lookup = {
        (r["candidate"], r["quant"], r["rows"], r["routing"]): r
        for r in calibration["measurements"]
    }
    selected = calibration["selected"]["name"]

    def stack(name, batch, route):
        return (
            47 * lookup[name, "mixed_q6_q8", batch, route]["seconds"]
            + lookup[name, "q8", batch, route]["seconds"]
        )

    residual = current["first"]["median_model_step_ms"] / 1000 - stack(selected, 8, "spread")
    layouts = []
    for candidate in calibration["candidates"]:
        name = candidate["name"]
        for batch in (1, 8, 16, 32):
            for routing in ("spread", "hot"):
                mixed = lookup[name, "mixed_q6_q8", batch, routing]
                eight = lookup[name, "q8", batch, routing]
                layouts.append(
                    dict(
                        candidate=name,
                        rows=batch,
                        active_rows=max(1, batch - 1),
                        routing=routing,
                        mixed_layer_ms=mixed["seconds"] * 1000,
                        all_q8_layer_ms=eight["seconds"] * 1000,
                        all_q8_to_mixed_latency_ratio=eight["seconds"] / mixed["seconds"],
                        mixed_temporary_bytes=mixed["temporary_bytes"],
                        stacked_expert_ms=stack(name, batch, routing) * 1000,
                        conditional_b8_model_ms=(stack(name, batch, routing) + residual) * 1000
                        if batch == 8
                        else None,
                        conditional_full_b8_tokens_per_second=8
                        / (stack(name, batch, routing) + residual)
                        if batch == 8
                        else None,
                    )
                )
    return dict(
        limitations=[
            "Uniform independent routes are a model, not measured routing histograms.",
            "Bandwidth ceilings assume perfect balancing and no duplicate reads or stalls.",
            "Critical-chip imbalance and sequential layers reduce attainable bandwidth.",
            "State traffic is a logical minimum; cache writes, selection, "
            "and communication are additional.",
            "HBM expansion scenarios are conditional; temporary allocation "
            "is not traffic evidence.",
            "Compute ceiling ignores FP32 emulation cost, padding, nonlinearities and dispatch.",
            "Synthetic layer stacks are not full-model benchmarks or isolated Q6 attribution.",
            "The first decode call is cold; warm sampling excludes calls above twice the median.",
        ],
        checked_at_utc=tracking["checked_at_utc"],
        hardware=dict(
            chips=8,
            bf16_peak_flops=8 * 197e12,
            int8_peak_ops=8 * 393e12,
            aggregate_bandwidth_bytes_per_second=bandwidths,
        ),
        model=dict(
            expert_parameters=expert_params,
            expert_packed_bytes=packed_bytes,
            q6_expert_parameters=q6_params,
            q8_expert_parameters=q8_params,
            packed_bits_per_expert_weight=8 * packed_bytes / expert_params,
            all_q8_experts_bytes=expert_params * 34 / 32,
            all_q8_additional_bytes=expert_params * 34 / 32 - packed_bytes,
            all_bf16_experts_bytes=2 * expert_params,
            dense_matrix_parameters_by_group=dense,
            active_expert_flops_per_token=active_expert_flops,
            dense_matrix_flops_per_token=dense_flops,
            recurrent_approximate_flops_per_token=recurrent_flops,
            sparse_attention_approximate_flops_per_token=qsa_flops,
            approximate_flops_per_token=approximate_flops,
            unattainable_peak_compute_tokens_per_second=8 * 197e12 / approximate_flops,
            ideal_compute_only_q8_int8_speedup=1
            / (1 - (2 * q8_params * top_k / experts) / approximate_flops / 2),
        ),
        rooflines=roofs,
        v6=old,
        current_decode_samples=current,
        calibration_selected=calibration["selected"],
        calibration_candidates=calibration["candidates"],
        b8_inferred_nonexpert_seconds=residual,
        layouts=layouts,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("audit", "v6-report", "tracking", "calibration", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = analyze(
        *[
            json.loads(path.read_text())
            for path in (args.audit, args.v6_report, args.tracking, args.calibration)
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                output=str(args.output),
                approximate_gflops_per_token=result["model"]["approximate_flops_per_token"] / 1e9,
                v6=result["v6"],
                current=result["current_decode_samples"],
                roofs=[
                    dict(
                        rows=r["active_rows"],
                        unique=round(r["expected_unique_experts_per_layer"], 2),
                        ceiling_tps={
                            n: round(s["aggregate_tokens_per_second"]["cloud_v5e"])
                            for n, s in r["scenarios"].items()
                        },
                    )
                    for r in result["rooflines"]
                ],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
