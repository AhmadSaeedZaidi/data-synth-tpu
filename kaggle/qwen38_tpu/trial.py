"""Full-batch replay workloads, measured memory gates and honest summaries."""

import json
import math
from collections import defaultdict
from pathlib import Path


def evaluation_sanity(rows):
    """Stop a repeat of v9's first-cohort collapse without shortening thinking.

    This detects a specific severe failure, not ordinary answer mistakes or
    a single difficult reasoning case. Preserve every completed case first.
    """
    asserted = [r for r in rows if r.get("asserted", True)]
    no_final = [r for r in asserted if r["truncated"] and r["answer_phase"] != "FINAL"]
    failed = len(asserted) >= 4 and len(no_final) == len(asserted)
    return dict(
        status="FAIL" if failed else "PASS",
        asserted_cases=len(asserted),
        truncated_without_final=len(no_final),
        reason="Every asserted case in the first cohort exhausted thinking without a final answer"
        if failed
        else "No complete first-cohort thinking collapse detected",
        limitation="Does not establish model accuracy; the full evaluation is still required",
    )


def validate_batches(batches, requests=None):
    if (
        not batches
        or any(type(b) is not int or not 1 <= b <= 64 for b in batches)
        or batches != sorted(set(batches))
    ):
        raise ValueError("Batch sizes must be increasing, distinct integers in [1, 64]")
    if requests is not None and (
        type(requests) is not int
        or not max(batches) <= requests <= 256
        or any(requests % b for b in batches)
    ):
        raise ValueError("Requests must be at most 256 and fill every requested batch")


def replay_workload(prompts, requests, batches):
    requests = len(prompts) if requests is None else requests
    validate_batches(batches, requests)
    if not prompts or requests % len(prompts):
        raise ValueError("Replay workload must contain complete copies of the labeled cohort")
    names = [p["case"]["name"] for p in prompts]
    if len(set(names)) != len(names):
        raise ValueError("Labeled cohort names must be unique")
    return [
        {
            **prompts[i % len(prompts)],
            "request_id": f"request-{i:04d}",
            "replay_index": i // len(prompts),
        }
        for i in range(requests)
    ]


def update_summaries(report):
    results = report["batch_results"]
    benchmark_rows = [row for result in results for row in result["cases"]]
    evaluation_rows = [
        row for result in report.get("evaluation_results", []) for row in result["cases"]
    ]
    rows = [*evaluation_rows, *benchmark_rows]
    asserted_rows = [row for row in rows if row.get("asserted", True)]
    grouped = defaultdict(list)
    for row in asserted_rows:
        grouped[row["name"], row["batch"]].append(row)
    failed = [
        f"{name}@batch{batch}"
        for (name, batch), group in grouped.items()
        if not all(row["passed"] for row in group)
    ]
    report["vision_test"] = [{**row, "asserted": row.get("asserted", True)} for row in rows]
    report["quality"] = dict(
        asserted_cases=len(grouped),
        failed_cases=failed,
        unique_cases=len({row["name"] for row in asserted_rows}),
        unique_cases_passed=sum(
            all(row["passed"] for row in asserted_rows if row["name"] == name)
            for name in {row["name"] for row in asserted_rows}
        ),
        request_checks=len(asserted_rows),
        passed_request_checks=sum(bool(row["passed"]) for row in asserted_rows),
        unasserted_controls=sum(not row.get("asserted", True) for row in rows),
        replay_answer_variations=[
            f"{name}@batch{batch}"
            for (name, batch), group in grouped.items()
            if len({row["answer"] for row in group}) > 1
        ],
        denominator="unique case/batch combinations; replays are separate request checks",
        limitation="Includes schema-only checks. Use evaluation groups for scoped accuracy; "
        "unlabelled event reviews do not establish event accuracy.",
    )
    if evaluation_rows or report.get("evaluation_contract"):
        groups = defaultdict(list)
        for row in evaluation_rows:
            groups[row.get("group", "unknown")].append(row)
        expected = report.get("evaluation_contract", {}).get("expected_cases")
        report["evaluation"] = dict(
            expected_cases=expected,
            completed_cases=len(evaluation_rows),
            coverage="COMPLETE" if len(evaluation_rows) == expected else "PARTIAL",
            groups={
                name: dict(
                    completed=len(group),
                    asserted=sum(row.get("asserted", True) for row in group),
                    passed=sum(
                        row.get("asserted", True) and row["passed"] is True for row in group
                    ),
                    failed=[
                        row["name"]
                        for row in group
                        if row.get("asserted", True) and row["passed"] is not True
                    ],
                    assertion_scope="schema_only"
                    if name == "beamng_pipeline_review"
                    else "format_and_answer",
                    syntax_and_schema_passed=sum(
                        row.get("matcher", {}).get("schema_valid", False) for row in group
                    )
                    if name in {"beamng_pipeline_review", "sft_formatting"}
                    else None,
                    content_passed=sum(
                        row.get("matcher", {}).get("semantic_valid") is True for row in group
                    )
                    if name == "sft_formatting"
                    else None,
                )
                for name, group in groups.items()
            },
            event_accuracy=None,
            event_reference_status="independent_event_labels_missing",
            training_ready=False,
        )
    report["benchmarks"] = []
    for batch in sorted({result["requested_batch"] for result in results}):
        measured = [r for r in results if r["requested_batch"] == batch]
        seconds = sum(r["decode_seconds"] for r in measured)
        prefill = sum(r["prefill_seconds"] for r in measured)
        count = sum(r["forward_tokens"] for r in measured)
        requests = sum(len(r["cases"]) for r in measured)
        slots = sum(r["allocated_row_steps"] for r in measured)
        report["benchmarks"].append(
            dict(
                batch_size=batch,
                actual_batches=sorted({r["batch"] for r in measured}),
                requests=requests,
                unique_cases=len({row["name"] for r in measured for row in r["cases"]}),
                prefill_seconds=prefill,
                decode_seconds=seconds,
                throughput_clock="decode_wall_time"
                if all(r.get("throughput_clock") == "decode_wall_time" for r in measured)
                else "legacy_active_step_time",
                timing_breakdown={
                    key: sum(r[key] for r in measured)
                    for key in (
                        "host_lookup_seconds",
                        "model_step_seconds",
                        "selection_seconds",
                        "decode_overhead_seconds",
                    )
                    if all(key in r for r in measured)
                },
                forward_tokens=count,
                output_tokens=sum(row["generated_tokens"] for r in measured for row in r["cases"]),
                aggregate_tokens_per_second=count / seconds if seconds else 0,
                prepared_requests_per_second=requests / (seconds + prefill)
                if seconds + prefill
                else 0,
                active_row_utilization=count / slots if slots else 0,
                tokens_include_reasoning=True,
                excludes="weight loading, vision preparation and warmup",
            )
        )


class MemoryTracker:
    """Read each chip's allocator counters; decline unknown or tight projections.

    Phase peaks are sampled bytes-in-use. Allocator peak counters are cumulative
    over the process, so neither is presented as an isolated kernel's peak.
    """

    def __init__(self, devices, path, *, min_free_bytes=512 << 20, observer=None):
        self.devices = devices
        self.path = Path(path)
        self.min_free_bytes = min_free_bytes
        self.records = []
        self.observed = []
        self.observer = observer

    def sample(self, stage, batch=None):
        chips = []
        for device in self.devices:
            stats = device.memory_stats() or {}
            chips.append(
                {
                    "device": str(device),
                    **{
                        key: int(stats[key])
                        for key in (
                            "bytes_limit",
                            "bytes_in_use",
                            "peak_bytes_in_use",
                            "bytes_reserved",
                            "peak_bytes_reserved",
                        )
                        if stats.get(key) is not None
                    },
                }
            )
        known = all("bytes_limit" in c and "bytes_in_use" in c for c in chips)
        record = dict(stage=stage, batch=batch, chips=chips, counters_available=known)
        if known:
            record["total_bytes_in_use"] = sum(c["bytes_in_use"] for c in chips)
            record["minimum_free_bytes"] = min(c["bytes_limit"] - c["bytes_in_use"] for c in chips)
        self.records.append(record)
        with self.path.open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        if self.observer:
            self.observer(record)
        return record

    def admit(self, batch, plan):
        before = self.sample("before_batch", batch)
        if not plan["fits"]:
            return dict(admitted=False, reason="logical_memory_plan_exceeds_hbm")
        if not before["counters_available"]:
            return dict(admitted=False, reason="allocator_counters_unavailable")
        # Initial estimate includes 50% beyond logical state for replication and
        # padding. Later shapes also extrapolate the largest measured delta,
        # with 25% margin. The minimum free space is required on EVERY chip.
        projected = []
        for chip, now in enumerate(before["chips"]):
            extra = 1.5 * plan["state_bytes"] / len(self.devices)
            for observation in self.observed:
                extra = max(
                    extra, 1.25 * observation["deltas"][chip] * batch / observation["batch"]
                )
            projected.append(now["bytes_limit"] - now["bytes_in_use"] - math.ceil(extra))
        return dict(
            admitted=min(projected) >= self.min_free_bytes,
            reason="headroom_available"
            if min(projected) >= self.min_free_bytes
            else "insufficient_per_chip_headroom",
            projected_free_bytes_per_chip=projected,
            minimum_required_free_bytes_per_chip=self.min_free_bytes,
        )

    def complete(self, batch, baseline_index):
        baseline = self.records[baseline_index]
        samples = [r for r in self.records[baseline_index + 1 :] if r["batch"] == batch]
        if (
            baseline["counters_available"]
            and samples
            and all(r["counters_available"] for r in samples)
        ):
            deltas = [
                max(
                    0,
                    max(r["chips"][i]["bytes_in_use"] for r in samples)
                    - baseline["chips"][i]["bytes_in_use"],
                )
                for i in range(len(self.devices))
            ]
            self.observed.append(dict(batch=batch, deltas=deltas))

    def summary(self):
        known = [r for r in self.records if r["counters_available"]]
        return dict(
            samples=len(self.records),
            sampled_peak_total_bytes_in_use=max(
                (r["total_bytes_in_use"] for r in known), default=None
            ),
            minimum_sampled_free_bytes=min((r["minimum_free_bytes"] for r in known), default=None),
            allocator_cumulative_peak_bytes_per_chip=[
                max((r["chips"][i].get("peak_bytes_in_use", 0) for r in known), default=0) or None
                for i in range(len(self.devices))
            ],
            limitation=(
                "Phase bytes-in-use peaks are sampled; "
                "allocator peak counters cover the whole process"
            ),
        )
