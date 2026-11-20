"""Bounded teacher-forced evidence for batch-sensitive OCR, outside speed tests."""

import time

import numpy as np
from glm53_offload import tracing
from glm53_offload.diagnostics import atomic_json


class DiagnosticBudgetExpired(Exception):
    pass


def diagnose(
    engine,
    prompts,
    hasher,
    ngrams,
    tokens,
    prefill,
    emit,
    destination,
    *,
    seconds=300,
    steps=64,
    clock=time.monotonic,
):
    """Compare identical target histories at B1, four copies and mixed B4.

    Other mixed-batch rows decode normally. Only the target is teacher-forced;
    these logits and forced tokens are diagnostic evidence, never quality
    scores, generated answers or throughput samples. Nonfinite logits remain
    a model correctness failure and propagate to the trial's hard gate.
    """
    target = next((p for p in prompts if p["case"]["name"] == "read_text_stop_7"), None)
    report = dict(
        status="RUNNING",
        scope="teacher_forced_batch_logit_comparison",
        included_in_quality=False,
        timed_benchmark=False,
        max_steps=steps,
        cases=[],
    )
    atomic_json(destination, report)
    if target is None or len(prompts) < 4:
        report.update(status="SKIPPED", reason="target_or_cohort_unavailable")
        atomic_json(destination, report)
        return report
    deadline = clock() + seconds

    def progress(**fields):
        if clock() >= deadline:
            raise DiagnosticBudgetExpired()
        emit(**fields)

    def replay(chunk, row, reference=None):
        engine.release_state()
        progress(stage="batch_divergence", detail=f"prefill batch={len(chunk)} row={row}")
        logits, history, _ = prefill(engine, chunk, hasher, ngrams, tokens, progress)
        active = np.ones(len(chunk), bool)
        records, scores, chosen = [], [], []
        for step in range(steps if reference is None else len(reference[0])):
            values = np.asarray(logits, dtype=np.float32)
            if values.ndim != 2 or values.shape[0] != len(chunk):
                raise ValueError("Diagnostic logits have the wrong batch shape")
            if not np.isfinite(values[active]).all():
                raise FloatingPointError("Nonfinite logits in batch divergence diagnostic")
            ids = np.argmax(values, axis=-1).astype(np.int64)
            vector = values[row].copy()
            selected = int(ids[row])
            top = np.argpartition(vector, -min(5, len(vector)))[-5:]
            top = top[np.argsort(vector[top])[::-1]]
            record = dict(
                step=step,
                argmax=selected,
                active_rows=int(active.sum()),
                top_ids=top.tolist(),
                top_logits=vector[top].tolist(),
            )
            if reference is not None:
                ids[row] = reference[0][step]
                base = reference[1][step]
                difference = vector.astype(np.float64) - base.astype(np.float64)
                record.update(
                    reference_token=int(ids[row]),
                    argmax_matches_reference=selected == int(ids[row]),
                    reference_token_rank=int(np.count_nonzero(vector > vector[ids[row]])) + 1,
                    max_logit_absolute_difference=float(np.max(np.abs(difference))),
                    relative_l2_difference=float(
                        np.linalg.norm(difference) / max(np.linalg.norm(base), 1e-30)
                    ),
                )
            records.append(record)
            scores.append(vector)
            chosen.append(int(ids[row]))
            if int(ids[row]) in engine.config.generation_eos:
                break
            active &= ~np.isin(ids, engine.config.generation_eos)
            progress(stage="batch_divergence", detail=f"batch={len(chunk)} step={step}")
            hashes, history = hasher.step(ids, history, active)
            positions = np.asarray([[p["next_position"] + step] * 3 for p in chunk])
            engine.forward(
                tokens.gather(ids[:, None], active=active),
                ngrams.gather(hashes, active=active),
                positions,
                active,
            )
            logits = engine.logits()
        return chosen, scores, records

    try:
        with tracing.span("batch_divergence_diagnostic") as span:
            reference = replay([target], 0)
            report["reference"] = dict(tokens=reference[0], steps=reference[2])
            atomic_json(destination, report)
            for name, chunk, row in (
                ("identical_b4", [target] * 4, 0),
                ("mixed_b4", [p for p in prompts if p is not target][:3] + [target], 3),
            ):
                _, _, records = replay(chunk, row, reference)
                mismatch = next(
                    (r["step"] for r in records if not r["argmax_matches_reference"]), None
                )
                report["cases"].append(
                    dict(name=name, target_row=row, steps=records, first_argmax_difference=mismatch)
                )
                atomic_json(destination, report)
            report["status"] = "MEASURED"
            span.update(status=report["status"], cases=len(report["cases"]))
    except DiagnosticBudgetExpired:
        report["status"] = "SKIPPED_BUDGET" if not report["cases"] else "PARTIAL_BUDGET"
    except Exception as error:
        report.update(status="ERROR", error_type=type(error).__name__)
        raise
    finally:
        engine.release_state()
        atomic_json(destination, report)
    return report
