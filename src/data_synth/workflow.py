"""Turn a model response into a separately stored annotation and decision."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import Annotation, Fact, ModelRun, Verification
from .verification import Evidence, Reason, StructuredAnnotation, verify


class AnnotationError(ValueError):
    """Invalid model output, with no raw model text exposed in the exception."""


def annotate_run(session: Session, run_id: str) -> Annotation:
    """Parse a stored response as strict JSON; never repairs/fabricates claims.

    Invalid responses stay on the model run for inspection; no annotation or
    verification is invented. Caller owns the transaction.
    """
    run = session.get(ModelRun, run_id)
    if run is None or run.scenario_id is None:
        raise AnnotationError("annotation requires a model run with a scenario")
    try:
        text = run.response["choices"][0]["message"]["content"]
        parsed = StructuredAnnotation.model_validate_json(text)
    except Exception:
        raise AnnotationError("model output is not a valid structured annotation") from None
    row = Annotation(
        scenario_id=run.scenario_id,
        model_run_id=run.id,
        subject_id=parsed.subject_id,
        timestamp_ms=parsed.timestamp_ms,
        claims=parsed.claims,
        confidence=parsed.confidence,
    )
    session.add(row)
    session.flush()
    return row


def verify_annotation(session: Session, annotation_id: str) -> Verification:
    """Load independent evidence from the annotation's scenario and store a decision."""
    row = session.get(Annotation, annotation_id)
    if row is None:
        raise ValueError("annotation does not exist")
    run = session.get(ModelRun, row.model_run_id)
    if run is None or run.scenario_id != row.scenario_id:
        raise ValueError("annotation and model run must belong to the same scenario")
    annotation = StructuredAnnotation(
        subject_id=row.subject_id,
        timestamp_ms=row.timestamp_ms,
        claims=row.claims,
        confidence=row.confidence,
    )
    facts = session.scalars(
        select(Fact)
        .where(
            Fact.scenario_id == row.scenario_id,
            Fact.subject_id == row.subject_id,
            Fact.timestamp_ms == row.timestamp_ms,
        )
        .order_by(Fact.id)
    ).all()
    evidence: list[Evidence] = []
    invalid_fact_ids: list[str] = []
    for fact in facts:
        try:
            evidence.append(
                Evidence(
                    id=fact.id,
                    source=fact.source,
                    subject_id=fact.subject_id,
                    timestamp_ms=fact.timestamp_ms,
                    values=fact.values,
                )
            )
        except Exception:
            # Keep malformed stored evidence out of rule evaluation while
            # preserving its identity in the persisted review reason.
            invalid_fact_ids.append(fact.id)

    result = verify(annotation, evidence)
    if invalid_fact_ids:
        fact_ids = list(dict.fromkeys([*result.fact_ids, *invalid_fact_ids]))
        reasons = [
            *result.reasons,
            Reason(code="invalid_evidence", fact_ids=invalid_fact_ids),
        ]
        result = result.model_copy(
            update={
                "status": result.status if result.status == "rejected" else "needs_review",
                "fact_ids": fact_ids,
                "reasons": reasons,
            }
        )
    verification = Verification(
        annotation_id=row.id,
        fact_ids=result.fact_ids,
        status=result.status,
        reasons=[r.model_dump(mode="json") for r in result.reasons],
        verifier_version=result.verifier_version,
    )
    session.add(verification)
    session.flush()
    return verification
