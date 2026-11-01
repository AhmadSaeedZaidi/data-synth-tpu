import pytest
from pydantic import ValidationError

from data_synth.verification import Evidence, StructuredAnnotation, verify


def annotation(**claims):
    return StructuredAnnotation(
        subject_id="car-1",
        timestamp_ms=100,
        claims={
            "traffic_light": "red",
            "crossed_stop_line": True,
            "event": "red_light_running",
            **claims,
        },
    )


def facts(**values):
    return Evidence(
        id="fact-1",
        source="manual",
        subject_id="car-1",
        timestamp_ms=100,
        values={"traffic_light": "red", "crossed_stop_line": True, **values},
    )


@pytest.mark.parametrize(
    ("light", "crossed", "event"),
    [
        ("red", True, "red_light_running"),
        ("red", False, "no_red_light_running"),
        ("green", True, "no_red_light_running"),
    ],
)
def test_positive_and_negative_examples(light, crossed, event):
    result = verify(
        annotation(traffic_light=light, crossed_stop_line=crossed, event=event),
        [facts(traffic_light=light, crossed_stop_line=crossed)],
    )
    assert result.status == "accepted"


def test_incompatible_claim_rejected_with_reason():
    result = verify(annotation(traffic_light="green"), [facts()])
    assert result.status == "rejected"
    mismatch = next(r for r in result.reasons if r.code == "mismatch")
    assert (mismatch.expected, mismatch.actual, mismatch.fact_ids) == ("red", "green", ["fact-1"])


def test_mismatch_references_only_facts_supporting_that_field():
    extra = Evidence(
        id="fact-2",
        source="detector",
        subject_id="car-1",
        timestamp_ms=100,
        values={"crossed_stop_line": True},
    )
    result = verify(annotation(traffic_light="green"), [facts(), extra])
    mismatch = next(reason for reason in result.reasons if reason.code == "mismatch")
    assert mismatch.field == "traffic_light"
    assert mismatch.fact_ids == ["fact-1"]


@pytest.mark.parametrize("change", [{"timestamp_ms": 101}, {"subject_id": "car-2"}])
def test_scope_never_silently_matches(change):
    result = verify(annotation(), [facts().model_copy(update=change)])
    assert result.status == "needs_review"
    assert result.fact_ids == []


def test_conflicting_evidence_and_unsupported_claim_require_review():
    second = facts(traffic_light="green").model_copy(update={"id": "fact-2"})
    assert verify(annotation(), [facts(), second]).status == "needs_review"
    assert verify(annotation(speed_mps=10), [facts()]).status == "needs_review"


@pytest.mark.parametrize("bad", [1, "true", None])
def test_boolean_claim_not_coerced(bad):
    assert verify(annotation(crossed_stop_line=bad), [facts()]).status == "needs_review"


def test_inconsistent_event_evidence_requires_review():
    result = verify(annotation(), [facts(event="no_red_light_running")])
    assert result.status == "needs_review"
    assert any(r.code == "inconsistent_event_evidence" for r in result.reasons)


def test_no_evidence_missing_claims_unknown_and_invalid_fact():
    assert verify(annotation(), []).status == "needs_review"
    partial = annotation().model_copy(update={"claims": {"event": "red_light_running"}})
    assert verify(partial, [facts()]).status == "needs_review"
    assert verify(annotation(), [facts(traffic_light="unknown")]).status == "needs_review"
    assert verify(annotation(), [facts(crossed_stop_line=1)]).status == "needs_review"


def test_schema_rejects_extra_keys_bad_confidence_and_time():
    for change in ({"confidence": 1.1}, {"timestamp_ms": -1}, {"ground_truth": True}):
        with pytest.raises(ValidationError):
            StructuredAnnotation.model_validate({**annotation().model_dump(), **change})


@pytest.mark.parametrize(
    ("model", "field", "value"),
    [
        (StructuredAnnotation, "subject_id", " "),
        (Evidence, "id", "\t"),
        (Evidence, "subject_id", "  "),
    ],
)
def test_schema_rejects_blank_identity_values(model, field, value):
    if model is StructuredAnnotation:
        payload = annotation().model_dump()
    else:
        payload = facts().model_dump()
    payload[field] = value
    with pytest.raises(ValidationError):
        model.model_validate(payload)


def test_confidence_is_retained_metadata_and_does_not_change_rule_status():
    low_confidence = annotation().model_copy(update={"confidence": 0.0})
    assert verify(low_confidence, [facts()]).status == "accepted"
