"""Small, deterministic consistency rules; never a ground-truth generator."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator

VERIFIER_VERSION = "traffic-consistency-v1"
FIELDS = ("traffic_light", "crossed_stop_line", "event")


class StructuredAnnotation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    subject_id: str = Field(min_length=1)
    timestamp_ms: int = Field(ge=0)
    claims: dict[str, JsonValue] = Field(min_length=1)
    confidence: float | None = Field(default=None, ge=0, le=1)

    @field_validator("subject_id")
    @classmethod
    def _nonblank_subject_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("subject_id must not be blank")
        return value


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    id: str = Field(min_length=1)
    source: Literal["simulator", "detector", "manual"]
    subject_id: str = Field(min_length=1)
    timestamp_ms: int = Field(ge=0)
    values: dict[str, JsonValue]

    @field_validator("id", "subject_id")
    @classmethod
    def _nonblank_identity(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("identity values must not be blank")
        return value


class Reason(BaseModel):
    code: str
    field: str | None = None
    expected: JsonValue = None
    actual: JsonValue = None
    fact_ids: list[str] = Field(default_factory=list)


class Decision(BaseModel):
    status: Literal["accepted", "rejected", "needs_review"]
    reasons: list[Reason]
    fact_ids: list[str]
    verifier_version: str = VERIFIER_VERSION


def _valid(field: str, value: JsonValue) -> bool:
    if field == "crossed_stop_line":
        return type(value) is bool
    if field == "traffic_light":
        return value in ("red", "green", "yellow")
    if field == "event":
        return value in ("red_light_running", "no_red_light_running")
    return False


def verify(annotation: StructuredAnnotation, evidence: list[Evidence]) -> Decision:
    """Compare exact subject/time; unknowns and conflicting evidence require review.

    Acceptance covers only these three claims. It does not establish evidence
    reliability, visibility, legal fault, or temporal event detection. Confidence
    is retained as uncalibrated model metadata and does not change this status.
    """
    scoped = [
        f
        for f in evidence
        if f.subject_id == annotation.subject_id and f.timestamp_ms == annotation.timestamp_ms
    ]
    ids = [f.id for f in scoped]
    reasons: list[Reason] = []
    expected: dict[str, JsonValue] = {}
    supporting_ids: dict[str, list[str]] = {}
    ambiguous: set[str] = set()
    for field in FIELDS:
        entries = [(f.id, f.values[field]) for f in scoped if field in f.values]
        if not entries:
            continue
        if any(not _valid(field, value) for _, value in entries):
            ambiguous.add(field)
            reasons.append(
                Reason(code="invalid_evidence", field=field, fact_ids=[i for i, _ in entries])
            )
        elif any(value != entries[0][1] for _, value in entries):
            ambiguous.add(field)
            reasons.append(
                Reason(code="conflicting_evidence", field=field, fact_ids=[i for i, _ in entries])
            )
        else:
            expected[field] = entries[0][1]
            supporting_ids[field] = [fact_id for fact_id, _ in entries]

    # Only derive a scoped red-light event when both primitives are known.
    light, crossed = expected.get("traffic_light"), expected.get("crossed_stop_line")
    derived = None
    if light == "red" and crossed is True:
        derived = "red_light_running"
    elif light in ("red", "green", "yellow") and crossed is False:
        derived = "no_red_light_running"
    elif light == "green" and crossed is True:
        derived = "no_red_light_running"
    if derived is not None and "event" not in ambiguous:
        primitive_ids = [
            fact.id
            for fact in scoped
            if fact.id in supporting_ids.get("traffic_light", [])
            or fact.id in supporting_ids.get("crossed_stop_line", [])
        ]
        event_ids = supporting_ids.get("event", [])
        derived_ids = list(dict.fromkeys([*event_ids, *primitive_ids]))
        if "event" in expected and expected["event"] != derived:
            ambiguous.add("event")
            expected.pop("event")
            reasons.append(
                Reason(code="inconsistent_event_evidence", field="event", fact_ids=derived_ids)
            )
        else:
            expected["event"] = derived
            supporting_ids["event"] = derived_ids

    for field, actual in annotation.claims.items():
        if field not in FIELDS:
            reasons.append(Reason(code="unsupported_claim", field=field, actual=actual))
        elif not _valid(field, actual):
            reasons.append(Reason(code="invalid_claim", field=field, actual=actual))
        elif field in ambiguous:
            continue
        elif field not in expected:
            reasons.append(Reason(code="missing_evidence", field=field, actual=actual))
        elif actual != expected[field]:
            reasons.append(
                Reason(
                    code="mismatch",
                    field=field,
                    expected=expected[field],
                    actual=actual,
                    fact_ids=supporting_ids.get(field, []),
                )
            )
    for field in FIELDS:
        if field not in annotation.claims:
            reasons.append(Reason(code="missing_claim", field=field))
    if not scoped:
        reasons.append(Reason(code="no_scoped_evidence"))
    if any(r.code == "mismatch" for r in reasons):
        status = "rejected"
    elif reasons:
        status = "needs_review"
    else:
        status = "accepted"
        reasons.append(Reason(code="claims_match_scoped_evidence", fact_ids=ids))
    return Decision(status=status, reasons=reasons, fact_ids=ids)
