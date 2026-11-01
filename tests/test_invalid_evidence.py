"""Workflow behavior when legacy/manual database facts violate the fact shape."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from data_synth.db import init_db, make_engine
from data_synth.models import Annotation, Fact, ModelRun, Scenario
from data_synth.workflow import verify_annotation


@pytest.fixture
def session(tmp_path: Path):
    engine = make_engine(f"sqlite:///{tmp_path / 'invalid-evidence.sqlite3'}")
    init_db(engine)
    with Session(engine) as db_session:
        yield db_session


def _annotation(session: Session) -> Annotation:
    scenario = Scenario(source="fixture", scenario_type="invalid-evidence", configuration={})
    session.add(scenario)
    session.flush()
    run = ModelRun(
        scenario_id=scenario.id,
        provider="fixture",
        model="fixture-model",
        endpoint="https://provider.example/v1",
        prompt_version="v1",
        request_hash="fixture-run",
        response={},
        messages=[],
    )
    session.add(run)
    session.flush()
    row = Annotation(
        scenario_id=scenario.id,
        model_run_id=run.id,
        subject_id="ego",
        timestamp_ms=100,
        claims={
            "traffic_light": "red",
            "crossed_stop_line": True,
            "event": "red_light_running",
        },
    )
    session.add(row)
    session.flush()
    return row


def _fact(session: Session, annotation: Annotation, values) -> Fact:
    fact = Fact(
        scenario_id=annotation.scenario_id,
        source="manual",
        source_version="fixture-v1",
        subject_id=annotation.subject_id,
        timestamp_ms=annotation.timestamp_ms,
        values=values,
    )
    session.add(fact)
    session.flush()
    return fact


def test_malformed_fact_values_become_needs_review(session: Session) -> None:
    annotation = _annotation(session)
    fact = _fact(session, annotation, ["not", "an", "object"])

    verification = verify_annotation(session, annotation.id)

    assert verification.status == "needs_review"
    assert verification.fact_ids == [fact.id]
    assert any(
        reason["code"] == "invalid_evidence" and reason["fact_ids"] == [fact.id]
        for reason in verification.reasons
    )


def test_invalid_fact_does_not_hide_a_definite_mismatch(session: Session) -> None:
    annotation = _annotation(session)
    invalid = _fact(session, annotation, None)
    valid = _fact(
        session,
        annotation,
        {"traffic_light": "green", "crossed_stop_line": True},
    )

    verification = verify_annotation(session, annotation.id)

    assert verification.status == "rejected"
    assert invalid.id in verification.fact_ids
    assert valid.id in verification.fact_ids
    assert any(reason["code"] == "invalid_evidence" for reason in verification.reasons)
