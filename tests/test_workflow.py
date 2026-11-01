"""Workflow verification and local snapshot export tests."""

from __future__ import annotations

import hashlib
import json
import socket
from copy import deepcopy
from pathlib import Path

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from data_synth.db import init_db, make_engine
from data_synth.export import export_snapshot
from data_synth.models import Annotation, Asset, Fact, ModelRun, Scenario, Verification
from data_synth.workflow import AnnotationError, annotate_run, verify_annotation


@pytest.fixture
def session(tmp_path: Path):
    engine = make_engine(f"sqlite:///{tmp_path / 'workflow.sqlite3'}")
    init_db(engine)
    with Session(engine) as db_session:
        yield db_session


def _scenario(session: Session, scenario_type: str = "workflow") -> Scenario:
    scenario = Scenario(source="fixture", scenario_type=scenario_type, configuration={})
    session.add(scenario)
    session.flush()
    return scenario


def _run(
    session: Session,
    scenario: Scenario,
    content: str,
    request_hash: str = "run-hash",
) -> ModelRun:
    run = ModelRun(
        scenario_id=scenario.id,
        provider="fixture",
        model="fixture-model",
        endpoint="https://provider.example/v1",
        prompt_version="prompt-v1",
        parameters={"temperature": 0},
        messages=[{"role": "user", "content": "RAW-MESSAGE"}],
        input_assets=[],
        request_hash=request_hash,
        response={"choices": [{"message": {"content": content}}]},
    )
    session.add(run)
    session.flush()
    return run


def _annotation_json(subject_id: str = "ego", timestamp_ms: int = 100) -> str:
    return json.dumps(
        {
            "subject_id": subject_id,
            "timestamp_ms": timestamp_ms,
            "claims": {
                "traffic_light": "red",
                "crossed_stop_line": True,
                "event": "red_light_running",
            },
            "confidence": 0.9,
        }
    )


def test_annotate_run_parses_only_strict_structured_json(session: Session) -> None:
    scenario = _scenario(session)
    run = _run(session, scenario, _annotation_json())

    annotation = annotate_run(session, run.id)

    assert annotation.scenario_id == scenario.id
    assert annotation.model_run_id == run.id
    assert annotation.subject_id == "ego"
    assert annotation.timestamp_ms == 100
    assert annotation.claims["event"] == "red_light_running"
    assert annotation.confidence == 0.9


def test_malformed_model_response_is_preserved_without_annotation(session: Session) -> None:
    scenario = _scenario(session)
    run = _run(session, scenario, "not valid JSON")
    original_response = deepcopy(run.response)

    with pytest.raises(AnnotationError, match="valid structured annotation"):
        annotate_run(session, run.id)

    assert session.get(ModelRun, run.id).response == original_response
    assert session.scalar(select(func.count(Annotation.id))) == 0


def test_verification_uses_correct_scope_and_does_not_mutate_facts(session: Session) -> None:
    scenario = _scenario(session)
    correct_run = _run(session, scenario, _annotation_json(), "correct-run")
    wrong_scope_run = _run(session, scenario, _annotation_json(timestamp_ms=200), "wrong-run")
    correct_annotation = annotate_run(session, correct_run.id)
    wrong_scope_annotation = annotate_run(session, wrong_scope_run.id)
    fact = Fact(
        scenario_id=scenario.id,
        source="manual",
        source_version="fixture-v1",
        subject_id="ego",
        timestamp_ms=100,
        values={
            "traffic_light": "red",
            "crossed_stop_line": True,
            "event": "red_light_running",
        },
        provenance={"fixture": "scope-test"},
    )
    wrong_scope_fact = Fact(
        scenario_id=scenario.id,
        source="manual",
        source_version="fixture-v1",
        subject_id="other",
        timestamp_ms=100,
        values={"traffic_light": "green"},
        provenance={"fixture": "wrong-subject"},
    )
    session.add_all([fact, wrong_scope_fact])
    session.flush()
    before_values = deepcopy(fact.values)
    before_provenance = deepcopy(fact.provenance)
    before_count = session.scalar(select(func.count(Fact.id)))

    accepted = verify_annotation(session, correct_annotation.id)
    needs_review = verify_annotation(session, wrong_scope_annotation.id)

    assert accepted.status == "accepted"
    assert accepted.fact_ids == [fact.id]
    assert needs_review.status == "needs_review"
    assert needs_review.fact_ids == []
    assert fact.values == before_values
    assert fact.provenance == before_provenance
    assert session.scalar(select(func.count(Fact.id))) == before_count


def test_mismatched_annotation_and_model_run_scenarios_are_rejected(session: Session) -> None:
    first = _scenario(session, "first")
    second = _scenario(session, "second")
    run = _run(session, second, _annotation_json(), "second-run")
    annotation = Annotation(
        scenario_id=first.id,
        model_run_id=run.id,
        subject_id="ego",
        timestamp_ms=100,
        claims={"event": "red_light_running"},
    )
    session.add(annotation)

    with pytest.raises(IntegrityError):
        session.flush()
    session.rollback()

    assert session.scalar(select(func.count(Verification.id))) == 0


def _export_rows(session: Session) -> Scenario:
    scenario = _scenario(session, "export")
    asset = Asset(
        scenario_id=scenario.id,
        kind="video",
        local_path="media/fixture.mp4",
        sha256="a" * 64,
    )
    fact = Fact(
        scenario_id=scenario.id,
        source="simulator",
        source_version="fixture-v1",
        timestamp_ms=100,
        values={"traffic_light": "red"},
        provenance={"simulator": "fixture"},
    )
    session.add_all([asset, fact])
    session.flush()
    for index, status in enumerate(("accepted", "rejected", "needs_review")):
        run = _run(
            session,
            scenario,
            _annotation_json(),
            request_hash=f"export-run-{index}",
        )
        annotation = Annotation(
            scenario_id=scenario.id,
            model_run_id=run.id,
            subject_id="ego",
            timestamp_ms=100,
            claims={"event": "red_light_running"},
        )
        session.add(annotation)
        session.flush()
        session.add(
            Verification(
                annotation_id=annotation.id,
                fact_ids=[fact.id],
                status=status,
                reasons=[{"code": f"reason-{index}"}],
                verifier_version="fixture-verifier-v1",
            )
        )
    session.commit()
    return scenario


def test_export_snapshot_is_local_lossless_metadata_with_hashes_and_statuses(
    session: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenario = _export_rows(session)

    def fail_network(*args, **kwargs):
        raise AssertionError("snapshot export must not use the network")

    monkeypatch.setattr(socket, "create_connection", fail_network)
    destination = tmp_path / "snapshot"
    manifest = export_snapshot(session, destination)

    assert manifest == destination / "scenarios.jsonl"
    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    record = records[0]
    assert record["scenario"]["id"] == scenario.id
    assert len(record["assets"]) == 1
    assert len(record["facts"]) == 1
    assert len(record["annotations"]) == 3
    assert {row["status"] for row in record["verifications"]} == {
        "accepted",
        "rejected",
        "needs_review",
    }
    assert all("messages" not in row and "response" not in row for row in record["model_runs"])
    raw_manifest = manifest.read_text(encoding="utf-8")
    assert "RAW-MESSAGE" not in raw_manifest
    assert "choices" not in raw_manifest

    metadata = json.loads((destination / "snapshot.json").read_text(encoding="utf-8"))
    assert metadata["scenario_count"] == 1
    assert metadata["manifest"] == "scenarios.jsonl"
    assert metadata["sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert metadata["media_included"] is False
    assert metadata["publish_ready"] is False


def test_export_snapshot_refuses_to_overwrite_existing_destination(
    session: Session, tmp_path: Path
) -> None:
    _export_rows(session)
    destination = tmp_path / "existing-snapshot"
    destination.mkdir()
    sentinel = destination / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError):
        export_snapshot(session, destination)

    assert sentinel.read_text(encoding="utf-8") == "keep"
