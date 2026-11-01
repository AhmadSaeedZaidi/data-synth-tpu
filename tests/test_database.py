"""Focused persistence and integrity tests for the bootstrap database."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import func, select, text, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from data_synth.db import init_db, make_engine
from data_synth.models import (
    Annotation,
    Asset,
    Fact,
    ModelRun,
    Scenario,
    SchemaVersion,
    Verification,
)


def _initialized_engine(tmp_path: Path):
    database_path = tmp_path / "scaffold.sqlite3"
    engine = make_engine(f"sqlite:///{database_path}")
    init_db(engine)
    assert database_path.exists()
    return engine


def _scenario(engine) -> Scenario:
    with Session(engine, expire_on_commit=False) as session:
        scenario = Scenario(
            source="beamng",
            scenario_type="rear_end",
            seed=42,
            configuration={"weather": "clear", "traffic": 3},
        )
        session.add(scenario)
        session.commit()
        return scenario


def _model_run(engine, scenario: Scenario) -> ModelRun:
    with Session(engine, expire_on_commit=False) as session:
        model_run = ModelRun(
            scenario_id=scenario.id,
            provider="local",
            model="test-vlm",
            endpoint="http://localhost/v1/chat/completions",
            prompt_version="v1",
            parameters={"temperature": 0},
            input_assets=[],
            request_hash="request-hash",
            response={"claims": []},
        )
        session.add(model_run)
        session.commit()
        return model_run


def test_persistence_survives_reopen_and_round_trips_json(tmp_path: Path) -> None:
    database_path = tmp_path / "reopen.sqlite3"
    engine = make_engine(f"sqlite:///{database_path}")
    init_db(engine)

    with Session(engine) as session:
        scenario = Scenario(
            source="beamng",
            scenario_type="intersection",
            seed=7,
            configuration={"weather": "rain", "lights": {"state": "red"}},
        )
        session.add(scenario)
        session.flush()

        asset = Asset(
            scenario_id=scenario.id,
            kind="video",
            local_path="clips/intersection.mp4",
            sha256="a" * 64,
        )
        fact = Fact(
            scenario_id=scenario.id,
            source="simulator",
            source_version="fixture-v1",
            timestamp_ms=1250,
            values={"speed_mps": 12.5},
            provenance={"tick": 25},
        )
        session.add_all([asset, fact])
        session.flush()

        model_run = ModelRun(
            scenario_id=scenario.id,
            provider="local",
            model="test-vlm",
            endpoint="http://localhost/v1/chat/completions",
            prompt_version="v1",
            parameters={"temperature": 0},
            input_assets=[{"id": asset.id, "sha256": asset.sha256}],
            request_hash="hash-1",
            response={"label": "unsafe_following"},
            messages=[{"role": "user", "content": "Review this clip."}],
            cache_hit=True,
        )
        session.add(model_run)
        session.flush()

        annotation = Annotation(
            scenario_id=scenario.id,
            model_run_id=model_run.id,
            subject_id="ego",
            timestamp_ms=1250,
            claims={"unsafe_following": True},
            confidence=0.9,
        )
        session.add(annotation)
        session.flush()
        session.add(
            Verification(
                annotation_id=annotation.id,
                fact_ids=[fact.id],
                status="accepted",
                reasons=["speed and distance agree"],
                verifier_version="rules-v1",
            )
        )
        session.commit()
        scenario_id = scenario.id

    engine.dispose()
    reopened = make_engine(f"sqlite:///{database_path}")
    init_db(reopened)
    with Session(reopened) as session:
        loaded = session.get(Scenario, scenario_id)
        assert loaded is not None
        assert loaded.configuration["lights"]["state"] == "red"
        assert loaded.created_at.tzinfo is not None
        assert loaded.created_at.utcoffset().total_seconds() == 0
        assert session.scalar(select(func.count(Asset.id))) == 1
        assert session.scalar(select(func.count(Fact.id))) == 1
        assert session.scalar(select(func.count(Annotation.id))) == 1
        assert session.scalar(select(func.count(Verification.id))) == 1
        loaded_run = session.scalar(select(ModelRun))
        assert loaded_run.messages == [{"role": "user", "content": "Review this clip."}]


def test_model_run_allows_text_smoke_without_scenario(tmp_path: Path) -> None:
    engine = _initialized_engine(tmp_path)
    with Session(engine) as session:
        run = ModelRun(
            provider="local",
            model="text-model",
            endpoint="http://localhost/v1/chat/completions",
            prompt_version="v1",
            parameters={},
            input_assets=[],
            request_hash="text-smoke",
            response={"text": "ok"},
        )
        session.add(run)
        session.commit()
        assert session.get(ModelRun, run.id).scenario_id is None


def test_annotation_cannot_reference_another_scenarios_model_run(tmp_path: Path) -> None:
    engine = _initialized_engine(tmp_path)
    first = _scenario(engine)
    second = _scenario(engine)
    run = _model_run(engine, first)
    with Session(engine) as session:
        session.add(Annotation(scenario_id=second.id, model_run_id=run.id))
        with pytest.raises(IntegrityError):
            session.commit()


def test_foreign_key_rejection_and_transaction_rollback(tmp_path: Path) -> None:
    engine = _initialized_engine(tmp_path)
    with Session(engine) as session:
        session.add(Asset(scenario_id="missing-scenario", kind="frame"))
        with pytest.raises(IntegrityError):
            session.commit()
        session.rollback()

    with Session(engine) as session:
        scenario = Scenario(source="manual", scenario_type="rollback")
        session.add(scenario)
        session.flush()
        session.add(Asset(scenario_id="missing-scenario", kind="frame"))
        with pytest.raises(IntegrityError):
            session.commit()
        session.rollback()
        assert session.get(Scenario, scenario.id) is None


def test_enum_and_range_constraints(tmp_path: Path) -> None:
    engine = _initialized_engine(tmp_path)
    scenario = _scenario(engine)
    model_run = _model_run(engine, scenario)

    with Session(engine) as session:
        session.add(Asset(scenario_id=scenario.id, kind="audio"))
        with pytest.raises(IntegrityError):
            session.commit()

    with Session(engine) as session:
        session.add(
            Fact(
                scenario_id=scenario.id,
                source="human",
                source_version="v1",
            )
        )
        with pytest.raises(IntegrityError):
            session.commit()

    with Session(engine) as session:
        session.add(
            Annotation(
                scenario_id=scenario.id,
                model_run_id=model_run.id,
                timestamp_ms=-1,
                confidence=1.1,
            )
        )
        with pytest.raises(IntegrityError):
            session.commit()

    with Session(engine) as session:
        annotation = Annotation(scenario_id=scenario.id, model_run_id=model_run.id, claims={})
        session.add(annotation)
        session.flush()
        session.add(
            Verification(
                annotation_id=annotation.id,
                status="unknown",
                verifier_version="v1",
            )
        )
        with pytest.raises(IntegrityError):
            session.commit()


def test_schema_init_is_idempotent_and_refuses_unknown_version(tmp_path: Path) -> None:
    engine = _initialized_engine(tmp_path)
    init_db(engine)
    with Session(engine) as session:
        assert session.scalar(select(func.count(SchemaVersion.id))) == 1

    with engine.begin() as connection:
        connection.execute(update(SchemaVersion).values(version=999))

    with pytest.raises(RuntimeError, match="Unsupported database schema version"):
        init_db(engine)


def test_schema_init_refuses_changed_columns(tmp_path: Path) -> None:
    engine = _initialized_engine(tmp_path)
    with engine.begin() as connection:
        connection.execute(text("ALTER TABLE scenarios RENAME COLUMN seed TO unexpected_seed"))
    with pytest.raises(RuntimeError, match="incompatible columns"):
        init_db(engine)


def test_schema_init_refuses_unversioned_or_partial_databases(tmp_path: Path) -> None:
    unversioned = make_engine(f"sqlite:///{tmp_path / 'unversioned.sqlite3'}")
    with unversioned.begin() as connection:
        connection.execute(text("CREATE TABLE unrelated (id INTEGER PRIMARY KEY)"))
    with pytest.raises(RuntimeError, match="no schema version marker"):
        init_db(unversioned)

    partial = _initialized_engine(tmp_path / "partial")
    with partial.begin() as connection:
        connection.execute(text("DROP TABLE assets"))
    with pytest.raises(RuntimeError, match="incomplete"):
        init_db(partial)


def test_sqlite_foreign_keys_are_enabled(tmp_path: Path) -> None:
    engine = _initialized_engine(tmp_path)
    with engine.connect() as connection:
        assert connection.scalar(text("PRAGMA foreign_keys")) == 1
