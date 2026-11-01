"""Offline CLI and scripted-demo integration tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from data_synth import __main__ as cli
from data_synth.config import Settings
from data_synth.db import make_engine
from data_synth.models import Annotation, Asset, Fact, ModelRun, Scenario, Verification


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        database_url=f"sqlite:///{tmp_path / 'research.sqlite3'}",
        cache_dir=tmp_path / "cache",
    )


def _use_settings(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> None:
    monkeypatch.setattr(cli.Settings, "load", staticmethod(lambda: settings))


def test_demo_cli_runs_offline_with_expected_statuses_and_provenance(
    tmp_path: Path, capsys
) -> None:
    destination = tmp_path / "demo"

    assert cli.main(["demo", str(destination)]) == 0
    result = json.loads(capsys.readouterr().out)

    assert result["fixture_only"] is True
    assert result["statuses"] == {
        "positive": "accepted",
        "negative": "accepted",
        "disagreement": "rejected",
        "missing_evidence": "needs_review",
    }
    manifest = Path(result["manifest"])
    assert manifest == destination / "snapshot" / "scenarios.jsonl"
    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 4
    assert {record["verifications"][0]["status"] for record in records} == {
        "accepted",
        "rejected",
        "needs_review",
    }

    engine = make_engine(f"sqlite:///{destination / 'research.db'}")
    with Session(engine) as session:
        assert session.scalar(select(func.count(Scenario.id))) == 4
        assert session.scalar(select(func.count(Asset.id))) == 4
        assert session.scalar(select(func.count(Fact.id))) == 4
        assert session.scalar(select(func.count(ModelRun.id))) == 4
        assert session.scalar(select(func.count(Annotation.id))) == 4
        assert session.scalar(select(func.count(Verification.id))) == 4
        assert all(row.source == "manual" for row in session.scalars(select(Scenario)))
        facts = session.scalars(select(Fact)).all()
        assert all(row.source == "manual" for row in facts)
        assert all(row.source_version == "demo-fixture-v1" for row in facts)
        assert all(row.provenance == {"fixture_only": True} for row in facts)

    metadata = json.loads((destination / "snapshot" / "snapshot.json").read_text())
    assert metadata["scenario_count"] == 4
    assert metadata["sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()


def test_demo_cli_refuses_overwrite(tmp_path: Path, capsys) -> None:
    destination = tmp_path / "demo"
    assert cli.main(["demo", str(destination)]) == 0
    capsys.readouterr()

    assert cli.main(["demo", str(destination)]) == 1
    output = capsys.readouterr().out
    assert "Destination already exists" in output


def test_init_db_uses_configured_temporary_database(monkeypatch, tmp_path: Path, capsys) -> None:
    settings = _settings(tmp_path)
    _use_settings(monkeypatch, settings)

    assert cli.main(["init-db"]) == 0
    assert "schema version 1" in capsys.readouterr().out
    database_path = tmp_path / "research.sqlite3"
    assert database_path.exists()
    engine = make_engine(settings.database_url)
    with Session(engine) as session:
        assert session.scalar(select(func.count(Scenario.id))) == 0


def test_export_empty_database_is_valid_and_hashes_manifest(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    settings = _settings(tmp_path)
    _use_settings(monkeypatch, settings)
    assert cli.main(["init-db"]) == 0
    capsys.readouterr()

    destination = tmp_path / "empty-snapshot"
    assert cli.main(["export", str(destination)]) == 0
    manifest = Path(capsys.readouterr().out.strip())
    assert manifest == destination / "scenarios.jsonl"
    assert manifest.read_text(encoding="utf-8") == ""
    metadata = json.loads((destination / "snapshot.json").read_text(encoding="utf-8"))
    assert metadata["scenario_count"] == 0
    assert metadata["sha256"] == hashlib.sha256(b"").hexdigest()


def test_annotate_run_invalid_id_has_sanitized_cli_failure(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    settings = _settings(tmp_path)
    _use_settings(monkeypatch, settings)
    assert cli.main(["init-db"]) == 0
    capsys.readouterr()

    missing_id = "secret-missing-run-id"
    assert cli.main(["annotate-run", missing_id]) == 1
    output = capsys.readouterr().out
    assert output.strip() == (
        "Operation failed. Check configuration, record IDs, schema, and file paths."
    )
    assert missing_id not in output


def test_verify_cli_can_verify_an_annotation_stored_by_demo(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    demo_dir = tmp_path / "demo"
    assert cli.main(["demo", str(demo_dir)]) == 0
    capsys.readouterr()
    settings = Settings(
        database_url=f"sqlite:///{demo_dir / 'research.db'}",
        cache_dir=tmp_path / "cache",
    )
    _use_settings(monkeypatch, settings)
    engine = make_engine(settings.database_url)
    with Session(engine) as session:
        annotation_id = session.scalar(select(Annotation.id).order_by(Annotation.id))
    assert annotation_id is not None

    assert cli.main(["verify", annotation_id]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["verification_id"]
    assert output["status"] in {"accepted", "rejected", "needs_review"}
