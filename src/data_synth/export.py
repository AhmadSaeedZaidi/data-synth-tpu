"""Local metadata snapshots: the future Hugging Face upload boundary."""

import hashlib
import json
from datetime import datetime
from pathlib import Path

from sqlalchemy import inspect, select
from sqlalchemy.orm import Session

from .models import Annotation, Asset, Fact, ModelRun, Scenario, Verification


def _record(row, *, exclude=()):
    values = {
        column.key: getattr(row, column.key)
        for column in inspect(type(row)).columns
        if column.key not in exclude
    }
    return {
        key: value.isoformat() if isinstance(value, datetime) else value
        for key, value in values.items()
    }


def export_snapshot(session: Session, destination: Path) -> Path:
    """Write a new local snapshot directory. No upload and no media copying.

    Includes every status for disagreement analysis. Never collapses facts and
    model claims into a shared label. The caller must export from a quiescent
    database (or a repeatable-read transaction on a future PostgreSQL server).
    Exact prompts/raw responses remain in the local DB, not the dataset manifest.
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    manifest = destination / "scenarios.jsonl"
    count = 0
    with manifest.open("w", encoding="utf-8") as handle:
        for scenario in session.scalars(select(Scenario).order_by(Scenario.id)):
            assets = session.scalars(
                select(Asset).where(Asset.scenario_id == scenario.id).order_by(Asset.id)
            ).all()
            facts = session.scalars(
                select(Fact).where(Fact.scenario_id == scenario.id).order_by(Fact.id)
            ).all()
            runs = session.scalars(
                select(ModelRun).where(ModelRun.scenario_id == scenario.id).order_by(ModelRun.id)
            ).all()
            annotations = session.scalars(
                select(Annotation)
                .where(Annotation.scenario_id == scenario.id)
                .order_by(Annotation.id)
            ).all()
            checks = session.scalars(
                select(Verification)
                .join(Annotation)
                .where(Annotation.scenario_id == scenario.id)
                .order_by(Verification.id)
            ).all()
            record = {
                "schema_version": 1,
                "scenario": _record(scenario),
                "assets": [_record(a) for a in assets],
                "facts": [_record(f) for f in facts],
                "model_runs": [_record(r, exclude=("messages", "response")) for r in runs],
                "annotations": [_record(a) for a in annotations],
                "verifications": [_record(v) for v in checks],
            }
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            count += 1
    metadata = {
        "schema_version": 1,
        "scenario_count": count,
        "manifest": manifest.name,
        "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "media_included": False,
        "publish_ready": False,
    }
    (destination / "snapshot.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return manifest
