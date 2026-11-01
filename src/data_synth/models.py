"""SQLAlchemy models for the data synthesis scaffold.

The schema is intentionally a small bootstrap schema.  It is versioned by
``data_synth.db.init_db`` and is not a migration system; schema changes should
be handled by an explicit migration plan before this version is changed.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import TypeDecorator


def utc_now() -> datetime:
    """Return an aware UTC timestamp suitable for ``created_at`` defaults."""

    return datetime.now(timezone.utc)


class UTCDateTime(TypeDecorator[datetime]):
    """Store UTC timestamps consistently across SQLite and other databases.

    SQLite does not retain timezone information for its native datetime type.
    Values are stored as UTC and re-hydrated with an explicit UTC timezone so
    callers observe the same behavior after reopening a database.
    """

    impl = DateTime
    cache_ok = True

    def load_dialect_impl(self, dialect: Any) -> Any:
        return dialect.type_descriptor(DateTime(timezone=True))

    def process_bind_param(self, value: datetime | None, dialect: Any) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def process_result_value(self, value: datetime | None, dialect: Any) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)


def uuid_string() -> str:
    """Generate the portable string representation used for entity IDs."""

    return str(uuid4())


class Base(DeclarativeBase):
    """Declarative base shared by all application tables."""


class SchemaVersion(Base):
    """Singleton bootstrap schema marker used by :func:`init_db`."""

    __tablename__ = "schema_version"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False)

    __table_args__ = (
        CheckConstraint("id = 1", name="ck_schema_version_singleton"),
        CheckConstraint("version > 0", name="ck_schema_version_positive"),
    )


class Scenario(Base):
    """A source scenario from which assets, facts, and annotations originate."""

    __tablename__ = "scenarios"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_string)
    source: Mapped[str] = mapped_column(String(128), nullable=False)
    scenario_type: Mapped[str] = mapped_column(String(128), nullable=False)
    seed: Mapped[int | None] = mapped_column(Integer, nullable=True)
    configuration: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now, nullable=False)

    assets: Mapped[list["Asset"]] = relationship(
        back_populates="scenario", cascade="all, delete-orphan"
    )
    facts: Mapped[list["Fact"]] = relationship(
        back_populates="scenario", cascade="all, delete-orphan"
    )
    model_runs: Mapped[list["ModelRun"]] = relationship(
        back_populates="scenario", cascade="all, delete-orphan"
    )
    annotations: Mapped[list["Annotation"]] = relationship(
        back_populates="scenario", cascade="all, delete-orphan"
    )

    __table_args__ = (
        CheckConstraint("length(trim(source)) > 0", name="ck_scenarios_source_nonempty"),
        CheckConstraint("length(trim(scenario_type)) > 0", name="ck_scenarios_type_nonempty"),
    )


class Asset(Base):
    """A rendered media asset associated with a scenario."""

    __tablename__ = "assets"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_string)
    scenario_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("scenarios.id", ondelete="CASCADE"), nullable=False
    )
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    local_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    remote_uri: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    timestamp_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    frame_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now, nullable=False)

    scenario: Mapped[Scenario] = relationship(back_populates="assets")

    __table_args__ = (
        CheckConstraint("kind IN ('frame', 'storyboard', 'video')", name="ck_assets_kind"),
        CheckConstraint(
            "timestamp_ms IS NULL OR timestamp_ms >= 0", name="ck_assets_timestamp_nonnegative"
        ),
        CheckConstraint(
            "frame_index IS NULL OR frame_index >= 0", name="ck_assets_frame_nonnegative"
        ),
    )


class Fact(Base):
    """A timestamped structured observation from a simulator or detector."""

    __tablename__ = "facts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_string)
    scenario_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("scenarios.id", ondelete="CASCADE"), nullable=False
    )
    source: Mapped[str] = mapped_column(String(16), nullable=False)
    source_version: Mapped[str] = mapped_column(String(128), nullable=False)
    subject_id: Mapped[str] = mapped_column(String(128), default="ego", nullable=False)
    timestamp_ms: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    values: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    provenance: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now, nullable=False)

    scenario: Mapped[Scenario] = relationship(back_populates="facts")

    __table_args__ = (
        CheckConstraint("source IN ('simulator', 'detector', 'manual')", name="ck_facts_source"),
        CheckConstraint("timestamp_ms >= 0", name="ck_facts_timestamp_nonnegative"),
        CheckConstraint("length(trim(source_version)) > 0", name="ck_facts_version_nonempty"),
        CheckConstraint("length(trim(subject_id)) > 0", name="ck_facts_subject_nonempty"),
    )


class ModelRun(Base):
    """A provider/model invocation and its cached response."""

    __tablename__ = "model_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_string)
    scenario_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("scenarios.id", ondelete="CASCADE"), nullable=True
    )
    provider: Mapped[str] = mapped_column(String(128), nullable=False)
    model: Mapped[str] = mapped_column(String(256), nullable=False)
    endpoint: Mapped[str] = mapped_column(String(2048), nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(128), nullable=False)
    parameters: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    input_assets: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list, nullable=False)
    request_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    response: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    messages: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list, nullable=False)
    cache_hit: Mapped[bool] = mapped_column(default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now, nullable=False)

    scenario: Mapped[Scenario | None] = relationship(back_populates="model_runs")
    annotations: Mapped[list["Annotation"]] = relationship(
        back_populates="model_run",
        cascade="all, delete-orphan",
        foreign_keys="Annotation.model_run_id",
    )

    __table_args__ = (
        UniqueConstraint("id", "scenario_id", name="uq_model_runs_id_scenario"),
        CheckConstraint("length(trim(provider)) > 0", name="ck_model_runs_provider_nonempty"),
        CheckConstraint("length(trim(model)) > 0", name="ck_model_runs_model_nonempty"),
        CheckConstraint(
            "length(trim(prompt_version)) > 0", name="ck_model_runs_prompt_version_nonempty"
        ),
        CheckConstraint("length(trim(request_hash)) > 0", name="ck_model_runs_hash_nonempty"),
    )


class Annotation(Base):
    """A model interpretation grounded in a scenario and model run."""

    __tablename__ = "annotations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_string)
    scenario_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("scenarios.id", ondelete="CASCADE"), nullable=False
    )
    model_run_id: Mapped[str] = mapped_column(String(36), nullable=False)
    subject_id: Mapped[str] = mapped_column(String(128), default="ego", nullable=False)
    timestamp_ms: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    claims: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    confidence: Mapped[float | None] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now, nullable=False)

    scenario: Mapped[Scenario] = relationship(back_populates="annotations")
    model_run: Mapped[ModelRun] = relationship(
        back_populates="annotations", foreign_keys=[model_run_id]
    )
    verifications: Mapped[list["Verification"]] = relationship(
        back_populates="annotation", cascade="all, delete-orphan"
    )

    __table_args__ = (
        ForeignKeyConstraint(
            ["model_run_id", "scenario_id"],
            ["model_runs.id", "model_runs.scenario_id"],
            ondelete="CASCADE",
            name="fk_annotation_run_scenario",
        ),
        CheckConstraint("timestamp_ms >= 0", name="ck_annotations_timestamp_nonnegative"),
        CheckConstraint(
            "confidence IS NULL OR (confidence >= 0.0 AND confidence <= 1.0)",
            name="ck_annotations_confidence_range",
        ),
        CheckConstraint("length(trim(subject_id)) > 0", name="ck_annotations_subject_nonempty"),
    )


class Verification(Base):
    """A consistency check of an annotation against stored facts."""

    __tablename__ = "verifications"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_string)
    annotation_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("annotations.id", ondelete="CASCADE"), nullable=False
    )
    fact_ids: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    reasons: Mapped[list[Any]] = mapped_column(JSON, default=list, nullable=False)
    verifier_version: Mapped[str] = mapped_column(String(128), nullable=False)
    manual_review_status: Mapped[str] = mapped_column(String(16), default="pending", nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now, nullable=False)

    annotation: Mapped[Annotation] = relationship(back_populates="verifications")

    __table_args__ = (
        CheckConstraint(
            "status IN ('accepted', 'rejected', 'needs_review')", name="ck_verifications_status"
        ),
        CheckConstraint(
            "manual_review_status IN ('pending', 'approved', 'rejected')",
            name="ck_verifications_manual_status",
        ),
        CheckConstraint(
            "length(trim(verifier_version)) > 0", name="ck_verifications_version_nonempty"
        ),
    )


__all__ = [
    "Annotation",
    "Asset",
    "Base",
    "Fact",
    "ModelRun",
    "Scenario",
    "SchemaVersion",
    "UTCDateTime",
    "Verification",
    "utc_now",
    "uuid_string",
]
