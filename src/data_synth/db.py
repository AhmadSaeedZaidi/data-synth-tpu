"""Database engine and bootstrap initialization helpers.

This module provides a deliberately small schema bootstrap, rather than a
migration framework.  ``init_db`` creates the version-1 schema once and
refuses to operate when an existing database reports an unknown version.
Future schema changes should introduce an explicit migration plan.
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import Engine, create_engine, event, insert, inspect, select
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from .models import Base, SchemaVersion

SCHEMA_VERSION = 1
DEFAULT_DATABASE_URL = "sqlite:///data/research.db"


def make_engine(database_url: str = DEFAULT_DATABASE_URL) -> Engine:
    """Create an engine with SQLite foreign-key enforcement enabled.

    The in-memory SQLite form uses a static pool so all sessions created from
    the returned engine share the same database.  Other database URLs remain
    portable SQLAlchemy URLs and use SQLAlchemy's normal pool selection.
    """

    url = make_url(database_url)
    kwargs: dict[str, object] = {"future": True}

    if url.get_backend_name() == "sqlite":
        if url.database in (None, "", ":memory:"):
            kwargs["poolclass"] = StaticPool
            kwargs["connect_args"] = {"check_same_thread": False}
        elif not url.database.startswith("file:"):
            database_path = Path(url.database).expanduser()
            database_path.parent.mkdir(parents=True, exist_ok=True)
            url = url.set(database=str(database_path))

    engine = create_engine(url, **kwargs)

    if url.get_backend_name() == "sqlite":

        @event.listens_for(engine, "connect")
        def _enable_sqlite_foreign_keys(
            dbapi_connection: object, connection_record: object
        ) -> None:
            del connection_record
            dbapi_connection.execute("PRAGMA foreign_keys=ON")  # type: ignore[attr-defined]

    return engine


def init_db(engine: Engine) -> None:
    """Create the version-1 schema or validate an existing schema marker.

    A fresh database is initialized with ``create_all`` and one singleton
    ``schema_version`` row.  Existing databases must contain exactly one
    supported version row; an unknown, missing, or duplicated marker raises
    ``RuntimeError`` instead of silently attempting a migration.
    """

    table_names = set(inspect(engine).get_table_names())

    if "schema_version" not in table_names:
        if table_names:
            raise RuntimeError(
                "Database contains tables but no schema version marker; "
                "migrations are not performed by init_db"
            )
        Base.metadata.create_all(engine)
        with engine.begin() as connection:
            connection.execute(insert(SchemaVersion).values(id=1, version=SCHEMA_VERSION))
        return

    with Session(engine) as session:
        versions = (
            session.execute(select(SchemaVersion.version).where(SchemaVersion.id == 1))
            .scalars()
            .all()
        )

    if len(versions) != 1:
        raise RuntimeError(
            "Database has no valid singleton schema version marker; "
            "migrations are not performed by init_db"
        )
    if versions[0] != SCHEMA_VERSION:
        raise RuntimeError(
            f"Unsupported database schema version {versions[0]!r}; "
            f"expected {SCHEMA_VERSION} (migrations are not performed by init_db)"
        )

    expected_tables = set(Base.metadata.tables)
    missing_tables = expected_tables - table_names
    if missing_tables:
        missing = ", ".join(sorted(missing_tables))
        raise RuntimeError(
            f"Database schema version {SCHEMA_VERSION} is incomplete; "
            f"missing tables: {missing}. Migrations are not performed by init_db"
        )

    inspector = inspect(engine)
    for table in Base.metadata.sorted_tables:
        actual_columns = {column["name"] for column in inspector.get_columns(table.name)}
        if actual_columns != set(table.columns.keys()):
            raise RuntimeError(f"Database table {table.name} has incompatible columns")


__all__ = ["DEFAULT_DATABASE_URL", "SCHEMA_VERSION", "init_db", "make_engine"]
