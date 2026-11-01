"""Import operator-recorded BeamNG media without manufacturing simulator facts."""

import mimetypes
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy.orm import Session

from ..blob_storage import LocalBlobStore
from ..models import Asset, Scenario


class CapturedMedia(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    path: str = Field(min_length=1)
    kind: Literal["frame", "storyboard", "video"]
    timestamp_ms: int = Field(ge=0)
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @field_validator("path")
    @classmethod
    def portable_path(cls, value: str) -> str:
        if "\\" in value or ":" in value or value.startswith("/"):
            raise ValueError("media path must be relative, using forward slashes")
        if any(part in {"", ".", ".."} for part in value.split("/")):
            raise ValueError("media path contains invalid components")
        return value


class CaptureManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal[1]
    simulator_product: Literal["BeamNG.drive", "BeamNG.tech"]
    simulator_version: str = Field(min_length=1)
    map_name: str = Field(min_length=1)
    scenario_type: str = Field(min_length=1)
    capture_method: Literal["screen_recording", "screenshot"]
    notes: str = ""
    media: list[CapturedMedia] = Field(min_length=1, max_length=64)


def import_capture(session: Session, manifest_path: Path, store: LocalBlobStore) -> Scenario:
    """Copy media into managed blob storage and register metadata; caller commits.

    All manifest fields are operator declarations, not independent facts. Files
    are validated before database writes. A SQL failure can leave unreferenced
    content-addressed blobs; it never deletes or changes the capture originals.
    """
    manifest_path = Path(manifest_path).resolve()
    if manifest_path.stat().st_size > 1024 * 1024:
        raise ValueError("capture manifest exceeds 1 MiB")
    manifest = CaptureManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    base = manifest_path.parent
    prepared = []
    seen = set()
    for media in manifest.media:
        source = (base / media.path).resolve()
        if not source.is_relative_to(base) or not source.is_file() or source in seen:
            raise ValueError("capture files must be distinct files inside the capture directory")
        seen.add(source)
        mime = mimetypes.guess_type(source.name)[0]
        allowed = (
            {"image/png", "image/jpeg", "image/webp"}
            if media.kind != "video"
            else {
                "video/mp4",
                "video/webm",
                "video/quicktime",
                "video/x-matroska",
                "video/x-msvideo",
            }
        )
        if mime not in allowed:
            raise ValueError("unsupported media extension for the declared kind")
        prepared.append((media, source, mime))
    stored = [
        (media, mime, store.put(source, expected_sha256=media.sha256))
        for media, source, mime in prepared
    ]
    scenario = Scenario(
        source=manifest.simulator_product.lower(),
        scenario_type=manifest.scenario_type,
        configuration={
            "capture_schema_version": 1,
            "provenance_authority": "operator_declared",
            "simulator_product": manifest.simulator_product,
            "simulator_version": manifest.simulator_version,
            "map_name": manifest.map_name,
            "capture_method": manifest.capture_method,
            "notes": manifest.notes,
            "independent_telemetry_included": False,
        },
    )
    session.add(scenario)
    session.flush()
    for media, mime, ref in stored:
        local_path = store.root / ref.uri.removeprefix("local://")
        session.add(
            Asset(
                scenario_id=scenario.id,
                kind=media.kind,
                local_path=str(local_path),
                sha256=ref.sha256,
                timestamp_ms=media.timestamp_ms,
                metadata_json={
                    "media_type": mime,
                    "original_relative_path": media.path,
                    "size_bytes": ref.size_bytes,
                    "capture_provenance": "operator_declared",
                },
            )
        )
    session.flush()
    return scenario
