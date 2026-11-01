"""Offline, explicitly scripted fixtures; no simulator or model inference."""

import base64
import hashlib
import json
from pathlib import Path

from sqlalchemy.orm import Session

from .cache import ResponseCache, image_to_data_url
from .db import init_db, make_engine
from .export import export_snapshot
from .model_service import run_model
from .models import Asset, Fact, Scenario
from .providers import ModelRequest, ModelResult
from .workflow import annotate_run, verify_annotation

# One-pixel PNG placeholder, never presented as simulator footage.
PIXEL = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a0xkAAAAASUVORK5CYII="
)


class FixtureProvider:
    """Exercise the persistence contract with declared fixture outputs."""

    def __init__(self, cache: ResponseCache, annotation: dict):
        self.cache, self.annotation = cache, annotation

    def complete(self, request: ModelRequest) -> ModelResult:
        key = self.cache.request_hash(request)
        with self.cache.locked(key):
            response = self.cache.get(request)
            hit = response is not None
            if response is None:
                response = {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "content": json.dumps(self.annotation),
                            },
                        }
                    ]
                }
                self.cache.put(request, response)
            return ModelResult(response=response, request_hash=key, cache_hit=hit)


def run_demo(directory: Path) -> dict:
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    media = directory / "fixture-placeholder.png"
    media.write_bytes(base64.b64decode(PIXEL))
    digest = hashlib.sha256(media.read_bytes()).hexdigest()
    engine = make_engine(f"sqlite:///{directory / 'research.db'}")
    init_db(engine)
    cache = ResponseCache(directory / "cache")
    positive = {"traffic_light": "red", "crossed_stop_line": True, "event": "red_light_running"}
    negative = {"traffic_light": "red", "crossed_stop_line": False, "event": "no_red_light_running"}
    examples = [
        ("positive", positive, positive),
        ("negative", negative, negative),
        ("disagreement", positive, {**positive, "traffic_light": "green"}),
        ("missing_evidence", {"traffic_light": "red"}, positive),
    ]
    statuses = {}
    try:
        with Session(engine) as session:
            for name, evidence, claims in examples:
                scenario = Scenario(
                    source="manual",
                    scenario_type=f"fixture_{name}",
                    configuration={"fixture_only": True},
                )
                session.add(scenario)
                session.flush()
                asset = Asset(
                    scenario_id=scenario.id,
                    kind="frame",
                    local_path=str(media),
                    sha256=digest,
                    timestamp_ms=0,
                    metadata_json={"fixture_only": True, "not_traffic_footage": True},
                )
                session.add_all(
                    [
                        asset,
                        Fact(
                            scenario_id=scenario.id,
                            source="manual",
                            source_version="demo-fixture-v1",
                            subject_id="ego",
                            timestamp_ms=0,
                            values=evidence,
                            provenance={"fixture_only": True},
                        ),
                    ]
                )
                session.flush()
                annotation = {
                    "subject_id": "ego",
                    "timestamp_ms": 0,
                    "claims": claims,
                    "confidence": None,
                }
                request = ModelRequest(
                    provider="fixture",
                    endpoint="https://fixture.invalid/v1",
                    model="scripted-fixture-not-a-model",
                    prompt_version="offline-demo-v1",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Scripted fixture: " + json.dumps(annotation),
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": image_to_data_url(media)},
                                },
                            ],
                        }
                    ],
                    input_assets=[{"id": asset.id, "sha256": digest}],
                )
                run = run_model(session, FixtureProvider(cache, annotation), request, scenario.id)
                row = annotate_run(session, run.id)
                statuses[name] = verify_annotation(session, row.id).status
            session.commit()
            manifest = export_snapshot(session, directory / "snapshot")
        return {"fixture_only": True, "statuses": statuses, "manifest": str(manifest)}
    finally:
        engine.dispose()
