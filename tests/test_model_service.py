"""Tests for provider invocation provenance and model-run persistence."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from data_synth.cache import image_to_data_url
from data_synth.db import init_db, make_engine
from data_synth.model_service import run_model
from data_synth.models import Asset, ModelRun, Scenario
from data_synth.providers import ModelRequest, ModelResult, ProviderError


class StubProvider:
    def __init__(self, results: list[ModelResult] | None = None, error: Exception | None = None):
        self.results = list(results or [])
        self.error = error
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelResult:
        self.requests.append(deepcopy(request))
        if self.error is not None:
            raise self.error
        if not self.results:
            raise AssertionError("stub provider ran out of results")
        return self.results.pop(0)


@pytest.fixture
def session(tmp_path: Path):
    engine = make_engine(f"sqlite:///{tmp_path / 'service.sqlite3'}")
    init_db(engine)
    with Session(engine) as db_session:
        yield db_session


def _request(input_assets: list[dict[str, str]] | None = None) -> ModelRequest:
    return ModelRequest(
        provider="fixture",
        endpoint="https://provider.example/v1",
        model="fixture-model",
        prompt_version="prompt-v1",
        messages=[{"role": "user", "content": "Review the traffic scene."}],
        parameters={"temperature": 0, "max_tokens": 32},
        input_assets=input_assets or [],
    )


def _scenario(session: Session, scenario_type: str = "fixture") -> Scenario:
    row = Scenario(source="fixture", scenario_type=scenario_type, configuration={})
    session.add(row)
    session.flush()
    return row


def _result(request_hash: str, *, cache_hit: bool) -> ModelResult:
    return ModelResult(
        request_hash=request_hash,
        cache_hit=cache_hit,
        response={"choices": [{"message": {"content": '{"event": "normal"}'}}]},
    )


def test_asset_scope_and_content_digest_are_checked_before_provider_call(
    session: Session, tmp_path: Path
) -> None:
    scenario = _scenario(session, "requested")
    other_scenario = _scenario(session, "other")
    media_path = tmp_path / "frame.png"
    media_path.write_bytes(b"fixture-frame")
    digest = sha256(media_path.read_bytes()).hexdigest()
    asset = Asset(
        scenario_id=other_scenario.id,
        kind="frame",
        local_path=str(media_path),
        sha256=digest,
    )
    session.add(asset)
    session.flush()

    provider = StubProvider([_result("unused", cache_hit=False)])
    request = _request([{"id": asset.id, "sha256": digest}])
    request.messages = [
        {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": image_to_data_url(media_path)}}],
        }
    ]
    with pytest.raises(ValueError, match="belong"):
        run_model(session, provider, request, scenario_id=scenario.id)
    assert provider.requests == []
    assert session.scalar(select(func.count(ModelRun.id))) == 0

    asset.scenario_id = scenario.id
    session.flush()
    media_path.write_bytes(b"changed-after-registration")
    with pytest.raises(ValueError, match="content changed"):
        run_model(session, provider, request, scenario_id=scenario.id)
    assert provider.requests == []
    assert session.scalar(select(func.count(ModelRun.id))) == 0


def test_run_model_stores_exact_messages_and_input_asset_references(
    session: Session, tmp_path: Path
) -> None:
    scenario = _scenario(session)
    media_path = tmp_path / "frame.png"
    media_path.write_bytes(b"fixture-frame")
    digest = sha256(media_path.read_bytes()).hexdigest()
    asset = Asset(
        scenario_id=scenario.id,
        kind="frame",
        local_path=str(media_path),
        sha256=digest,
    )
    session.add(asset)
    session.flush()
    request = _request([{"id": asset.id, "sha256": digest}])
    request.messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Review this fixture."},
                {"type": "image_url", "image_url": {"url": image_to_data_url(media_path)}},
            ],
        }
    ]
    provider = StubProvider([_result("a" * 64, cache_hit=False)])

    run = run_model(session, provider, request, scenario_id=scenario.id)
    session.commit()

    assert provider.requests == [request]
    assert run.messages == request.messages
    assert run.input_assets == request.input_assets
    assert run.parameters == request.parameters
    assert run.request_hash == "a" * 64
    assert run.cache_hit is False
    assert run.scenario_id == scenario.id


def test_cache_hit_is_recorded_as_a_distinct_model_run(session: Session) -> None:
    scenario = _scenario(session)
    request = _request()
    provider = StubProvider(
        [
            _result("same-request", cache_hit=False),
            _result("same-request", cache_hit=True),
        ]
    )

    first = run_model(session, provider, request, scenario_id=scenario.id)
    second = run_model(session, provider, request, scenario_id=scenario.id)
    session.commit()

    assert first.id != second.id
    rows = session.scalars(select(ModelRun).order_by(ModelRun.id)).all()
    assert {row.request_hash for row in rows} == {"same-request"}
    assert {row.cache_hit for row in rows} == {False, True}
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert len(provider.requests) == 2


def test_failed_provider_call_does_not_create_a_model_run(session: Session) -> None:
    scenario = _scenario(session)
    provider = StubProvider(error=ProviderError("safe provider failure"))

    with pytest.raises(ProviderError, match="safe provider failure"):
        run_model(session, provider, _request(), scenario_id=scenario.id)

    assert len(provider.requests) == 1
    assert session.scalar(select(func.count(ModelRun.id))) == 0


def test_missing_scenario_or_unregistered_image_fails_before_provider_call(session):
    provider = StubProvider()
    with pytest.raises(ValueError, match="scenario does not exist"):
        run_model(session, provider, _request(), scenario_id="missing")
    request = _request()
    request.messages = [
        {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}],
        }
    ]
    with pytest.raises(ValueError, match="exact embedded image bytes"):
        run_model(session, provider, request)
    assert provider.requests == []
