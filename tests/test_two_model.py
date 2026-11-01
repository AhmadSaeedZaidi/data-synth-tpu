import hashlib
import json

import httpx
import pytest
from openai import OpenAI
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from data_synth.cache import ResponseCache
from data_synth.config import Settings
from data_synth.db import init_db, make_engine
from data_synth.models import Annotation, Asset, Fact, ModelRun, Scenario
from data_synth.providers import OpenAIChatProvider
from data_synth.two_model import run_two_model
from data_synth.workflow import AnnotationError

CLAIMS = {"traffic_light": "red", "crossed_stop_line": True, "event": "red_light_running"}


@pytest.fixture
def setup(tmp_path):
    engine = make_engine("sqlite:///:memory:")
    init_db(engine)
    with Session(engine) as session:
        scenario = Scenario(source="manual", scenario_type="fixture")
        session.add(scenario)
        session.flush()
        image = tmp_path / "frame.png"
        image.write_bytes(b"fixture image")
        asset = Asset(
            scenario_id=scenario.id,
            kind="frame",
            local_path=str(image),
            sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
            timestamp_ms=1000,
        )
        session.add(asset)
        session.flush()
        yield session, scenario, asset, Settings(cache_dir=tmp_path / "cache")
    engine.dispose()


def provider(settings, verdict="agree", malformed=False, scope_change=False):
    requests = []

    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        if payload["model"] == settings.annotator_model:
            result = {
                "subject_id": "other" if scope_change else "ego",
                "timestamp_ms": 1000,
                "claims": CLAIMS,
                "confidence": 0.8,
            }
        else:
            result = {"verdict": verdict, "reasons": ["fixture review"]}
        content = "invalid JSON" if malformed and len(requests) == 2 else json.dumps(result)
        return httpx.Response(
            200,
            json={
                "id": "fixture",
                "object": "chat.completion",
                "created": 0,
                "model": payload["model"],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": content},
                    }
                ],
            },
        )

    client = OpenAI(
        api_key="fake",
        base_url=settings.nvidia_base_url,
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
    )
    return OpenAIChatProvider(
        api_key="fake", cache=ResponseCache(settings.cache_dir), client=client
    ), requests


def evidence(session, scenario, *, conflict=False):
    session.add(
        Fact(
            scenario_id=scenario.id,
            source="manual",
            source_version="fixture-v1",
            subject_id="ego",
            timestamp_ms=1000,
            values={**CLAIMS, "traffic_light": "green"} if conflict else CLAIMS,
        )
    )
    session.flush()


def invoke(setup, adapter):
    session, scenario, asset, settings = setup
    return run_two_model(
        session,
        adapter,
        scenario_id=scenario.id,
        asset_ids=[asset.id],
        subject_id="ego",
        timestamp_ms=1000,
        settings=settings,
    )


@pytest.mark.parametrize(
    "with_facts,conflict,verdict,malformed,expected",
    [
        (True, False, "agree", False, "accepted"),
        (False, False, "agree", False, "needs_review"),
        (True, True, "agree", False, "rejected"),
        (True, False, "disagree", False, "needs_review"),
        (True, False, "uncertain", False, "needs_review"),
        (True, False, "agree", True, "needs_review"),
    ],
)
def test_two_model_policy_and_provenance(setup, with_facts, conflict, verdict, malformed, expected):
    session, scenario, asset, settings = setup
    if with_facts:
        evidence(session, scenario, conflict=conflict)
    adapter, calls = provider(settings, verdict=verdict, malformed=malformed)
    decision = invoke(setup, adapter)
    session.commit()
    assert decision.status == expected
    assert [call["model"] for call in calls] == [settings.annotator_model, settings.reviewer_model]
    runs = session.scalars(select(ModelRun).order_by(ModelRun.created_at)).all()
    assert len(runs) == 2
    assert runs[0].input_assets == [{"id": asset.id, "sha256": asset.sha256}]
    assert runs[1].input_assets == []
    assert session.scalar(select(func.count(Fact.id))) == int(with_facts)
    assert session.scalar(select(func.count(Annotation.id))) == 1
    assert decision.reasons[-1]["reviewer_run_id"] == runs[1].id
    assert decision.reasons[-1]["reviewer_observed_images"] is False
    assert "independent_evidence" not in json.dumps(calls[0])


def test_repeat_workflow_reuses_both_api_responses(setup):
    session, scenario, _, settings = setup
    evidence(session, scenario)
    adapter, calls = provider(settings)
    first = invoke(setup, adapter)
    second = invoke(setup, adapter)
    assert len(calls) == 2
    assert first.id != second.id
    runs = session.scalars(select(ModelRun)).all()
    assert sum(run.cache_hit for run in runs) == 2


def test_changed_media_is_rejected_before_model_call(setup):
    _, _, asset, settings = setup
    from pathlib import Path

    Path(asset.local_path).write_bytes(b"changed")
    adapter, calls = provider(settings)
    with pytest.raises(ValueError, match="exact embedded image bytes"):
        invoke(setup, adapter)
    assert not calls


def test_changed_scope_is_not_verified(setup):
    adapter, calls = provider(setup[-1], scope_change=True)
    with pytest.raises(AnnotationError, match="subject or timestamp"):
        invoke(setup, adapter)
    assert len(calls) == 1


def test_duplicate_models_fail_before_calls(setup):
    settings = setup[-1].model_copy(update={"reviewer_model": setup[-1].annotator_model})
    adapter, calls = provider(settings)
    with pytest.raises(ValueError, match="distinct configured models"):
        invoke((*setup[:-1], settings), adapter)
    assert not calls
