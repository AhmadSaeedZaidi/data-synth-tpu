"""Offline tests for the explicit opt-in model smoke command."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import SecretStr
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from data_synth import providers, smoke
from data_synth.config import Settings
from data_synth.db import make_engine
from data_synth.models import ModelRun


class FakeOpenAI:
    def __init__(
        self,
        response: dict,
        calls: list[dict],
        failures: list[Exception] | None = None,
        **kwargs,
    ):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self.response = response
        self.calls = calls
        self.failures = failures if failures is not None else []
        self.kwargs = kwargs

    def _create(self, **payload):
        self.calls.append(payload)
        if self.failures:
            raise self.failures.pop(0)
        return self.response


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        database_url=f"sqlite:///{tmp_path / 'smoke.sqlite3'}",
        cache_dir=tmp_path / "cache",
        nvidia_base_url="https://provider.example/v1",
        nvidia_model="fixture-model",
        nvidia_api_key=SecretStr("fixture-secret-key"),
    )


def _completion() -> dict:
    return {
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "9.8 is larger."},
                "finish_reason": "stop",
            }
        ]
    }


def test_live_is_mandatory() -> None:
    with pytest.raises(SystemExit) as error:
        smoke.main([])

    assert error.value.code == 2


def test_help_is_available_without_loading_configuration(capsys, monkeypatch) -> None:
    monkeypatch.setattr(
        smoke.Settings,
        "load",
        staticmethod(lambda: pytest.fail("--help must not load settings")),
    )

    with pytest.raises(SystemExit) as error:
        smoke.main(["--help"])

    assert error.value.code == 0
    assert "--live" in capsys.readouterr().out


def test_mocked_success_replays_cache_and_records_each_model_run(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    settings = _settings(tmp_path)
    monkeypatch.setattr(smoke.Settings, "load", staticmethod(lambda: settings))
    sdk_calls: list[dict] = []
    factory_calls: list[dict] = []
    fake_clients: list[FakeOpenAI] = []

    def fake_factory(**kwargs):
        factory_calls.append(kwargs)
        client = FakeOpenAI(_completion(), sdk_calls)
        fake_clients.append(client)
        return client

    monkeypatch.setattr(providers, "OpenAI", fake_factory)

    assert smoke.main(["--live"]) == 0
    first_output = json.loads(capsys.readouterr().out)
    assert first_output["cache_hit"] is False

    assert smoke.main(["--live"]) == 0
    second_output = json.loads(capsys.readouterr().out)
    assert second_output["cache_hit"] is True
    assert first_output["run_id"] != second_output["run_id"]
    assert len(factory_calls) == 1
    assert len(fake_clients) == 1
    assert len(sdk_calls) == 1
    assert sdk_calls[0]["model"] == "fixture-model"
    assert sdk_calls[0]["messages"][0]["content"].startswith("Which number")

    engine = make_engine(settings.database_url)
    with Session(engine) as session:
        runs = session.scalars(select(ModelRun)).all()
        assert len(runs) == 2
        assert {run.cache_hit for run in runs} == {False, True}
        assert session.scalar(select(func.count(ModelRun.id))) == 2


def test_provider_failure_is_sanitized_and_does_not_echo_secret(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    settings = _settings(tmp_path)
    monkeypatch.setattr(smoke.Settings, "load", staticmethod(lambda: settings))
    secret = "provider-secret-that-must-not-leak"
    sdk_calls: list[dict] = []

    def fake_factory(**kwargs):
        return FakeOpenAI(_completion(), sdk_calls, failures=[RuntimeError(secret)])

    monkeypatch.setattr(providers, "OpenAI", fake_factory)

    assert smoke.main(["--live"]) == 1
    output = capsys.readouterr().out
    assert "model completion request failed" in output
    assert secret not in output
    assert len(sdk_calls) == 1

    engine = make_engine(settings.database_url)
    with Session(engine) as session:
        assert session.scalar(select(func.count(ModelRun.id))) == 0
