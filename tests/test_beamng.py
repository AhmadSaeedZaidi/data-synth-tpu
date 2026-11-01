import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from data_synth import __main__ as cli
from data_synth.blob_storage import LocalBlobStore
from data_synth.config import Settings
from data_synth.db import init_db, make_engine
from data_synth.models import Annotation, Asset, Fact, Scenario, Verification
from data_synth.simulators import beamng
from data_synth.simulators.capture import import_capture


def manifest(tmp_path, **overrides):
    path = tmp_path / "capture.json"
    (tmp_path / "frame.png").write_bytes(b"operator image fixture")
    values = {
        "schema_version": 1,
        "simulator_product": "BeamNG.drive",
        "simulator_version": "0.39.4",
        "map_name": "operator-declared-map",
        "scenario_type": "non_violation",
        "capture_method": "screenshot",
        "media": [{"path": "frame.png", "kind": "frame", "timestamp_ms": 1000}],
        **overrides,
    }
    path.write_text(json.dumps(values), encoding="utf-8")
    return path


@pytest.fixture
def session():
    engine = make_engine("sqlite:///:memory:")
    init_db(engine)
    with Session(engine) as value:
        yield value
    engine.dispose()


def test_import_keeps_operator_declarations_separate_from_facts(session, tmp_path):
    source = manifest(tmp_path)
    scenario = import_capture(session, source, LocalBlobStore(tmp_path / "blobs"))
    session.commit()
    asset = session.scalar(select(Asset))
    assert scenario.source == "beamng.drive"
    assert scenario.configuration["provenance_authority"] == "operator_declared"
    assert scenario.configuration["independent_telemetry_included"] is False
    assert Path(asset.local_path).read_bytes() == (tmp_path / "frame.png").read_bytes()
    assert asset.metadata_json["media_type"] == "image/png"
    assert asset.timestamp_ms == 1000
    for model in (Fact, Annotation, Verification):
        assert session.scalar(select(func.count(model.id))) == 0


@pytest.mark.parametrize("path", ["../frame.png", "C:/capture.png", "folder\\frame.png", ".env"])
def test_import_rejects_unsafe_or_nonmedia_paths(session, tmp_path, path):
    source = manifest(tmp_path, media=[{"path": path, "kind": "frame", "timestamp_ms": 0}])
    with pytest.raises(ValueError):
        import_capture(session, source, LocalBlobStore(tmp_path / "blobs"))
    assert session.scalar(select(func.count(Scenario.id))) == 0


def test_import_bad_checksum_registers_no_rows(session, tmp_path):
    source = manifest(
        tmp_path,
        media=[
            {
                "path": "frame.png",
                "kind": "frame",
                "timestamp_ms": 0,
                "sha256": "0" * 64,
            }
        ],
    )
    with pytest.raises(RuntimeError, match="checksum"):
        import_capture(session, source, LocalBlobStore(tmp_path / "blobs"))
    assert session.scalar(select(func.count(Scenario.id))) == 0


def test_inspection_does_not_equate_source_markers_with_live_capabilities(tmp_path):
    exe = tmp_path / "Bin64/BeamNG.drive.x64.exe"
    exe.parent.mkdir()
    exe.touch()
    source = tmp_path / "lua/ge/extensions/tech/techCore.lua"
    source.parent.mkdir(parents=True)
    source.write_text(
        "M.handleHello = function(request) end\n"
        "M.handleGetSystemInfo = function(request) end\n"
        "M.handleOpenCamera = function(request)\n"
        "if not ResearchVerifier.isTechLicenseVerified() then return end\nend"
    )
    communication = tmp_path / "lua/common/tech/techCommunication.lua"
    communication.parent.mkdir(parents=True)
    communication.write_text("M.protocolVersion = 'v1.26'")
    report = beamng.inspect_installation(str(tmp_path), "drive")
    assert report["installation_found"] is True
    assert report["camera_license_gate_observed"] is True
    assert report["tech_enabled"] is None
    assert report["live_connection_tested"] is False
    assert report["installed_protocol"] == "v1.26"


def test_probe_never_launches_controls_or_quits(monkeypatch):
    client = Mock()
    client.host_os.return_value = "windows"
    client.tech_enabled.return_value = False
    factory = Mock(return_value=client)
    monkeypatch.setattr(beamng, "version", lambda package: "1.35.1")
    result = beamng._probe_sdk("127.0.0.1", 25252, factory=factory)
    factory.assert_called_once_with("127.0.0.1", 25252, quit_on_close=False)
    client.open.assert_called_once_with(launch=False)
    client.disconnect.assert_called_once()
    client.close.assert_not_called()
    assert result["connected"] is True
    assert result["tech_enabled"] is False
    assert result["camera_capture_tested"] is False


def test_probe_failure_disconnects_and_sanitizes():
    client = Mock()
    client.open.side_effect = RuntimeError("private raw detail")
    with pytest.raises(beamng.BeamNGError) as caught:
        beamng._probe_sdk("127.0.0.1", 25252, factory=Mock(return_value=client))
    assert "private raw detail" not in str(caught.value)
    client.disconnect.assert_called_once()
    client.close.assert_not_called()


def test_probe_sdk_contract():
    from inspect import signature

    BeamNGpy = pytest.importorskip("beamngpy").BeamNGpy
    from beamngpy.connection import Connection

    signature(BeamNGpy).bind("127.0.0.1", 25252, quit_on_close=False)
    signature(BeamNGpy.open).bind(None, launch=False)
    assert callable(BeamNGpy.disconnect)
    assert callable(BeamNGpy.tech_enabled)
    assert callable(BeamNGpy.host_os)
    assert Connection.PROTOCOL_VERSION == "v1.26"


def test_probe_deadline_is_enforced_and_error_sanitized(monkeypatch):
    import subprocess

    runner = Mock(side_effect=subprocess.TimeoutExpired(cmd="private detail", timeout=1))
    monkeypatch.setattr(beamng.subprocess, "run", runner)
    with pytest.raises(beamng.BeamNGError, match="timed out") as caught:
        beamng.probe_connection("127.0.0.1", 25252, timeout=1)
    assert "private detail" not in str(caught.value)
    assert runner.call_args.kwargs["timeout"] == 1


def test_cli_capture_import(monkeypatch, tmp_path, capsys):
    settings = Settings(
        database_url=f"sqlite:///{tmp_path / 'research.db'}", blob_dir=tmp_path / "blobs"
    )
    monkeypatch.setattr(cli.Settings, "load", staticmethod(lambda: settings))
    assert cli.main(["import-capture", str(manifest(tmp_path))]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["scenario_id"]
    assert len(report["asset_ids"]) == 1


def test_cli_probe_requires_live():
    with pytest.raises(SystemExit) as caught:
        cli.main(["beamng-probe"])
    assert caught.value.code == 2


def test_imported_blob_images_enter_two_model_workflow(session, tmp_path):
    from data_synth.providers import ModelResult
    from data_synth.two_model import run_two_model

    scenario = import_capture(session, manifest(tmp_path), LocalBlobStore(tmp_path / "blobs"))
    asset = session.scalar(select(Asset))
    outputs = [
        {
            "subject_id": "ego",
            "timestamp_ms": 1000,
            "claims": {
                "traffic_light": "red",
                "crossed_stop_line": None,
                "event": None,
            },
        },
        {"verdict": "agree", "reasons": ["scripted review"]},
    ]
    provider = Mock()
    provider.complete.side_effect = [
        ModelResult(
            response={"choices": [{"message": {"content": json.dumps(value)}}]},
            request_hash=str(index) * 64,
            cache_hit=False,
        )
        for index, value in enumerate(outputs)
    ]
    result = run_two_model(
        session,
        provider,
        scenario_id=scenario.id,
        asset_ids=[asset.id],
        subject_id="ego",
        timestamp_ms=1000,
        settings=Settings(),
    )
    assert result.status == "needs_review"
    assert provider.complete.call_count == 2
    assert session.scalar(select(func.count(Fact.id))) == 0
