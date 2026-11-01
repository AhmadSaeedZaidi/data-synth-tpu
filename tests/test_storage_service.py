"""Asset/blob integration: offline round trips, provenance, and failure boundaries."""

import hashlib
import json

import pytest
from sqlalchemy.orm import Session

from data_synth import __main__ as cli
from data_synth.blob_storage import LocalBlobStore, StorageError
from data_synth.config import Settings
from data_synth.db import init_db, make_engine
from data_synth.export import export_snapshot
from data_synth.models import Asset, Scenario
from data_synth.storage_service import download_asset, upload_asset


@pytest.fixture
def asset_db(tmp_path):
    settings = Settings(
        database_url=f"sqlite:///{tmp_path / 'research.db'}", blob_dir=tmp_path / "blobs"
    )
    engine = make_engine(settings.database_url)
    init_db(engine)
    media = tmp_path / "capture.mp4"
    media.write_bytes(b"synthetic fixture bytes, not a real video")
    with Session(engine) as session:
        scenario = Scenario(source="manual", scenario_type="fixture")
        session.add(scenario)
        session.flush()
        asset = Asset(
            scenario_id=scenario.id,
            kind="video",
            local_path=str(media),
            sha256=hashlib.sha256(media.read_bytes()).hexdigest(),
            metadata_json={"fixture_only": True},
        )
        session.add(asset)
        session.commit()
        yield settings, session, asset
    engine.dispose()


def test_round_trip_persists_blob_provenance_and_export(asset_db, tmp_path):
    settings, session, asset = asset_db
    store = LocalBlobStore(settings.blob_dir)
    ref = upload_asset(session, asset.id, store)
    session.commit()
    session.expire_all()
    assert asset.remote_uri == ref.uri
    assert asset.metadata_json["fixture_only"] is True
    destination = tmp_path / "restored.mp4"
    download_asset(session, asset.id, destination, store)
    session.commit()
    assert hashlib.sha256(destination.read_bytes()).hexdigest() == asset.sha256
    assert asset.local_path == str(destination.resolve())
    manifest = export_snapshot(session, tmp_path / "snapshot")
    record = json.loads(manifest.read_text().strip())
    assert record["assets"][0]["remote_uri"] == ref.uri
    assert record["assets"][0]["metadata_json"]["blob"]["size_bytes"] == ref.size_bytes
    assert record["facts"] == record["annotations"] == record["verifications"] == []


def test_failed_upload_preserves_asset(asset_db):
    settings, session, asset = asset_db
    asset.sha256 = "0" * 64
    session.commit()
    with pytest.raises(StorageError):
        upload_asset(session, asset.id, LocalBlobStore(settings.blob_dir))
    assert asset.remote_uri is None
    assert asset.metadata_json == {"fixture_only": True}


def test_failed_download_preserves_original_local_path(asset_db, tmp_path):
    settings, session, asset = asset_db
    store = LocalBlobStore(settings.blob_dir)
    upload_asset(session, asset.id, store)
    original = asset.local_path
    asset.metadata_json = {"blob": {"sha256": asset.sha256, "size_bytes": 9999}}
    with pytest.raises(StorageError):
        download_asset(session, asset.id, tmp_path / "invalid.mp4", store)
    assert asset.local_path == original
    assert not (tmp_path / "invalid.mp4").exists()


def test_local_cli_round_trip(asset_db, tmp_path, monkeypatch, capsys):
    settings, session, asset = asset_db
    monkeypatch.setattr(cli.Settings, "load", staticmethod(lambda: settings))
    assert cli.main(["upload-asset", asset.id, "--backend", "local"]) == 0
    assert json.loads(capsys.readouterr().out)["uri"].startswith("local://")
    destination = tmp_path / "cli-restored.mp4"
    assert cli.main(["download-asset", asset.id, str(destination), "--backend", "local"]) == 0
    session.expire_all()
    assert asset.local_path == str(destination.resolve())


def test_hf_cli_requires_explicit_live_flag():
    with pytest.raises(SystemExit) as caught:
        cli.main(["upload-asset", "unused"])
    assert caught.value.code == 2


def test_bucket_init_missing_credentials_is_sanitized(monkeypatch, capsys):
    monkeypatch.setattr(cli.Settings, "load", staticmethod(lambda: Settings()))
    assert cli.main(["blob-init", "--live"]) == 1
    assert capsys.readouterr().out.startswith("Operation failed.")
