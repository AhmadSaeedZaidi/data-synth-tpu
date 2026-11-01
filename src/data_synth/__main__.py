"""Research CLI; network storage operations require an explicit --live flag."""

import argparse
import json
from pathlib import Path

from sqlalchemy.orm import Session

from .config import Settings
from .db import init_db, make_engine


def _blob_store(settings: Settings, backend: str):
    from .blob_storage import HFBucketBlobStore, LocalBlobStore, StorageError

    if backend == "local":
        return LocalBlobStore(settings.blob_dir)
    if not settings.hf_bucket or not settings.hf_token:
        raise StorageError("Configure HF_BUCKET and HF_TOKEN privately before using HF storage.")
    return HFBucketBlobStore(settings.hf_bucket, token=settings.hf_token.get_secret_value())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init-db", help="initialize/check the configured database")
    commands.add_parser("demo", help="run offline manual fixtures in a NEW directory").add_argument(
        "directory", type=Path
    )
    commands.add_parser("export", help="export local metadata to a NEW directory").add_argument(
        "directory", type=Path
    )
    commands.add_parser("annotate-run", help="parse a stored model response").add_argument("run_id")
    commands.add_parser("verify", help="verify a stored annotation").add_argument("annotation_id")
    commands.add_parser("beamng-inspect", help="read installation files; no connection or launch")
    commands.add_parser(
        "beamng-probe", help="inspect an already-running SDK endpoint"
    ).add_argument("--live", action="store_true", required=True)
    commands.add_parser(
        "import-capture", help="import operator-recorded BeamNG media"
    ).add_argument("manifest", type=Path)
    two_model = commands.add_parser("verify-media", help="Kimi annotation plus GLM evidence review")
    two_model.add_argument("scenario_id")
    two_model.add_argument("--asset", action="append", dest="asset_ids", required=True)
    two_model.add_argument("--subject", default="ego")
    two_model.add_argument("--timestamp-ms", type=int, required=True)
    two_model.add_argument("--live", action="store_true", required=True)
    commands.add_parser("blob-init", help="create the configured private HF bucket").add_argument(
        "--live", action="store_true", required=True
    )
    for command in ("upload-asset", "download-asset"):
        transfer = commands.add_parser(command, help="transfer an existing checksummed asset")
        transfer.add_argument("asset_id")
        if command == "download-asset":
            transfer.add_argument("destination", type=Path)
        transfer.add_argument("--backend", choices=("hf", "local"), default="hf")
        transfer.add_argument("--live", action="store_true")
    args = parser.parse_args(argv)
    if (
        args.command in ("upload-asset", "download-asset")
        and args.backend == "hf"
        and not args.live
    ):
        parser.error("Hugging Face transfers require --live")
    engine = None
    try:
        if args.command == "demo":
            from .demo import run_demo

            print(json.dumps(run_demo(args.directory), indent=2))
            return 0
        settings = Settings.load()
        if args.command == "beamng-inspect":
            from .simulators.beamng import inspect_installation

            if not settings.beamng_home:
                raise ValueError("BEAMNG_HOME is required")
            report = inspect_installation(settings.beamng_home, settings.beamng_edition)
            print(json.dumps(report, indent=2))
            return 0 if report["installation_found"] else 1
        if args.command == "beamng-probe":
            from .simulators.beamng import probe_connection

            report = probe_connection(
                settings.beamng_host, settings.beamng_port, timeout=settings.beamng_timeout
            )
            print(json.dumps(report, indent=2))
            return 0
        if args.command == "blob-init":
            _blob_store(settings, "hf").create_bucket()
            print("Hugging Face bucket is ready.")
            return 0
        engine = make_engine(settings.database_url)
        init_db(engine)
        with Session(engine) as session:
            if args.command == "init-db":
                print("Database schema version 1 is ready.")
            elif args.command == "import-capture":
                from .blob_storage import LocalBlobStore
                from .simulators.capture import import_capture

                row = import_capture(session, args.manifest, LocalBlobStore(settings.blob_dir))
                session.commit()
                print(json.dumps({"scenario_id": row.id, "asset_ids": [a.id for a in row.assets]}))
            elif args.command == "export":
                from .export import export_snapshot

                print(export_snapshot(session, args.directory))
            elif args.command == "annotate-run":
                from .workflow import annotate_run

                row = annotate_run(session, args.run_id)
                session.commit()
                print(json.dumps({"annotation_id": row.id}))
            elif args.command == "verify":
                from .workflow import verify_annotation

                row = verify_annotation(session, args.annotation_id)
                session.commit()
                print(json.dumps({"verification_id": row.id, "status": row.status}))
            elif args.command == "verify-media":
                from .cache import ResponseCache
                from .providers import OpenAIChatProvider
                from .two_model import run_two_model

                if not settings.nvidia_api_key:
                    raise ValueError("NVIDIA credentials are required")
                provider = OpenAIChatProvider(
                    api_key=settings.nvidia_api_key.get_secret_value(),
                    cache=ResponseCache(settings.cache_dir),
                    endpoint=settings.nvidia_base_url,
                )
                row = run_two_model(
                    session,
                    provider,
                    scenario_id=args.scenario_id,
                    asset_ids=args.asset_ids,
                    subject_id=args.subject,
                    timestamp_ms=args.timestamp_ms,
                    settings=settings,
                )
                session.commit()
                print(json.dumps({"verification_id": row.id, "status": row.status}))
            elif args.command in ("upload-asset", "download-asset"):
                from .storage_service import download_asset, upload_asset

                store = _blob_store(settings, args.backend)
                if args.command == "upload-asset":
                    ref = upload_asset(session, args.asset_id, store)
                    result = {"uri": ref.uri, "sha256": ref.sha256, "size_bytes": ref.size_bytes}
                else:
                    path = download_asset(session, args.asset_id, args.destination, store)
                    result = {"local_path": str(path)}
                session.commit()
                print(json.dumps(result))
        return 0
    except FileExistsError:
        print("Destination already exists; choose a new destination.")
        return 1
    except Exception:
        print("Operation failed. Check configuration, record IDs, schema, and file paths.")
        return 1
    finally:
        if engine is not None:
            engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
