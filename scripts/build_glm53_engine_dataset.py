#!/usr/bin/env python3
"""Stage a minimal notebook-matched dataset, optionally retaining legacy files."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from io import BytesIO
from pathlib import Path, PurePosixPath

from build_glm53_offload_notebook import NOTEBOOK_PATH, REPO_ROOT, _archive_sources


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def archive(files: dict[str, bytes]) -> bytes:
    result = BytesIO()
    with zipfile.ZipFile(result, "w", compression=zipfile.ZIP_DEFLATED) as target:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            target.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return result.getvalue()


def build(
    legacy_zip: Path | None,
    out: Path,
    previous_version: int,
    *,
    notebook_path: Path = NOTEBOOK_PATH,
    beamng_bundle: Path | None = None,
) -> dict:
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Refusing to mix a dataset with existing files: {out}")
    legacy = {}
    if legacy_zip is not None:
        with zipfile.ZipFile(legacy_zip) as source:
            for name in source.namelist():
                if name.endswith("/"):
                    continue
                path = PurePosixPath(name)
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError(f"Unsafe legacy path: {name}")
                if name in legacy:
                    raise ValueError(f"Duplicate legacy path: {name}")
                legacy[name] = source.read(name)

    runtime, records = _archive_sources()
    notebook = json.loads(notebook_path.read_text())
    metadata = notebook["metadata"]["glm53_offload"]
    if metadata["bundle_sha256"] != digest(runtime):
        raise ValueError("Rebuild the notebook before staging its dataset")
    with zipfile.ZipFile(BytesIO(runtime)) as source:
        manifest_hash = digest(source.read("bundle-manifest.json"))
    if metadata["source_manifest_sha256"] != manifest_hash:
        raise ValueError("Notebook source manifest differs from dataset runtime")

    out.mkdir(parents=True, exist_ok=True)
    engine = {}
    uploaded = {}
    for name, data in legacy.items():
        path = PurePosixPath(name)
        if len(path.parts) == 1:
            uploaded[name] = data
        elif path.parts[0] == "engine":
            # Kaggle expands engine.zip under engine/. Preserve the v19 paths.
            engine[str(PurePosixPath(*path.parts[1:]))] = data
        else:
            raise ValueError(f"Unexpected legacy directory; inspect before packaging: {name}")
    if engine:
        uploaded["engine.zip"] = archive(engine)
    uploaded["glm53-offload-runtime.bundle"] = runtime
    if metadata.get("beamng_fixture_sha256"):
        if beamng_bundle is None:
            raise ValueError("This notebook requires --beamng-bundle")
        scene = beamng_bundle.read_bytes()
        if digest(scene) != metadata["beamng_fixture_sha256"]:
            raise ValueError("BeamNG fixtures differ from the notebook")
        uploaded["glm53-beamng-fixtures.bundle"] = scene
    test_names = (
        "conftest.py",
        "test_gguf_quant.py",
        "test_integer_kernel.py",
        "test_weights.py",
        "test_run.py",
        "test_offload_engine.py",
        "test_expert_cache.py",
        "test_routing.py",
        "test_vision.py",
        "test_text_reference.py",
        "test_monitoring.py",
        "test_tracking_check.py",
        "test_scene_fixtures.py",
        "test_supervisor.py",
        "test_comparison.py",
    )
    uploaded["glm53-offload-tests.bundle"] = archive(
        {f"tests/{name}": (REPO_ROOT / "kaggle/tests" / name).read_bytes() for name in test_names}
    )
    uploaded["README.md"] = b"""# GLM-5.3-Flash source engine dataset

This dataset contains source code and small test fixtures, not model weights.
`glm53-offload-runtime.bundle` is a ZIP with a deliberate `.bundle` extension
to preserve its exact bytes on Kaggle. It matches the notebook's embedded
runtime byte for byte. The notebook checks SHA-256, extracts the runtime to
working storage, then prepends that directory and its `vendor` subdirectory to
the worker's PYTHONPATH. A mismatched attached bundle stops before downloading
weights. If the dataset is absent, the notebook uses its embedded copy.

The active runtime provides Q6/Q4 host RAM expert offload, native vision and
joint decode batches of 1, 4 and 8. The default notebook compares Q4_K_XL then
Q6_K_XL in separate workers under one six-hour session budget. Checkpoint
revisions and the upstream MIT license are included in the runtime bundle.
Direct resident-cache tile reads avoid an explicit compact-table copy. Bounded
integer-dot and real-Q4_K projection diagnostics preserve GGUF affine metadata;
integer activation math is experimental and not enabled in model generation.
Fixed-token matched image-state replays compare the same work at batches 1/4/8.
Completed replay groups are checkpointed separately. A post-warmup image
diagnostic samples sparse-layer timings and a native device trace outside
throughput measurements. A budget-aware higher-resolution road-text retry
preserves the original quality gate.

The notebook uses glm53_offload and vendor/glm53 from its verified bundle.
By default this dataset contains only the active runtime, tests and manifests.
Legacy probes are included only when the builder receives --legacy-zip.

For CPU verification, extract the runtime and test bundles into the same fresh
directory using Python's zipfile module, then run `python -m pytest tests` from
that directory with JAX 0.11.2, NumPy, ml-dtypes, Pillow and pytest installed.
Tests cover startup, independent batch state, packed cache eviction, actual
mixed expert formats and numerical vision references. See the repository audit
for measured validation results. Full-checkpoint TPU memory fit, image quality
and throughput require a completed target run.

DATASET_MANIFEST.json records every expected mounted file hash, the runtime
bundle hash, and the matching local notebook artifact hash. Verify the mounted
files after upload; the engine archive is expected to expand under engine/.
"""
    expected = dict(legacy)
    expected.update({name: data for name, data in uploaded.items() if name != "engine.zip"})
    manifest = {
        "format": 1,
        "dataset": "your-kaggle-username/glm53-spill-engine",
        "previous_version": previous_version,
        "preserved_legacy_file_count": len(legacy),
        "runtime_source_file_count": len(records),
        "runtime_bundle_sha256": digest(runtime),
        "runtime_source_manifest_sha256": manifest_hash,
        "notebook_file": notebook_path.name,
        "notebook_sha256": digest(notebook_path.read_bytes()),
        "files": [
            {"path": name, "size": len(data), "sha256": digest(data)}
            for name, data in sorted(expected.items())
        ],
    }
    uploaded["DATASET_MANIFEST.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    for name, data in uploaded.items():
        (out / name).write_bytes(data)
    dataset_metadata = {
        "id": manifest["dataset"],
        "title": "GLM 5.3 Spill Engine Fork",
        "subtitle": "Q6/Q4 RAM expert offload and vision; notebook-matched runtime source bundle",
        "description": uploaded["README.md"].decode(),
        "licenses": [{"name": "unknown"}],
        "keywords": ["tpu"],
    }
    (out / "dataset-metadata.json").write_text(json.dumps(dataset_metadata, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-zip", type=Path, help="Optional prior dataset to retain")
    parser.add_argument("--previous-version", required=True, type=int)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--notebook", type=Path, default=NOTEBOOK_PATH)
    parser.add_argument("--beamng-bundle", type=Path)
    args = parser.parse_args()
    manifest = build(
        args.legacy_zip,
        args.out,
        args.previous_version,
        notebook_path=args.notebook,
        beamng_bundle=args.beamng_bundle,
    )
    print(json.dumps({key: value for key, value in manifest.items() if key != "files"}, indent=2))


if __name__ == "__main__":
    main()
