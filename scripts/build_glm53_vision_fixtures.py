#!/usr/bin/env python3
"""Package inspected BeamNG frames separately so notebooks stay below 1 MB."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

from build_glm53_engine_dataset import archive


def build(root, plan_path, output):
    root = Path(root).resolve()
    manifest = json.loads(Path(plan_path).read_text())
    files = {}
    for entry in manifest["images"]:
        relative = PurePosixPath(entry["source"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("fixture source must be relative to the capture root")
        source = root.joinpath(*relative.parts).resolve()
        if not source.is_relative_to(root):
            raise ValueError("fixture source escapes the capture root")
        data = source.read_bytes()
        entry["path"] = f"images/{entry['name']}{source.suffix.lower()}"
        entry["sha256"] = hashlib.sha256(data).hexdigest()
        entry["size"] = len(data)
        files[entry["path"]] = data
    files["manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    payload = archive(files)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    return {
        "path": str(output),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "case_names": [c["name"] for c in manifest["cases"]],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "kaggle/beamng_fixture_plan.json",
    )
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.root, args.plan, args.out), indent=2))
