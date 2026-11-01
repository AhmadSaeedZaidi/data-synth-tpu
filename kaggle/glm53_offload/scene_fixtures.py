"""Small, checksummed real-scene fixtures, separate from the embedded runtime."""

import hashlib
import json
import zipfile
from pathlib import Path, PurePosixPath

from . import quality


def load_scene_bundle(path, expected_sha256=None):
    path = Path(path)
    if path.stat().st_size > 32 << 20:
        raise ValueError("scene fixture bundle exceeds 32 MiB")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected_sha256 and digest != expected_sha256:
        raise ValueError("scene fixture bundle SHA-256 differs from notebook")
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if sum(x.file_size for x in members) > 32 << 20:
            raise ValueError("expanded scene fixtures exceed 32 MiB")
        names = [x.filename for x in members]
        if len(names) != len(set(names)):
            raise ValueError("duplicate scene fixture archive member")
        if any(PurePosixPath(x).is_absolute() or ".." in PurePosixPath(x).parts for x in names):
            raise ValueError("unsafe scene fixture archive path")
        manifest = json.loads(archive.read("manifest.json"))
        if manifest["format"] != 1 or manifest["label_scope"] != "visible_image_properties":
            raise ValueError("scene fixtures require visible-image labels")
        images = {}
        for entry in manifest["images"]:
            data = archive.read(entry["path"])
            if hashlib.sha256(data).hexdigest() != entry["sha256"]:
                raise ValueError("scene image checksum differs from manifest")
            if entry["name"] in images:
                raise ValueError("duplicate scene image name")
            images[entry["name"]] = data
        cases = manifest["cases"]
        case_names = [case["name"] for case in cases]
        if not cases or len(cases) > 16 or len(case_names) != len(set(case_names)):
            raise ValueError("scene fixture case names must be distinct and bounded")
        for case in cases:
            if (
                not case["name"].startswith("beamng_")
                or not case["asserted"]
                or len(case["images"]) != 1
                or not set(case["images"]) <= set(images)
                or not isinstance(case["prompt"], str)
            ):
                raise ValueError("invalid scene fixture case")
            quality.match(case["match"], "")  # Reject unsupported matchers before model loading.
        expected = {"manifest.json"} | {item["path"] for item in manifest["images"]}
        if set(names) != expected:
            raise ValueError("unexpected scene fixture bundle member")
    return images, cases, manifest, digest
