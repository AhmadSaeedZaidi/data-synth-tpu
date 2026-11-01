"""Reject altered capture inputs before spending model-loading time."""

import hashlib
import json
import zipfile

import pytest
from glm53_offload.scene_fixtures import load_scene_bundle


def bundle(tmp_path, *, bad_image=False, extra=None):
    image = b"checksummed capture bytes"
    manifest = {
        "format": 1,
        "label_scope": "visible_image_properties",
        "label_authority": "inspection",
        "images": [
            {
                "name": "beamng_frame",
                "path": "images/frame.png",
                "sha256": hashlib.sha256(image).hexdigest(),
            }
        ],
        "cases": [
            {
                "name": "beamng_count",
                "asserted": True,
                "group": "beamng_count",
                "images": ["beamng_frame"],
                "prompt": "Count visible cars.",
                "match": {"kind": "first_int", "value": 2},
            }
        ],
    }
    path = tmp_path / "scenes.bundle"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        archive.writestr("images/frame.png", b"altered" if bad_image else image)
        if extra:
            archive.writestr(extra, b"unexpected")
    return path


def test_scene_hashes_and_visible_labels_are_preserved(tmp_path):
    path = bundle(tmp_path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    images, cases, manifest, got = load_scene_bundle(path, digest)
    assert got == digest
    assert images == {"beamng_frame": b"checksummed capture bytes"}
    assert cases[0]["match"]["value"] == 2
    assert manifest["label_scope"] == "visible_image_properties"


def test_altered_scene_archive_or_image_fails_integrity_check(tmp_path):
    with pytest.raises(ValueError, match="SHA-256"):
        load_scene_bundle(bundle(tmp_path), "wrong digest")
    with pytest.raises(ValueError, match="image checksum"):
        load_scene_bundle(bundle(tmp_path, bad_image=True))


@pytest.mark.parametrize("extra", ["../escape.png", "/absolute.png", "unexpected.png"])
def test_unexpected_or_unsafe_archive_members_are_rejected(tmp_path, extra):
    with pytest.raises(ValueError, match="unsafe|unexpected"):
        load_scene_bundle(bundle(tmp_path, extra=extra))
