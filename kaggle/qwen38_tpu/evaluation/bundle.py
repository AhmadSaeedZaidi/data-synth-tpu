"""Bounded, checksummed evaluation bundle with an explicit coverage contract."""

import hashlib
import io
import json
import zipfile
from pathlib import Path, PurePosixPath

from glm53_offload import quality

from .scoring import validate_spec

LIMIT = 32 << 20
TEXT_CONTROLS = {
    "text_collision_time",
    "text_braking_distance",
    "text_telemetry_order",
    "text_frame_sampling",
}


def validate_coverage(cases, coverage):
    names = [c["name"] for c in cases]
    required = (
        {c["name"] for c in quality.cases()}
        | TEXT_CONTROLS
        | {
            "beamng_grid_count",
            "beamng_contact_blue_side",
            "beamng_city_road_text",
            "beamng_roadside_count",
        }
    )
    if len(names) != len(set(names)) or not required <= set(names):
        raise ValueError("missing_or_duplicate_glm_coverage")
    pipeline = [c for c in cases if c["group"] == "beamng_pipeline_review"]
    cells = coverage["glm_pipeline_cells"]
    if len(cells) != 4 or len(pipeline) != 4:
        raise ValueError("missing_glm_pipeline_cells")
    if {c["provenance"]["cell_id"] for c in pipeline} != set(cells):
        raise ValueError("pipeline_cell_coverage_differs")
    for case in pipeline:
        p = case["provenance"]
        if len(case["images"]) != 8 or p["frame_indices"] != [0, 11, 23, 34, 45, 56, 68, 79]:
            raise ValueError("incomplete_pipeline_frame_selection")
        if p["input_sha256"] != p["reconstructed_input_sha256"]:
            raise ValueError("original_glm_input_hash_differs")
        if case["match"]["kind"] != "pair_markdown":
            raise ValueError("pipeline_requires_original_parser")
    if {c["provenance"]["condition"] for c in pipeline} != {"image_only", "damage_grounded"}:
        raise ValueError("missing_pipeline_condition")
    if len({c["provenance"]["episode_id"] for c in pipeline}) != 2:
        raise ValueError("missing_pipeline_episode")
    if not any(c["group"] == "general_spatial" for c in cases) or not any(
        c["group"] == "sft_formatting" for c in cases
    ):
        raise ValueError("missing_spatial_or_formatting_coverage")
    control = next(c for c in cases if c["name"] == "no_image_control")
    if control["asserted"] or control["images"]:
        raise ValueError("invalid_no_image_control")


def load_bundle(path, expected_sha256=None):
    data = Path(path).read_bytes()
    if len(data) > LIMIT:
        raise ValueError("evaluation_archive_too_large")
    sha = hashlib.sha256(data).hexdigest()
    if expected_sha256 and sha != expected_sha256:
        raise ValueError("evaluation_checksum_differs")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        members = archive.infolist()
        names = [x.filename for x in members]
        if len(names) != len(set(names)) or sum(x.file_size for x in members) > LIMIT:
            raise ValueError("duplicate_or_oversized_evaluation_archive")
        for name in names:
            p = PurePosixPath(name)
            if p.is_absolute() or ".." in p.parts or "\\" in name:
                raise ValueError("unsafe_evaluation_member")
        manifest = json.loads(archive.read("manifest.json"))
        if manifest["format_version"] != 1 or not 1 <= len(manifest["cases"]) <= 128:
            raise ValueError("unsupported_evaluation_manifest")
        refs = manifest["images"]
        if set(names) != {"manifest.json"} | {r["path"] for r in refs.values()}:
            raise ValueError("evaluation_members_differ")
        images = {}
        for name, ref in refs.items():
            blob = archive.read(ref["path"])
            if hashlib.sha256(blob).hexdigest() != ref["sha256"]:
                raise ValueError("evaluation_image_checksum_differs")
            images[name] = blob
    cases = manifest["cases"]
    for case in cases:
        if not set(case["images"]) <= images.keys() or len(case["images"]) > 8:
            raise ValueError("missing_or_excess_images")
        if not isinstance(case["prompt"], str) or type(case["asserted"]) is not bool:
            raise ValueError("invalid_evaluation_case")
        if "content" in case:
            ordered = [x["image"] for x in case["content"] if x["type"] == "image"]
            if ordered != case["images"] or any(
                x["type"] not in {"image", "text"} for x in case["content"]
            ):
                raise ValueError("image_order_differs")
        if (
            type(case.get("max_new_tokens", 512)) is not int
            or not 1 <= case.get("max_new_tokens", 512) <= 4096
        ):
            raise ValueError("invalid_output_budget")
        validate_spec(case["match"])
    validate_coverage(cases, manifest["coverage"])
    return images, cases, manifest, sha
