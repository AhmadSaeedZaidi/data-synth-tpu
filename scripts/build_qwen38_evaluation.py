"""Freeze all GLM controls, exact BeamNG pipeline inputs and SFT format tests."""

import argparse
import base64
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kaggle"))

from glm53_offload import probe_images, quality  # noqa: E402
from glm53_offload.scene_fixtures import load_scene_bundle  # noqa: E402
from qwen38_tpu.evaluation.bundle import load_bundle  # noqa: E402
from qwen38_tpu.evaluation.fixtures import extra_fixtures, text_controls  # noqa: E402
from qwen38_tpu.evaluation.pair_review_reference import make_pair_frames_prompt  # noqa: E402

WINDOWS_DATA = Path("data")  # Override the input paths for your local captures.
BEAM_SHA = "129352904eed775b2ea9820245cb1613b36d67ccf5ed31b75062e7f342629840"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def json_hash(value):
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def verified_blob(root, ref):
    path = (root / ref["path"]).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("snapshot_path_escape")
    blob = path.read_bytes()
    if len(blob) != ref["size_bytes"] or sha(blob) != ref["sha256"]:
        raise ValueError("snapshot_blob_differs")
    return blob


def pipeline_cases(snapshot, protocol_path):
    protocol_blob = protocol_path.read_bytes()
    protocol = json.loads(protocol_blob)
    episodes_blob = (snapshot / "episodes.jsonl").read_bytes()
    episodes = {row["episode_id"]: row for row in map(json.loads, episodes_blob.splitlines())}
    cells = [c for c in protocol["cells"] if c["teacher"] == "glm-flash"]
    if len(cells) != 4 or protocol["reference_status"] != "independent_event_labels_missing":
        raise ValueError("unexpected_comparison_protocol")
    images, cases = {}, []
    for cell in cells:
        episode = episodes[cell["scenario_id"]]
        frames = episode["public_student_input"]["frames"]
        ref = episode["private_verifier_evidence"]["capture_report"]
        if ref["sha256"] != cell["report_sha256"]:
            raise ValueError("capture_report_provenance_differs")
        report = json.loads(verified_blob(snapshot, ref))
        indices = [f["frame_index"] for f in frames]
        if indices != cell["frame_indices"] or [f["image"]["sha256"] for f in frames] != [
            a["sha256"] for a in cell["input_assets"]
        ]:
            raise ValueError("glm_selected_frames_differ")
        samples = {sample["frame_index"]: sample for sample in report["samples"]}
        if any(
            f["timestamp_ms"] != samples[f["frame_index"]]["simulation_time_ms"] for f in frames
        ):
            raise ValueError("glm_frame_timestamps_differ")
        grounding = None
        if cell["condition"] == "damage_grounded":
            first, last = (
                samples[indices[0]]["before_camera"],
                samples[indices[-1]]["before_camera"],
            )
            grounding = dict(
                red_first=first["rear"]["damage"]["damage"],
                red_last=last["rear"]["damage"]["damage"],
                blue_first=first["front"]["damage"]["damage"],
                blue_last=last["front"]["damage"]["damage"],
            )
        times = [f["timestamp_ms"] / 1000 for f in frames]
        prompt = (
            make_pair_frames_prompt(times, grounding)
            + f"\nInput provenance SHA-256: {ref['sha256']}."
        )
        original, ordered, names = (
            [{"type": "text", "text": prompt}],
            [{"type": "text", "text": prompt}],
            [],
        )
        for frame in frames:
            name = "pipeline_" + frame["image"]["sha256"]
            payload = verified_blob(snapshot, frame["image"])
            images[name] = payload
            names.append(name)
            text = dict(
                type="text", text=f"Simulation time {frame['timestamp_ms'] / 1000:.2f} seconds:"
            )
            original.extend(
                [
                    text,
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,"
                            + base64.b64encode(payload).decode("ascii")
                        },
                    },
                ]
            )
            ordered.extend([text, {"type": "image", "image": name}])
        fingerprint = json_hash(
            dict(
                prompt_version=cell["prompt_version"],
                messages=[dict(role="user", content=original)],
                input_assets=cell["input_assets"],
            )
        )
        if fingerprint != cell["input_sha256"]:
            raise ValueError("exact_glm_input_hash_differs")
        cases.append(
            dict(
                name=f"pipeline_glm_cell_{cell['cell_id']:04d}",
                group="beamng_pipeline_review",
                images=names,
                content=ordered,
                prompt=prompt,
                asserted=True,
                match=dict(kind="pair_markdown", duration_seconds=times[-1]),
                max_new_tokens=3072,
                min_pixels=65536,
                max_pixels=589824,
                provenance=dict(
                    cell_id=cell["cell_id"],
                    episode_id=episode["episode_id"],
                    run_id=episode["run_id"],
                    condition=cell["condition"],
                    frame_indices=indices,
                    timestamps_ms=[f["timestamp_ms"] for f in frames],
                    report_sha256=ref["sha256"],
                    input_sha256=cell["input_sha256"],
                    reconstructed_input_sha256=fingerprint,
                    input_assets=cell["input_assets"],
                    prompt_version=cell["prompt_version"],
                    grounding_given=grounding,
                    reference_status=protocol["reference_status"],
                    training_ready=False,
                ),
            )
        )
    return (
        images,
        cases,
        dict(
            glm_pipeline_cells=[c["cell_id"] for c in cells],
            protocol_sha256=sha(protocol_blob),
            episodes_jsonl_sha256=sha(episodes_blob),
            source_task=protocol["task"],
            comparison_limitations="Native image resolution, context and thinking budgets "
            "differ from NIM. "
            "Original image bytes, prompt, frame order, timestamps and telemetry are preserved. "
            "Event correctness is unlabelled; schema-only passes are not training approval.",
        ),
    )


def build(out, *, snapshot, protocol, beamng_bundle):
    images = probe_images.fixtures()
    cases = quality.cases(images)
    real_images, real_cases, _, _ = load_scene_bundle(beamng_bundle, BEAM_SHA)
    images.update(real_images)
    cases.extend(real_cases)
    # GLM also retried failed road OCR with a larger image-token cap. Keep that
    # control separate so it cannot replace or rescue the original gate.
    road = next(c for c in real_cases if c["name"] == "beamng_city_road_text")
    cases.append(
        {
            **road,
            "name": "beamng_city_road_text_high_resolution",
            "group": "beamng_resolution_control",
            "max_pixels": 1048576,
            "max_new_tokens": 2048,
            "provenance": {
                "control_for": road["name"],
                "scope": "unchanged image, prompt and label; native higher resolution",
            },
        }
    )
    cases.extend(text_controls())
    extra_images, extra_cases = extra_fixtures()
    images.update(extra_images)
    cases.extend(extra_cases)
    pipeline_images, pipeline, coverage = pipeline_cases(snapshot, protocol)
    images.update(pipeline_images)
    cases.extend(pipeline)
    manifest = dict(
        format_version=1,
        cases=cases,
        coverage=coverage,
        images={
            name: dict(path=f"images/{name}.image", sha256=sha(blob), bytes=len(blob))
            for name, blob in images.items()
        },
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        members = {
            "manifest.json": (json.dumps(manifest, indent=2, allow_nan=False) + "\n").encode()
        }
        members.update({manifest["images"][name]["path"]: blob for name, blob in images.items()})
        for name, blob in sorted(members.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 12, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, blob)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(buffer.getvalue())
    load_bundle(out)
    return dict(
        cases=len(cases),
        asserted=sum(c["asserted"] for c in cases),
        images=len(images),
        sha256=sha(buffer.getvalue()),
        glm_pipeline_cells=coverage["glm_pipeline_cells"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--snapshot", type=Path, default=WINDOWS_DATA / "pair-clean-rgb-final-snapshot-20261004"
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=WINDOWS_DATA / "nvidia-flash-comparison-low-20261008/protocol.json",
    )
    parser.add_argument(
        "--beamng-bundle",
        type=Path,
        default=ROOT / "output/current/inputs/qwen38-beamng-fixtures.bundle",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                args.output,
                snapshot=args.snapshot,
                protocol=args.protocol,
                beamng_bundle=args.beamng_bundle,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
