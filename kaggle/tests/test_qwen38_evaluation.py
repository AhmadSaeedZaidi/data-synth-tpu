"""Output contracts, original GLM coverage, frame order and decode masks."""

import copy
import importlib.util
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from glm53_offload import quality
from qwen38_tpu.evaluation.bundle import load_bundle, validate_coverage
from qwen38_tpu.evaluation.fixtures import extra_fixtures, text_controls
from qwen38_tpu.evaluation.scoring import match, strict_json

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "output/current/inputs/qwen38-evaluation.bundle"


@pytest.fixture
def full_suite():
    if not BUNDLE.exists():
        pytest.skip("local private BeamNG snapshot bundle is not in the repository")
    return load_bundle(BUNDLE)


@pytest.mark.parametrize(
    "answer",
    [
        '{"n":1,"n":1}',
        '{"n":NaN}',
        '{"n":Infinity}',
        '{"n":1e999}',
        '```json\n{"n":1}\n```',
        '{"n":1} extra',
    ],
)
def test_rejects_unsafe_or_non_json_outputs(answer):
    with pytest.raises(ValueError):
        strict_json(answer)


@pytest.mark.parametrize("answer", ['{"n":true}', '{"n":"1"}', '{"n":1.0}', '{"n":1,"extra":0}'])
def test_json_types_and_keys_must_match(answer):
    assert not match(dict(kind="json_exact", expected={"n": 1}), answer)[0]


def test_jsonl_order_blank_lines_and_sft_roles():
    expected = [
        {
            "messages": [
                {"role": "user", "content": "سڑک\nSTOP"},
                {"role": "assistant", "content": "red"},
            ]
        }
    ]
    spec = dict(kind="jsonl_exact", expected=expected)
    good = json.dumps(expected[0], ensure_ascii=True)
    assert match(spec, good)[0]
    for bad in [good + "\n\n", good + "\n{}", good.replace('"assistant"', '"user"')]:
        assert not match(spec, bad)[0]


def test_production_annotation_contract_is_used():
    expected = dict(subject_id="car", timestamp_ms=100, claims={"event": None}, confidence=None)
    spec = dict(kind="annotation_json", expected=expected)
    assert match(spec, json.dumps(expected))[0]
    for change in [
        dict(timestamp_ms=True),
        dict(timestamp_ms="100"),
        dict(timestamp_ms=-1),
        dict(confidence=1.1),
        dict(subject_id=" "),
        dict(extra=1),
    ]:
        assert not match(spec, json.dumps({**expected, **change}))[0]


PAIR = """# Pair review

Vehicle contact: uncertain
First contact time seconds: unknown
Red front damage visible: uncertain
Blue rear damage visible: uncertain
Red stopped short: uncertain

## Reasons
- Sparse frames do not establish contact.
"""


def test_original_pair_parser_scores_schema_without_inventing_truth():
    spec = dict(kind="pair_markdown", duration_seconds=4.0)
    passed, detail = match(spec, PAIR)
    assert passed and detail["schema_valid"]
    assert detail["semantic_valid"] is None
    assert detail["assertion_scope"] == "schema_only" and not detail["training_ready"]
    for bad in [
        "```\n" + PAIR + "```",
        PAIR + "extra",
        PAIR.replace("unknown", "2.0"),
        PAIR.replace("unknown", "5.0").replace(
            "Vehicle contact: uncertain", "Vehicle contact: yes"
        ),
        PAIR.replace("unknown", "2.0")
        .replace("Vehicle contact: uncertain", "Vehicle contact: yes")
        .replace("Red stopped short: uncertain", "Red stopped short: yes"),
    ]:
        assert not match(spec, bad)[0]


def test_constructed_spatial_and_formatting_expected_answers():
    images, cases = extra_fixtures()
    assert len(images) == 8
    assert sum(c["group"] == "general_spatial" for c in cases) == 11
    assert sum(c["group"] == "sft_formatting" for c in cases) == 12
    for c in cases:
        expected = c["match"]["expected"]
        answer = (
            "\n".join(json.dumps(r) for r in expected)
            if c["match"]["kind"] == "jsonl_exact"
            else expected
            if c["match"]["kind"] == "exact"
            else json.dumps(expected)
        )
        assert match(c["match"], answer)[0], c["name"]


def test_all_original_glm_prompts_and_text_controls_are_identical(full_suite):
    _, cases, manifest, _ = full_suite
    assert len(cases) == 49 and sum(c["asserted"] for c in cases) == 48
    names = {c["name"]: c for c in cases}
    for original in quality.cases():
        assert names[original["name"]] == original
    spec = importlib.util.spec_from_file_location(
        "nim_comparison", ROOT / "scripts/compare_glm53_deepseek41_nim.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _, nim_cases, _, _ = module.fixtures()
    for original in nim_cases:
        assert names[original["name"]] == original
    assert text_controls() == [c for c in nim_cases if c["group"] == "text_control"]
    assert manifest["coverage"]["glm_pipeline_cells"] == [1, 2, 5, 6]


def test_original_beamng_cells_preserve_eight_frames_and_hashes(full_suite):
    images, cases, manifest, _ = full_suite
    pipeline = [c for c in cases if c["group"] == "beamng_pipeline_review"]
    for case in pipeline:
        provenance = case["provenance"]
        assert provenance["input_sha256"] == provenance["reconstructed_input_sha256"]
        assert provenance["timestamps_ms"] == [50, 600, 1200, 1750, 2300, 2850, 3450, 4000]
        assert len(case["images"]) == 8 and len(case["content"]) == 17
        assert case["content"][0]["text"] == case["prompt"]
        for index, name in enumerate(case["images"]):
            assert manifest["images"][name]["sha256"] == provenance["input_assets"][index]["sha256"]
            assert images[name].startswith(b"\x89PNG")
    broken = copy.deepcopy(cases)
    broken[-1]["images"].pop()
    with pytest.raises(ValueError, match="incomplete_pipeline"):
        validate_coverage(broken, manifest["coverage"])
    with pytest.raises(ValueError, match="glm_coverage"):
        validate_coverage(cases[1:], manifest["coverage"])


def test_bundle_rejects_corruption_and_extra_members(full_suite, tmp_path):
    _, _, _, sha = full_suite
    data = BUNDLE.read_bytes()
    path = tmp_path / "bad.bundle"
    path.write_bytes(data + b"changed")
    with pytest.raises(ValueError, match="checksum"):
        load_bundle(path, sha)
    path.write_bytes(data)
    with zipfile.ZipFile(path, "a") as archive:
        archive.writestr("extra", "unexpected")
    with pytest.raises(ValueError, match="members"):
        load_bundle(path)


def test_decode_per_row_caps_eos_and_unasserted_control(monkeypatch):
    from qwen38_tpu import run

    monkeypatch.setattr(
        run,
        "checked_argmax",
        lambda logits: np.stack((np.asarray(logits), np.zeros(len(np.asarray(logits)), int)), -1),
    )
    masks = []
    next_logits = iter([np.array([10, 20, 99]), np.array([10, 99, 99])])

    class ReadyArray:
        def __init__(self, data):
            self.data = data

        def block_until_ready(self):
            return None

        def __array__(self, dtype=None):
            return np.asarray(self.data, dtype=dtype)

    engine = SimpleNamespace(
        config=SimpleNamespace(generation_eos={99}),
        forward=lambda *args: masks.append(args[-1].copy()),
        logits=lambda: ReadyArray(next(next_logits)),
    )
    tokenizer = SimpleNamespace(
        decode=lambda tokens, **kw: "</think>\n" + " ".join(map(str, tokens))
    )
    prompts = [
        dict(
            case=dict(
                name=f"case-{i}",
                images=[],
                asserted=i != 2,
                match=dict(kind="first_int", value=10 if i == 0 else 20),
            ),
            ids=np.array([1]),
            next_position=1,
            max_new_tokens=1 if i == 0 else 4,
        )
        for i in range(3)
    ]
    hasher = SimpleNamespace(step=lambda ids, history, active: (ids, history))
    lookup_masks = []

    def gather(ids, *, active):
        lookup_masks.append(active.copy())
        return np.zeros((3, 1))

    table = SimpleNamespace(gather=gather, hits=0, misses=0)
    rows, _ = run.generate(
        engine,
        prompts,
        np.array([10, 20, 99]),
        None,
        hasher,
        table,
        table,
        tokenizer,
        4,
        lambda **kw: None,
        record_cases=False,
    )
    assert rows[0]["truncated"] and not rows[0]["passed"]
    assert rows[1]["generated_tokens"] == 2 and rows[1]["passed"]
    assert rows[2]["passed"] is None and not rows[2]["asserted"]
    assert all(mask.tolist() == [False, True, False] for mask in masks)
    assert len(lookup_masks) == 2 * len(masks)
    assert all(mask.tolist() == [False, True, False] for mask in lookup_masks)


def test_prepare_preserves_timestamps_and_reuses_visual_features(monkeypatch):
    from glm53_offload import probe_images
    from qwen38_tpu import run

    captured, encoded = [], []

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            captured.append(messages[0]["content"])
            return "<think>\n"

        def encode(self, text, **kwargs):
            return [5, 100, 6, 100, 7]

    monkeypatch.setattr(
        run, "text_positions", lambda ids, grids, c: (np.zeros((len(ids), 3)), len(ids))
    )
    monkeypatch.setattr(run, "preprocess", lambda *args, **kw: (np.zeros((4, 1)), (1, 2, 2)))

    class Feature:
        def block_until_ready(self):
            return None

        def __array__(self, dtype=None):
            return np.asarray([[1, 2]], dtype=dtype)

    def encode(patches, grid):
        encoded.append(grid)
        return Feature()

    case = dict(
        name="ordered",
        images=["frame_1", "frame_1"],
        prompt="initial",
        content=[
            dict(type="text", text="0.05 seconds"),
            dict(type="image", image="frame_1"),
            dict(type="text", text="0.60 seconds"),
            dict(type="image", image="frame_1"),
        ],
    )
    config = SimpleNamespace(image_token=100, max_context=2048)
    result = run.prepare_prompt(
        Tokenizer(),
        case,
        probe_images.fixtures(),
        SimpleNamespace(encode_patches=encode),
        config,
        max_new_tokens=512,
        effort="medium",
        vision_cache={},
    )
    assert len(encoded) == 1 and len(result["visual"]) == 2
    assert captured[0] == [
        dict(type="text", text="0.05 seconds"),
        dict(type="image"),
        dict(type="text", text="0.60 seconds"),
        dict(type="image"),
    ]
