"""Completion budgets must preserve strict scoring, state updates and source evidence."""

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from qwen38_tpu.evaluation.bundle import load_bundle
from qwen38_tpu.evaluation.policy import QUALITY_POLICY, AnswerReserve, quality_case

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def local_bundle():
    path = ROOT / "output/current/inputs/qwen38-evaluation.bundle"
    if not path.exists():
        pytest.skip("Private BeamNG evaluation bundle is only available in the workspace")
    return path


def test_policy_contract_is_independent_of_reference_values():
    original = dict(
        name="annotation",
        group="sft_formatting",
        images=["image"],
        prompt="Return an annotation",
        match=dict(kind="annotation_json", expected={"secret_reference": 1}),
        asserted=True,
    )
    snapshot = copy.deepcopy(original)
    changed = copy.deepcopy(original)
    changed["match"]["expected"] = {"unrelated_reference": False}
    first, second = quality_case(original), quality_case(changed)
    assert first["system_prompt"] == second["system_prompt"]
    assert first["runtime_policy"] == second["runtime_policy"]
    assert "secret_reference" not in first["system_prompt"]
    assert original == snapshot and first["match"] == original["match"]


def test_quality_policy_preserves_all_original_inputs_and_matchers(local_bundle):
    _, cases, _, _ = load_bundle(local_bundle)
    before = copy.deepcopy(cases)
    for original in cases:
        case = quality_case(original)
        assert case["runtime_policy"]["name"] == QUALITY_POLICY
        if case["match"]["kind"] == "pair_markdown":
            assert "one and six reason bullets" in case["system_prompt"]
            assert "angle-bracket" in case["system_prompt"]
        for key in ("prompt", "images", "content", "match", "provenance", "asserted"):
            assert case.get(key) == original.get(key)
        assert case["max_new_tokens"] > case["answer_reserve_tokens"] + 1
        if case["group"] == "beamng_pipeline_review":
            assert case["max_new_tokens"] == 8192 and len(case["images"]) == 8
        if case["images"] and case["group"] in {"text_reading", "sft_formatting"}:
            assert case["min_pixels"] == 1048576 and case["max_pixels"] >= 1048576
        changed = copy.deepcopy(original)
        changed["match"]["expected"] = {"arbitrary": "different reference"}
        other = quality_case(changed)
        assert other["system_prompt"] == case["system_prompt"]
        assert other["runtime_policy"] == case["runtime_policy"]
    assert cases == before


def test_final_reserve_handles_natural_boundaries_inactive_rows_and_small_caps():
    tokenizer = SimpleNamespace(encode=lambda *a, **kw: [50], decode=lambda *a, **kw: "</think>")
    prompts = [dict(case={"answer_reserve_tokens": 3}), dict(case={})]
    reserve = AnswerReserve(tokenizer, prompts, [8, 1], [99])
    ids = np.array([11, 20])
    actual = reserve.select(ids, np.array([True, False]), [[11] * 4, []])
    assert actual.tolist() == [50, 20] and ids.tolist() == [11, 20]
    assert reserve.forced == [True, False]
    assert reserve.select(ids, np.array([True, False]), [[11] * 7, []]).tolist() == [11, 20]
    with pytest.raises(ValueError, match="reserve"):
        AnswerReserve(tokenizer, [dict(case={"answer_reserve_tokens": 7})], [8], [99])
    with pytest.raises(ValueError, match="non-EOS"):
        AnswerReserve(tokenizer, prompts, [8, 1], [50])


def test_forced_delimiter_updates_real_decode_state_and_keeps_strict_scoring(monkeypatch):
    from qwen38_tpu import run

    monkeypatch.setattr(
        run, "checked_argmax", lambda logits: np.stack((np.asarray(logits), np.zeros(2, int)), -1)
    )
    forwarded = []
    last = np.array([11, 50])

    class Ready:
        def block_until_ready(self):
            return self

        def __array__(self, dtype=None):
            return np.asarray(np.where(last == 50, 20, np.where(last == 20, 99, 11)), dtype)

    def forward(embeddings, features, positions, active):
        last[:] = np.asarray(embeddings).reshape(-1)
        forwarded.append((last.copy(), active.copy()))

    engine = SimpleNamespace(
        config=SimpleNamespace(generation_eos=(99,)), forward=forward, logits=Ready
    )
    tokenizer = SimpleNamespace(
        encode=lambda *a, **kw: [50],
        decode=lambda ids, **kw: (
            "</think>"
            if ids == [50]
            else ("reasoning</think>OK" if 50 in ids and 20 in ids else "reasoning")
        ),
    )
    prompts = [
        dict(
            case=dict(
                name=str(i), match={"kind": "exact", "expected": "OK"}, answer_reserve_tokens=3
            ),
            ids=np.array([1]),
            positions=np.zeros((1, 3), int),
            next_position=1,
            max_new_tokens=8,
        )
        for i in range(2)
    ]
    table = SimpleNamespace(gather=lambda ids, **kw: ids, hits=0, misses=0)
    rows, _ = run.generate(
        engine,
        prompts,
        last.copy(),
        None,
        SimpleNamespace(step=lambda ids, state, active: (ids, state)),
        table,
        table,
        tokenizer,
        8,
        lambda **kw: None,
        record_cases=False,
    )
    assert [r["passed"] for r in rows] == [True, True]
    assert [r["reasoning_budget_forced"] for r in rows] == [True, False]
    assert rows[0]["generated_tokens"] == 6 and not rows[0]["truncated"]
    assert any(ids[0] == 50 and active[0] for ids, active in forwarded)
    assert all(not active[1] for _, active in forwarded[2:])
    # The real three-token warm-up must not inherit a request's answer reserve
    # and abort after evaluation has completed successfully.
    warm = run.warmup_prompt(prompts[0])
    last[:] = [11, 11]
    warm_rows, _ = run.generate(
        engine,
        [warm, warm],
        last.copy(),
        None,
        SimpleNamespace(step=lambda ids, state, active: (ids, state)),
        table,
        table,
        tokenizer,
        3,
        lambda **kw: None,
        record_cases=False,
    )
    assert all(r["final_answer_reserve_tokens"] == 0 for r in warm_rows)
    assert prompts[0]["case"]["answer_reserve_tokens"] == 3


@pytest.mark.parametrize("bundled", [False, True])
def test_quality_notebook_bootstraps_full_suite_with_long_context(
    tmp_path, monkeypatch, local_bundle, bundled
):
    spec = importlib.util.spec_from_file_location(
        "quality_builder", ROOT / "scripts/build_qwen38_notebook.py"
    )
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    bundle = local_bundle
    stage = builder.build(
        tmp_path / "stage",
        evaluation_bundle=bundle,
        quality_first=True,
        rounding_probe=bundled,
        batch_sizes=[1, 2, 4, 8] if bundled else None,
    )
    document = json.loads((stage / "notebook/qwen38-v5e8.ipynb").read_text())
    working = tmp_path / "working"
    working.mkdir()
    code = "".join(next(c for c in document["cells"] if c["cell_type"] == "code")["source"])
    code = code.replace("/kaggle/input", str(stage / "dataset")).replace(
        "/kaggle/working", str(working)
    )
    import sys

    monkeypatch.setattr(sys, "path", list(sys.path))
    env = {}
    exec(code, env)
    config = env["CONFIG"]
    assert config["max_context"] == 16384 and config["kv_cache_dtype"] == "q8"
    assert config["batch_sizes"] == ([1, 2, 4, 8] if bundled else [1])
    assert config["evaluation_batch_size"] == 1
    assert config["evaluation_expected_cases"] == 49 and config["evaluation_max_new_tokens"] == 8192
    assert config["max_new_tokens"] == 4096
    assert config["stop_on_evaluation_collapse"] is False
    assert config["reasoning_effort"] == "xhigh" and config["require_live_tracking"]
    assert not config["optimization_probes"] and not config["packed_q6_probe"]
    assert config["rounding_probe"] is bundled and config["deadline_seconds"] == 21600
    assert config["quant"] == "q6" and config["expert_compute"] == "bf16"
    assert config["requests"] is None
