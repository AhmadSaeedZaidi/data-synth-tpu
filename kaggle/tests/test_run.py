import hashlib
import io
import json
import sys
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from glm53_offload import quality
from glm53_offload import run as worker
from glm53_offload.run import VisionChat, generate


class Tokenizer:
    def convert_tokens_to_ids(self, token):
        return 154854

    def apply_chat_template(self, messages, **kwargs):
        content = messages[0]["content"]
        count = sum(x["type"] == "image" for x in content)
        return "image" * count + "<think>"

    def __call__(self, text, **kwargs):
        assert text.endswith("<think></think>")
        return SimpleNamespace(input_ids=[1, 154854, 2, 154854, 3])

    def decode(self, tokens, **kwargs):
        return "".join(map(str, tokens))


def test_render_preserves_image_order_and_expands_positions():
    chat = VisionChat(Tokenizer(), None, thinking_budget=0)
    vectors = {b"first": np.ones((2, 4), np.float32), b"second": np.full((3, 4), 2, np.float32)}
    chat.image = vectors.__getitem__
    tokens, (positions, overrides) = chat.render("question", [b"first", b"second"])
    np.testing.assert_array_equal(tokens, [1, 154854, 154854, 2, 154854, 154854, 154854, 3])
    np.testing.assert_array_equal(positions, [1, 2, 4, 5, 6])
    np.testing.assert_array_equal(overrides[:2], vectors[b"first"])
    np.testing.assert_array_equal(overrides[2:], vectors[b"second"])
    with pytest.raises(ValueError, match="more image markers"):
        chat.render("question", [b"first"])


def test_generation_uses_independent_positions_and_excludes_finished_rows():
    class Chat:
        def render(self, text, images):
            return np.arange(int(text), dtype=np.int32), None

        def answer(self, ids):
            return str(ids)

    class Engine:
        max_len = 32
        cfg = SimpleNamespace(vocab=12)
        store = SimpleNamespace(stats={"bytes_transferred": 0})

        def prefill(self, tokens, override):
            logits = np.zeros((1, 12), np.float32)
            # One stream ends immediately; others produce one prefill token.
            logits[0, 9 if len(tokens) == 2 else len(tokens)] = 1
            return logits, [len(tokens)]

        def decode_rows(self, tokens, caches, positions):
            np.testing.assert_array_equal(positions, [3, 4, 5, 3])
            np.testing.assert_array_equal(tokens, [3, 4, 5, 3])
            assert caches == [[3], [4], [5], [3]]
            self.store.stats["bytes_transferred"] += 1024
            logits = np.zeros((4, 12), np.float32)
            logits[:, 9] = 1
            return logits, caches

    result = generate(Engine(), Chat(), [(str(n), []) for n in (2, 3, 4, 5)], 4, [9])
    assert result["output_ids"] == [[], [3], [4], [5]]
    assert result["stopped"] == [True] * 4
    assert result["generated_tokens"] == 3
    assert result["decode_generated_tokens"] == 0
    assert result["decode_calls"] == 1
    assert result["decode_bytes_transferred"] == 1024


def test_quality_gate_matches_final_answers_and_control_is_not_asserted():
    cases = quality.cases()
    assert len([x for x in cases if x["asserted"]]) == 12
    seven = next(x for x in cases if x["name"] == "count_circles_7")
    assert quality.match(seven["match"], "7")[0]
    assert not quality.match(seven["match"], "8")[0]
    assert not next(x for x in cases if x["name"] == "no_image_control")["asserted"]


def test_compacted_decode_matches_fixed_width_and_preserves_independent_state():
    lengths = [1, 2, 3, 4, 8, 9, 13, 18]

    class Engine:
        cfg = SimpleNamespace(vocab=16)
        store = SimpleNamespace(stats={"bytes_transferred": 0})

        def __init__(self, compact):
            self.compact_decode_batches = compact
            self.widths = []

        def decode_rows(self, tokens, caches, positions):
            self.widths.append(len(tokens))
            logits = np.zeros((len(tokens), 16), np.float32)
            updated = []
            for b, (cache, pos) in enumerate(zip(caches, positions, strict=True)):
                row, step = cache[0]
                assert pos == 100 + row + step
                updated.append([(row, step + 1)])
                logits[b, 0 if step + 1 >= lengths[row] else row + 1] = 1
            return logits, updated

    class Chat:
        def answer(self, ids):
            return str(ids)

    logits = np.zeros((8, 16), np.float32)
    logits[np.arange(8), np.arange(1, 9)] = 1
    prepared = worker.PreparedBatch(
        logits, [[(row, 0)] for row in range(8)], np.arange(100, 108), {}
    )
    fixed, compact = Engine(False), Engine(True)
    a = worker.decode_prepared(fixed, Chat(), prepared, 32, [0])
    b = worker.decode_prepared(compact, Chat(), prepared, 32, [0])
    assert (
        a["output_ids"]
        == b["output_ids"]
        == [[row + 1] * length for row, length in enumerate(lengths)]
    )
    assert a["stopped"] == b["stopped"] == [True] * 8
    assert a["decode_generated_tokens"] == b["decode_generated_tokens"]
    assert b["processed_decode_row_steps"] < a["processed_decode_row_steps"]
    assert {1, 2, 4, 8} == set(compact.widths)
    assert prepared.caches == [[(row, 0)] for row in range(8)]
    np.testing.assert_array_equal(prepared.positions, np.arange(100, 108))
    np.testing.assert_array_equal(prepared.logits, logits)


def test_hbm_guard_reduces_unallocated_cache_to_tightest_chip(monkeypatch):
    import jax
    from glm53_offload.expert_cache import ExpertCache
    from glm53_offload.weights import PackedExperts
    from jax.sharding import Mesh

    cfg = SimpleNamespace(
        layer_types=("linear_attention",),
        n_layers=1,
        kda_heads=8,
        kda_hd=4,
        conv_k=4,
        kv_lora=8,
        idx_kpool=4,
        idx_hd=4,
    )
    raw = np.zeros((8, 8, 1, 1, 4), np.uint8)
    store = PackedExperts({0: (raw,) * 3}, {0: ("F32",) * 3}, 8)
    mesh = Mesh(np.array(jax.devices()), ("x",))
    engine = SimpleNamespace(cfg=cfg, n=8, store=store, mesh=mesh, transport=lambda x: x)
    engine.expert_cache = ExpertCache(store, mesh, engine.transport, store.nbytes)
    # Reserve too much to exercise reduction; no payload has been allocated.
    engine.expert_cache.stats["reserved_bytes"] = 64 << 30
    devices = [
        SimpleNamespace(
            id=i,
            memory_stats=lambda: {
                "bytes_limit": 8 << 30,
                "bytes_in_use": 1 << 30,
            },
        )
        for i in range(8)
    ]
    devices[-1].memory_stats = lambda: {"bytes_limit": 6 << 30, "bytes_in_use": 1 << 30}
    monkeypatch.setattr(jax, "devices", lambda: devices)
    plan = worker.check_hbm_budget(engine, {"batch_sizes": [8], "max_context": 1024})
    assert plan["cache_bytes_per_device"] < plan["requested_cache_bytes_per_device"]
    assert plan["cache_safety_bytes_per_device"] == 1 << 30
    assert engine.expert_cache.stats["allocated_bytes"] == 0


def test_vision_batch_warmup_replays_complete_response_without_new_prefill():
    class Engine:
        cfg = SimpleNamespace(vocab=2)
        store = SimpleNamespace(stats={"bytes_transferred": 0})
        expert_cache = None

        def __init__(self):
            self.widths = set()

        @property
        def program_count(self):
            return len(self.widths)

        def decode_rows(self, tokens, caches, positions):
            self.widths.add(len(tokens))
            logits = np.zeros((len(tokens), 2), np.float32)
            logits[np.arange(len(tokens)), np.where(positions >= 4, 0, 1)] = 1
            return logits, caches

    class Chat:
        def answer(self, ids):
            return "1"

    metrics = {
        key: 0
        for key in (
            "prefill_seconds",
            "render_and_vision_seconds",
            "language_prefill_seconds",
            "prefill_bytes_transferred",
        )
    }
    state = worker.PreparedBatch(np.array([[0.0, 1.0]]), [[0]], np.array([0]), metrics)
    bank = [
        {
            "state": state,
            "single_answer": "1",
            "case": {
                "name": f"case_{i}",
                "match": {"kind": "first_int", "value": 1},
            },
        }
        for i in range(8)
    ]
    report = {}
    worker.batch_benchmarks(
        Engine(),
        Chat(),
        {
            "batch_sizes": [1, 4, 8],
            "max_new_tokens": 16,
            "full_decode_warmup": True,
        },
        [0],
        report,
        bank,
        lambda: None,
        lambda: None,
    )
    for measured in report["benchmarks"]:
        assert measured["warmup_decode_calls"] == measured["decode_calls"] == 5
        assert measured["warmup_completed"] and measured["fully_warmed_program_variants"]
        assert measured["new_programs"] == 0
        assert measured["prefill_reused"] and measured["status"] == "PASS"


def test_matched_replay_keeps_identical_image_rows_tokens_and_work_at_every_width():
    class Cache:
        stats = {"hits": 0, "misses": 0, "evictions": 0}

        def __init__(self):
            self.clears = 0

        def clear_entries(self):
            self.clears += 1

    class Engine:
        max_len = 128
        program_count = 0

        def __init__(self):
            self.store = SimpleNamespace(stats={"bytes_transferred": 0, "host_gather_seconds": 0})
            self.expert_cache = Cache()
            self.rows_seen = {1: {}, 4: {}, 8: {}}

        def decode_rows(self, tokens, caches, positions):
            batch = len(tokens)
            updated = []
            for token, cache, pos in zip(tokens, caches, positions, strict=True):
                row, step = cache[0]
                assert pos == 30 + row + step
                self.rows_seen[batch].setdefault(row, []).append((int(token), int(pos)))
                updated.append([(row, step + 1)])
            self.store.stats["bytes_transferred"] += batch * 7
            self.expert_cache.stats["hits"] += batch
            return np.zeros((batch, 2)), updated

    metrics = {
        name: 0
        for name in (
            "prefill_seconds",
            "render_and_vision_seconds",
            "language_prefill_seconds",
            "prefill_bytes_transferred",
        )
    }
    bank = [
        {
            "case": {"name": f"image_{row}"},
            "state": worker.PreparedBatch(
                np.zeros((1, 2)),
                [[(row, 0)]],
                np.array([30 + row]),
                metrics,
            ),
        }
        for row in range(8)
    ]
    chat = SimpleNamespace(
        tokenizer=lambda *args, **kwargs: SimpleNamespace(input_ids=list(range(64)))
    )
    engine, report = Engine(), {}
    worker.matched_replay_benchmarks(
        engine,
        chat,
        {"batch_sizes": [1, 4, 8], "matched_replay_steps": 5, "profile_matched_decode": False},
        report,
        bank,
        lambda: None,
        lambda: None,
    )
    assert engine.expert_cache.clears == 3
    results = report["matched_replay_benchmarks"]
    assert len({item["forced_tokens_sha256"] for item in results}) == 1
    for item in results:
        assert item["decode_row_steps"] == 40
        assert item["bytes_transferred"] == 40 * 7
        assert item["expert_cache_hits"] == 40
        assert item["case_names"] == [f"image_{i}" for i in range(8)]
        assert item["fully_warmed_program_variants"] and item["status"] == "MEASURED"
        assert item["quality_gate"] == "NOT_RUN"
    assert engine.rows_seen[1] == engine.rows_seen[4] == engine.rows_seen[8]
    for row, entry in enumerate(bank):
        assert entry["state"].caches == [[(row, 0)]]
        np.testing.assert_array_equal(entry["state"].positions, [30 + row])


@pytest.fixture
def replay_trial():
    class Engine:
        max_len = 128
        program_count = 0
        expert_cache = None
        cfg = SimpleNamespace(mlp_types=("sparse", "dense", "sparse", "sparse"))
        purpose = "startup"
        sampling = True

        def __init__(self):
            self.store = SimpleNamespace(stats={"bytes_transferred": 0})
            self.calls = []

        @contextmanager
        def execution_sampling(self, purpose, *, enabled=True, layers=None):
            previous = self.purpose, self.sampling
            self.purpose, self.sampling = purpose, enabled
            try:
                yield
            finally:
                self.purpose, self.sampling = previous

        def decode_rows(self, tokens, caches, positions):
            self.calls.append((self.purpose, self.sampling, len(tokens)))
            return np.zeros((len(tokens), 2)), caches

    metrics = dict.fromkeys(
        (
            "prefill_seconds",
            "render_and_vision_seconds",
            "language_prefill_seconds",
            "prefill_bytes_transferred",
        ),
        0,
    )
    bank = [
        {
            "case": {"name": f"image_{i}"},
            "state": worker.PreparedBatch(
                np.zeros((1, 2)), [[np.zeros(1)]], np.array([30]), metrics
            ),
        }
        for i in range(8)
    ]
    chat = SimpleNamespace(tokenizer=lambda *a, **kw: SimpleNamespace(input_ids=list(range(64))))
    return Engine(), chat, bank


def test_replay_checkpoints_completed_groups_before_deadline_interrupt(replay_trial):
    engine, chat, bank = replay_trial
    report, checkpoints = {}, []

    def deadline():
        if len(engine.calls) == 9:  # Eight warm groups, then one measured group.
            raise TimeoutError("quota budget")

    with pytest.raises(TimeoutError, match="quota budget"):
        worker.matched_replay_benchmarks(
            engine,
            chat,
            {"batch_sizes": [1, 4, 8], "matched_replay_steps": 1, "profile_matched_decode": False},
            report,
            bank,
            lambda: checkpoints.append(json.loads(json.dumps(report))),
            deadline,
        )
    item = report["matched_replay_benchmarks"][0]
    assert item["status"] == "INTERRUPTED"
    assert len(item["warmup_cycles"][0]["groups"]) == 8
    assert len(item["groups"]) == 1 and item["groups"][0]["decode_row_steps"] == 1
    assert item["active_group"]["number"] == 2
    assert "tokens_per_second" not in item
    assert any(
        c["matched_replay_benchmarks"] and len(c["matched_replay_benchmarks"][0]["groups"]) == 1
        for c in checkpoints
    )
    assert all(not sampling for _, sampling, _ in engine.calls)
    assert engine.purpose == "startup" and engine.sampling


def test_image_profile_runs_after_measurements_with_separate_synchronization(
    replay_trial, monkeypatch, tmp_path
):
    engine, chat, bank = replay_trial
    engine.device_trace_dir = tmp_path / "device-trace"
    captured = []

    @contextmanager
    def trace(directory):
        captured.append((directory, engine.purpose, len(engine.calls)))
        yield

    monkeypatch.setattr(worker, "optional_device_trace", trace)
    report = {}
    worker.matched_replay_benchmarks(
        engine,
        chat,
        {"batch_sizes": [1, 4, 8], "matched_replay_steps": 2},
        report,
        bank,
        lambda: None,
        lambda: None,
        lambda: 1000,
    )
    assert all(r["decode_row_steps"] == 16 for r in report["matched_replay_benchmarks"])
    diagnostic = report["matched_decode_diagnostic"]
    assert diagnostic["status"] == "MEASURED" and diagnostic["quality_gate"] == "NOT_RUN"
    assert diagnostic["sampled_layers"] == [0, 2, 3]
    assert diagnostic["replay"]["decode_calls"] == 2
    assert (
        engine.calls[-4:]
        == [("image_diagnostic_warmup", False, 8)] * 2 + [("matched_images", True, 8)] * 2
    )
    assert all(not sampling for _, sampling, _ in engine.calls[:-4])
    assert captured == [(engine.device_trace_dir, "matched_images", len(engine.calls) - 2)]
    assert engine.purpose == "startup" and engine.sampling


def test_image_profile_skips_before_restoring_states_when_budget_is_low(replay_trial):
    engine, _, bank = replay_trial
    report = {}
    worker.matched_decode_diagnostic(
        engine,
        bank,
        np.zeros((8, 2), np.int32),
        report,
        lambda: None,
        lambda: None,
        lambda: 100,
    )
    assert report["matched_decode_diagnostic"]["status"] == "SKIPPED_BUDGET"
    assert engine.calls == []


def test_vision_cache_keys_include_resolution_and_restore_original_embeddings():
    from PIL import Image

    encoded = io.BytesIO()
    Image.new("RGB", (448, 448), "red").save(encoded, format="PNG")
    blob = encoded.getvalue()
    calls = []

    def forward(patches, grids):
        calls.append(grids)
        return np.full((len(patches) // 4, 8), len(calls), np.float32)

    chat = VisionChat(Tokenizer(), forward, max_image_tokens=16, hidden_size=8)
    digest = hashlib.sha256(blob).hexdigest()
    original = chat.image(blob)
    chat.image_token_limits[digest] = 64
    higher = chat.image(blob)
    assert len(higher) > len(original)
    assert chat.image(blob) is higher
    chat.image_token_limits.pop(digest)
    assert chat.image(blob) is original
    assert len(calls) == 2
    np.testing.assert_array_equal(original, 1)
    np.testing.assert_array_equal(higher, 2)


@pytest.mark.parametrize("outcome", ["pass", "error", "budget"])
def test_resolution_retry_preserves_original_gate_and_restores_cap(monkeypatch, outcome):
    blob = b"unchanged image"
    digest = hashlib.sha256(blob).hexdigest()
    chat = SimpleNamespace(
        max_image_tokens=1024,
        image_token_limits={digest: 384},
        image=lambda image: np.zeros((768, 1)),
    )
    case = {
        "name": "road_text",
        "group": "beamng_visible_text",
        "images": ["road"],
        "prompt": "Read the word",
        "match": {"kind": "contains_all", "words": ["stop"]},
    }
    original = {
        "name": "road_text",
        "status": "FAIL",
        "answer": "SLOW",
        "prompt_tokens": [500],
        "language_prefill_seconds": 20,
        "decode_seconds": 2,
    }
    report = {"quality": {"status": "FAIL", "failed_cases": ["road_text"], "cases": [original]}}
    unchanged = json.loads(json.dumps(report["quality"]))

    def generate(engine, actual_chat, requests, max_new, eos):
        assert outcome != "budget", "must not start a prefill without budget"
        assert requests == [(case["prompt"], [blob])] and actual_chat is chat
        assert max_new == 256 and chat.image_token_limits[digest] == 1024
        if outcome == "error":
            raise RuntimeError("failed prefill")
        return {
            "answers": ["STOP"],
            "stopped": [True],
            "prompt_tokens": [900],
            "prefill_seconds": 30,
            "decode_seconds": 2,
            "reasoning_complete": [True],
            "reasoning_tokens": [10],
            "answer_tokens": [1],
        }

    monkeypatch.setattr(worker, "generate", generate)
    arguments = (
        None,
        chat,
        {"max_new_tokens": 256},
        0,
        {"road": blob},
        [case],
        report,
        lambda: None,
        lambda: 10 if outcome == "budget" else 3600,
    )
    if outcome == "error":
        with pytest.raises(RuntimeError, match="failed prefill"):
            worker.vision_resolution_controls(*arguments)
    else:
        worker.vision_resolution_controls(*arguments)
    assert chat.image_token_limits == {digest: 384}
    assert report["quality"] == unchanged
    control = report["vision_resolution_controls"][0]
    assert (
        control["status"]
        == {"pass": "PASS", "error": "INTERRUPTED", "budget": "SKIPPED_BUDGET"}[outcome]
    )
    assert control["quality_gate"] == "NOT_RUN" and control["image_sha256"] == [digest]


def test_fixed_replay_deadline_is_checked_each_step_and_context_is_bounded():
    prepared = worker.PreparedBatch(np.zeros((1, 2)), [[0]], np.array([3]), {})
    engine = SimpleNamespace(max_len=4)
    with pytest.raises(ValueError, match="exceeds model context"):
        worker.fixed_token_replay(engine, prepared, np.zeros((1, 2), np.int32), lambda: None)
    calls = []

    def decode(tokens, caches, positions):
        calls.append(positions.copy())
        return np.zeros((1, 2)), caches

    def deadline():
        if len(calls) == 2:
            raise TimeoutError("quota budget")

    engine = SimpleNamespace(
        max_len=16,
        program_count=0,
        expert_cache=None,
        decode_rows=decode,
        store=SimpleNamespace(stats={"bytes_transferred": 0}),
    )
    with pytest.raises(TimeoutError, match="quota budget"):
        worker.fixed_token_replay(engine, prepared, np.zeros((1, 4), np.int32), deadline)
    assert len(calls) == 2
    np.testing.assert_array_equal(calls, [[3], [4]])
    np.testing.assert_array_equal(prepared.positions, [3])


class ReasoningTokenizer:
    pieces = {1: "I count seven circles.", 2: "</think>", 3: "7", 4: "<|assistant|>"}

    def convert_tokens_to_ids(self, token):
        return 154854

    def apply_chat_template(self, messages, **kwargs):
        self.options = kwargs
        return "question<think>"

    def __call__(self, text, **kwargs):
        self.rendered = text
        return SimpleNamespace(input_ids=[1])

    def decode(self, tokens, **kwargs):
        return "".join(self.pieces[token] for token in tokens)


def test_native_thinking_is_default_and_only_completed_final_answer_is_matched():
    tokenizer = ReasoningTokenizer()
    chat = VisionChat(tokenizer, None, reasoning_effort="high")
    chat.render("How many circles?", [])
    assert tokenizer.rendered.endswith("<think>")
    assert tokenizer.options["reasoning_effort"] == "high"
    assert tokenizer.options["clear_thinking"] is True
    interrupted = chat.generation_details([1])
    assert interrupted == {
        "answer": "",
        "reasoning_complete": False,
        "reasoning_tokens": 1,
        "answer_tokens": 0,
    }
    completed = chat.generation_details([1, 2, 3])
    assert completed == {
        "answer": "7",
        "reasoning_complete": True,
        "reasoning_tokens": 1,
        "answer_tokens": 1,
    }
    spec = {"kind": "first_int", "value": 7}
    assert not quality.match(spec, interrupted["answer"])[0]
    assert quality.match(spec, completed["answer"])[0]


def test_forced_close_remains_explicit_and_split_closing_tokens_are_counted():
    tokenizer = ReasoningTokenizer()
    chat = VisionChat(tokenizer, None, thinking_budget=0)
    chat.render("count", [])
    assert tokenizer.rendered.endswith("<think></think>")
    assert chat.generation_details([3])["reasoning_tokens"] == 0
    tokenizer.pieces = {1: "reasoning", 2: "</", 3: "think>", 4: "7"}
    details = chat.generation_details([1, 2, 3, 4])
    assert details["answer"] == "7"
    assert details["reasoning_tokens"] == details["answer_tokens"] == 1


def test_native_decode_records_truncation_without_leaking_thoughts_as_answers():
    class Engine:
        max_len = 32
        cfg = SimpleNamespace(vocab=5)
        store = SimpleNamespace(stats={"bytes_transferred": 0})

        def prefill(self, tokens, override):
            return np.array([[0, 1, 0, 0, 0]], np.float32), [0]

    result = generate(Engine(), VisionChat(ReasoningTokenizer(), None), [("count", [])], 1, 0)
    assert result["answers"] == [""]
    assert result["reasoning_tokens"] == [1]
    assert result["answer_tokens"] == [0]
    assert result["reasoning_complete"] == [False]
    assert result["generation_limit_reached"] == [True]


def test_profile_worker_does_not_load_or_claim_full_model(monkeypatch, tmp_path):
    from glm53_offload import profiling

    monkeypatch.setattr(worker, "memory_snapshot", lambda: {})
    monkeypatch.setattr(worker, "environment", lambda work: {"devices": [{"platform": "tpu"}] * 8})
    monkeypatch.setattr(worker, "decoder_gate", lambda: {"decoder": "PASS"})
    monkeypatch.setattr(worker, "vision_gate", lambda: {"vision": "PASS"})
    monkeypatch.setattr(worker, "expert_cache_gate", lambda: {"expert_cache": "PASS"})

    def unexpected(*args, **kwargs):
        raise AssertionError("profile must never load the full model")

    monkeypatch.setattr(worker, "load_runtime", unexpected)
    calls = []

    def probe(config, report, save, deadline):
        calls.append(config["profile_experts"])
        deadline()
        report["expert_profile"] = {"cases": [{"status": "PASS"}]}
        save()

    monkeypatch.setattr(profiling, "run_expert_profile", probe)
    report = worker.run(
        {
            "experiment_mode": "expert_profile",
            "deadline_seconds": 900,
            "wandb_mode": "disabled",
            "weave_enabled": False,
        },
        tmp_path / "report.json",
    )
    assert calls == [96]
    assert report["status"] == "PROFILE_COMPLETE"
    assert report["model_check"] == "NOT_RUN"
    assert "quality" not in report


def test_full_trial_records_batches_before_remaining_images_without_reprefill(
    monkeypatch, tmp_path
):
    class Engine:
        max_len = 8192
        cfg = SimpleNamespace(vocab=2)
        expert_cache = None
        program_count = 0
        routing_stats = {}
        routing_by_layer = {}
        store = SimpleNamespace(stats={"bytes_transferred": 0})

        def __init__(self):
            self.prefill_calls = 0

        def prefill(self, tokens, override=None):
            self.prefill_calls += 1
            self.store.stats["bytes_transferred"] += 1024
            return np.array([[0.0, 1.0]], np.float32), [np.zeros((1, 1), np.float32)]

        def decode_rows(self, tokens, caches, positions):
            self.store.stats["bytes_transferred"] += 256
            logits = np.zeros((len(tokens), 2), np.float32)
            logits[:, 0] = 1
            return logits, caches

    class Chat:
        def render(self, text, images):
            return np.arange(4, dtype=np.int32), None

        def answer(self, ids):
            return "ok"

    engine = Engine()
    monkeypatch.setattr(worker, "memory_snapshot", lambda: {})
    monkeypatch.setattr(worker, "environment", lambda work: {"devices": [{"platform": "tpu"}] * 8})
    monkeypatch.setattr(worker, "decoder_gate", lambda: {"decoder": "PASS"})
    monkeypatch.setattr(worker, "vision_gate", lambda: {"vision": "PASS"})
    monkeypatch.setattr(worker, "load_runtime", lambda *args: (engine, Chat(), 0))
    monkeypatch.setattr(worker, "check_hbm_budget", lambda *args: {})
    monkeypatch.setattr(worker.quality, "match", lambda *args: (True, {}))
    checkpoints = []
    original_write = worker.write_report

    def record(path, report):
        checkpoints.append((report["stage"], len(report.get("quality", {}).get("cases", []))))
        original_write(path, report)

    monkeypatch.setattr(worker, "write_report", record)
    report = worker.run(
        {
            "wandb_mode": "disabled",
            "integer_dot_probe": False,
            "integer_projection_probe": False,
            "matched_replay_steps": 0,
        },
        tmp_path / "report.json",
    )
    assert report["status"] == report["model_check"] == "PASS"
    assert [item["batch_size"] for item in report["benchmarks"]] == [1, 4, 8]
    assert len(report["quality"]["cases"]) == 13
    assert engine.prefill_calls == 15
    assert all(item["prefill_reused"] for item in report["benchmarks"])
    assert next(count for stage, count in checkpoints if stage == "BATCH_BENCHMARKS") == 8


def test_long_image_trial_skips_expensive_prefills_when_budget_is_low(monkeypatch):
    monkeypatch.setattr(
        worker, "prepare_requests", lambda *args: pytest.fail("must not start an expensive prefill")
    )
    images = worker.probe_images.fixtures()
    report = {"quality": {"cases": [{"prompt_tokens": [100], "language_prefill_seconds": 1000}]}}
    worker.long_image_benchmarks(
        None,
        SimpleNamespace(render=lambda text, blobs: (np.arange(100), None)),
        {"batch_sizes": [1, 4, 8], "max_new_tokens": 32},
        0,
        images,
        report,
        lambda: None,
        lambda: None,
        lambda: 60,
    )
    assert [row["batch_size"] for row in report["long_image_benchmarks"]] == [1, 4, 8]
    assert all(row["status"] == "SKIPPED_BUDGET" for row in report["long_image_benchmarks"])
    assert all("tokens_per_second" not in row for row in report["long_image_benchmarks"])


@pytest.mark.parametrize("enough_ram", [True, False])
def test_runtime_loading_initializes_weight_report_and_checks_ram(
    monkeypatch, tmp_path, enough_ram
):
    """Exercise the full startup wiring without network access or large tensors."""
    from glm53 import checkpoint, vision
    from glm53_offload import engine as engine_module

    cfg = SimpleNamespace(n_layers=1, mlp_types=["sparse"], layer_types=["linear_attention"])
    closed = []
    tensor_names = [f"blk.0.ffn_{kind}_exps.weight" for kind in ("gate", "up", "down")]
    source = SimpleNamespace(
        where=dict.fromkeys(tensor_names, "shard"),
        info=lambda name: {"type": "Q6_K", "nbytes": 100},
        shards={"shard": SimpleNamespace(pf=SimpleNamespace(close=lambda: closed.append(True)))},
    )
    reader = SimpleNamespace(bytes_read=47)
    reader_loads = []
    monkeypatch.setattr(worker, "fetch_json", lambda url: {"text_config": {"eos_token_id": 9}})
    monkeypatch.setattr(worker, "cfg_from_json", lambda native: cfg)
    monkeypatch.setattr(worker, "open_gguf", lambda *args: source)
    monkeypatch.setattr(worker, "available_ram", lambda: (64 << 30) if enough_ram else 0)
    monkeypatch.setattr(worker, "BF16Reader", lambda: reader_loads.append(True) or reader)
    monkeypatch.setattr(checkpoint, "load_top", lambda *args, **kwargs: {})
    monkeypatch.setattr(checkpoint, "load_layer", lambda *args, **kwargs: {})
    store = object()
    monkeypatch.setattr(worker.PackedExperts, "from_gguf", lambda *args, **kwargs: store)
    engine = SimpleNamespace(params={}, mesh=None)
    monkeypatch.setattr(engine_module, "PackedOffloadEngine", lambda *args, **kwargs: engine)
    monkeypatch.setattr(vision, "load_vision", lambda *args, **kwargs: {"n_heads": 1})
    monkeypatch.setattr(vision, "to_sharded", lambda params, mesh: params)
    monkeypatch.setattr(vision, "make_forward_sharded", lambda *args: None)

    hub = ModuleType("huggingface_hub")
    hub.snapshot_download = lambda *args, **kwargs: None
    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: Tokenizer()
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    report = {}
    saves = []
    config = {
        "quant": "Q6_K_XL",
        "expert_tile": 4,
        "prefill_chunk": 32,
        "max_context": 8192,
        "vision_max_tokens": 1024,
        "thinking_budget": 0,
        "work_dir": str(tmp_path),
    }

    def save():
        saves.append((report["stage"], "weights" in report))

    if not enough_ram:
        with pytest.raises(MemoryError, match="base-load reserve"):
            worker.load_runtime(config, report, save)
        assert reader_loads == []
        return

    loaded_engine, chat, eos = worker.load_runtime(config, report, save)
    assert loaded_engine is engine
    assert isinstance(chat, VisionChat)
    assert eos == 9
    assert report["weights"]["packed_expert_bytes"] == 300
    assert report["weights"]["bf16_bytes_read"] == 47
    assert report["weights"]["types"] == ["Q6_K"]
    assert not report["weights"]["expert_requantization"]
    assert ("BF16_NONEXPERT_LOAD", True) in saves
    assert closed == [True]


@pytest.mark.parametrize(
    "settings,message",
    [
        ({"expert_tile": 3}, "expert_tile"),
        ({"prefill_chunk": 16}, "prefill_chunk"),
        ({"prefill_moe_window": 512}, "prefill_moe_window"),
        ({"tile_routed_experts": "true"}, "tile_routed_experts"),
        ({"expert_cache_mib_per_device": -1}, "expert_cache"),
        ({"vision_max_tokens": 1}, "vision_max_tokens"),
        ({"quant": "unknown"}, "quant"),
        ({"reasoning_effort": "none"}, "reasoning_effort"),
        ({"quality_case_names": []}, "quality_case_names"),
        ({"quality_case_names": ["unknown"] * 8}, "quality_case_names"),
        ({"quality_case_names": [None] * 8}, "quality_case_names"),
        ({"max_context": 16, "max_new_tokens": 32}, "max_new_tokens"),
        ({"experiment_mode": "unknown"}, "experiment_mode"),
        ({"experiment_mode": "expert_profile", "deadline_seconds": 901}, "deadline"),
        ({"experiment_mode": "expert_profile", "profile_experts": 97}, "profile_experts"),
        ({"experiment_mode": "expert_profile", "profile_repeats": 6}, "profile_repeats"),
    ],
)
def test_invalid_settings_fail_before_environment_or_model_loading(
    monkeypatch, tmp_path, settings, message
):
    def unexpected(*args, **kwargs):
        raise AssertionError("validation must happen before model/environment access")

    monkeypatch.setattr(worker, "environment", unexpected)
    monkeypatch.setattr(worker, "load_runtime", unexpected)
    monkeypatch.setattr(worker, "memory_snapshot", lambda: {})
    report = worker.run({"run_model": False, **settings}, tmp_path / "report.json")
    assert report["status"] == "ERROR"
    assert report["error"]["type"] == "ValueError"
    assert message in report["error"]["message"]


def test_hbm_budget_uses_individual_device_limit(monkeypatch):
    import jax

    cfg = SimpleNamespace(
        layer_types=("linear_attention", "deepseek_sparse_attention"),
        n_layers=2,
        kda_heads=8,
        kda_hd=4,
        conv_k=4,
        kv_lora=8,
        idx_kpool=4,
        idx_hd=4,
    )
    engine = SimpleNamespace(
        cfg=cfg, n=8, expert_cache=SimpleNamespace(stats={"reserved_bytes": 8 << 30})
    )
    config = {"batch_sizes": [1, 4, 8], "max_context": 1024}
    devices = [
        SimpleNamespace(
            id=i, memory_stats=lambda: {"bytes_limit": 16 << 30, "bytes_in_use": 1 << 30}
        )
        for i in range(8)
    ]
    monkeypatch.setattr(jax, "devices", lambda: devices)
    plan = worker.check_hbm_budget(engine, config)
    assert plan["cache_bytes_per_device"] == 1 << 30
    assert plan["batch_state_bytes_per_device"] > 0
    devices[-1].memory_stats = lambda: {"bytes_limit": 4 << 30, "bytes_in_use": 1 << 30}
    with pytest.raises(MemoryError, match="device 7"):
        worker.check_hbm_budget(engine, config)
