"""Numerical and allocation contracts for the inference performance paths."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh
from qwen38_tpu.ngram import HostNGramTable, NGramHash
from qwen38_tpu.routing_validation import validate_expert_routing
from qwen38_tpu.validation import tiny_model


def test_mixed_quant_routing_on_eight_devices_preserves_hot_and_duplicate_routes():
    mesh = Mesh(np.asarray(jax.devices()), ("expert",))
    assert mesh.size == 8
    result = validate_expert_routing(mesh)
    assert result["status"] == "PASS"
    assert all(item["hot_expert_rows"] > 2 * item["row_tile"] for item in result["shapes"])
    assert {item["policy"] for item in result["shapes"]} == {"bf16", "int8_q8"}


def test_cache_donation_aliases_every_state_buffer_and_preserves_layout():
    mesh = Mesh(np.asarray(jax.devices()), ("expert",))
    engine, _, _ = tiny_model(mesh, "long-reference", kv_cache_dtype="q8")
    engine.reset(8)
    c = engine.config
    for kind, layer in (("gdn", 0), ("qsa", 3)):
        for chunk in (False, True):
            cache = engine.state["layers"][layer]
            shape = (8, 2) if chunk else (8,)
            arguments = (
                jnp.zeros((*shape, c.streams * c.hidden)),
                engine.layers[layer],
                cache,
                jnp.zeros(8, jnp.int32),
                jnp.zeros((*shape, 3), jnp.int32),
                jnp.ones(shape, bool),
            )
            function = getattr(engine, ("chunk_" if chunk else "") + kind)
            compiled = function.lower(*arguments).compile()
            stats = compiled.memory_analysis()
            assert stats.alias_size_in_bytes == sum(
                leaf.addressable_shards[0].data.nbytes for leaf in jax.tree.leaves(cache)
            )
            before = [leaf.sharding for leaf in jax.tree.leaves(cache)]
            updated = compiled(*arguments)[-1]
            assert all(
                old.is_equivalent_to(new.sharding, new.ndim)
                for old, new in zip(before, jax.tree.leaves(updated), strict=True)
            )
            engine.state["layers"][layer] = updated
    engine.release_state()


def test_prefill_batched_ram_gathers_match_serial_model_with_images_and_padding():
    from qwen38_tpu.run import prefill

    engine, _, a = tiny_model(Mesh(np.asarray(jax.devices()[:1]), ("expert",)))
    c = engine.config
    host = a["model.language_model.embed_tokens.weight"]
    ngram = a["model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight"]

    def f32_table(array):
        raw = array.astype("<f4").view(np.uint8).reshape(len(array), array.shape[1], 4)
        return HostNGramTable(raw, "F32", array.shape[1])

    tokens, ngrams = f32_table(host), f32_table(ngram)
    hasher = NGramHash(c)
    positions = a["positions"].transpose(1, 2, 0)
    visual = a["vision"].reshape(2, 4, c.hidden)
    prompts = [
        dict(
            ids=a["ids"][row, :length],
            positions=positions[row, :length],
            visual={step: visual[row, step - 2] for step in range(2, 6)},
        )
        for row, length in enumerate((12, 9))
    ]
    engine.reset(2)
    history = hasher.initial_history(2)
    for step in range(12):
        active = np.array([step < len(p["ids"]) for p in prompts])
        ids = np.array([p["ids"][step] if active[row] else c.eos for row, p in enumerate(prompts)])
        hashes, history = hasher.step(ids, history, active)
        embedding = host[ids].copy()
        if 2 <= step < 6:
            embedding[:] = visual[:, step - 2]
        engine.forward(embedding, ngram[hashes].reshape(2, -1), positions[:, step], active)
    expected = np.asarray(engine.logits()).copy()
    logits, actual_history, _ = prefill(engine, prompts, hasher, ngrams, tokens, lambda **kw: None)
    np.testing.assert_allclose(np.asarray(logits), expected, rtol=3e-6, atol=3e-6)
    np.testing.assert_array_equal(actual_history, history)
    np.testing.assert_array_equal(engine.lengths, [12, 9])
    assert tokens.hits + tokens.misses == 21
    assert ngrams.hits + ngrams.misses == 21 * len(hasher.sizes)


def test_malformed_inputs_do_not_consume_donated_state():
    engine, _, _ = tiny_model(Mesh(np.asarray(jax.devices()[:1]), ("expert",)))
    engine.reset(2)
    c = engine.config
    with pytest.raises(ValueError, match="mask"):
        engine.forward(np.zeros((1, c.hidden)), np.zeros((1, c.ple_dim)), np.zeros((1, 3)), [True])
    with pytest.raises(ValueError, match="positions"):
        engine.forward_chunk(
            np.zeros((2, 1, c.hidden)),
            np.zeros((2, 1, c.ple_dim)),
            np.zeros((2, 3)),
            [[True], [True]],
        )
    np.testing.assert_array_equal(engine.lengths, [0, 0])
    # An input error must happen before JAX donates any cache buffers.
    for leaf in jax.tree.leaves(engine.state):
        assert not leaf.is_deleted()


@pytest.mark.parametrize("profile", [False, True])
def test_throughput_counts_tracking_and_final_selection_wall_time(monkeypatch, tmp_path, profile):
    from contextlib import nullcontext
    from types import SimpleNamespace

    from qwen38_tpu import run

    now = [0.0]

    def spend(seconds, result=None):
        now[0] += seconds
        return result

    monkeypatch.setattr(run.time, "perf_counter", lambda: now[0])
    monkeypatch.setattr(run.tracing, "span", lambda *args, **kw: nullcontext({}))
    captures = []

    def capture(directory):
        captures.append((directory, now[0]))
        return nullcontext()

    monkeypatch.setattr(run, "optional_device_trace", capture)
    monkeypatch.setattr(run, "checked_argmax", lambda _: spend(1, np.array([[10, 0], [10, 0]])))
    logits = SimpleNamespace(block_until_ready=lambda: None)
    engine = SimpleNamespace(
        config=SimpleNamespace(generation_eos=(999,)),
        forward=lambda *args: spend(2),
        logits=lambda: logits,
    )
    lookup = SimpleNamespace(gather=lambda ids, **kw: spend(1, np.zeros((2, 32))), hits=0, misses=0)
    hasher = SimpleNamespace(step=lambda ids, history, active: (ids[:, None], history))
    tokenizer = SimpleNamespace(decode=lambda *args, **kw: "reasoning</think>ok")
    prompts = [
        dict(
            next_position=4,
            ids=np.zeros(4),
            case=dict(name=str(i), match=dict(kind="exact", expected="ok")),
        )
        for i in range(2)
    ]
    _, metrics = run.generate(
        engine,
        prompts,
        logits,
        None,
        hasher,
        lookup,
        lookup,
        tokenizer,
        3,
        lambda **kw: spend(3),
        record_cases=False,
        profile_directory=tmp_path / "device-trace" if profile else None,
    )
    assert metrics["decode_seconds"] == 17
    assert metrics["decode_active_seconds"] == 10
    assert metrics["host_lookup_seconds"] == metrics["model_step_seconds"] == 4
    assert metrics["selection_seconds"] == 3
    assert metrics["decode_overhead_seconds"] == 6
    assert metrics["aggregate_tokens_per_second"] == 4 / 17
    assert len(captures) == int(profile)
    if profile:
        assert captures[0][1] > 2, "Trace must follow the first model step"
