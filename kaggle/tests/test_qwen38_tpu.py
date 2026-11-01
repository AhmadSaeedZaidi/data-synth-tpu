"""Numerical/reference and placement gates for the new engine."""

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh
from qwen38_tpu.checkpoint import AdaptiveRangeFile, Checkpoint, undo_value_head_order
from qwen38_tpu.config import ModelConfig
from qwen38_tpu.math import gdn_chunk, gdn_step
from qwen38_tpu.ngram import NGramHash
from qwen38_tpu.quant import IQ4_VALUES, dequantize_rows
from qwen38_tpu.run import final_answer
from qwen38_tpu.validation import tiny_model, validate
from qwen38_tpu.vision import text_positions

ROOT = Path(__file__).resolve().parents[2]


def test_full_official_reference():
    result = validate()
    assert result["status"] == "PASS"
    assert result["max_logit_error"] < 2e-6


def test_chunk_prefill_cache_and_padding():
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("expert",))
    engine, _, a = tiny_model(mesh)
    c = engine.config
    ids = a["ids"]
    hasher = NGramHash(c)
    history = hasher.initial_history(2)
    table = a["model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight"]
    embeddings = a["model.language_model.embed_tokens.weight"][ids].copy()
    embeddings[:, 2:6] = a["vision"].reshape(2, 4, c.hidden)
    features = []
    for step in range(12):
        indices, history = hasher.step(ids[:, step], history)
        features.append(table[indices].reshape(2, c.ple_dim))
    features = np.stack(features, 1)
    positions = a["positions"].transpose(1, 2, 0)
    mask = np.ones((2, 12), bool)
    mask[1, 9:] = False
    engine.reset(2)
    for step in range(12):
        engine.forward(embeddings[:, step], features[:, step], positions[:, step], mask[:, step])
    serial = jax.tree.map(lambda x: np.asarray(x).copy(), engine.state)
    logits = np.asarray(engine.logits()).copy()
    engine.reset(2)
    for first, last in ((0, 4), (4, 8), (8, 12)):
        engine.forward_chunk(
            embeddings[:, first:last],
            features[:, first:last],
            positions[:, first:last],
            mask[:, first:last],
        )
    np.testing.assert_allclose(np.asarray(engine.logits()), logits, atol=3e-6, rtol=3e-6)
    for old, new in zip(jax.tree.leaves(serial), jax.tree.leaves(engine.state), strict=True):
        np.testing.assert_allclose(np.asarray(new), old, atol=3e-6, rtol=3e-6)
    np.testing.assert_array_equal(engine.lengths, [12, 9])


def test_eight_chip_reference_and_expert_no_all_gather():
    env = {
        **os.environ,
        "XLA_FLAGS": "--xla_force_host_platform_device_count=8",
        "JAX_PLATFORMS": "cpu",
        "OPENBLAS_NUM_THREADS": "2",
        "OMP_NUM_THREADS": "2",
        "PYTHONPATH": str(ROOT / "kaggle"),
    }
    code = """import json, jax, numpy as np
from jax.sharding import Mesh
from qwen38_tpu.validation import validate, tiny_model
mesh=Mesh(np.asarray(jax.devices()),('expert',))
result=validate(mesh,batch_sizes=(8,16,32))
assert [r['batch'] for r in result['batch_shapes']]==[8,16,32]
engine,_,_=tiny_model(mesh)
packed=engine.experts[0]
x=jax.numpy.zeros((8,32)); ids=jax.numpy.zeros((8,2),dtype='int32'); weights=jax.numpy.ones((8,2))/2
hlo=str(engine.expert_functions[0].lower(packed['gate'],packed['up'],packed['down'],x,ids,weights).compiler_ir())
assert 'all_gather' not in hlo.lower(), 'Resident experts were all-gathered'
assert 'all_reduce' in hlo.lower(), 'Expert owner outputs were not reduced'
print(json.dumps(result))
"""
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=90
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1])["devices"] == 8


def test_ram_ngram_plan_q6_and_q8():
    root = ROOT / "kaggle/qwen38_tpu/checkpoints"
    q6 = Checkpoint(root / "q6.json").memory_plan(ModelConfig(), 8, hbm_bytes=8 * 16909332480)
    q8 = Checkpoint(root / "q8.json").memory_plan(ModelConfig(), 8, hbm_bytes=8 * 16909332480)
    assert q6["fits"] and not q8["fits"]
    assert 50 < q6["host_ngram_bytes"] / 2**30 < 51
    hash = NGramHash(ModelConfig())
    metadata = Checkpoint(root / "q6.json").audit["metadata"]
    assert hash.multipliers == metadata["qwen4exp.ple.layer_multipliers"]
    assert hash.offsets.tolist() == metadata["qwen4exp.ple.head_offsets"]
    assert hash.sizes.tolist() == metadata["qwen4exp.ple.head_vocab_sizes"]
    for q in ("q4", "q6", "q8"):
        assert Checkpoint(root / (q + ".json")).validate(ModelConfig()) == 1510


def test_adaptive_ranges_preserve_exact_bytes_and_reject_wrong_ranges(monkeypatch):
    from glm53_offload.weights import RangeFile

    def flaky(file, offset, length):
        if length > 256 << 10:
            raise OSError("truncated response")
        return bytes((offset + i) % 251 for i in range(length))

    monkeypatch.setattr(RangeFile, "_range", flaky)
    reader = AdaptiveRangeFile("https://example.test/weights", 2 << 20)
    assert reader._range(37, 1 << 20) == bytes((37 + i) % 251 for i in range(1 << 20))

    def invalid(file, offset, length):
        raise ValueError("server ignored requested range")

    monkeypatch.setattr(RangeFile, "_range", invalid)
    with pytest.raises(ValueError, match="ignored"):
        reader._range(0, 1 << 20)


@pytest.mark.parametrize("qtype", ["Q5_1", "IQ4_NL"])
def test_new_quant_layouts(qtype):
    width = 640 if qtype == "Q5_1" else 160
    size = 24 if qtype == "Q5_1" else 18
    rng = np.random.default_rng(74)
    raw = rng.integers(0, 256, (2, width // 32, size), dtype=np.uint8)
    raw[..., :2] = np.asarray([1.0], dtype="<f2").view(np.uint8)
    if qtype == "Q5_1":
        raw[..., 2:4] = np.asarray([-2.0], dtype="<f2").view(np.uint8)
    expected = []
    for row in raw:
        values = []
        for block in row:
            if qtype == "Q5_1":
                high = int.from_bytes(block[4:8].tobytes(), "little")
                low = block[8:]
                values.extend(
                    [
                        ((int(low[i % 16]) >> (4 * (i // 16))) & 15) + (((high >> i) & 1) << 4) - 2
                        for i in range(32)
                    ]
                )
            else:
                low = block[2:]
                values.extend(
                    [IQ4_VALUES[(int(low[i % 16]) >> (4 * (i // 16))) & 15] for i in range(32)]
                )
        expected.append(values)
    for backend in ("numpy", "jax"):
        np.testing.assert_array_equal(
            np.asarray(dequantize_rows(raw, qtype, width, backend=backend)), expected
        )


def test_gguf_value_head_permutation_and_thinking_budget():
    c = ModelConfig()
    original = np.arange(c.value_heads * 2).reshape(c.key_heads, 3, 2)
    tiled = original.swapaxes(0, 1).reshape(-1)
    np.testing.assert_array_equal(undo_value_head_order(tiled, 0, c, 2), original.reshape(-1))
    assert final_answer("thinking only") == ("", "THINKING_TRUNCATED")
    assert final_answer("thinking</think> STOP 7") == ("STOP 7", "FINAL")
    assert ModelConfig(max_context=8192, kv_cache_dtype="q8").sparse_qsa
    with pytest.raises(ValueError, match="capacity"):
        ModelConfig(max_context=32769)
    with pytest.raises(ValueError, match="placeholder"):
        text_positions([c.image_token], [], c)


@pytest.mark.parametrize(
    "steps,tail,state_dim,continuation",
    [
        (48, 0, 16, False),
        (96, 13, 16, False),
        (128, 0, 16, False),
        (7, 3, 16, False),
        (129, 17, 128, True),
    ],
)
def test_gdn_chunk_prefill_matches_serial_scan(steps, tail, state_dim, continuation):
    rng = np.random.default_rng(3200 + steps)
    c = replace(ModelConfig(), hidden=64, key_heads=2, value_heads=6, state_dim=state_dim)
    conv_channels = c.conv_channels

    def weight(shape, scale, dtype=jnp.bfloat16):
        return jnp.asarray(rng.normal(size=shape).astype(np.float32) * scale).astype(dtype)

    w = dict(
        qkv=weight((conv_channels, c.hidden), 0.25),
        conv=weight((conv_channels, c.conv_kernel), 0.35),
        alpha=weight((c.value_heads, c.hidden), 0.25),
        dt=weight((c.value_heads,), 0.25, jnp.float32),
        a=-jnp.abs(weight((c.value_heads,), 0.5, jnp.float32)) - 0.05,
        beta=weight((c.value_heads, c.hidden), 0.25),
        gate=weight((c.value_heads * c.state_dim, c.hidden), 0.25),
        norm=jnp.asarray(0.5 + rng.normal(size=(c.state_dim,)) * 0.1, jnp.float32),
        out=weight((c.hidden, c.value_heads * c.state_dim), 0.2),
    )
    batch = 3 if continuation else 2
    x = jnp.asarray(rng.normal(size=(batch, steps, c.hidden)).astype(np.float32) * 0.7).astype(
        jnp.bfloat16
    )
    active = np.ones((batch, steps), bool)
    if tail:
        active[1, steps - tail :] = False
    if continuation:
        active[2] = False
    active = jnp.asarray(active)

    conv = jnp.zeros((batch, c.conv_kernel - 1, conv_channels), jnp.bfloat16)
    state = jnp.zeros((batch, c.value_heads, c.state_dim, c.state_dim), jnp.float32)
    if continuation:
        conv = weight(conv.shape, 0.1)
        state = weight(state.shape, 0.1, jnp.float32)
    initial_cache = conv, state

    def serial(x, w, cache, mask):
        def step(carry, row):
            value, carry = gdn_step(row[0], w, carry, row[1], c)
            return carry, value

        cache, values = jax.lax.scan(step, cache, (x.swapaxes(0, 1), mask.T))
        return values.swapaxes(0, 1), cache

    expected, (conv, state) = jax.jit(serial)(x, w, initial_cache, active)
    result, chunk_cache = jax.jit(lambda x, w, cache, mask: gdn_chunk(x, w, cache, mask, c))(
        x, w, initial_cache, active
    )

    keep = np.asarray(active)
    np.testing.assert_allclose(
        np.asarray(result, np.float32)[keep],
        np.asarray(expected, np.float32)[keep],
        atol=6e-2,
        rtol=6e-2,
        err_msg="chunked GDN outputs diverge from the serial scan",
    )
    serial_conv, serial_state = conv, state
    chunk_conv, chunk_state = chunk_cache
    finished = np.asarray(active.sum(1)) > 0
    np.testing.assert_allclose(
        np.asarray(chunk_conv, np.float32)[finished],
        np.asarray(serial_conv, np.float32)[finished],
        atol=1e-4,
        rtol=1e-4,
    )
    np.testing.assert_allclose(
        np.asarray(chunk_state, np.float32)[finished],
        np.asarray(serial_state, np.float32)[finished],
        atol=2e-3,
        rtol=2e-3,
    )
    if continuation:
        for old, new in zip(initial_cache, chunk_cache, strict=True):
            np.testing.assert_array_equal(np.asarray(old)[2], np.asarray(new)[2])
