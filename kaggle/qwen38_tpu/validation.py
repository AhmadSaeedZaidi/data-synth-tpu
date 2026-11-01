"""Small, pinned full-model CPU/TPU numerical gate, also shipped in the notebook."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from . import numerics
from .attention_cache import quantize, read_quantized, write_quantized
from .config import ModelConfig, VisionConfig
from .engine import Engine
from .experts import expert_parallel
from .ngram import NGramHash
from .parallel import expert_mesh, put_expert_bytes
from .vision import VisionEncoder, text_positions


def validate_q8_codec(mesh, diagnostics=None):
    """Check compiled cache codes/scales against NumPy, including cast elision.

    The oracle rounds using ml_dtypes on the host, independent of the JAX
    integer rounding implementation. Both the real head width and tiny gate
    width are tested, with values that distinguish rounded/unrounded divisors.
    """
    details = []
    rng = np.random.default_rng(118)
    replica = NamedSharding(mesh, P())
    for width in (8, 256):
        group = min(32, width)
        source = rng.normal(size=(2, 2, width)).astype(np.float32)
        source[0, 0, :group] = 0
        source[0, 1, :group] = np.linspace(-0.99, 0.99, group, dtype=np.float32)
        source[0, 1, 0] = 1
        stored = np.float32(1 / 127).astype(np.dtype(jnp.bfloat16)).astype(np.float32)
        source[0, 1, 1] = np.float32(64.501) * stored
        for dtype in (np.float32, np.dtype(jnp.bfloat16)):
            host = source.astype(dtype)
            grouped = host.astype(np.float32).reshape(2, 2, -1, group)
            full = np.max(np.abs(grouped), -1) / np.float32(127)
            scale = full.astype(np.dtype(jnp.bfloat16))
            divisor = np.where(scale == 0, 1, scale.astype(np.float32))
            codes = np.clip(np.rint(grouped / divisor[..., None]), -127, 127).astype(np.int8)
            unrounded = np.clip(
                np.rint(grouped / np.where(full == 0, 1, full)[..., None]), -127, 127
            ).astype(np.int8)
            x = jax.device_put(host, replica)
            actual_codes, actual_scale = jax.jit(lambda x: quantize(x, group))(x)
            actual_codes, actual_scale = np.asarray(actual_codes), np.asarray(actual_scale)
            item = dict(
                width=width,
                dtype=str(host.dtype),
                code_mismatches=int(np.count_nonzero(actual_codes.reshape(codes.shape) != codes)),
                scale_mismatches=int(np.count_nonzero(actual_scale != scale)),
                unrounded_divisor_changes=int(np.count_nonzero(unrounded != codes)),
            )
            details.append(item)
            if diagnostics is not None:
                diagnostics["codec_shapes"] = details
            np.testing.assert_array_equal(actual_codes, codes.reshape(host.shape))
            np.testing.assert_array_equal(actual_scale, scale)
            c = ModelConfig(
                max_context=8,
                kv_cache_dtype="q8",
                kv_heads=2,
                head_dim=width,
                rotary_dim=min(64, width),
            )
            groups = 2 * (width // group)
            cache = {
                name: jax.device_put(
                    np.zeros((2, 8, 2 * width), np.int8),
                    NamedSharding(mesh, P(None, "expert", None)),
                )
                for name in ("k", "v")
            }
            cache.update(
                {
                    name: jax.device_put(
                        np.zeros((2 * groups, 8), np.dtype(jnp.bfloat16)),
                        NamedSharding(mesh, P(None, "expert")),
                    )
                    for name in ("k_scale", "v_scale")
                }
            )
            insert = jax.jit(
                lambda cache, x, active: write_quantized(cache, x, x, jnp.array([3, 7]), active, c)
            )
            cache = insert(cache, x, jnp.array([True, True]))
            decode = jax.jit(
                lambda cache: read_quantized(cache, "k", jnp.array([[3], [7]]), c, jnp.float32)
            )
            reconstructed = codes.astype(np.float32) * scale.astype(np.float32)[..., None]
            np.testing.assert_array_equal(
                np.asarray(decode(cache))[:, 0], reconstructed.reshape(host.shape)
            )
            changed = insert(cache, x * 0.5, jnp.array([True, False]))
            for name in ("k", "v", "k_scale", "v_scale"):
                old = np.asarray(cache[name]).reshape(2, -1)
                new = np.asarray(changed[name]).reshape(2, -1)
                np.testing.assert_array_equal(old[1], new[1])
    if not any(item["unrounded_divisor_changes"] for item in details):
        raise ValueError("Codec oracle did not distinguish the v3 rounding failure")
    return dict(
        status="PASS",
        codec_shapes=details,
        independent_reference="numpy_ml_dtypes_bf16_rounding",
        scale_rounding="explicit_integer_round_to_nearest_even",
        cache_write_read="PASS",
        finished_row_cache="PASS",
    )


def tiny_model(
    mesh,
    fixture="tiny-reference",
    *,
    kv_cache_dtype=None,
    expert_compute="bf16",
    dtype=jnp.float32,
    expert_group_size=1,
    expert_tensor_parallel=1,
    fuse_layers=True,
    gdn_prefill="serial",
    prefill_expert_group_size=None,
    kv_scale_gather="indexed",
):
    directory = Path(__file__).with_name("fixtures")
    manifest = json.loads((directory / (fixture + ".json")).read_text())
    path = directory / (fixture + ".npz")
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["fixture_sha256"]:
        raise ValueError("Golden fixture checksum differs")
    a = np.load(path)
    fields = manifest["config"]
    for key in ("mrope_sections", "generation_eos"):
        fields[key] = tuple(fields[key])
    c, v = ModelConfig(**fields), VisionConfig(**manifest["vision_config"])
    c = replace(c, kv_scale_gather=kv_scale_gather)
    if kv_cache_dtype is not None:
        c = replace(c, kv_cache_dtype=kv_cache_dtype)
    if dtype == jnp.bfloat16 and fixture == "long-reference":
        c = replace(c, qsa_budget=64)
    replica = NamedSharding(mesh, P())
    moe_mesh = expert_mesh(mesh, expert_tensor_parallel)
    original_intermediate = c.intermediate
    if c.intermediate % (32 * expert_tensor_parallel):
        # The frozen toy has only one 32-value down block. Append exact zero
        # channels to exercise TP without splitting a quantization block or
        # changing its function/reference. Production I=640 needs no padding.
        padded = (
            ((c.intermediate + 32 * expert_tensor_parallel - 1) // (32 * expert_tensor_parallel))
            * 32
            * expert_tensor_parallel
        )
        c = replace(c, intermediate=padded)

    def expert_bytes(layer, kind):
        value = a[f"{'bf16_packed' if dtype == jnp.bfloat16 else 'packed'}.{layer}.{kind}"]
        extra = c.intermediate - original_intermediate
        if extra:
            padding = [(0, 0)] * value.ndim
            padding[2 if kind == "down" else 1] = (0, extra // 32 if kind == "down" else extra)
            value = np.pad(value, padding)
        return put_expert_bytes(value, kind, "Q8_0", c.hidden, c.intermediate, moe_mesh)

    def array(name, plus_one=False, squeeze=False):
        value = a[name]
        if plus_one:
            value = value + 1
        if squeeze:
            value = np.squeeze(value)
        # Production loader keeps one-dimensional norms/state coefficients
        # and router logits FP32; main linear and convolution weights are BF16.
        rounded = (
            value.ndim >= 2
            or name.endswith("shared_expert_gate.weight")
            or (name.startswith("model.visual") and name.endswith("bias") and "norm" not in name)
        )
        if dtype == jnp.bfloat16 and rounded and not name.endswith("mlp.gate.weight"):
            value = jnp.asarray(value, dtype)
        return jax.device_put(value, replica)

    def gr(prefix):
        result = {
            "norm": array(prefix + ".hc_norm.weight", True),
            "down": array(prefix + ".input_mix_weight_down.weight"),
            "up": array(prefix + ".input_mix_weight_up.weight"),
        }
        if prefix + ".block_inject_weight.weight" in a:
            result["inject"] = array(prefix + ".block_inject_weight.weight")
        return result

    layers, packed, functions = [], [], []
    for i in range(c.layers):
        prefix = f"model.language_model.layers.{i}."
        w = dict(
            attn_gr=gr(prefix + "attn_hyper_connection"), ffn_gr=gr(prefix + "mlp_hyper_connection")
        )
        w["mlp"] = {
            target: array(prefix + source, squeeze=target == "shared_router")
            for target, source in (
                ("router", "mlp.gate.weight"),
                ("shared_router", "mlp.shared_expert_gate.weight"),
                ("shared_gate", "mlp.shared_expert.gate_proj.weight"),
                ("shared_up", "mlp.shared_expert.up_proj.weight"),
                ("shared_down", "mlp.shared_expert.down_proj.weight"),
            )
        }
        if c.is_qsa(i):
            w["attn"] = {
                target: array(prefix + "self_attn." + source, plus_one=target.endswith("norm"))
                for target, source in (
                    ("q", "q_proj.weight"),
                    ("k", "k_proj.weight"),
                    ("v", "v_proj.weight"),
                    ("out", "o_proj.weight"),
                    ("q_norm", "q_norm.weight"),
                    ("k_norm", "k_norm.weight"),
                )
            }
            if c.sparse_qsa:
                index = prefix + "self_attn.indexer."
                combined = array(index + "index_qk_proj.weight")
                split = c.indexer_heads * c.indexer_dim
                w["attn"].update(
                    index_q=combined[:split],
                    index_k=combined[split:],
                    index_q_norm=array(index + "q_layernorm.weight", True),
                    index_k_norm=array(index + "k_layernorm.weight", True),
                )
        else:
            w["attn"] = {
                target: array(prefix + "linear_attn." + source, squeeze=target == "conv")
                for target, source in (
                    ("qkv", "in_proj_qkv.weight"),
                    ("gate", "in_proj_z.weight"),
                    ("out", "out_proj.weight"),
                    ("alpha", "in_proj_a.weight"),
                    ("beta", "in_proj_b.weight"),
                    ("dt", "dt_bias"),
                    ("conv", "conv1d.weight"),
                    ("norm", "norm.weight"),
                )
            }
            w["attn"]["a"] = -numerics.exp(array(prefix + "linear_attn.A_log"))
        if i == c.ple_layer:
            w["ple"] = {
                target: array(
                    prefix + "ple." + source,
                    plus_one=target.endswith("norm"),
                    squeeze=target == "conv",
                )
                for target, source in (
                    ("key", "key_proj.weight"),
                    ("value", "value_proj.weight"),
                    ("key_norm", "norm_key.weight"),
                    ("query_norm", "norm_query.weight"),
                    ("conv_norm", "norm_conv.weight"),
                    ("conv", "conv1d.weight"),
                )
            }
        layers.append(w)
        packed.append({kind: expert_bytes(i, kind) for kind in ("gate", "up", "down")})
        functions.append(
            expert_parallel(
                mesh,
                gate_type="Q8_0",
                up_type="Q8_0",
                down_type="Q8_0",
                hidden=c.hidden,
                intermediate=c.intermediate,
                compute=expert_compute,
                group_size=expert_group_size,
                tensor_parallel=expert_tensor_parallel,
            )
        )
    engine = Engine(
        c,
        mesh,
        layers,
        packed,
        functions,
        gr("model.language_model.hyper_connection_mixer"),
        array("lm_head.weight"),
        log=lambda *args: None,
        fuse_layers=fuse_layers,
        gdn_prefill=gdn_prefill,
        prefill_expert_functions=[
            expert_parallel(
                mesh,
                gate_type="Q8_0",
                up_type="Q8_0",
                down_type="Q8_0",
                hidden=c.hidden,
                intermediate=c.intermediate,
                compute=expert_compute,
                group_size=prefill_expert_group_size,
                tensor_parallel=expert_tensor_parallel,
            )
        ]
        * c.layers
        if prefill_expert_group_size is not None
        else None,
    )

    def pair(prefix):
        return {suffix: array(prefix + "." + suffix) for suffix in ("weight", "bias")}

    patch = pair("model.visual.patch_embed.proj")
    patch["weight"] = patch["weight"].reshape(v.hidden, -1)
    vision = dict(
        patch=patch,
        position=array("model.visual.pos_embed.weight"),
        post_norm=pair("model.visual.merger.norm"),
        merge_up=pair("model.visual.merger.linear_fc1"),
        merge_down=pair("model.visual.merger.linear_fc2"),
        blocks=[
            {
                target: pair(f"model.visual.blocks.{i}.{source}")
                for target, source in (
                    ("ln1", "norm1"),
                    ("ln2", "norm2"),
                    ("qkv", "attn.qkv"),
                    ("out", "attn.proj"),
                    ("up", "mlp.linear_fc1"),
                    ("down", "mlp.linear_fc2"),
                )
            }
            for i in range(v.layers)
        ],
    )
    return engine, VisionEncoder(vision, v), a


def validate_batch_shapes(mesh, batches):
    engine, _, a = tiny_model(mesh)
    c = engine.config
    table = a["model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight"]
    tokens = a["model.language_model.embed_tokens.weight"]
    results = []
    for batch in batches:
        source = np.arange(batch) % 2
        ids = a["ids"][source]
        hasher = NGramHash(c)
        history = hasher.initial_history(batch)
        features = []
        for step in range(ids.shape[1]):
            hashes, history = hasher.step(ids[:, step], history)
            features.append(table[hashes].reshape(batch, c.ple_dim))
        embedding = tokens[ids].copy()
        embedding[:, 2:6] = a["vision"].reshape(2, 4, c.hidden)[source]
        engine.reset(batch)
        engine.forward_chunk(
            embedding,
            np.stack(features, 1),
            a["positions"].transpose(1, 2, 0)[source],
            np.ones(ids.shape, bool),
        )
        expected = a["prefill_logits"][source, -1]
        actual = np.asarray(engine.logits())
        np.testing.assert_allclose(actual, expected, atol=4e-5, rtol=4e-5)
        maximum = float(np.max(np.abs(actual - expected)))
        for step in range(3):
            decoded = a["decode_ids"][source, step]
            hashes, history = hasher.step(decoded, history)
            pos = np.repeat((12 + step + a["rope_delta"][source]).reshape(batch, 1), 3, axis=1)
            engine.forward(
                tokens[decoded], table[hashes].reshape(batch, c.ple_dim), pos, np.ones(batch, bool)
            )
            expected = a["decode_logits"][source, step]
            actual = np.asarray(engine.logits())
            np.testing.assert_allclose(actual, expected, atol=4e-5, rtol=4e-5)
            maximum = max(maximum, float(np.max(np.abs(actual - expected))))
        before = jax.tree.map(lambda x: np.asarray(x).copy(), engine.state)
        active = np.arange(batch) % 2 == 0
        engine.forward(
            tokens[np.full(batch, 32)],
            np.zeros((batch, c.ple_dim), np.float32),
            np.full((batch, 3), 16),
            active,
        )
        for old, new in zip(jax.tree.leaves(before), jax.tree.leaves(engine.state), strict=True):
            np.testing.assert_array_equal(old[~active], np.asarray(new)[~active])
        results.append(dict(batch=batch, max_logit_error=maximum, finished_row_cache="PASS"))
    engine.release_state()
    return results


def quantized_cache_diagnostics(engine, reference=None):
    """Bounded failure evidence from the actual compiled cache, without replay.

    Retain the last eight tokens and compare the entire written prefix when
    the frozen fixture has matching cache values. This would have localized
    v5's first key-code and value-scale changes immediately.
    """
    result = {}
    c = engine.config
    for layer, cache in enumerate(engine.state["layers"]):
        if not c.is_qsa(layer) or c.kv_cache_dtype != "q8":
            continue
        batch = len(engine.lengths)
        stop = int(np.max(engine.lengths))
        start = max(0, stop - 8)
        item = dict(token_start=start, token_stop=stop)
        indices = jnp.broadcast_to(jnp.arange(stop), (batch, stop))
        for name, golden_name in (("k", "q8_keys"), ("v", "q8_values")):
            codes = np.asarray(cache[name])[:, :stop]
            scales = (
                np.asarray(cache[name + "_scale"])
                .reshape(batch, -1, c.max_context)[:, :, :stop]
                .transpose(0, 2, 1)
                .astype(np.float32)
            )
            reconstructed = np.asarray(read_quantized(cache, name, indices, c, jnp.float32))
            entry = dict(codes=codes[:, start:].tolist(), scales=scales[:, start:].tolist())
            if reference is not None:
                expected = reference[golden_name][:, :, :stop].transpose(0, 2, 1, 3)
                error = reconstructed - expected
                changed = np.argwhere(error != 0)
                entry.update(
                    reference_mismatches=len(changed),
                    max_reconstruction_error=float(np.max(np.abs(error))),
                    first_changes=[
                        dict(
                            index=ix.tolist(),
                            actual=float(reconstructed[tuple(ix)]),
                            expected=float(expected[tuple(ix)]),
                        )
                        for ix in changed[:32]
                    ],
                )
            item[name] = entry
        result[str(layer)] = item
    return result


def validate_long_context(
    mesh,
    kv_cache_dtype="q8",
    *,
    expert_compute="bf16",
    dtype=jnp.float32,
    diagnostics=None,
    expert_group_size=1,
    expert_tensor_parallel=1,
    gdn_prefill="serial",
    prefill_expert_group_size=None,
    fuse_layers=True,
    kv_scale_gather="indexed",
):
    """Exercise sparse selection beyond its budget against independent Torch.

    Q8 uses separately generated quantized Torch K/V logits. It is never
    compared to an unquantized reference with a relaxed tolerance.
    """
    engine, vision, a = tiny_model(
        mesh,
        "long-reference",
        kv_cache_dtype=kv_cache_dtype,
        expert_compute=expert_compute,
        dtype=dtype,
        expert_group_size=expert_group_size,
        expert_tensor_parallel=expert_tensor_parallel,
        gdn_prefill=gdn_prefill,
        prefill_expert_group_size=prefill_expert_group_size,
        fuse_layers=fuse_layers,
        kv_scale_gather=kv_scale_gather,
    )
    c = engine.config
    ids, batch = a["ids"], a["ids"].shape[0]
    steps = ids.shape[1]
    table = a["model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight"]
    tokens = a["model.language_model.embed_tokens.weight"]
    hasher, history = NGramHash(c), NGramHash(c).initial_history(batch)
    features = []
    for i in range(steps):
        hashes, history = hasher.step(ids[:, i], history)
        features.append(table[hashes].reshape(batch, c.ple_dim))
    features = np.stack(features, 1)
    embeddings = tokens[ids].copy()
    encoded = a["vision"]
    if dtype == jnp.bfloat16:
        encoded = np.concatenate(
            [
                np.asarray(
                    vision.encode_patches(a["pixels"][i * 16 : (i + 1) * 16], (1, 4, 4)), np.float32
                )
                for i in range(batch)
            ]
        )
        np.testing.assert_allclose(encoded, a["bf16_vision"], atol=0.015, rtol=0.015)
    embeddings[:, 2:6] = encoded.reshape(batch, 4, c.hidden)
    positions = a["positions"].transpose(1, 2, 0)
    prefix = "q8_" if kv_cache_dtype == "q8" else ""
    if expert_compute == "int8_q8":
        if kv_cache_dtype != "q8":
            raise ValueError("Integer full-model oracle requires Q8 KV")
        prefix = "int8_q8_"
    if dtype == jnp.bfloat16:
        if kv_cache_dtype != "q8" or expert_compute != "bf16":
            raise ValueError("BF16 precision oracle requires Q8 KV and BF16 experts")
        prefix = "bf16_q8_"
    # A8 rounding is discontinuous: a few-micro-ULP float reduction difference
    # can move an activation by one code. Exact integer dots are checked
    # separately, and the full-model bound remains far below a wrong policy.
    tolerance = 0.025 if dtype == jnp.bfloat16 else (1e-4 if expert_compute == "int8_q8" else 4e-5)
    # Chunked prefill is algebraically identical to the serial scan but
    # re-associates fp32 reductions over 64-token chunks, so the
    # serial cross-check budget is wider than the golden-oracle tolerance; a
    # wrong chunk rule still shows O(1) differences. Recurrent state drifts
    # further than logits across 64 steps (the CPU chunk-vs-scan unit test
    # allows 2e-3 on state).
    chunk_tolerance = (
        0.025 if dtype == jnp.bfloat16 else (5e-4 if gdn_prefill == "chunked" else tolerance)
    )
    chunk_state_tolerance = (
        0.025 if dtype == jnp.bfloat16 else (5e-3 if gdn_prefill == "chunked" else tolerance)
    )
    engine.reset(batch)
    maximum = 0.0
    maximum_rms = 0.0

    def check_logits(actual, expected, label, limit=tolerance):
        nonlocal maximum_rms
        actual = np.asarray(actual).astype(np.float32)
        expected = np.asarray(expected).astype(np.float32)
        error = actual - expected
        rms_error = float(np.sqrt(np.mean(error**2)))
        close = np.allclose(actual, expected, atol=limit, rtol=limit)
        rms_close = dtype != jnp.bfloat16 or rms_error <= 0.010
        if diagnostics is not None:
            comparison = dict(
                label=label,
                absolute_tolerance=limit,
                relative_tolerance=limit,
                max_absolute_error=float(np.max(np.abs(error))),
                rms_error=rms_error,
            )
            diagnostics.setdefault("comparisons", []).append(dict(comparison))
            diagnostics["last_comparison"] = comparison
            if not close or not rms_close:
                comparison.update(actual_logits=actual.tolist(), expected_logits=expected.tolist())
                diagnostics["failure_q8_cache"] = quantized_cache_diagnostics(
                    engine, a if prefix == "q8_" and label.startswith("q8 serial") else None
                )
        np.testing.assert_allclose(actual, expected, atol=limit, rtol=limit, err_msg=label)
        maximum_rms = max(maximum_rms, rms_error)
        if dtype == jnp.bfloat16 and rms_error > 0.010:
            raise ValueError(f"{label} exceeds fixed 0.010 RMS precision budget: {rms_error}")
        return float(np.max(np.abs(error)))

    for i in range(steps):
        engine.forward(embeddings[:, i], features[:, i], positions[:, i], [True] * batch)
        actual, expected = np.asarray(engine.logits()), a[prefix + "prefill_logits"][:, i]
        if i == 0 and diagnostics is not None:
            first = dict(actual_logits=actual.tolist(), expected_logits=expected.tolist())
            if kv_cache_dtype == "q8":
                first["q8_cache"] = {}
                indices = jnp.zeros((batch, 1), jnp.int32)
                for layer, cache in enumerate(engine.state["layers"]):
                    if not c.is_qsa(layer):
                        continue
                    first["q8_cache"][str(layer)] = {
                        name: dict(
                            codes=np.asarray(cache[name])[:, 0].tolist(),
                            scales=np.asarray(cache[name + "_scale"])[:, 0]
                            .reshape(batch, -1)
                            .astype(np.float32)
                            .tolist(),
                            reconstructed=np.asarray(
                                read_quantized(cache, name, indices, c, jnp.float32)
                            )[:, 0].tolist(),
                        )
                        for name in ("k", "v")
                    }
            diagnostics["first_token"] = first
        maximum = max(
            maximum, check_logits(actual, expected, f"{kv_cache_dtype} serial prefill token {i}")
        )
    serial = jax.tree.map(lambda x: np.asarray(x).copy(), engine.state)
    engine.reset(batch)
    for first in range(0, steps, 16):
        last = min(first + 16, steps)
        engine.forward_chunk(
            embeddings[:, first:last],
            features[:, first:last],
            positions[:, first:last],
            np.ones((batch, last - first), bool),
        )
    maximum = max(
        maximum,
        check_logits(
            engine.logits(),
            a[prefix + "prefill_logits"][:, -1],
            "chunked prefill",
            limit=chunk_tolerance,
        ),
    )
    # Quantized caches can legitimately cross a rounding boundary after tiny
    # CPU reduction differences; logits still match their independent oracle.
    for old, new in zip(jax.tree.leaves(serial), jax.tree.leaves(engine.state), strict=True):
        if old.dtype != np.int8:
            np.testing.assert_allclose(
                np.asarray(new).astype(np.float32),
                old.astype(np.float32),
                atol=chunk_state_tolerance,
                rtol=chunk_state_tolerance,
            )
    for i in range(3):
        decoded = a["decode_ids"][:, i]
        hashes, history = hasher.step(decoded, history)
        pos = np.repeat((steps + i + a["rope_delta"]).reshape(batch, 1), 3, axis=1)
        engine.forward(
            tokens[decoded], table[hashes].reshape(batch, c.ple_dim), pos, [True] * batch
        )
        actual, expected = np.asarray(engine.logits()), a[prefix + "decode_logits"][:, i]
        maximum = max(maximum, check_logits(actual, expected, f"cached decode {i}"))
    before = jax.tree.map(lambda x: np.asarray(x).copy(), engine.state)
    engine.forward(
        tokens[[32, 33]],
        np.zeros((batch, c.ple_dim), np.float32),
        np.full((batch, 3), steps + 4),
        [True, False],
    )
    for old, new in zip(jax.tree.leaves(before), jax.tree.leaves(engine.state), strict=True):
        # Q8 scales flatten batch/KV/groups along their first axis.
        old = old.reshape(batch, -1)
        new = np.asarray(new).reshape(batch, -1)
        np.testing.assert_array_equal(old[1], new[1])
    engine.release_state()
    drift = float(np.max(np.abs(a["q8_prefill_logits"] - a["prefill_logits"])))
    if drift > 0.005:
        raise ValueError("Q8 full-model fixture drift exceeds fixed 0.005 logit budget")
    return dict(
        status="PASS",
        kv_cache_dtype=kv_cache_dtype,
        kv_scale_gather=kv_scale_gather,
        expert_compute=expert_compute,
        expert_group_size=expert_group_size,
        expert_tensor_parallel=expert_tensor_parallel,
        gdn_prefill=gdn_prefill,
        prefill_expert_group_size=prefill_expert_group_size,
        fuse_layers=fuse_layers,
        dtype=str(dtype),
        oracle_scope="BF16 compute precision with equal experts/all-token toy attention"
        if dtype == jnp.bfloat16
        else "random routed experts and nontrivial sparse selection",
        logit_tolerance=tolerance,
        chunk_logit_tolerance=chunk_tolerance,
        context_capacity=c.max_context,
        token_budget=c.qsa_budget,
        prompt_tokens=steps,
        max_logit_error=maximum,
        max_rms_error=maximum_rms,
        bf16_rms_budget=0.010 if dtype == jnp.bfloat16 else None,
        independent_reference="pinned_torch_quantized_kv" if prefix else "pinned_torch",
        q8_reference_drift=drift,
        chunk_prefill="PASS",
        finished_row_cache="PASS",
    )


def validate(mesh=None, batch_sizes=()):
    mesh = mesh or Mesh(np.asarray(jax.devices()[:1]), ("expert",))
    engine, vision, a = tiny_model(mesh)
    c = engine.config
    encoded = np.concatenate(
        [
            np.asarray(vision.encode_patches(a["pixels"][i * 16 : (i + 1) * 16], (1, 4, 4)))
            for i in range(2)
        ]
    )
    np.testing.assert_allclose(encoded, a["vision"], atol=3e-5, rtol=3e-5)
    positions = [text_positions(ids, [(1, 4, 4)], c) for ids in a["ids"]]
    np.testing.assert_array_equal(
        np.stack([p[0] for p in positions], 1).transpose(2, 1, 0), a["positions"]
    )
    table = a["model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight"]
    tokens = a["model.language_model.embed_tokens.weight"]
    hasher = NGramHash(c)
    history = hasher.initial_history(2)
    engine.reset(2)
    maximum = 0.0
    for step in range(a["ids"].shape[1]):
        ids = a["ids"][:, step]
        hashes, history = hasher.step(ids, history)
        embedding = tokens[ids].copy()
        if 2 <= step < 6:
            embedding = encoded.reshape(2, 4, c.hidden)[:, step - 2]
        engine.forward(
            embedding,
            table[hashes].reshape(2, c.ple_dim),
            a["positions"][:, :, step].T,
            [True, True],
        )
        logits = np.asarray(engine.logits())
        expected = a["prefill_logits"][:, step]
        np.testing.assert_allclose(logits, expected, atol=4e-5, rtol=4e-5)
        maximum = max(maximum, float(np.max(np.abs(logits - expected))))
    for step in range(3):
        ids = a["decode_ids"][:, step]
        hashes, history = hasher.step(ids, history)
        pos = np.repeat((12 + step + a["rope_delta"]).reshape(2, 1), 3, axis=1)
        engine.forward(tokens[ids], table[hashes].reshape(2, c.ple_dim), pos, [True, True])
        logits = np.asarray(engine.logits())
        np.testing.assert_allclose(logits, a["decode_logits"][:, step], atol=4e-5, rtol=4e-5)
        maximum = max(maximum, float(np.max(np.abs(logits - a["decode_logits"][:, step]))))
    # Finished rows retain every cache and output; active row advances alone.
    before = jax.tree.map(lambda x: np.asarray(x).copy(), engine.state)
    engine.forward(
        tokens[[32, 33]], np.zeros((2, c.ple_dim), np.float32), np.full((2, 3), 16), [True, False]
    )
    after = jax.tree.map(np.asarray, engine.state)
    for previous, current in zip(jax.tree.leaves(before), jax.tree.leaves(after), strict=True):
        np.testing.assert_array_equal(previous[1], current[1])
    engine.release_state()
    return dict(
        status="PASS",
        devices=mesh.size,
        max_logit_error=maximum,
        vision_max_error=float(np.max(np.abs(encoded - a["vision"]))),
        prefill_tokens=24,
        cached_decode_tokens=6,
        finished_row_cache="PASS",
        batch_shapes=validate_batch_shapes(mesh, batch_sizes) if batch_sizes else [],
    )


if __name__ == "__main__":
    print(json.dumps(validate(Mesh(np.asarray(jax.devices()), ("expert",)))))
