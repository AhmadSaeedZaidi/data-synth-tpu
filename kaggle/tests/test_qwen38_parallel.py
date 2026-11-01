"""EP/TP byte placement, native arithmetic, routing and full-model contracts."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh
from qwen38_tpu.checkpoint import Checkpoint
from qwen38_tpu.expert_calibration import CANDIDATES, select_candidate
from qwen38_tpu.parallel import expert_mesh, split_expert_bytes
from qwen38_tpu.routing_validation import validate_expert_routing
from qwen38_tpu.validation import tiny_model, validate_long_context


@pytest.mark.parametrize("tp,group", [(1, 4), (2, 4), (2, 1), (2, 2)])
@pytest.mark.parametrize("batch", [1, 17, 273])
def test_grouped_hybrid_preserves_routes_independent_numpy(tp, group, batch):
    result = validate_expert_routing(
        Mesh(np.asarray(jax.devices()), ("expert",)),
        group_size=group,
        tensor_parallel=tp,
        batch=batch,
    )
    assert result["status"] == "PASS"
    assert not result["expert_weights_all_gathered"]


@pytest.mark.parametrize(
    "tp,policy,dtype",
    [
        (1, "bf16", jnp.float32),
        (1, "bf16", jnp.bfloat16),
        (1, "int8_q8", jnp.float32),
        (2, "bf16", jnp.float32),
        (2, "bf16", jnp.bfloat16),
        (2, "int8_q8", jnp.float32),
    ],
)
def test_grouped_hybrid_matches_frozen_full_model(tp, policy, dtype):
    result = validate_long_context(
        Mesh(np.asarray(jax.devices()), ("expert",)),
        expert_compute=policy,
        dtype=dtype,
        expert_group_size=4,
        expert_tensor_parallel=tp,
    )
    assert result["status"] == "PASS"


@pytest.mark.parametrize("intermediate", [64, 640])
def test_loader_owns_unique_bytes_and_matching_intermediate_slices(intermediate):
    mesh = expert_mesh(Mesh(np.asarray(jax.devices()), ("expert",)), 2)
    hidden, count = 256, 24
    rng = np.random.default_rng(101)
    checkpoint = Checkpoint.__new__(Checkpoint)
    checkpoint.log = lambda *args: None
    checkpoint.tensors = {}
    raw = {}
    for kind in ("gate", "up", "down"):
        name = f"blk.0.ffn_{kind}_exps.weight"
        width, rows, qtype, block, size = (
            (intermediate, hidden, "Q8_0", 32, 34)
            if kind == "down"
            else (hidden, intermediate, "Q6_K", 256, 210)
        )
        raw[name] = rng.integers(0, 256, (count, rows, width // block, size), np.uint8)
        checkpoint.tensors[name] = dict(
            dims=[width, rows, count], type=qtype, nbytes=raw[name].nbytes
        )
    reads = []

    def read(name, offset, length):
        reads.append((name, offset, length))
        return raw[name].tobytes()[offset : offset + length]

    checkpoint.read = read
    loaded = checkpoint.resident_experts(0, mesh)
    for kind, array in loaded.items():
        name = f"blk.0.ffn_{kind}_exps.weight"
        assert array.nbytes == raw[name].nbytes
        assert sum(s.data.nbytes for s in array.addressable_shards) == raw[name].nbytes
        assert sum(n for key, _, n in reads if key == name) == raw[name].nbytes
        pieces = []
        for s in array.addressable_shards:
            if kind == "down":
                pieces.append(np.asarray(s.data).reshape(6, hidden, intermediate // 64, 34))
            else:
                pieces.append(np.asarray(s.data).reshape(6, intermediate // 2, 1, 210))
        reconstructed = np.concatenate(
            [
                np.concatenate(pieces[i : i + 2], axis=2 if kind == "down" else 1)
                for i in range(0, 8, 2)
            ],
            axis=0,
        )
        np.testing.assert_array_equal(reconstructed, raw[name])


def test_rejects_quantization_blocks_split_between_tensor_chips():
    with pytest.raises(ValueError, match="whole down"):
        split_expert_bytes(np.zeros((8, 32 * 34), np.uint8), "down", "Q8_0", 32, 32, 2)


def test_fused_layer_donates_all_cache_buffers():
    engine, _, _ = tiny_model(
        Mesh(np.asarray(jax.devices()), ("expert",)),
        "long-reference",
        kv_cache_dtype="q8",
        expert_group_size=4,
        expert_tensor_parallel=2,
    )
    engine.reset(8)
    c = engine.config
    for layer in (0, 3):
        for chunk in (False, True):
            shape = (8, 2) if chunk else (8,)
            cache = engine.state["layers"][layer]
            args = (
                jnp.zeros((*shape, c.streams * c.hidden)),
                engine.layers[layer],
                cache,
                jnp.zeros(8, jnp.int32),
                jnp.zeros((*shape, 3), jnp.int32),
                jnp.ones(shape, bool),
                engine.experts[layer],
            )
            fn = (engine.prefill_layers if chunk else engine.decode_layers)[layer]
            compiled = fn.lower(*args).compile()
            assert compiled.memory_analysis().alias_size_in_bytes == sum(
                leaf.addressable_shards[0].data.nbytes for leaf in jax.tree.leaves(cache)
            )
            before = [leaf.sharding for leaf in jax.tree.leaves(cache)]
            _, updated = compiled(*args)
            assert all(
                old.is_equivalent_to(new.sharding, new.ndim)
                for old, new in zip(before, jax.tree.leaves(updated), strict=True)
            )
            engine.state["layers"][layer] = updated
    engine.release_state()


def test_calibration_requires_accuracy_complete_coverage_and_no_large_regression():
    measurements = []
    for candidate in CANDIDATES:
        for quant in ("mixed_q6_q8", "q8"):
            for rows in (1, 16, 128):
                measurements.append(
                    dict(
                        candidate=candidate["name"],
                        quant=quant,
                        rows=rows,
                        routing="spread",
                        accuracy_pass=True,
                        temporary_bytes=1024,
                        seconds=0.6 if candidate["name"].startswith("grouped_") else 1,
                    )
                )
    assert select_candidate(measurements)["selected"]["group_size"] == 4
    # Even a very fast path is not eligible if it gets the output wrong.
    for row in measurements:
        if row["candidate"] == "grouped_ep8":
            row["accuracy_pass"] = False
    assert select_candidate(measurements)["selected"]["tensor_parallel"] == 2
    # A large B1 regression rejects an otherwise fast hybrid layout.
    bad = next(r for r in measurements if r["candidate"] == "grouped_ep4_tp2")
    bad["seconds"] = 1.3
    assert select_candidate(measurements)["selected"]["name"] == "serial_ep8"
    # Missing a slow measurement cannot make that candidate win.
    assert (
        select_candidate([r for r in measurements if r is not bad])["selected"]["name"]
        == "serial_ep8"
    )
    bad["seconds"] = 0.6
    bad["temporary_bytes"] = 257 << 20
    assert select_candidate(measurements)["selected"]["name"] == "serial_ep8"
