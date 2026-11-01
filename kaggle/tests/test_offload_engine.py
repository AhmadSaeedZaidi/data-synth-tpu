"""CPU checks for packed host expert selection, image overrides, and independent decode rows."""

import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import dataclasses  # noqa: E402
import json  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from glm53 import model as M  # noqa: E402
from glm53.engine import AXIS, R, shard_map  # noqa: E402
from glm53_offload.engine import PackedOffloadEngine  # noqa: E402
from glm53_offload.expert_kernel import tiled_experts  # noqa: E402
from glm53_offload.gguf_quant import block_layout, dequantize_rows  # noqa: E402
from glm53_offload.routing import expert_tile_capacities, expert_token_rows  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

N_DEV = 8
D = 8
VOCAB = 16
N_EXPERTS = 8
MOE_INTER = 16
TOPK = 2
MAX_LEN = 64


def test_execution_sampling_has_independent_bounded_purposes_and_restores_on_error(tmp_path):
    engine = object.__new__(PackedOffloadEngine)
    engine.execution_log_path = tmp_path / "execution.jsonl"
    engine.execution_samples, engine._sample_counts, engine._offload_progs = [], {}, {}
    engine._sample_layer, engine._sample_layers = 0, (0,)
    engine._sample_purpose, engine._sample_enabled = "startup", True
    engine.expert_cache = None
    engine.store = SimpleNamespace(stats={"bytes_transferred": 0})

    def complete(layer):
        sample = engine._start_sample("decode", layer, 8)
        assert sample is not None
        engine._sample_router_ready(sample)
        engine._sample_packed_ready(sample, np.zeros(1), 8)
        engine._finish_sample(sample, np.zeros(1))

    complete(0)
    complete(0)
    assert engine._start_sample("decode", 0, 8) is None
    with engine.execution_sampling("measurement", enabled=False):
        assert engine._start_sample("decode", 0, 8) is None
    with pytest.raises(RuntimeError, match="interrupted"):
        with engine.execution_sampling("matched_images", layers=[0, 21, 44]):
            for layer in (0, 21, 44):
                complete(layer)
                complete(layer)
                assert engine._start_sample("decode", layer, 8) is None
            raise RuntimeError("interrupted")
    assert (engine._sample_purpose, engine._sample_enabled, engine._sample_layers) == (
        "startup",
        True,
        (0,),
    )
    assert engine._start_sample("decode", 0, 8) is None
    assert len(engine.execution_samples) == 8
    assert [(s["purpose"], s["layer"]) for s in engine.execution_samples] == (
        [("startup", 0)] * 2
        + [("matched_images", layer) for layer in (0, 21, 44) for _ in range(2)]
    )
    records = [json.loads(line) for line in engine.execution_log_path.read_text().splitlines()]
    assert [r["status"] for r in records] == ["START", "DONE"] * 8


def test_expert_usage_separates_image_text_and_decode_without_changing_routes():
    engine = object.__new__(PackedOffloadEngine)
    engine.cfg = SimpleNamespace(n_exp=8)
    engine.expert_usage_by_layer = {}
    routes = np.array([[1, 3], [1, 2], [3, 4]], np.int32)
    original = routes.copy()
    engine._record_expert_usage(3, routes, "prefill", [True, False, True])
    engine._record_expert_usage(3, routes, "decode")
    image = engine.expert_usage_by_layer[3]["image_positions"]
    text = engine.expert_usage_by_layer[3]["text_positions"]
    assert image["tokens"] == 2
    assert image["assignments"] == [0, 1, 0, 2, 1, 0, 0, 0]
    assert image["calls_with_expert"] == [0, 1, 0, 1, 1, 0, 0, 0]
    assert text["assignments"] == [0, 1, 1, 0, 0, 0, 0, 0]
    assert engine.expert_usage_by_layer[3]["decode"]["assignments"] == [0, 2, 1, 2, 1, 0, 0, 0]
    np.testing.assert_array_equal(routes, original)


def _cfg():
    return M.Cfg(
        hidden=D,
        vocab=VOCAB,
        n_layers=1,
        layer_types=("linear_attention",),
        mlp_types=("sparse",),
        eps=1e-5,
        kda_heads=8,
        kda_hd=1,
        conv_k=3,
        gate_lb=None,
        n_heads=8,
        q_lora=4,
        kv_lora=4,
        qk_nope=1,
        v_hd=1,
        idx_heads=1,
        idx_hd=2,
        idx_topk=0,
        idx_kpool=1,
        hc=1,
        hc_iters=2,
        hc_eps=1e-6,
        inter=8,
        moe_inter=MOE_INTER,
        n_exp=N_EXPERTS,
        topk=TOPK,
        n_shared=1,
        scaling=1.0,
        norm_topk=False,
        swiglu_limit=7.0,
        dtype=jnp.float32,
    )


def _lin(rng, shape, scale=0.08):
    return rng.normal(0.0, scale, shape).astype(np.float32)


def _make_params(seed=19):
    rng = np.random.default_rng(seed)
    H, hd, conv_k = 8, 1, 3
    layer = {
        "ln1": np.ones((D,), np.float32),
        "ln2": np.ones((D,), np.float32),
        "hc_attn": {
            "fn": _lin(rng, (D, 3)),
            "base": _lin(rng, (3,)),
            "scale": _lin(rng, (3,)),
        },
        "hc_ffn": {
            "fn": _lin(rng, (D, 3)),
            "base": _lin(rng, (3,)),
            "scale": _lin(rng, (3,)),
        },
        "attn": {
            "q": _lin(rng, (D, H * hd)),
            "k": _lin(rng, (D, H * hd)),
            "v": _lin(rng, (D, H * hd)),
            "conv_q": _lin(rng, (H * hd, conv_k)),
            "conv_k": _lin(rng, (H * hd, conv_k)),
            "conv_v": _lin(rng, (H * hd, conv_k)),
            "f_a": _lin(rng, (D, hd)),
            "f_b": _lin(rng, (hd, H * hd)),
            "dt_bias": np.zeros((H * hd,), np.float32),
            "A_log": np.zeros((H,), np.float32),
            "b": _lin(rng, (D, H)),
            "g_a": _lin(rng, (D, hd)),
            "g_b": _lin(rng, (hd, H * hd)),
            "o_norm": np.ones((hd,), np.float32),
            "o": _lin(rng, (H * hd, D)),
        },
        "mlp": {
            "router_w": _lin(rng, (D, N_EXPERTS)),
            "router_bias": _lin(rng, (N_EXPERTS,), scale=0.2),
            "shared": {
                "gate": _lin(rng, (D, 8)),
                "up": _lin(rng, (D, 8)),
                "down": _lin(rng, (8, D)),
            },
        },
    }
    mi = MOE_INTER
    experts = {
        "gate_up": _lin(rng, (N_EXPERTS, D, 2 * mi), scale=0.06),
        "down": _lin(rng, (N_EXPERTS, mi, D), scale=0.06),
    }
    layer["mlp"].update(experts)
    return {
        "embed": _lin(rng, (VOCAB, D), scale=0.2),
        "norm": np.ones((D,), np.float32),
        "lm_head": _lin(rng, (D, VOCAB), scale=0.2),
        "layers": [layer],
    }


def _make_mla_params(seed=41):
    params = _make_params(seed)
    rng = np.random.default_rng(seed + 1)
    cfg = dataclasses.replace(
        _cfg(),
        layer_types=("deepseek_sparse_attention",),
        idx_topk=4,
        idx_kpool=1,
    )
    layer = dict(params["layers"][0])
    layer["attn"] = {
        "q_a": _lin(rng, (D, cfg.q_lora)),
        "q_a_norm": np.ones((cfg.q_lora,), np.float32),
        "q_b": _lin(rng, (cfg.q_lora, cfg.n_heads * cfg.qk_nope)),
        "kv_a": _lin(rng, (D, cfg.kv_lora)),
        "kv_a_norm": np.ones((cfg.kv_lora,), np.float32),
        "kv_b": _lin(rng, (cfg.kv_lora, cfg.n_heads * (cfg.qk_nope + cfg.v_hd))),
        "o": _lin(rng, (cfg.n_heads * cfg.v_hd, D)),
        "indexer": {
            "wk": _lin(rng, (D, cfg.idx_hd)),
            "k_norm_w": np.ones((cfg.idx_hd,), np.float32),
            "k_norm_b": np.zeros((cfg.idx_hd,), np.float32),
            "gate": _lin(rng, (D, cfg.idx_hd)),
            "wq_b": _lin(rng, (cfg.q_lora, cfg.idx_heads * cfg.idx_hd)),
            "weights_proj": _lin(rng, (D, cfg.idx_heads)),
            "ape": _lin(rng, (cfg.idx_hd,)),
        },
    }
    params = {**params, "layers": [layer]}
    return cfg, params


def _make_wide_quant_params(seed=67):
    hidden, moe_inter = 256, 2048
    cfg = dataclasses.replace(_cfg(), hidden=hidden, moe_inter=moe_inter)
    rng = np.random.default_rng(seed)
    heads, hd, conv_k = cfg.kda_heads, cfg.kda_hd, cfg.conv_k
    layer = {
        "ln1": np.ones((hidden,), np.float32),
        "ln2": np.ones((hidden,), np.float32),
        "hc_attn": {
            "fn": _lin(rng, (hidden, 3)),
            "base": _lin(rng, (3,)),
            "scale": _lin(rng, (3,)),
        },
        "hc_ffn": {
            "fn": _lin(rng, (hidden, 3)),
            "base": _lin(rng, (3,)),
            "scale": _lin(rng, (3,)),
        },
        "attn": {
            "q": _lin(rng, (hidden, heads * hd)),
            "k": _lin(rng, (hidden, heads * hd)),
            "v": _lin(rng, (hidden, heads * hd)),
            "conv_q": _lin(rng, (heads * hd, conv_k)),
            "conv_k": _lin(rng, (heads * hd, conv_k)),
            "conv_v": _lin(rng, (heads * hd, conv_k)),
            "f_a": _lin(rng, (hidden, hd)),
            "f_b": _lin(rng, (hd, heads * hd)),
            "dt_bias": np.zeros((heads * hd,), np.float32),
            "A_log": np.zeros((heads,), np.float32),
            "b": _lin(rng, (hidden, heads)),
            "g_a": _lin(rng, (hidden, hd)),
            "g_b": _lin(rng, (hd, heads * hd)),
            "o_norm": np.ones((hd,), np.float32),
            "o": _lin(rng, (heads * hd, hidden)),
        },
        "mlp": {
            "router_w": _lin(rng, (hidden, N_EXPERTS)),
            "router_bias": _lin(rng, (N_EXPERTS,), scale=0.2),
            "shared": {
                "gate": _lin(rng, (hidden, cfg.inter)),
                "up": _lin(rng, (hidden, cfg.inter)),
                "down": _lin(rng, (cfg.inter, hidden)),
            },
            "gate_up": _lin(rng, (N_EXPERTS, hidden, 2 * moe_inter), scale=0.06),
            "down": _lin(rng, (N_EXPERTS, moe_inter, hidden), scale=0.06),
        },
    }
    params = {
        "embed": _lin(rng, (VOCAB, hidden), scale=0.2),
        "norm": np.ones((hidden,), np.float32),
        "lm_head": _lin(rng, (hidden, VOCAB), scale=0.2),
        "layers": [layer],
    }
    return cfg, params


def _encode_f32_rows(rows):
    rows = np.ascontiguousarray(rows, dtype="<f4")
    return rows.view(np.uint8).reshape(rows.shape + (4,))


class _F32HostStore:
    """Tiny exact host store using the same packed row contract as production."""

    def __init__(self, params, tile=4, layer=0):
        self.params = params
        self.tile = tile
        self.layer = layer
        self.selections = []
        self.ep_history = []

    def qtypes(self, layer):
        assert layer == self.layer
        return ("F32", "F32", "F32")

    def gather(self, layer, global_idx):
        assert layer == self.layer
        global_idx = np.asarray(global_idx, np.int32)
        self.selections.append(global_idx.copy())
        unique = np.unique(global_idx)
        local = np.searchsorted(unique, global_idx).astype(np.int32)
        # Keep two expert tiles live even when routes touch only one tile, so
        # the CPU tests execute the tiled scan and its zero-padded rows.
        ep = N_EXPERTS
        self.ep_history.append(ep)
        ml = MOE_INTER // N_DEV
        gu = self.params["layers"][self.layer]["mlp"]["gate_up"]
        dn = self.params["layers"][self.layer]["mlp"]["down"]

        tables = [[], [], []]
        for chip in range(N_DEV):
            gates, ups, downs = [], [], []
            for expert in unique:
                gates.append(_encode_f32_rows(gu[expert, :, chip * ml : (chip + 1) * ml].T))
                ups.append(
                    _encode_f32_rows(
                        gu[expert, :, MOE_INTER + chip * ml : MOE_INTER + (chip + 1) * ml].T
                    )
                )
                downs.append(_encode_f32_rows(dn[expert, chip * ml : (chip + 1) * ml, :].T))
            chip_tables = (gates, ups, downs)
            for j, rows in enumerate(chip_tables):
                current = np.stack(rows, axis=0)
                padded = np.zeros((ep,) + current.shape[1:], dtype=np.uint8)
                padded[: len(unique)] = current
                tables[j].append(padded)
        return tuple(np.stack(t, axis=0) for t in tables), local, ep


class _Q6HostStore:
    """Small packed Q6_K store whose reference matrices are decoded from bytes."""

    def __init__(self, seed=21):
        rng = np.random.default_rng(seed)
        local_mi = MOE_INTER // N_DEV
        gate = np.empty((N_DEV, N_EXPERTS, local_mi, 1, 210), np.uint8)
        up = np.empty_like(gate)
        down = np.empty((N_DEV, N_EXPERTS, D, 1, 210), np.uint8)

        def make_rows(shape):
            raw = rng.integers(0, 256, size=shape, dtype=np.uint8)
            raw[..., 192:208] = rng.integers(-3, 4, size=shape[:-1] + (16,), dtype=np.int8).view(
                np.uint8
            )
            raw[..., 208:210] = np.asarray([0.015], dtype="<f2").view(np.uint8)
            return raw

        for chip in range(N_DEV):
            gate[chip] = make_rows((N_EXPERTS, local_mi, 1, 210))
            up[chip] = make_rows((N_EXPERTS, local_mi, 1, 210))
            down[chip] = make_rows((N_EXPERTS, D, 1, 210))
        self.tables = (gate, up, down)

    def qtypes(self, layer):
        assert layer == 0
        return ("Q6_K", "Q6_K", "Q6_K")

    def gather(self, layer, global_idx):
        assert layer == 0
        global_idx = np.asarray(global_idx, np.int32)
        unique = np.unique(global_idx)
        local = np.searchsorted(unique, global_idx).astype(np.int32)
        return tuple(table[:, unique] for table in self.tables), local, len(unique)

    def dense_matrices(self):
        local_mi = MOE_INTER // N_DEV
        gate_up = np.zeros((N_EXPERTS, D, 2 * MOE_INTER), np.float32)
        down = np.zeros((N_EXPERTS, MOE_INTER, D), np.float32)
        for chip, (gate, up, dn) in enumerate(zip(*self.tables)):
            gate_up[:, :, chip * local_mi : (chip + 1) * local_mi] = dequantize_rows(
                gate, "Q6_K", D
            ).transpose(0, 2, 1)
            gate_up[:, :, MOE_INTER + chip * local_mi : MOE_INTER + (chip + 1) * local_mi] = (
                dequantize_rows(up, "Q6_K", D).transpose(0, 2, 1)
            )
            down[:, chip * local_mi : (chip + 1) * local_mi, :] = dequantize_rows(
                dn, "Q6_K", local_mi
            ).transpose(0, 2, 1)
        return gate_up, down


class _MixedQuantHostStore:
    """Tiny chip-major packed store with an independently chosen type per matrix."""

    def __init__(self, qtypes, hidden=256, moe_inter=2048, seed=53, tile=4):
        self._qtypes = tuple(qtypes)
        self.hidden = hidden
        self.moe_inter = moe_inter
        self.tile = tile
        rng = np.random.default_rng(seed)
        local_mi = moe_inter // N_DEV

        def make_rows(qtype, shape):
            block_weights, block_bytes = block_layout(qtype)
            raw = np.zeros(shape + (block_bytes,), np.uint8)
            if qtype in ("Q4_K", "Q5_K"):
                raw[...] = rng.integers(0, 256, size=raw.shape, dtype=np.uint8)
                raw[..., :2] = np.asarray([0.003], dtype="<f2").view(np.uint8)
                raw[..., 2:4] = np.asarray([0.001], dtype="<f2").view(np.uint8)
                raw[..., 4:16] = rng.integers(0, 12, size=raw[..., 4:16].shape, dtype=np.uint8)
            elif qtype == "Q6_K":
                raw[..., :192] = rng.integers(0, 256, size=raw[..., :192].shape, dtype=np.uint8)
                scales = rng.integers(-2, 3, size=raw[..., 192:208].shape, dtype=np.int8)
                raw[..., 192:208] = scales.view(np.uint8)
                raw[..., 208:210] = np.asarray([0.01], dtype="<f2").view(np.uint8)
            elif qtype == "Q8_0":
                raw[..., :2] = np.asarray([0.02], dtype="<f2").view(np.uint8)
                values = rng.integers(-6, 7, size=raw[..., 2:].shape, dtype=np.int8)
                raw[..., 2:] = values.view(np.uint8)
            else:  # the parameterized cases below intentionally cover these three GGUF formats
                raise ValueError(f"unsupported test quantization type: {qtype}")
            return raw

        tables = []
        for qtype in self._qtypes:
            block_weights, _ = block_layout(qtype)
            gate_blocks = (hidden + block_weights - 1) // block_weights
            down_blocks = (local_mi + block_weights - 1) // block_weights
            if len(tables) == 0:
                shape = (N_DEV, N_EXPERTS, local_mi, gate_blocks)
            elif len(tables) == 1:
                shape = (N_DEV, N_EXPERTS, local_mi, gate_blocks)
            else:
                shape = (N_DEV, N_EXPERTS, hidden, down_blocks)
            tables.append(make_rows(qtype, shape))
        self.tables = tuple(tables)

    def qtypes(self, layer):
        assert layer == 0
        return self._qtypes

    def gather(self, layer, global_idx):
        assert layer == 0
        global_idx = np.asarray(global_idx, np.int32)
        selected, local = np.unique(global_idx, return_inverse=True)
        ep = max(self.tile, int(np.ceil(len(selected) / self.tile) * self.tile))
        packed = []
        for table in self.tables:
            staged = np.zeros((N_DEV, ep) + table.shape[2:], np.uint8)
            staged[:, : len(selected)] = table[:, selected]
            packed.append(staged)
        return tuple(packed), local.reshape(global_idx.shape).astype(np.int32), ep

    def dense_matrices(self):
        local_mi = self.moe_inter // N_DEV
        gate_up = np.zeros((N_EXPERTS, self.hidden, 2 * self.moe_inter), np.float32)
        down = np.zeros((N_EXPERTS, self.moe_inter, self.hidden), np.float32)
        for kind, (raw, qtype) in enumerate(zip(self.tables, self._qtypes, strict=True)):
            if kind < 2:
                # [chip, expert, local_mi, D] -> [expert, D, chip, local_mi]
                decoded = dequantize_rows(raw, qtype, self.hidden).transpose(1, 3, 0, 2)
                start = 0 if kind == 0 else self.moe_inter
                for chip in range(N_DEV):
                    gate_up[:, :, start + chip * local_mi : start + (chip + 1) * local_mi] = (
                        decoded[:, :, chip, :]
                    )
            else:
                # [chip, expert, D, local_mi] -> [expert, local_mi, chip, D]
                decoded = dequantize_rows(raw, qtype, local_mi).transpose(1, 3, 0, 2)
                for chip in range(N_DEV):
                    down[:, chip * local_mi : (chip + 1) * local_mi, :] = decoded[:, :, chip, :]
        return gate_up, down


@pytest.fixture(scope="module")
def model_and_engine():
    assert jax.device_count() == N_DEV, jax.devices()
    cfg = _cfg()
    params = _make_params()
    store = _F32HostStore(params)
    engine = PackedOffloadEngine(
        cfg,
        params,
        store,
        devices=jax.devices(),
        max_len=MAX_LEN,
        expert_tile=4,
        prefill_chunk=32,
        seq_shard=True,
    )
    return cfg, params, store, engine


def _reference_logits(cfg, params, tokens, override=None):
    p = jax.tree.map(jnp.asarray, params)
    ids = jnp.asarray(tokens, jnp.int32)
    if ids.ndim == 1:
        ids = ids[None, :]
    embeds = p["embed"][ids]
    if override is not None:
        positions, vectors = override
        embeds = embeds.at[0, np.asarray(positions, np.int32)].set(jnp.asarray(vectors))
    h, _ = M.forward(p, ids, cfg, embeds=embeds)
    return np.asarray(M.logits(p, h[:, -1]))


def test_prefill_matches_dense_and_only_gathers_real_routes(model_and_engine):
    cfg, params, store, engine = model_and_engine
    tokens = np.arange(1, 39, dtype=np.int32) % VOCAB

    got, caches = engine.prefill(tokens)
    want = _reference_logits(cfg, params, tokens)
    np.testing.assert_allclose(np.asarray(got), want, rtol=2e-3, atol=2e-3)
    assert len(caches) == cfg.n_layers

    # The last prefill chunk is padded to 32 on device, but host I/O includes
    # only the six real rows in that chunk.
    assert store.selections[-1].shape == (6, TOPK)
    assert len(np.unique(store.selections[-1])) <= store.selections[-1].size
    assert store.ep_history[-1] == 8  # two fixed four-expert tiles, with unused rows zero padded


@pytest.mark.parametrize("pattern", ["spread", "hot", "duplicates", "padded"])
@pytest.mark.parametrize("tile_buckets", [False, True])
def test_routed_expert_kernel_matches_scalar_reference(model_and_engine, pattern, tile_buckets):
    cfg, params, store, engine = model_and_engine
    rng = np.random.default_rng(109)
    count = 32 if pattern == "padded" else 8
    x = rng.normal(size=(count, D)).astype(np.float32)
    routes = np.stack((np.arange(8), (np.arange(8) + 1) % 8), axis=1).astype(np.int32)
    if pattern == "hot":
        routes[:, 0] = 0
    if pattern == "duplicates":
        routes[0, 1] = routes[0, 0]
    packed, local, ep = store.gather(0, routes)
    ids = np.full((count, TOPK), -1, np.int32)
    ids[:8] = local
    weights = rng.uniform(0.1, 1.0, size=ids.shape).astype(np.float32)
    rows = expert_token_rows(ids, ep)
    capacities = expert_tile_capacities(rows, 4)
    if pattern in ("spread", "duplicates", "padded"):
        assert rows.size < ep * count
    device_packed = engine.transport(packed)

    def kernel(x, ids, weights, rows, packed):
        return engine.lcfg.reduce(
            tiled_experts(x, packed, ids, weights, engine.lcfg, store.qtypes(0), 4, rows)
            if not tile_buckets
            else tiled_experts(
                x,
                packed,
                ids,
                weights,
                engine.lcfg,
                store.qtypes(0),
                4,
                rows,
                jnp.asarray(capacities),
            )
        )

    program = jax.jit(
        shard_map(
            kernel,
            mesh=engine.mesh,
            in_specs=(R, R, R, R, (P(AXIS),) * 3),
            out_specs=R,
            check_vma=False,
        )
    )
    actual = np.asarray(program(x, ids, weights, rows, device_packed))
    expected = np.zeros_like(x)
    gu = params["layers"][0]["mlp"]["gate_up"]
    down = params["layers"][0]["mlp"]["down"]
    for token in range(8):
        for route, expert in enumerate(routes[token]):
            gate = np.minimum(x[token] @ gu[expert, :, :MOE_INTER], cfg.swiglu_limit)
            up = np.clip(x[token] @ gu[expert, :, MOE_INTER:], -cfg.swiglu_limit, cfg.swiglu_limit)
            hidden = gate / (1.0 + np.exp(-gate)) * up
            expected[token] += weights[token, route] * (hidden @ down[expert])
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
    np.testing.assert_array_equal(actual[8:], 0)


def test_isolated_profile_uses_actual_mesh_and_validates_outputs(model_and_engine):
    from glm53_offload.profiling import profile_case
    from glm53_offload.weights import PackedExperts

    cfg, _, store, engine = model_and_engine
    packed, _, _ = store.gather(0, np.arange(N_EXPERTS, dtype=np.int32)[None])
    host = PackedExperts({0: packed}, {0: ("F32",) * 3}, N_DEV)
    checkpoints = []

    def publish(partial):
        checkpoints.append((partial["phase"], tuple(partial["measurements"])))

    result = profile_case(cfg, host, 0, engine.mesh, 8, "spread", 1, lambda: None, publish)
    assert result["status"] == "PASS"
    assert result["route_capacity"] == 2
    assert result["computed_token_expert_pairs"] == 16
    assert result["all_token_expert_pairs"] == 64
    assert result["host_gather_seconds"] >= 0
    assert result["blocked_transfer_seconds"] >= 0
    assert result["old_kernel_seconds"] > 0
    assert result["routed_kernel_seconds"] > 0
    assert result["tile_routed_kernel_seconds"] > 0
    assert result["tile_computed_token_expert_pairs"] <= result["computed_token_expert_pairs"]
    assert result["tile_max_output_error"] < 5e-3
    assert ("warmup all_tokens", ("all_tokens",)) in checkpoints
    assert "compiler_estimates" in result["measurements"]["routed_tokens"]


def test_prompt_replay_restores_sharding_and_preserves_initial_caches(model_and_engine):
    from glm53_offload.run import PreparedBatch, decode_prepared, prepare_requests

    cfg, _, store, engine = model_and_engine
    store.stats = {"bytes_transferred": 0}

    class Chat:
        def render(self, text, images):
            return np.arange(int(text), dtype=np.int32) % cfg.vocab, None

        def answer(self, ids):
            return str(ids)

    chat = Chat()
    singles = [prepare_requests(engine, chat, [(str(n), [])], 3).to_host() for n in (5, 8, 11, 14)]
    frozen = [np.array(a, copy=True) for s in singles for a in jax.tree.leaves(s.caches)]
    prepared = PreparedBatch.combine(singles)
    warm = decode_prepared(engine, chat, prepared, 2, [-1])
    first = decode_prepared(engine, chat, prepared, 3, [-1])
    second = decode_prepared(engine, chat, prepared, 3, [-1])
    assert first["output_ids"] == second["output_ids"]
    assert warm["output_ids"] == [ids[:2] for ids in first["output_ids"]]
    np.testing.assert_array_equal(prepared.positions, [5, 8, 11, 14])
    for actual, expected in zip(
        (a for s in singles for a in jax.tree.leaves(s.caches)), frozen, strict=True
    ):
        np.testing.assert_array_equal(actual, expected)
    assert all(isinstance(a, jax.Array) for a in jax.tree.leaves(prepared.caches))


def test_image_embedding_override_matches_dense(model_and_engine):
    cfg, params, _, engine = model_and_engine
    tokens = np.arange(1, 41, dtype=np.int32) % VOCAB
    positions = np.array([3, 34, 39], np.int32)
    vectors = np.linspace(-0.3, 0.4, len(positions) * D, dtype=np.float32).reshape(-1, D)

    got, _ = engine.prefill(tokens, override=(positions, vectors))
    want = _reference_logits(cfg, params, tokens[None, :], (positions, vectors))
    np.testing.assert_allclose(np.asarray(got), want, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("batch", [2, 4, 8])
def test_decode_rows_preserves_independent_positions(model_and_engine, batch):
    cfg, params, _, engine = model_and_engine
    rng = np.random.default_rng(batch)
    prompts = [rng.integers(0, VOCAB, size=(1, 33 + i), dtype=np.int32) for i in range(batch)]
    cache_sets, positions, tokens = [], [], []
    for prompt in prompts:
        logits, caches = engine.prefill(prompt)
        np.testing.assert_allclose(
            np.asarray(logits), _reference_logits(cfg, params, prompt), rtol=4e-3, atol=4e-3
        )
        cache_sets.append(caches)
        positions.append(prompt.shape[1])
        tokens.append(int(np.argmax(np.asarray(logits)[0])))

    decode_tokens = np.asarray(tokens, np.int32)
    got, updated = engine.decode_rows(decode_tokens, cache_sets, positions)
    expected = []
    for prompt, token in zip(prompts, decode_tokens):
        full = np.concatenate((prompt, np.asarray([[token]], np.int32)), axis=1)
        expected.append(_reference_logits(cfg, params, full)[0])
    np.testing.assert_allclose(np.asarray(got), np.asarray(expected), rtol=3e-3, atol=3e-3)
    assert len(updated) == batch
    assert all(len(cache_set) == cfg.n_layers for cache_set in updated)


def test_store_result_padding_and_remap_validation(model_and_engine):
    _, _, _, engine = model_and_engine
    arrays = tuple(np.zeros((N_DEV, 3, 2, 1, 4), np.uint8) for _ in range(3))
    packed, local, ep = engine._pad_store_result(
        arrays, np.array([[0, 2], [2, 1]], np.int32), 3, (2, 2)
    )
    assert ep == 4
    assert all(a.shape[1] == 4 for a in packed)
    assert np.all(packed[0][:, 3] == 0)
    np.testing.assert_array_equal(local, [[0, 2], [2, 1]])

    with pytest.raises(ValueError, match="remapped expert ids"):
        engine._pad_store_result(arrays, np.array([[3, 0]], np.int32), 3, (1, 2))


def test_dense_layer_down_projection_survives_mixed_layer_engine():
    base_cfg = _cfg()
    cfg = dataclasses.replace(
        base_cfg,
        n_layers=2,
        layer_types=("linear_attention", "linear_attention"),
        mlp_types=("dense", "sparse"),
    )
    params = _make_params()
    rng = np.random.default_rng(72)
    dense = dict(params["layers"][0])
    dense["mlp"] = {
        "gate": _lin(rng, (D, cfg.inter)),
        "up": _lin(rng, (D, cfg.inter)),
        "down": _lin(rng, (cfg.inter, D)),
    }
    params = {**params, "layers": [dense, params["layers"][0]]}
    store = _F32HostStore(params, layer=1)
    engine = PackedOffloadEngine(
        cfg,
        params,
        store,
        devices=jax.devices(),
        max_len=MAX_LEN,
        expert_tile=4,
        prefill_chunk=32,
        seq_shard=True,
    )

    # The base constructor must keep the first layer's resident dense MLP.
    assert "down" in engine.params["layers"][0]["mlp"]
    assert engine.params["layers"][0]["mlp"]["down"].shape == (cfg.inter, D)
    tokens = np.arange(1, 35, dtype=np.int32) % VOCAB
    got, _ = engine.prefill(tokens)
    want = _reference_logits(cfg, params, tokens)
    np.testing.assert_allclose(np.asarray(got), want, rtol=2e-3, atol=2e-3)


def test_q6_routes_match_reference_from_decoded_packed_matrices():
    assert jax.device_count() == N_DEV, jax.devices()
    cfg = _cfg()
    params = _make_params(seed=38)
    store = _Q6HostStore()
    gate_up, down = store.dense_matrices()
    params["layers"][0]["mlp"]["gate_up"] = gate_up
    params["layers"][0]["mlp"]["down"] = down
    engine = PackedOffloadEngine(
        cfg,
        params,
        store,
        devices=jax.devices(),
        max_len=MAX_LEN,
        expert_tile=4,
        prefill_chunk=32,
        seq_shard=True,
    )

    tokens = np.arange(2, 39, dtype=np.int32) % VOCAB
    got, _ = engine.prefill(tokens)
    want = _reference_logits(cfg, params, tokens)
    np.testing.assert_allclose(np.asarray(got), want, rtol=3e-3, atol=3e-3)


@pytest.mark.parametrize(
    "qtypes",
    [
        ("Q6_K", "Q6_K", "Q8_0"),
        ("Q8_0", "Q8_0", "Q8_0"),
        ("Q4_K", "Q4_K", "Q5_K"),
        ("Q5_K", "Q5_K", "Q6_K"),
        ("Q4_K", "Q4_K", "Q6_K"),
    ],
)
def test_mixed_quant_routes_match_dense_reference(qtypes):
    """Exercise the real GGUF row/block layouts through TP expert projection and reduction."""
    assert jax.device_count() == N_DEV, jax.devices()
    cfg, params = _make_wide_quant_params()
    store = _MixedQuantHostStore(qtypes)
    gate_up, down = store.dense_matrices()
    params["layers"][0]["mlp"]["gate_up"] = gate_up
    params["layers"][0]["mlp"]["down"] = down
    engine = PackedOffloadEngine(
        cfg,
        params,
        store,
        devices=jax.devices(),
        max_len=MAX_LEN,
        expert_tile=4,
        prefill_chunk=32,
        seq_shard=False,
    )

    tokens = np.arange(1, 10, dtype=np.int32) % VOCAB
    got, _ = engine.prefill(tokens)
    want = _reference_logits(cfg, params, tokens)
    np.testing.assert_allclose(np.asarray(got), want, rtol=5e-3, atol=5e-3)


@pytest.mark.parametrize("qtypes", [("Q4_K", "Q5_K", "Q6_K"), ("Q6_K", "Q6_K", "Q8_0")])
def test_direct_resident_cache_matches_compact_kernel_through_prefill_and_decode(qtypes):
    """Real packed formats, eight tensor shards, over-capacity misses and hits."""
    from glm53_offload.weights import PackedExperts

    cfg, params = _make_wide_quant_params()
    source = _MixedQuantHostStore(qtypes)
    outputs = []
    for direct in (False, True):
        store = PackedExperts({0: source.tables}, {0: qtypes}, N_DEV)
        per_expert = sum(a[:, :1].nbytes for a in source.tables)
        engine = PackedOffloadEngine(
            cfg,
            params,
            store,
            devices=jax.devices(),
            max_len=MAX_LEN,
            expert_cache_bytes=3 * per_expert,
            direct_expert_cache=direct,
            prefill_moe_window=64,
            seq_shard=False,
        )
        logits, state = engine.prefill(np.arange(1, 38, dtype=np.int32) % VOCAB)
        values = [np.asarray(logits)]
        # Independent contexts exercise cache donation after a consuming post.
        for batch in (1, 4, 8):
            caches = [state] * batch
            token = np.full(batch, 3, np.int32)
            positions = np.full(batch, 37, np.int32)
            for _ in range(2):
                logits, caches = engine.decode_rows(token, caches, positions)
                values.append(np.asarray(logits))
                positions += 1
        assert engine.expert_cache.stats["hits"] > 0
        assert engine.expert_cache.stats["misses"] > 0
        assert (
            engine.expert_cache.stats["compact_table_bytes_avoided"] > 0
            if direct
            else (engine.expert_cache.stats["compact_table_bytes_avoided"] == 0)
        )
        outputs.append(values)
    for compact, direct in zip(*outputs, strict=True):
        np.testing.assert_array_equal(direct, compact)


@pytest.mark.parametrize("batch", [2, 4, 8])
def test_mla_indexer_decode_rows_use_independent_absolute_positions(batch):
    assert jax.device_count() == N_DEV, jax.devices()
    cfg, params = _make_mla_params()
    store = _F32HostStore(params)
    engine = PackedOffloadEngine(
        cfg,
        params,
        store,
        devices=jax.devices(),
        max_len=MAX_LEN,
        expert_tile=4,
        prefill_chunk=32,
        seq_shard=False,
    )

    rng = np.random.default_rng(batch + 14)
    prompts = [rng.integers(0, VOCAB, size=(1, 33 + i), dtype=np.int32) for i in range(batch)]
    cache_sets, positions, tokens = [], [], []
    for prompt in prompts:
        logits, caches = engine.prefill(prompt)
        np.testing.assert_allclose(
            np.asarray(logits), _reference_logits(cfg, params, prompt), rtol=4e-3, atol=4e-3
        )
        cache_sets.append(caches)
        positions.append(prompt.shape[1])
        tokens.append(int(np.argmax(np.asarray(logits)[0])))

    decode_tokens = np.asarray(tokens, np.int32)
    got, updated = engine.decode_rows(decode_tokens, cache_sets, positions)
    expected = []
    for prompt, token in zip(prompts, decode_tokens):
        full = np.concatenate((prompt, np.asarray([[token]], np.int32)), axis=1)
        expected.append(_reference_logits(cfg, params, full)[0])
    np.testing.assert_allclose(np.asarray(got), np.asarray(expected), rtol=4e-3, atol=4e-3)
    assert len(updated) == batch
    assert all(len(cache_set) == cfg.n_layers for cache_set in updated)


@pytest.mark.parametrize("capacity,length", [(40, 39), (17, 15)])
def test_mla_padded_prefill_at_capacity_keeps_absolute_cache_positions(capacity, length):
    cfg, params = _make_mla_params()
    engine = PackedOffloadEngine(
        cfg, params, _F32HostStore(params), devices=jax.devices(), max_len=capacity
    )
    prompt = np.arange(length, dtype=np.int32)[None] % VOCAB
    logits, caches = engine.prefill(prompt)
    np.testing.assert_allclose(
        np.asarray(logits), _reference_logits(cfg, params, prompt), rtol=4e-3, atol=4e-3
    )
    token = int(np.argmax(np.asarray(logits)[0]))
    got, _, _ = engine.decode([token], caches, length)
    full = np.concatenate((prompt, [[token]]), axis=1)
    np.testing.assert_allclose(
        np.asarray(got), _reference_logits(cfg, params, full), rtol=4e-3, atol=4e-3
    )


def test_equal_shaped_layers_share_programs_but_use_their_own_weights():
    cfg = dataclasses.replace(
        _cfg(), n_layers=2, layer_types=("linear_attention",) * 2, mlp_types=("sparse",) * 2
    )
    params = _make_params()
    params["layers"].append(_make_params(seed=56)["layers"][0])
    from glm53_offload.weights import PackedExperts

    tables = {
        i: _F32HostStore(params, layer=i).gather(i, np.arange(N_EXPERTS)[None])[0] for i in range(2)
    }
    store = PackedExperts(tables, {i: ("F32",) * 3 for i in tables}, N_DEV)
    engine = PackedOffloadEngine(
        cfg,
        params,
        store,
        devices=jax.devices(),
        max_len=64,
        expert_cache_bytes=store.nbytes,
    )
    assert engine._prefill_pre(0, 32, False) is engine._prefill_pre(1, 32, False)
    assert engine._decode_pre(0, 4) is engine._decode_pre(1, 4)
    assert engine._post_program(0, 4, 1, 8) is engine._post_program(1, 4, 1, 8)
    prompt = np.array([[1, 3, 5]], np.int32)
    logits, _ = engine.prefill(prompt)
    np.testing.assert_allclose(
        np.asarray(logits), _reference_logits(cfg, params, prompt), rtol=4e-3, atol=4e-3
    )
    # Repeating a prompt must reuse the exact packed bytes and outputs.
    repeated, _ = engine.prefill(prompt)
    np.testing.assert_allclose(repeated, logits, rtol=4e-3, atol=4e-3)
    assert engine.expert_cache.stats["hits"] > 0


@pytest.mark.parametrize("window,length", [(64, 75), (256, 269)])
def test_prefill_windows_reuse_experts_and_preserve_mixed_layer_caches_and_decode(
    tmp_path, window, length
):
    from glm53_offload.weights import PackedExperts

    base, mla = _make_mla_params()
    cfg = dataclasses.replace(
        base,
        n_layers=3,
        layer_types=("linear_attention", "deepseek_sparse_attention", "linear_attention"),
        mlp_types=("sparse", "dense", "sparse"),
        idx_kpool=4,
    )
    params = _make_params()
    dense = dict(mla["layers"][0])
    dense["attn"] = {**dense["attn"], "indexer": dict(dense["attn"]["indexer"])}
    dense["attn"]["indexer"]["ape"] = np.broadcast_to(
        dense["attn"]["indexer"]["ape"], (cfg.idx_kpool, cfg.idx_hd)
    ).copy()
    rng = np.random.default_rng(137)
    dense["mlp"] = {
        "gate": _lin(rng, (D, cfg.inter)),
        "up": _lin(rng, (D, cfg.inter)),
        "down": _lin(rng, (cfg.inter, D)),
    }
    params["layers"] = [params["layers"][0], dense, _make_params(seed=59)["layers"][0]]
    tables = {
        i: _F32HostStore(params, layer=i).gather(i, np.arange(N_EXPERTS)[None])[0] for i in (0, 2)
    }
    engines = []
    for size in (32, window):
        store = PackedExperts(tables, {i: ("F32",) * 3 for i in tables}, N_DEV)
        engines.append(
            PackedOffloadEngine(
                cfg,
                params,
                store,
                max_len=length + 16,
                prefill_moe_window=size,
                tile_routed_experts=size != 32,
                execution_log_path=tmp_path / "execution.jsonl" if size != 32 else None,
            )
        )
    prompt = np.arange(length, dtype=np.int32)[None] % VOCAB
    positions = np.unique([31, 32, 63, 64, window - 1, window, length - 1])
    override = (positions, _lin(rng, (len(positions), D), scale=0.2))
    old_logits, old_caches = engines[0].prefill(prompt, override)
    new_logits, new_caches = engines[1].prefill(prompt, override)
    want = _reference_logits(cfg, params, prompt, override)
    np.testing.assert_allclose(new_logits, want, rtol=3e-3, atol=3e-3)
    np.testing.assert_allclose(new_logits, old_logits, rtol=2e-5, atol=2e-6)
    for old_cache, new_cache in zip(old_caches, new_caches, strict=True):
        for key in old_cache:
            old, new = old_cache[key], new_cache[key]
            if key in ("tk", "tg"):
                # Only the occupied prefix of an incomplete DSA pool is live.
                # Padding outside it is overwritten before it can be read.
                old, new = old[:, : length % cfg.idx_kpool], new[:, : length % cfg.idx_kpool]
            np.testing.assert_allclose(new, old, rtol=3e-5, atol=3e-6)
    old_decoded, _, _ = engines[0].decode(np.array([3]), old_caches, length)
    new_decoded, _, _ = engines[1].decode(np.array([3]), new_caches, length)
    np.testing.assert_allclose(new_decoded, old_decoded, rtol=3e-5, atol=3e-6)
    # Two sparse layers: small chunks versus 2 windows. These are actual store
    # transfers, not merely a counter predicting a speedup.
    assert engines[0].store.stats["gathers"] == 2 * ((length + 31) // 32 + 1)
    assert engines[1].store.stats["gathers"] == 6  # 4 prefill + 2 decode
    assert engines[1].store.stats["bytes_transferred"] < engines[0].store.stats["bytes_transferred"]
    samples = engines[1].execution_samples
    assert [(s["phase"], s["tokens"]) for s in samples] == [
        ("prefill", window),
        ("prefill", 32),
        ("decode", 1),
    ]
    evidence = [
        json.loads(line) for line in (tmp_path / "execution.jsonl").read_text().splitlines()
    ]
    assert [s["status"] for s in evidence] == ["START", "DONE"] * 3
    assert all(s["packed_bytes_submitted"] > 0 for s in samples)
