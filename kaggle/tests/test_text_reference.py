"""Independent HF reference for KDA, DSA, mHC, dense/MoE and cached decode."""

import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from glm53 import checkpoint
from glm53.model import Cfg
from glm53_offload import run
from glm53_offload.engine import PackedOffloadEngine
from glm53_offload.weights import PackedExperts


@pytest.mark.parametrize("window", [32, 256])
def test_hybrid_text_and_image_overrides_match_independent_hf(window):
    path = Path(run.__file__).parent / "data/text_reference.npz"
    with np.load(path, allow_pickle=False) as data:
        cfg = Cfg.from_hf(SimpleNamespace(**json.loads(str(data["config_json"]))), jnp.float32)

        class Reader:
            wm = {k.removeprefix("weight/"): "fixture" for k in data if k.startswith("weight/")}

            def get(self, name, dtype=np.float32):
                return data["weight/" + name].astype(dtype)

            def has(self, name):
                return "weight/" + name in data

        reader = Reader()
        params = checkpoint.load_top(reader)
        params["layers"] = [
            checkpoint.load_layer(reader, i, cfg.layer_types[i], cfg.mlp_types[i]) for i in range(2)
        ]
        tables = []
        n = len(jax.devices())
        for kind in ("gate", "up", "down"):
            weights = np.stack(
                [
                    reader.get(f"model.language_model.layers.1.mlp.experts.{i}.{kind}_proj.weight")
                    for i in range(cfg.n_exp)
                ]
            )
            # Split output rows for gate/up, input columns for down, without
            # using the implementation's store builder to form the reference.
            axis = 2 if kind == "down" else 1
            local = np.ascontiguousarray(np.stack(np.split(weights, n, axis=axis)))
            tables.append(local.view(np.uint8).reshape(*local.shape, 4))
        store = PackedExperts({1: tuple(tables)}, {1: ("F32",) * 3}, n)
        engine = PackedOffloadEngine(
            cfg,
            params,
            store,
            max_len=40,
            expert_cache_bytes=store.nbytes,
            prefill_moe_window=window,
        )
        logits, caches = engine.prefill(
            data["tokens"][:, :35], (data["positions"], data["vectors"])
        )
        np.testing.assert_allclose(logits, data["expected_35"], rtol=2e-4, atol=2e-5)
        decoded, _, _ = engine.decode(data["tokens"][:, 35], caches, 35)
        np.testing.assert_allclose(decoded, data["expected_36"], rtol=2e-4, atol=2e-5)
