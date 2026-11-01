#!/usr/bin/env python3
"""Freeze a tiny HF hybrid-text reference, including an image-embedding override.

Requires torch 2.12.0 and transformers 5.18.0; no checkpoint download.
Rename HF's fused in-memory parameters back to the published checkpoint layout
so the test exercises the real checkpoint loader as well as the JAX engine.
"""

import json
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import Glm5NextTextConfig
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextModel


def main():
    if transformers.__version__ != "5.18.0":
        raise RuntimeError("Use Transformers 5.18.0")
    torch.set_num_threads(2)
    torch.manual_seed(719)
    config = Glm5NextTextConfig(
        hidden_size=32,
        vocab_size=32,
        pad_token_id=0,
        num_hidden_layers=2,
        layer_types=["linear_attention", "indexed_attention"],
        mlp_layer_types=["dense", "sparse"],
        indexer_types=["full", "full"],
        intermediate_size=64,
        moe_intermediate_size=16,
        n_routed_experts=8,
        num_experts_per_tok=2,
        num_attention_heads=8,
        num_key_value_heads=8,
        q_lora_rank=16,
        kv_lora_rank=8,
        qk_nope_head_dim=4,
        v_head_dim=4,
        index_head_dim=4,
        index_n_heads=2,
        index_topk=8,
        index_kpool=2,
        linear_num_heads=8,
        linear_head_dim=4,
    )
    config._attn_implementation = "eager"
    model = Glm5NextTextModel(config).eval()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if ".attn_hc." in name or ".ffn_hc." in name:
                parameter.normal_(0, 0.1)
    data = {}
    prefix = "weight/model.language_model."
    for name, value in model.state_dict().items():
        value = value.detach().numpy()
        name = name.replace("attn_hc.", "hc_attn_").replace("ffn_hc.", "hc_ffn_")
        name = name.replace("self_attn.forget_gate.", "self_attn.")
        if name.endswith("self_attn.conv1d.weight"):
            for kind, split in zip(("q", "k", "v"), np.split(value, 3), strict=True):
                data[prefix + name.replace("conv1d", kind + "_conv1d")] = split
        elif name.endswith("experts.gate_up_proj"):
            for i, expert in enumerate(value):
                for kind, split in zip(("gate", "up"), np.split(expert, 2), strict=True):
                    key = name.replace("experts.gate_up_proj", f"experts.{i}.{kind}_proj.weight")
                    data[prefix + key] = split
        elif name.endswith("experts.down_proj"):
            for i, expert in enumerate(value):
                key = name.replace("experts.down_proj", f"experts.{i}.down_proj.weight")
                data[prefix + key] = expert
        else:
            data[prefix + name] = value
    head = torch.randn(32, 32) * 0.1
    data["weight/lm_head.weight"] = head.numpy()
    tokens = torch.arange(36)[None] % 32
    positions = torch.tensor([1, 31, 32])
    vectors = torch.randn(3, 32) * 0.2
    data.update(tokens=tokens.numpy(), positions=positions.numpy(), vectors=vectors.numpy())
    with torch.no_grad():
        for count in (35, 36):
            emb = model.embed_tokens(tokens[:, :count]).clone()
            emb[:, positions] = vectors
            hidden = model(inputs_embeds=emb, use_cache=False).last_hidden_state
            data[f"expected_{count}"] = (hidden[:, -1] @ head.T).numpy()
    data["config_json"] = np.array(json.dumps(config.to_dict()))
    path = Path(__file__).resolve().parents[1] / "kaggle/glm53_offload/data/text_reference.npz"
    np.savez_compressed(path, **data)
    print(f"Wrote {path}: {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
