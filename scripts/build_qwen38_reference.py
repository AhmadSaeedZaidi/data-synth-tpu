"""Generate small full-model golden fixtures using the pinned official PyTorch implementation.

Run in a CPU-only reference environment; this never downloads model weights.
"""

import argparse
import hashlib
import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kaggle"))
from qwen38_tpu.config import ModelConfig, VisionConfig  # noqa: E402


def load_pinned(name):
    path = ROOT / "output/qwen38-strata-research/transformers" / (name + ".py")
    fullname = "transformers.models.qwen4_exp." + name
    spec = importlib.util.spec_from_file_location(fullname, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[fullname] = module
    spec.loader.exec_module(module)
    return module, hashlib.sha256(path.read_bytes()).hexdigest()


def q8(array):
    shape = array.shape
    rows = array.reshape(-1, shape[-1] // 32, 32)
    scale = (np.max(np.abs(rows), -1) / 127).astype("<f2")
    codes = np.clip(np.rint(rows / scale[..., None]), -127, 127).astype(np.int8)
    raw = np.concatenate(
        (scale.view(np.uint8).reshape(scale.shape + (2,)), codes.view(np.uint8)), -1
    )
    decoded = (scale.astype(np.float32)[..., None] * codes).reshape(shape)
    return raw.reshape(shape[:-1] + (shape[-1] // 32, 34)), decoded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--long-context", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(317)
    from transformers.models.qwen4_exp import configuration_qwen4_exp as configs

    config_sha = hashlib.sha256(Path(configs.__file__).read_bytes()).hexdigest()
    modeling, model_sha = load_pinned("modeling_qwen4_exp")
    c = ModelConfig(
        hidden=32,
        layers=4,
        streams=2,
        lowrank=8,
        experts=8,
        top_k=2,
        intermediate=32,
        vocab=64,
        key_heads=2,
        value_heads=8,
        state_dim=8,
        query_heads=8,
        kv_heads=2,
        head_dim=8,
        rotary_dim=8,
        mrope_sections=(2, 1, 1),
        qsa_budget=32,
        max_context=96 if args.long_context else 32,
        indexer_heads=4 if args.long_context else 2,
        indexer_dim=8,
        ple_dim=32,
        heads_per_ngram=2,
        ngram_base=7,
        eos=1,
        generation_eos=(2, 1),
        image_token=62,
        vision_start=60,
        vision_end=61,
    )
    v = VisionConfig(
        hidden=16,
        heads=2,
        intermediate=32,
        layers=2,
        patch=2,
        temporal=2,
        merge=2,
        grid_side=4,
        output=32,
    )
    tc = configs.Qwen4ExpTextConfig(
        hidden_size=c.hidden,
        num_hidden_layers=c.layers,
        intermediate_size=c.intermediate,
        moe_intermediate_size=c.intermediate,
        shared_expert_intermediate_size=c.intermediate,
        num_experts=c.experts,
        num_experts_per_tok=c.top_k,
        vocab_size=c.vocab,
        linear_num_key_heads=c.key_heads,
        linear_num_value_heads=c.value_heads,
        linear_key_head_dim=c.state_dim,
        linear_value_head_dim=c.state_dim,
        linear_conv_kernel_dim=c.conv_kernel,
        num_attention_heads=c.query_heads,
        num_key_value_heads=c.kv_heads,
        head_dim=c.head_dim,
        hc_count=c.streams,
        hc_lowrank=c.lowrank,
        ple_layer_ids=[2],
        ple_embed_dim=c.ple_dim,
        ngram_size=c.ngram_size,
        heads_per_ngram=c.heads_per_ngram,
        ngram_vocab_size_base=c.ngram_base,
        split_ngram_parts=1,
        eos_token_id=c.eos,
        pad_token_id=c.eos,
        output_gate_type="sigmoid",
        indexer_n_heads=c.indexer_heads,
        indexer_kv_heads=1,
        indexer_head_dim=c.indexer_dim,
        indexer_budget=c.qsa_budget,
        indexer_compress_ratio=4,
        rope_parameters=dict(
            rope_type="default",
            rope_theta=c.rope_theta,
            partial_rotary_factor=1.0,
            mrope_section=list(c.mrope_sections),
            mrope_interleaved=True,
        ),
    )
    vc = configs.Qwen4ExpVisionConfig(
        hidden_size=v.hidden,
        num_heads=v.heads,
        intermediate_size=v.intermediate,
        depth=v.layers,
        patch_size=v.patch,
        temporal_patch_size=v.temporal,
        spatial_merge_size=v.merge,
        num_position_embeddings=v.grid_side**2,
        out_hidden_size=v.output,
        rope_parameters=dict(rope_type="axial", rope_theta=v.theta),
    )
    config = configs.Qwen4ExpConfig(
        text_config=tc,
        vision_config=vc,
        image_token_id=c.image_token,
        vision_start_token_id=c.vision_start,
        vision_end_token_id=c.vision_end,
    )
    config._attn_implementation = "eager"
    tc._attn_implementation = "eager"
    vc._attn_implementation = "eager"
    model = modeling.Qwen4ExpForConditionalGeneration(config).eval()
    arrays = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.normal_(0, 0.08)
            if name.endswith("A_log"):
                param.add_(-1)
            if (name.startswith("model.visual") and "norm.weight" in name) or name.endswith(
                "linear_attn.norm.weight"
            ):
                param.add_(1)
        for i, layer in enumerate(model.model.language_model.layers):
            gate, up = layer.mlp.experts.gate_up_proj.chunk(2, dim=-2)
            for kind, param in (("gate", gate), ("up", up), ("down", layer.mlp.experts.down_proj)):
                raw, decoded = q8(param.numpy())
                arrays[f"packed.{i}.{kind}"] = raw
                param.copy_(torch.from_numpy(decoded))
            if args.long_context and c.is_qsa(i):
                # ReLU gives exact zero-score ties with random independent
                # heads. Top-k tie order is unspecified across backends. Paired
                # opposite query heads make this full-logit oracle unambiguous;
                # the runtime also tests all-zero ties as valid tied selections.
                projection = layer.self_attn.indexer.index_qk_proj.weight
                projection[c.indexer_dim : 2 * c.indexer_dim].copy_(-projection[: c.indexer_dim])
                projection[3 * c.indexer_dim : 4 * c.indexer_dim].copy_(
                    -projection[2 * c.indexer_dim : 3 * c.indexer_dim]
                )
        arrays.update({name: tensor.numpy() for name, tensor in model.state_dict().items()})
        ids = torch.tensor(
            [
                [11, 60, 62, 62, 62, 62, 61, 18, 1, 20, 21, 22],
                [13, 60, 62, 62, 62, 62, 61, 19, 1, 23, 24, 25],
            ]
        )
        if args.long_context:
            ids = torch.cat((ids, torch.arange(52).view(1, -1).expand(2, -1) % 40 + 10), dim=1)
        pixels = torch.randn(32, 24)
        grids = torch.tensor([[1, 4, 4], [1, 4, 4]])
        mm = (ids == c.image_token).int()
        positions, delta = model.model.get_rope_index(
            ids, mm_token_type_ids=mm, image_grid_thw=grids
        )
        vision = model.model.visual(pixels, grids).pooler_output
        output = model(
            input_ids=ids,
            pixel_values=pixels,
            image_grid_thw=grids,
            mm_token_type_ids=mm,
            use_cache=True,
        )
        arrays.update(
            ids=ids.numpy(),
            pixels=pixels.numpy(),
            grids=grids.numpy(),
            positions=positions.numpy(),
            rope_delta=delta.numpy(),
            vision=vision.numpy(),
            prefill_logits=output.logits.numpy(),
        )
        generated = torch.tensor([[26, 27, 28], [29, 30, 31]])
        cache, results = output.past_key_values, []
        for step in range(3):
            pos = (ids.shape[1] + step + delta).view(1, 2, 1).expand(3, -1, -1)
            output = model(
                input_ids=generated[:, step : step + 1],
                position_ids=pos,
                past_key_values=cache,
                use_cache=True,
            )
            cache = output.past_key_values
            results.append(output.logits.numpy())
        arrays["decode_ids"] = generated.numpy()
        arrays["decode_logits"] = np.concatenate(results, axis=1)
        if args.long_context:
            # Independent Torch INT8/BF16-scale cache policy. Only newly inserted
            # K/V values are rounded; indexer keys and GDN state stay untouched.
            from transformers.cache_utils import DynamicCache

            update = DynamicCache.update

            def quantized_update(self, key, value, layer_idx, *rest, **kwargs):
                def round_cache(tensor):
                    grouped = tensor.float().reshape(*tensor.shape[:-1], -1, c.cache_group)
                    scale = (grouped.abs().amax(dim=-1) / 127).to(torch.bfloat16).float()
                    divisor = torch.where(scale == 0, 1, scale)
                    codes = (grouped / divisor[..., None]).round().clamp(-127, 127)
                    return (codes * scale[..., None]).reshape(tensor.shape).to(tensor.dtype)

                return update(
                    self, round_cache(key), round_cache(value), layer_idx, *rest, **kwargs
                )

            DynamicCache.update = quantized_update
            try:
                output = model(
                    input_ids=ids,
                    pixel_values=pixels,
                    image_grid_thw=grids,
                    mm_token_type_ids=mm,
                    use_cache=True,
                )
                arrays["q8_prefill_logits"] = output.logits.numpy()
                arrays["q8_keys"] = output.past_key_values.layers[3].keys.numpy()
                arrays["q8_values"] = output.past_key_values.layers[3].values.numpy()
                cache, results = output.past_key_values, []
                for step in range(3):
                    pos = (ids.shape[1] + step + delta).view(1, 2, 1).expand(3, -1, -1)
                    output = model(
                        input_ids=generated[:, step : step + 1],
                        position_ids=pos,
                        past_key_values=cache,
                        use_cache=True,
                    )
                    cache = output.past_key_values
                    results.append(output.logits.numpy())
                arrays["q8_decode_logits"] = np.concatenate(results, axis=1)
                # Actual BF16 model, with the production FP32 router/norm and
                # recurrence coefficient policy. This is a separate precision
                # oracle, in addition to strict FP32 equation comparisons.
                from types import MethodType

                router_methods = []
                fp32_buffers = {
                    name: value.detach().clone() for name, value in model.named_buffers()
                }
                model.to(torch.bfloat16)
                # Rotary frequencies are FP32 coefficients, just as in the
                # JAX policy. .to(BF16) otherwise silently rounds those buffers
                # before their later .float(), changing long-context angles.
                for name, saved in fp32_buffers.items():
                    if saved.is_floating_point():
                        model.get_buffer(name).data = saved.clone()
                for name, parameter in model.named_parameters():
                    if "norm" in name or name.endswith(("A_log", "dt_bias", "mlp.gate.weight")):
                        parameter.data = torch.from_numpy(arrays[name].copy())
                # Isolate compute precision from discontinuous choices: tiny
                # random routers have near ties that can flip after one BF16
                # ULP. Keep all experts exercised but equal for this precision
                # oracle only. Strict FP32/W8A8 goldens retain random experts
                # and sparse selection, and integer/routing tests are separate.
                for i, layer in enumerate(model.model.language_model.layers):
                    for kind, param in (
                        ("gate", layer.mlp.experts.gate_up_proj[:, : c.intermediate]),
                        ("up", layer.mlp.experts.gate_up_proj[:, c.intermediate :]),
                        ("down", layer.mlp.experts.down_proj),
                    ):
                        original = arrays[f"packed.{i}.{kind}"]
                        packed = np.repeat(original[:1], c.experts, axis=0)
                        arrays[f"bf16_packed.{i}.{kind}"] = packed
                        raw = np.ascontiguousarray(packed)
                        scale = (
                            np.ascontiguousarray(raw[..., :2]).view("<f2").reshape(raw.shape[:-1])
                        )
                        decoded = (
                            scale.astype(np.float32)[..., None] * raw[..., 2:].view(np.int8)
                        ).reshape(param.shape)
                        param.copy_(torch.from_numpy(decoded).to(torch.bfloat16))
                    if c.is_qsa(i):
                        layer.self_attn.indexer.token_budget = 64
                        layer.self_attn.indexer.block_topk = 16

                def fp32_router(self, hidden):
                    logits = torch.nn.functional.linear(
                        hidden.float().reshape(-1, c.hidden), self.weight
                    )
                    values, indices = logits.softmax(dim=-1).topk(c.top_k, dim=-1)
                    values = values / values.sum(dim=-1, keepdim=True)
                    if hidden.numel() == 2 * 64 * c.hidden:
                        arrays[f"bf16_router_ids.{self.reference_index}"] = indices.reshape(
                            2, 64, -1
                        ).numpy()
                        arrays[f"bf16_router_input.{self.reference_index}"] = (
                            hidden.float().reshape(2, 64, -1).numpy()
                        )
                    return logits, values.to(hidden.dtype), indices

                for number, layer in enumerate(model.model.language_model.layers):
                    router = layer.mlp.gate
                    router.reference_index = number
                    router_methods.append(router.forward)
                    router.forward = MethodType(fp32_router, router)
                arrays["bf16_vision"] = (
                    model.model.visual(pixels.to(torch.bfloat16), grids)
                    .pooler_output.float()
                    .numpy()
                )
                output = model(
                    input_ids=ids,
                    pixel_values=pixels.to(torch.bfloat16),
                    image_grid_thw=grids,
                    mm_token_type_ids=mm,
                    use_cache=True,
                )
                arrays["bf16_q8_prefill_logits"] = output.logits.float().numpy()
                cache, results = output.past_key_values, []
                for step in range(3):
                    pos = (ids.shape[1] + step + delta).view(1, 2, 1).expand(3, -1, -1)
                    output = model(
                        input_ids=generated[:, step : step + 1],
                        position_ids=pos,
                        past_key_values=cache,
                        use_cache=True,
                    )
                    cache = output.past_key_values
                    results.append(output.logits.float().numpy())
                arrays["bf16_q8_decode_logits"] = np.concatenate(results, axis=1)
                model.to(torch.float32)
                for name, saved in fp32_buffers.items():
                    model.get_buffer(name).copy_(saved)
                for name, parameter in model.named_parameters():
                    parameter.copy_(torch.from_numpy(arrays[name]))
                for layer, method in zip(
                    model.model.language_model.layers, router_methods, strict=True
                ):
                    layer.mlp.gate.forward = method
                    if hasattr(layer, "self_attn"):
                        layer.self_attn.indexer.token_budget = c.qsa_budget
                        layer.self_attn.indexer.block_topk = c.qsa_budget // c.indexer_compress

                # Independent INT8 expert arithmetic, with the same Q8 cache.
                # Packed weights are read directly, independently of JAX code.
                def integer_project(x, packed):
                    raw = np.ascontiguousarray(packed)
                    scales = torch.from_numpy(
                        np.ascontiguousarray(raw[..., :2])
                        .view("<f2")
                        .reshape(raw.shape[:2])
                        .astype(np.float32)
                    )
                    codes = torch.from_numpy(
                        np.ascontiguousarray(raw[..., 2:]).view(np.int8).astype(np.int32)
                    )
                    groups = x.float().reshape(x.shape[0], -1, 32)
                    act_scale = groups.abs().amax(dim=-1) / 127
                    denominator = torch.where(act_scale == 0, 1, act_scale)
                    activation = (groups / denominator[..., None]).round().clamp(-127, 127).int()
                    partial = torch.einsum("bgk,ogk->bgo", activation, codes).float()
                    return (partial * act_scale[..., None] * scales.T[None]).sum(dim=1).to(x.dtype)

                def expert_forward(self, hidden, ids, weights):
                    result = torch.zeros_like(hidden)
                    for expert in range(c.experts):
                        token, route = torch.where(ids == expert)
                        if token.numel() == 0:
                            continue
                        x = hidden[token]
                        gate = integer_project(x, self.reference_packed["gate"][expert])
                        up = integer_project(x, self.reference_packed["up"][expert])
                        value = torch.nn.functional.silu(gate) * up
                        value = integer_project(value, self.reference_packed["down"][expert])
                        result.index_add_(0, token, value * weights[token, route, None])
                    return result

                for i, layer in enumerate(model.model.language_model.layers):
                    layer.mlp.experts.reference_packed = {
                        kind: arrays[f"packed.{i}.{kind}"] for kind in ("gate", "up", "down")
                    }
                    layer.mlp.experts.forward = MethodType(expert_forward, layer.mlp.experts)
                output = model(
                    input_ids=ids,
                    pixel_values=pixels,
                    image_grid_thw=grids,
                    mm_token_type_ids=mm,
                    use_cache=True,
                )
                arrays["int8_q8_prefill_logits"] = output.logits.numpy()
                cache, results = output.past_key_values, []
                for step in range(3):
                    pos = (ids.shape[1] + step + delta).view(1, 2, 1).expand(3, -1, -1)
                    output = model(
                        input_ids=generated[:, step : step + 1],
                        position_ids=pos,
                        past_key_values=cache,
                        use_cache=True,
                    )
                    cache = output.past_key_values
                    results.append(output.logits.numpy())
                arrays["int8_q8_decode_logits"] = np.concatenate(results, axis=1)
            finally:
                DynamicCache.update = update
    out = ROOT / "kaggle/qwen38_tpu/fixtures"
    out.mkdir(parents=True, exist_ok=True)
    name = "long-reference" if args.long_context else "tiny-reference"
    np.savez_compressed(out / (name + ".npz"), **arrays)
    manifest = dict(
        config=asdict(c),
        vision_config=asdict(v),
        reference="huggingface/transformers@536ecc007387a50e77603bb5d92100e9b07514cc",
        modeling_sha256=model_sha,
        configuration_sha256=config_sha,
        torch_version=torch.__version__,
        fixture_sha256=hashlib.sha256((out / (name + ".npz")).read_bytes()).hexdigest(),
        sparse_oracle_tie_policy="opposite query heads avoid zero-score ties"
        if args.long_context
        else None,
        bf16_precision_oracle="identical routed experts and 64-token budget isolate BF16 rounding",
    )
    (out / (name + ".json")).write_text(json.dumps(manifest, indent=2) + "\n")
    print("Pinned full-model vision + prefill + cached decode fixture generated")


if __name__ == "__main__":
    main()
