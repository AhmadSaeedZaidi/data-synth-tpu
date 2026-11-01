"""Audited GGUF access, host placement and explicit conversion back to HF layouts."""

import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from glm53_offload.weights import RangeFile, hf_url
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from .config import GGUF_REPO, GGUF_REVISION, VisionConfig
from .ngram import HostNGramTable, NGramHash
from .parallel import packed_spec, split_expert_bytes
from .quant import block_layout, dequantize_rows


class AdaptiveRangeFile(RangeFile):
    """Keep exact range checks; shrink truncated/failed reads to bounded pieces."""

    def _range(self, offset, length):
        try:
            return super()._range(offset, length)
        except OSError:
            if length <= 256 << 10:
                raise
            middle = length // 2
            return self._range(offset, middle) + self._range(offset + middle, length - middle)


class Checkpoint:
    def __init__(self, audit_path, local_dir=None, log=print):
        self.audit = json.loads(Path(audit_path).read_text())
        if self.audit["repo"] != GGUF_REPO or self.audit["revision"] != GGUF_REVISION:
            raise ValueError("Checkpoint audit does not match pinned revision")
        self.tensors = self.audit["tensors"]
        self.local_dir, self.log = local_dir, log
        self.files = {
            f["file"]: AdaptiveRangeFile(
                hf_url(GGUF_REPO, GGUF_REVISION, f["file"]),
                f["file_bytes"],
                chunk_bytes=16 << 20,
                attempts=2,
            )
            for f in self.audit["files"]
        }
        self.bytes_read = 0

    def validate(self, c):
        """Check tensor coverage, shapes and source bounds before any large read."""
        expected = {
            "token_embd.weight": [c.hidden, c.vocab],
            "output.weight": [c.hidden, c.vocab],
            "output_hc_norm.weight": [c.streams * c.hidden],
            "output_hc_down.weight": [c.streams * c.hidden, c.lowrank],
            "output_hc_up.weight": [c.lowrank, c.streams * c.hidden],
        }
        for layer in range(c.layers):
            prefix = f"blk.{layer}."
            for block in ("hc_attn", "hc_ffn"):
                for kind, shape in (
                    ("norm", [c.streams * c.hidden]),
                    ("down", [c.streams * c.hidden, c.lowrank]),
                    ("up", [c.lowrank, c.streams * c.hidden]),
                    ("inject", [c.streams * c.hidden, c.streams]),
                ):
                    expected[prefix + block + "_" + kind + ".weight"] = shape
            for kind in ("gate", "up"):
                expected[prefix + f"ffn_{kind}_exps.weight"] = [c.hidden, c.intermediate, c.experts]
                expected[prefix + f"ffn_{kind}_shexp.weight"] = [c.hidden, c.intermediate]
            expected[prefix + "ffn_down_exps.weight"] = [c.intermediate, c.hidden, c.experts]
            expected[prefix + "ffn_down_shexp.weight"] = [c.intermediate, c.hidden]
            expected[prefix + "ffn_gate_inp.weight"] = [c.hidden, c.experts]
            expected[prefix + "ffn_gate_inp_shexp.weight"] = [c.hidden]
            if c.is_qsa(layer):
                for kind, out in (
                    ("q", 2 * c.query_heads * c.head_dim),
                    ("k", c.kv_heads * c.head_dim),
                    ("v", c.kv_heads * c.head_dim),
                    ("output", c.hidden),
                ):
                    expected[prefix + f"attn_{kind}.weight"] = [
                        c.query_heads * c.head_dim if kind == "output" else c.hidden,
                        out,
                    ]
                for kind in ("q", "k"):
                    expected[prefix + f"attn_{kind}_norm.weight"] = [c.head_dim]
                if c.sparse_qsa:
                    expected[prefix + "indexer.q_proj.weight"] = [
                        c.hidden,
                        c.indexer_heads * c.indexer_dim,
                    ]
                    expected[prefix + "indexer.k_proj.weight"] = [c.hidden, c.indexer_dim]
                    expected[prefix + "indexer.q_norm.weight"] = [c.indexer_dim]
                    expected[prefix + "indexer.k_norm.weight"] = [c.indexer_dim]
            else:
                for name, shape in (
                    ("attn_qkv.weight", [c.hidden, c.conv_channels]),
                    ("attn_gate.weight", [c.hidden, c.value_heads * c.state_dim]),
                    ("ssm_out.weight", [c.value_heads * c.state_dim, c.hidden]),
                    ("ssm_alpha.weight", [c.hidden, c.value_heads]),
                    ("ssm_beta.weight", [c.hidden, c.value_heads]),
                    ("ssm_a", [c.value_heads]),
                    ("ssm_dt.bias", [c.value_heads]),
                    ("ssm_conv1d.weight", [c.conv_kernel, c.conv_channels]),
                    ("ssm_norm.weight", [c.state_dim]),
                ):
                    expected[prefix + name] = shape
        prefix = f"blk.{c.ple_layer}."
        for name, shape in (
            ("ple_key.weight", [c.ple_dim, c.streams * c.hidden]),
            ("ple_value.weight", [c.ple_dim, c.hidden]),
            ("ple_norm_key.weight", [c.streams * c.hidden]),
            ("ple_norm_query.weight", [c.streams * c.hidden]),
            ("ple_norm_conv.weight", [c.streams * c.hidden]),
            ("ple_conv1d.weight", [c.conv_kernel, c.streams * c.hidden]),
        ):
            expected[prefix + name] = shape
        hasher = NGramHash(c)
        expected["per_layer_token_embd.weight"] = [c.ple_dim // len(hasher.sizes), hasher.rows]
        v = VisionConfig()
        expected.update(
            {
                "v.position_embd.weight": [v.hidden, v.grid_side**2],
                "v.patch_embd.weight": [v.patch, v.patch, 3, v.hidden],
                "v.patch_embd.weight.1": [v.patch, v.patch, 3, v.hidden],
                "v.patch_embd.bias": [v.hidden],
                "v.post_ln.weight": [v.hidden],
                "v.post_ln.bias": [v.hidden],
                "mm.0.weight": [v.hidden * v.merge**2, v.hidden * v.merge**2],
                "mm.0.bias": [v.hidden * v.merge**2],
                "mm.2.weight": [v.hidden * v.merge**2, c.hidden],
                "mm.2.bias": [c.hidden],
            }
        )
        for layer in range(v.layers):
            for name, shape in (
                ("ln1", [v.hidden]),
                ("ln2", [v.hidden]),
                ("attn_qkv", [v.hidden, 3 * v.hidden]),
                ("attn_out", [v.hidden, v.hidden]),
                ("ffn_up", [v.hidden, v.intermediate]),
                ("ffn_down", [v.intermediate, v.hidden]),
            ):
                expected[f"v.blk.{layer}.{name}.weight"] = shape
                expected[f"v.blk.{layer}.{name}.bias"] = [shape[-1]]
        for key, shape in expected.items():
            if key not in self.tensors or self.tensors[key]["dims"] != shape:
                raise ValueError(f"Missing or incompatible tensor: {key}")
        for key, t in self.tensors.items():
            if (
                t["data0"] + t["offset"] + t["nbytes"] > self.files[t["file"]].size
                or min(t["data0"], t["offset"], t["nbytes"]) < 0
            ):
                raise ValueError(f"Tensor outside source shard: {key}")
            width = t["dims"][0]
            values, size = block_layout(t["type"])
            if width % values or math.prod(t["dims"]) // values * size != t["nbytes"]:
                raise ValueError(f"Invalid quantized tensor size: {key}")
        return len(expected)

    def read(self, name, offset=0, length=None):
        t = self.tensors[name]
        length = t["nbytes"] - offset if length is None else length
        if offset < 0 or length < 0 or offset + length > t["nbytes"]:
            raise ValueError("Read outside tensor")
        absolute = t["data0"] + t["offset"] + offset
        if self.local_dir:
            import os

            path = Path(self.local_dir) / t["file"]
            with path.open("rb") as f:
                data = os.pread(f.fileno(), length, absolute)
            if len(data) != length:
                raise OSError("Truncated local tensor")
        else:
            data = self.files[t["file"]].pread(absolute, length)
        self.bytes_read += length
        return data

    def packed_shape(self, name):
        t = self.tensors[name]
        width, *others = t["dims"]
        values, size = block_layout(t["type"])
        if width % values:
            raise ValueError(f"{name}: GGUF rows do not divide into quant blocks")
        return tuple(reversed(others)) + (width // values, size)

    def packed(self, name):
        return np.frombuffer(self.read(name), np.uint8).reshape(self.packed_shape(name))

    def dense(self, name):
        t = self.tensors[name]
        out = dequantize_rows(self.packed(name), t["type"], t["dims"][0])
        return out.reshape(tuple(reversed(t["dims"])))

    def host_ngram(self, c, *, cache_rows=16384):
        name = "per_layer_token_embd.weight"
        t = self.tensors[name]
        hasher = NGramHash(c)
        if t["dims"] != [c.ple_dim // len(hasher.sizes), hasher.rows]:
            raise ValueError("N-gram dimensions differ from architecture")
        # Validate checkpoint-carried integer constants, not rounded floats.
        kv = self.audit["metadata"]
        for suffix, expected in (
            ("ple.layer_multipliers", hasher.multipliers),
            ("ple.head_offsets", hasher.offsets.tolist()),
            ("ple.head_vocab_sizes", hasher.sizes.tolist()),
        ):
            actual = kv.get("qwen4exp." + suffix)
            if actual is not None and list(actual) != list(expected):
                raise ValueError(f"Checkpoint hash constant differs: {suffix}")
        raw = np.empty(self.packed_shape(name), np.uint8)
        view = memoryview(raw).cast("B")
        chunk = 64 << 20
        for offset in range(0, len(view), chunk):
            end = min(len(view), offset + chunk)
            view[offset:end] = self.read(name, offset, end - offset)
            if offset % (1 << 30) == 0:
                self.log(f"ngram_ram_load {end}/{len(view)} bytes")
        return HostNGramTable(raw, t["type"], t["dims"][0], cache_rows), hasher

    def resident_experts(self, layer, mesh):
        result = {}
        for kind in ("gate", "up", "down"):
            name = f"blk.{layer}.ffn_{kind}_exps.weight"
            count = self.packed_shape(name)[0]
            stride = self.tensors[name]["nbytes"] // count
            # Flatten bytes per expert to avoid TPU padding of block axes
            # such as 10x210. Restore row/block axes after selecting an expert.
            tp = mesh.shape.get("tensor", 1)
            ep = mesh.shape["expert"]
            shape = (count * tp, stride // tp)
            devices = list(mesh.devices.flat)
            if count % ep:
                raise ValueError("Expert count does not divide into expert groups")
            per_group = count // ep
            dims = self.tensors[name]["dims"]
            hidden, intermediate = (dims[1], dims[0]) if kind == "down" else (dims[0], dims[1])
            arrays = []
            for group in range(ep):
                raw = np.frombuffer(
                    self.read(name, group * per_group * stride, per_group * stride), np.uint8
                ).reshape(per_group, stride)
                pieces = split_expert_bytes(
                    raw, kind, self.tensors[name]["type"], hidden, intermediate, tp
                )
                for tensor_rank, piece in enumerate(pieces):
                    array = jax.device_put(piece, devices[group * tp + tensor_rank])
                    array.block_until_ready()
                    arrays.append(array)
                del raw, pieces  # bound host staging to one EP group's packed matrix
            result[kind] = jax.make_array_from_single_device_arrays(
                shape, NamedSharding(mesh, packed_spec(mesh)), arrays
            )
            self.log(
                f"expert_load layer={layer} matrix={kind} bytes={self.tensors[name]['nbytes']}"
            )
        return result

    def memory_plan(self, c, batch, *, chips=8, hbm_bytes, reserve_bytes=8 << 30):
        experts = dense = vision = token_embedding = 0
        for name, t in self.tensors.items():
            if name == "per_layer_token_embd.weight":
                continue
            if name.startswith("blk."):
                layer = int(name.split(".")[1])
                if layer >= c.layers or (".indexer." in name and not c.sparse_qsa):
                    continue  # unused MTP; short contexts do not need an indexer
            if "_exps." in name:
                experts += t["nbytes"]
            elif name == "token_embd.weight":
                token_embedding += t["nbytes"]  # packed host token lookup too
            else:
                expanded = math.prod(t["dims"]) * (4 if t["type"] == "F32" else 2)
                dense += expanded
                if name.startswith(("v.", "mm.")):
                    vision += expanded
        states = c.state_bytes(batch)
        required = experts + dense + states + reserve_bytes
        return dict(
            experts_bytes=experts,
            dense_bytes=dense,
            vision_bytes=vision,
            language_dense_bytes=dense - vision,
            state_bytes=states,
            state_components=c.state_components(batch),
            kv_cache_dtype=c.kv_cache_dtype,
            qsa_token_budget=c.qsa_budget,
            sparse_qsa=c.sparse_qsa,
            state_accounting="logical; reserve also covers replication and TPU padding",
            reserve_bytes=reserve_bytes,
            required_hbm_bytes=required,
            available_hbm_bytes=hbm_bytes,
            per_chip_bytes=required / chips,
            batch_size=batch,
            max_context_tokens=c.max_context,
            chips=chips,
            host_ngram_bytes=self.tensors["per_layer_token_embd.weight"]["nbytes"],
            host_token_embedding_bytes=token_embedding,
            fits=required <= hbm_bytes,
        )


def undo_value_head_order(array, axis, c, head_dim):
    """GGUF tiles V heads; restore contiguous repetitions used by Hugging Face."""
    shape = array.shape
    axis = axis % array.ndim
    repeat = c.value_heads // c.key_heads
    expanded = array.reshape(shape[:axis] + (repeat, c.key_heads, head_dim) + shape[axis + 1 :])
    return np.swapaxes(expanded, axis, axis + 1).reshape(shape)


def load_layer(checkpoint, layer, c, mesh):
    prefix = f"blk.{layer}."
    replicated = NamedSharding(mesh, P())

    def get(suffix, *, shard=True):
        array = checkpoint.dense(prefix + suffix)
        # GGUF deliberately retains the routed expert gate in FP32. Preserve
        # that precision: changing top-k membership is not a harmless rounding.
        full_precision = suffix == "ffn_gate_inp.weight" or (
            array.ndim == 1 and "shexp" not in suffix
        )
        dtype = jnp.float32 if full_precision else jnp.bfloat16
        # Large linear matrices partition their output rows. Runtime x stays
        # replicated; XLA propagates row sharding through projections.
        layout = (
            NamedSharding(mesh, P("expert", None))
            if shard and array.ndim == 2 and array.shape[0] % mesh.size == 0
            else replicated
        )
        return jax.device_put(array.astype(np.float32), layout).astype(dtype)

    result = {}
    for target, source in (("attn_gr", "hc_attn"), ("ffn_gr", "hc_ffn")):
        result[target] = {
            key: get(source + "_" + suffix, shard=key in ("down", "up"))
            for key, suffix in (
                ("norm", "norm.weight"),
                ("down", "down.weight"),
                ("up", "up.weight"),
                ("inject", "inject.weight"),
            )
        }
    result["mlp"] = {
        key: get(suffix)
        for key, suffix in (
            ("router", "ffn_gate_inp.weight"),
            ("shared_router", "ffn_gate_inp_shexp.weight"),
            ("shared_gate", "ffn_gate_shexp.weight"),
            ("shared_up", "ffn_up_shexp.weight"),
            ("shared_down", "ffn_down_shexp.weight"),
        )
    }
    if c.is_qsa(layer):
        result["attn"] = {
            key: get(suffix)
            for key, suffix in (
                ("q", "attn_q.weight"),
                ("k", "attn_k.weight"),
                ("v", "attn_v.weight"),
                ("out", "attn_output.weight"),
                ("q_norm", "attn_q_norm.weight"),
                ("k_norm", "attn_k_norm.weight"),
            )
        }
        if c.sparse_qsa:
            result["attn"].update(
                {
                    key: get(suffix)
                    for key, suffix in (
                        ("index_q", "indexer.q_proj.weight"),
                        ("index_k", "indexer.k_proj.weight"),
                        ("index_q_norm", "indexer.q_norm.weight"),
                        ("index_k_norm", "indexer.k_norm.weight"),
                    )
                }
            )
    else:
        result["attn"] = {}
        for key, suffix in (
            ("qkv", "attn_qkv.weight"),
            ("gate", "attn_gate.weight"),
            ("out", "ssm_out.weight"),
            ("alpha", "ssm_alpha.weight"),
            ("beta", "ssm_beta.weight"),
            ("a", "ssm_a"),
            ("dt", "ssm_dt.bias"),
            ("conv", "ssm_conv1d.weight"),
            ("norm", "ssm_norm.weight"),
        ):
            array = checkpoint.dense(prefix + suffix)
            if key in ("gate", "alpha", "beta", "a", "dt"):
                array = undo_value_head_order(array, 0, c, c.state_dim if key == "gate" else 1)
            elif key in ("qkv", "conv"):
                cut = 2 * c.key_heads * c.state_dim
                array = np.concatenate(
                    (array[:cut], undo_value_head_order(array[cut:], 0, c, c.state_dim))
                )
            elif key == "out":
                array = undo_value_head_order(array, 1, c, c.state_dim)
            layout = (
                NamedSharding(mesh, P("expert", None))
                if array.ndim == 2 and array.shape[0] % mesh.size == 0
                else replicated
            )
            dtype = jnp.float32 if key in ("a", "dt", "norm") else jnp.bfloat16
            result["attn"][key] = jax.device_put(array, layout).astype(dtype)
    if layer == c.ple_layer:
        result["ple"] = {
            key: get(suffix)
            for key, suffix in (
                ("key", "ple_key.weight"),
                ("value", "ple_value.weight"),
                ("key_norm", "ple_norm_key.weight"),
                ("query_norm", "ple_norm_query.weight"),
                ("conv_norm", "ple_norm_conv.weight"),
                ("conv", "ple_conv1d.weight"),
            )
        }
    return result
