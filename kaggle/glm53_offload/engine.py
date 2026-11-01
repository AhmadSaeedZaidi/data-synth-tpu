"""Layer-segmented GLM engine with selected GGUF experts streamed from host RAM.

The base engine owns model parameter placement, tensor-parallel specs, cache
layouts, token embedding and the vocab-sharded output head. This class splits
each decoder layer around expert routing: attention and routing run on device,
the store reads only routed experts, then a small fixed expert tile is decoded
and multiplied at a time. No host callback runs inside a JIT program.
"""

from __future__ import annotations

import json
import math
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from glm53 import model as M
from glm53.engine import AXIS, Engine, R
from jax import lax
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from . import progress
from .expert_kernel import tiled_experts
from .gguf_quant import block_layout
from .routing import expert_tile_capacities, expert_token_rows

try:
    from jax import shard_map
except ImportError:  # pragma: no cover - older JAX images
    from jax.experimental.shard_map import shard_map


class PackedOffloadEngine(Engine):
    """Run GGUF experts from an on-demand host store, in bounded expert tiles.

    ``store`` implements two methods:

    * ``qtypes(layer)`` returns ``(gate_qtype, up_qtype, down_qtype)``.
    * ``gather(layer, global_idx)`` returns ``(packed, remapped_idx, Ep)``.
      ``packed`` contains three uint8 arrays, each chip-major with shape
      ``[n_devices, Ep, output_rows_local, input_blocks, block_bytes]``;
      gate/up shard output rows and down shards input blocks. ``remapped_idx``
      has the same shape as ``global_idx`` and addresses the compact packed
      expert axis. ``Ep`` includes zero-padded rows.

    Prompts are prefetched one stream at a time. Decode accepts independent
    cache sets and positions, so requests at different context lengths can be
    routed together. ``prefill`` accepts an optional image override as
    ``(token_positions, vectors)``. MLA/indexer caches stay replicated by
    default: sequence-sharded DSA can break ties differently when selecting
    global pools, which changes attention results.
    """

    PREFILL_BUCKETS = (32, 64)
    PREFILL_WINDOWS = (32, 64, 128, 256)

    def __init__(
        self,
        cfg: M.Cfg,
        params: dict,
        store: Any,
        devices=None,
        max_len: int = 8192,
        *,
        expert_tile: int = 4,
        prefill_chunk: int = 32,
        prefill_moe_window: int = 32,
        tile_routed_experts: bool = True,
        seq_shard: bool = False,
        expert_cache_bytes: int = 0,
        direct_expert_cache: bool = True,
        transport: Callable[[tuple[np.ndarray, ...]], tuple[Any, ...]] | None = None,
        routing_log_path: str | Path | None = None,
        execution_log_path: str | Path | None = None,
    ):
        if (
            store is None
            or not callable(getattr(store, "gather", None))
            or not callable(getattr(store, "qtypes", None))
        ):
            raise TypeError("store must provide qtypes(layer) and gather(layer, global_idx)")
        if expert_tile not in (4, 8):
            raise ValueError("expert_tile must be 4 or 8")
        if prefill_chunk not in self.PREFILL_BUCKETS:
            raise ValueError("prefill_chunk must be 32 or 64")
        if prefill_moe_window not in self.PREFILL_WINDOWS:
            raise ValueError("prefill_moe_window must be 32, 64, 128 or 256")
        if not isinstance(tile_routed_experts, bool):
            raise ValueError("tile_routed_experts must be boolean")

        # The parent Engine removes any dense/plane expert tensors before
        # placing parameters. Keep only router and shared-expert weights here.
        clean = dict(params)
        layers = []
        for i, layer in enumerate(params["layers"]):
            item = dict(layer)
            mlp = dict(layer["mlp"])
            if cfg.mlp_types[i] == "sparse":
                for key in ("gate_up", "down", "gate_q", "up_q", "down_q"):
                    mlp.pop(key, None)
            item["mlp"] = mlp
            layers.append(item)
        clean["layers"] = layers

        super().__init__(
            cfg,
            clean,
            host_experts=store,
            devices=devices,
            max_len=max_len,
            int8_nonexpert=False,
            seq_shard=seq_shard,
            cache_q8=False,
        )
        self.store = store
        self.expert_tile = int(expert_tile)
        self.prefill_chunk = min(int(prefill_chunk), max_len)
        self.prefill_moe_window = min(max(int(prefill_moe_window), prefill_chunk), max_len)
        self.tile_routed_experts = tile_routed_experts
        self.direct_expert_cache = direct_expert_cache
        self.transport = transport or self._device_transport
        self._offload_progs: dict[tuple[Any, ...], Any] = {}
        self._layer_signatures = {}
        self.routing_stats = {
            "calls": 0,
            "routed_token_expert_pairs": 0,
            "computed_token_expert_pairs": 0,
            "global_capacity_token_expert_pairs": 0,
            "all_token_expert_pairs": 0,
            "max_tokens_per_expert": 0,
        }
        self.routing_by_layer = {}
        self.expert_usage_by_layer = {}
        self._global_routes = {}
        self.routing_log_path = Path(routing_log_path) if routing_log_path is not None else None
        self.routing_log_bytes = 0
        self.execution_log_path = (
            Path(execution_log_path) if execution_log_path is not None else None
        )
        self.execution_samples = []
        self._sample_counts = {}
        self._sample_layer = next(
            (i for i, kind in enumerate(self.cfg.mlp_types) if kind == "sparse"), None
        )
        self._sample_purpose = "startup"
        self._sample_layers = (self._sample_layer,)
        self._sample_enabled = True
        self.expert_cache = None
        if expert_cache_bytes:
            from .expert_cache import ExpertCache

            self.expert_cache = ExpertCache(store, self.mesh, self.transport, expert_cache_bytes)

    def _write_execution_sample(self, sample, status):
        record = {key: value for key, value in sample.items() if not key.startswith("_")}
        record["status"] = status
        self.execution_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.execution_log_path.open("a") as stream:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
        return record

    @contextmanager
    def execution_sampling(self, purpose, *, enabled=True, layers=None):
        """Scope extra synchronizations to a diagnostic outside timed benchmarks."""
        previous = self._sample_purpose, self._sample_enabled, self._sample_layers
        self._sample_purpose, self._sample_enabled = purpose, enabled
        if layers is not None:
            self._sample_layers = tuple(layers)
        try:
            yield
        finally:
            self._sample_purpose, self._sample_enabled, self._sample_layers = previous

    def _start_sample(self, phase, layer, tokens):
        """Bound synchronization to two samples per purpose/size/phase/layer.

        These are wall timings of dependent work, not isolated device or link
        bandwidth. Cold samples may include compilation. START survives a
        later crash or timeout; DONE is written as soon as post work completes.
        """
        if (
            self.execution_log_path is None
            or not self._sample_enabled
            or layer not in self._sample_layers
        ):
            return None
        key = (self._sample_purpose, phase, layer, tokens)
        count = self._sample_counts.get(key, 0)
        if count >= 2 or len(self.execution_samples) >= 64:
            return None
        self._sample_counts[key] = count + 1
        sample = {
            "purpose": self._sample_purpose,
            "phase": phase,
            "layer": layer,
            "tokens": tokens,
            "sample_number": count + 1,
            "program_count_before": self.program_count,
            "scope": "selected sparse layer; dependent wall timing, may include compilation",
            "_start": time.perf_counter(),
            "_bytes_before": self.store.stats.get("bytes_transferred", 0),
        }
        self._write_execution_sample(sample, "START")
        return sample

    def _sample_router_ready(self, sample):
        if sample is not None:
            now = time.perf_counter()
            sample["attention_router_ready_seconds"] = now - sample["_start"]
            sample["_router_ready"] = now

    def _sample_packed_ready(self, sample, packed, ep):
        if sample is not None:
            now = time.perf_counter()
            sample["host_gather_dispatch_seconds"] = now - sample["_router_ready"]
            jax.block_until_ready(packed)
            sample["staging_cache_wait_seconds"] = time.perf_counter() - now
            sample["experts"] = ep
            sample["_packed_ready"] = time.perf_counter()

    def _finish_sample(self, sample, streams):
        if sample is not None:
            jax.block_until_ready(streams)
            now = time.perf_counter()
            sample["routing_plan_and_expert_post_seconds"] = now - sample["_packed_ready"]
            sample["elapsed_seconds"] = now - sample["_start"]
            sample["program_count_after"] = self.program_count
            sample["packed_bytes_submitted"] = (
                self.store.stats.get("bytes_transferred", 0) - sample["_bytes_before"]
            )
            self.execution_samples.append(self._write_execution_sample(sample, "DONE"))

    def _device_transport(self, packed: tuple[np.ndarray, ...]):
        """Place each chip-major packed table with its chip slice on the mesh."""
        sharding = NamedSharding(self.mesh, P(AXIS))
        # Keep full rows as the minor matrix axes. A [blocks, block_bytes]
        # suffix can waste TPU tiles when local down matrices have one block.
        return tuple(
            jax.device_put(np.ascontiguousarray(a).reshape(*a.shape[:3], -1), sharding)
            for a in packed
        )

    def _pad_store_result(self, packed, idx_local, ep: int, idx_shape):
        """Check the host-store contract and make Ep a multiple of the tile."""
        if ep <= 0 or ep > 288:
            raise ValueError(f"packed expert count Ep must be in [1, 288], got {ep}")
        arrays = tuple(np.asarray(a) for a in packed)
        if len(arrays) != 3:
            raise ValueError("store.gather must return gate, up, and down packed arrays")
        remapped = np.asarray(idx_local, dtype=np.int32)
        if remapped.shape != idx_shape:
            raise ValueError(f"remapped ids have shape {remapped.shape}; expected {idx_shape}")
        if np.any(remapped < -1) or np.any(remapped >= ep):
            raise ValueError("remapped expert ids must be -1 or within the packed expert axis")

        padded_ep = int(math.ceil(ep / self.expert_tile) * self.expert_tile)
        padded = []
        for a in arrays:
            if a.dtype != np.uint8:
                raise TypeError(f"packed GGUF arrays must be uint8; got {a.dtype}")
            if a.ndim != 5 or a.shape[0] != self.n or a.shape[1] != ep:
                raise ValueError(
                    "packed tables must have shape [n_devices, Ep, rows, blocks, block_bytes]"
                )
            if padded_ep != ep:
                z = np.zeros((self.n, padded_ep) + a.shape[2:], dtype=np.uint8)
                z[:, :ep] = a
                a = z
            padded.append(np.ascontiguousarray(a))
        return tuple(padded), remapped, padded_ep

    def _record_expert_usage(self, layer, idx, phase, image_mask=None):
        """Count original router IDs without another device read or changing routing.

        Image counts refer to image-embedding positions, not dedicated vision
        experts: deeper hidden states mix image and text context.
        """
        groups = {phase: np.ones(len(idx), dtype=np.bool_)}
        if phase == "prefill":
            image_mask = np.asarray(image_mask, dtype=np.bool_)
            if image_mask.shape != (len(idx),):
                raise ValueError("image routing mask must match real prefill tokens")
            groups.update(image_positions=image_mask, text_positions=~image_mask)
        layer_stats = self.expert_usage_by_layer.setdefault(layer, {})
        for name, mask in groups.items():
            selected = idx[mask].reshape(-1)
            if not selected.size:
                continue
            counts = np.bincount(selected, minlength=self.cfg.n_exp)
            present = (counts > 0).astype(np.int64)
            group = layer_stats.setdefault(
                name,
                {
                    "calls": 0,
                    "tokens": 0,
                    "assignments": [0] * self.cfg.n_exp,
                    "calls_with_expert": [0] * self.cfg.n_exp,
                },
            )
            group["calls"] += 1
            group["tokens"] += int(mask.sum())
            group["assignments"] = (np.asarray(group["assignments"]) + counts).tolist()
            group["calls_with_expert"] = (np.asarray(group["calls_with_expert"]) + present).tolist()

    def _gather(self, layer: int, idx, *, phase=None, image_mask=None):
        """Host gather only the unique valid routed IDs for a layer."""
        idx_np = np.asarray(idx, dtype=np.int32)
        if idx_np.ndim != 2:
            raise ValueError(f"router IDs must have shape [tokens, topk], got {idx_np.shape}")
        if phase is not None:
            self._record_expert_usage(layer, idx_np, phase, image_mask)
        self._global_routes[layer] = idx_np
        if self.expert_cache is not None:
            packed, local, ep = self.expert_cache.gather(
                layer, idx_np, direct=self.direct_expert_cache
            )
            return packed, local, ep, int(np.unique(local).size)
        packed, local, ep = self.store.gather(layer, idx_np)
        packed, local, ep = self._pad_store_result(packed, local, int(ep), idx_np.shape)
        gate_q, up_q, down_q = self._qtypes(layer)
        ml = self.lcfg.moe_inter
        expected = (
            (ml, math.ceil(self.cfg.hidden / block_layout(gate_q)[0]), block_layout(gate_q)[1]),
            (ml, math.ceil(self.cfg.hidden / block_layout(up_q)[0]), block_layout(up_q)[1]),
            (self.cfg.hidden, math.ceil(ml / block_layout(down_q)[0]), block_layout(down_q)[1]),
        )
        for name, array, shape in zip(("gate", "up", "down"), packed, expected):
            if array.shape[2:] != shape:
                raise ValueError(
                    f"{name} packed shard has trailing shape {array.shape[2:]}; expected {shape}"
                )
        active = int(np.unique(local[local >= 0]).size)
        if active == 0:
            raise ValueError("a sparse layer produced no routed experts")
        if active > ep:
            raise ValueError("host store returned fewer packed rows than active expert IDs")
        return self.transport(packed), local, ep, active

    def _qtypes(self, layer: int) -> tuple[str, str, str]:
        qtypes = self.store.qtypes(layer)
        if isinstance(qtypes, dict):
            qtypes = tuple(qtypes[k] for k in ("gate", "up", "down"))
        qtypes = tuple(str(q) for q in qtypes)
        if len(qtypes) != 3:
            raise ValueError("store.qtypes(layer) must return three GGUF types")
        return qtypes

    def _cache_spec(self, i):
        return super()._cache_spec(i)

    def _embed_program(self, B: int, T: int):
        key = ("embed", B, T)
        if key not in self._offload_progs:

            def prog(embed, tokens, override_vec, override_mask):
                x = self._embed(embed, tokens).astype(self.lcfg.dtype)
                x = jnp.where(override_mask[..., None], override_vec.astype(x.dtype), x)
                return jnp.broadcast_to(x[:, :, None, :], (B, T, self.cfg.hc, x.shape[-1]))

            self._offload_progs[key] = jax.jit(
                shard_map(
                    prog,
                    mesh=self.mesh,
                    in_specs=(self.specs["embed"], R, R, R),
                    out_specs=R,
                    check_vma=False,
                )
            )
        return self._offload_progs[key]

    def _prefill_pre(self, i: int, T: int, has_cache: bool):
        key = ("prefill_pre", self._layer_signature(i), T, has_cache)
        if key not in self._offload_progs:
            lt, mt = self.cfg.layer_types[i], self.cfg.mlp_types[i]
            cs = self._cache_spec(i)

            def prog(p, streams, pos0, length, cache):
                streams, h, post, comb, new_cache, idx, weights = M.layer_pre(
                    p,
                    streams,
                    self.lcfg,
                    lt,
                    mt,
                    cache,
                    pos0,
                    False,
                    "auto",
                    self.max_len,
                    length,
                )
                if idx is None:
                    idx = jnp.zeros((T, 1), jnp.int32)
                    weights = jnp.zeros((T, 1), jnp.float32)
                return streams, h, post, comb, new_cache, idx, weights

            self._offload_progs[key] = jax.jit(
                shard_map(
                    prog,
                    mesh=self.mesh,
                    in_specs=(self.specs["layers"][i], R, R, R, cs if has_cache else None),
                    out_specs=(R, R, R, R, cs, R, R),
                    check_vma=False,
                )
            )
        return self._offload_progs[key]

    def _decode_pre(self, i: int, B: int):
        key = ("decode_pre", self._layer_signature(i), B)
        if key not in self._offload_progs:
            lt, mt = self.cfg.layer_types[i], self.cfg.mlp_types[i]
            cs = self._cache_spec(i)

            def prog(p, streams, pos, caches):
                post, comb, h = M.hc_pre(p["hc_attn"], streams, self.lcfg)
                h = M.rmsnorm(h, p["ln1"], self.lcfg.eps)
                if lt == "linear_attention":
                    h, new_caches = M.kda_attention_rows(p["attn"], h, self.lcfg, caches)
                else:
                    h, new_caches = M.mla_attention_rows(
                        p["attn"], h, self.lcfg, caches, pos, "auto", self.max_len
                    )
                streams = M.hc_post(post, comb, h, streams)
                post, comb, h = M.hc_pre(p["hc_ffn"], streams, self.lcfg)
                h = M.rmsnorm(h, p["ln2"], self.lcfg.eps)
                if mt == "sparse":
                    idx, weights = M.moe_route(p["mlp"], h.reshape(B, -1), self.lcfg)
                else:
                    idx = jnp.zeros((B, 1), jnp.int32)
                    weights = jnp.zeros((B, 1), jnp.float32)
                return streams, h, post, comb, new_caches, idx, weights

            self._offload_progs[key] = jax.jit(
                shard_map(
                    prog,
                    mesh=self.mesh,
                    in_specs=(self.specs["layers"][i], R, R, [cs] * B),
                    out_specs=(R, R, R, R, [cs] * B, R, R),
                    check_vma=False,
                )
            )
        return self._offload_progs[key]

    def _routing_plan(self, local_ids, ep, layer=None):
        rows = expert_token_rows(local_ids, ep)
        capacities = expert_tile_capacities(rows, self.expert_tile)
        computed = (
            int(capacities.sum()) * self.expert_tile if self.tile_routed_experts else rows.size
        )
        stats = self.routing_stats
        stats["calls"] += 1
        stats["routed_token_expert_pairs"] += int(np.count_nonzero(rows >= 0))
        stats["computed_token_expert_pairs"] += int(computed)
        stats["global_capacity_token_expert_pairs"] += int(rows.size)
        stats["all_token_expert_pairs"] += int(ep * len(local_ids))
        stats["max_tokens_per_expert"] = max(
            stats["max_tokens_per_expert"], int(np.count_nonzero(rows >= 0, axis=1).max())
        )
        if layer is not None:
            layer_stats = self.routing_by_layer.setdefault(
                layer,
                {
                    "calls": 0,
                    "computed_pairs": 0,
                    "routed_pairs": 0,
                    "capacity_counts": {},
                },
            )
            layer_stats["calls"] += 1
            layer_stats["computed_pairs"] += int(computed)
            layer_stats["routed_pairs"] += int(np.count_nonzero(rows >= 0))
            capacity = str(rows.shape[1])
            counts = layer_stats["capacity_counts"]
            counts[capacity] = counts.get(capacity, 0) + 1
            if self.routing_log_path is not None and self.routing_log_bytes < (8 << 20):
                record = {
                    "layer": layer,
                    "tokens": len(local_ids),
                    "ep": ep,
                    "capacity": rows.shape[1],
                    "occupancy": np.count_nonzero(rows >= 0, axis=1).tolist(),
                    "tile_capacities": capacities.tolist(),
                }
                global_ids = self._global_routes[layer]
                remapped = local_ids[: len(global_ids)]
                valid = remapped >= 0
                ids_by_local = np.full(ep, -1, np.int32)
                ids_by_local[remapped[valid]] = global_ids[valid]
                record["global_experts"] = ids_by_local.tolist()
                line = json.dumps(record, separators=(",", ":")) + "\n"
                with self.routing_log_path.open("a") as stream:
                    stream.write(line)
                self.routing_log_bytes += len(line)
        progress.update(
            expert_route_capacity=rows.shape[1],
            packed_experts=ep,
            computed_token_expert_pairs=stats["computed_token_expert_pairs"],
            all_token_expert_pairs=stats["all_token_expert_pairs"],
        )
        return rows

    def _post_program(
        self,
        i: int,
        B: int,
        T: int,
        ep: int | None = None,
        route_capacity: int | None = None,
        direct=False,
        packed_shapes=None,
    ):
        sparse = self.cfg.mlp_types[i] == "sparse"
        qtypes = self._qtypes(i) if sparse else None
        # Direct miss-buffer shapes change independently of the routed union.
        # Count these variants explicitly instead of hiding JIT recompilation
        # inside an already-counted post function during a throughput sample.
        key = (
            "post",
            self._layer_signature(i),
            B,
            T,
            ep,
            qtypes,
            route_capacity,
            self.tile_routed_experts,
            direct,
            packed_shapes,
        )
        if key not in self._offload_progs:
            D = self.cfg.hidden

            def prog(
                p, streams, h, post, comb, weights, idx_local, token_rows, tile_capacities, *packed
            ):
                xf = h.reshape(B * T, D)
                if not sparse:
                    return M.layer_post(p, streams, h, post, comb, self.lcfg, "dense")

                expert_y = tiled_experts(
                    xf,
                    packed,
                    idx_local,
                    weights,
                    self.lcfg,
                    qtypes,
                    self.expert_tile,
                    token_rows,
                    tile_capacities if self.tile_routed_experts else None,
                )
                shared = M.mlp_partial(p["mlp"]["shared"], xf, self.lcfg)
                y = self.lcfg.reduce(expert_y.astype(xf.dtype) + shared).reshape(B, T, D)
                return M.hc_post(post, comb, y, streams)

            packed_specs = ((P(AXIS),) * 6 + (R,) if direct else (P(AXIS),) * 3) if sparse else ()
            self._offload_progs[key] = jax.jit(
                shard_map(
                    prog,
                    mesh=self.mesh,
                    in_specs=(self.specs["layers"][i], R, R, R, R, R, R, R, R) + packed_specs,
                    out_specs=R,
                    check_vma=False,
                )
            )
        return self._offload_progs[key]

    def _layer_signature(self, i):
        """Reuse programs across equal-shaped layers; weights remain arguments."""
        if i in self._layer_signatures:
            return self._layer_signatures[i]
        leaves, tree = jax.tree.flatten(self.params["layers"][i])
        specs = tuple(jax.tree.leaves(self.specs["layers"][i]))
        signature = (
            self.cfg.layer_types[i],
            self.cfg.mlp_types[i],
            tree,
            tuple((a.shape, str(a.dtype)) for a in leaves),
            specs,
        )
        self._layer_signatures[i] = signature
        return signature

    @property
    def program_count(self):
        cache_programs = len(self.expert_cache.programs) if self.expert_cache else 0
        return len(self._offload_progs) + cache_programs

    def _head_program(self, B: int, T: int):
        key = ("head", B, T)
        if key not in self._offload_progs:

            def prog(norm, lm_head, streams, length):
                h = M.rmsnorm(streams.mean(2).astype(self.lcfg.dtype), norm, self.cfg.eps)
                last = lax.dynamic_index_in_dim(h, length - 1, axis=1, keepdims=False)
                return self._logits(lm_head, last)

            self._offload_progs[key] = jax.jit(
                shard_map(
                    prog,
                    mesh=self.mesh,
                    in_specs=(R, self.specs["lm_head"], R, R),
                    out_specs=R,
                    check_vma=False,
                )
            )
        return self._offload_progs[key]

    def _bucket(self, length: int) -> int:
        for bucket in self.PREFILL_BUCKETS:
            if length <= bucket:
                return bucket
        raise ValueError(f"prefill chunk length must be <= 64, got {length}")

    def _prepare_override(self, override, start: int, length: int, bucket: int):
        vec = np.zeros((1, bucket, self.cfg.hidden), dtype=np.float32)
        mask = np.zeros((1, bucket), dtype=np.bool_)
        if override is None:
            return vec, mask
        if not isinstance(override, (tuple, list)) or len(override) != 2:
            raise TypeError("override must be (token_positions, vectors)")
        positions = np.asarray(override[0], dtype=np.int64).reshape(-1)
        vectors = np.asarray(override[1])
        if vectors.ndim != 2 or vectors.shape != (len(positions), self.cfg.hidden):
            raise ValueError("override vectors must have shape [len(token_positions), hidden]")
        selected = (positions >= start) & (positions < start + length)
        if selected.any():
            local = positions[selected] - start
            vec[0, local] = vectors[selected]
            mask[0, local] = True
        return vec, mask

    def prefill(self, tokens, override=None):
        """Prefill one stream, optionally replacing image-token embeddings.

        ``tokens`` may be ``[T]`` or ``[1,T]``. Returns ``(logits [1,V],
        caches)``; each cache entry has batch dimension one and can be placed
        directly into a stream's cache set for ``decode_rows``.
        """
        tokens = np.asarray(tokens, dtype=np.int32)
        if tokens.ndim == 1:
            tokens = tokens[None, :]
        if tokens.ndim != 2 or tokens.shape[0] != 1:
            raise ValueError("prefill accepts one stream with shape [T] or [1,T]")
        _, total = tokens.shape
        if total <= 0 or total > self.max_len:
            raise ValueError(f"prompt length must be in [1, {self.max_len}]")
        if override is not None:
            if not isinstance(override, (tuple, list)) or len(override) != 2:
                raise TypeError("override must be (token_positions, vectors)")
            override_positions = np.asarray(override[0], dtype=np.int64).reshape(-1)
            override_vectors = np.asarray(override[1])
            if override_vectors.shape != (len(override_positions), self.cfg.hidden):
                raise ValueError("override vectors must have shape [len(token_positions), hidden]")
            if np.any(override_positions < 0) or np.any(override_positions >= total):
                raise ValueError("override positions must refer to prompt tokens")
            if len(np.unique(override_positions)) != len(override_positions):
                raise ValueError("override positions must be unique")
            if not np.all(np.isfinite(override_vectors)):
                raise ValueError("override vectors must be finite")
            override = (override_positions, override_vectors)

        if self.prefill_moe_window > self.prefill_chunk:
            return self._prefill_windows(tokens, override)

        caches = None
        logits = None
        for start in range(0, total, self.prefill_chunk):
            stop = min(total, start + self.prefill_chunk)
            real_len = stop - start
            image_mask = (
                np.isin(np.arange(start, stop), override[0])
                if override is not None
                else np.zeros(real_len, np.bool_)
            )
            bucket = self._bucket(real_len)
            padded = np.zeros((1, bucket), dtype=np.int32)
            padded[:, :real_len] = tokens[:, start:stop]
            vec, mask = self._prepare_override(override, start, real_len, bucket)
            streams = self._embed_program(1, bucket)(
                self.params["embed"],
                jnp.asarray(padded),
                jnp.asarray(vec, dtype=self.lcfg.dtype),
                jnp.asarray(mask),
            )
            next_caches = []
            for i in range(self.cfg.n_layers):
                sample = self._start_sample("prefill", i, bucket)
                progress.update(
                    detail=f"prefill tokens {start}:{stop}, layer {i + 1}/{self.cfg.n_layers}"
                )
                pre = self._prefill_pre(i, bucket, caches is not None)
                old = None if caches is None else caches[i]
                streams, h, post, comb, cache, idx, weights = pre(
                    self.params["layers"][i],
                    streams,
                    jnp.int32(start),
                    jnp.int32(real_len),
                    old,
                )
                next_caches.append(cache)

                if self.cfg.mlp_types[i] == "sparse":
                    valid_idx = np.asarray(idx, dtype=np.int32)[:real_len]
                    self._sample_router_ready(sample)
                    dev_packed, local_idx, ep, _ = self._gather(
                        i, valid_idx, phase="prefill", image_mask=image_mask
                    )
                    self._sample_packed_ready(sample, dev_packed, ep)
                    full_local = np.full((bucket, valid_idx.shape[1]), -1, np.int32)
                    full_local[:real_len] = local_idx
                    token_rows = self._routing_plan(full_local, ep, i)
                    post_prog = self._post_program(
                        i,
                        1,
                        bucket,
                        ep,
                        token_rows.shape[1],
                        direct=len(dev_packed) == 7,
                        packed_shapes=tuple(a.shape for a in dev_packed),
                    )
                    streams = post_prog(
                        self.params["layers"][i],
                        streams,
                        h,
                        post,
                        comb,
                        weights,
                        jnp.asarray(full_local),
                        jnp.asarray(token_rows),
                        jnp.asarray(expert_tile_capacities(token_rows, self.expert_tile)),
                        *dev_packed,
                    )
                    self._finish_sample(sample, streams)
                else:
                    streams = self._post_program(i, 1, bucket)(
                        self.params["layers"][i],
                        streams,
                        h,
                        post,
                        comb,
                        weights,
                        jnp.zeros((bucket, 1), jnp.int32),
                        jnp.zeros((1, 1), jnp.int32),
                        jnp.zeros((1,), jnp.int32),
                    )
            caches = next_caches
            logits = self._head_program(1, bucket)(
                self.params["norm"], self.params["lm_head"], streams, jnp.int32(real_len)
            )
        return logits, caches

    def _prefill_windows(self, tokens, override):
        """Run each layer over a window, keeping attention's original small chunks.

        Attention depends on this layer's input and its preceding cache, not
        this layer's MLP output. Finish its causal chunks first, then stage the
        union of routed experts once for the whole window. No routing choices
        are dropped or changed; only the execution order changes.
        """
        total = tokens.shape[1]
        caches = None
        logits = None
        for start in range(0, total, self.prefill_moe_window):
            stop = min(total, start + self.prefill_moe_window)
            length = stop - start
            bucket = next(n for n in self.PREFILL_WINDOWS if n >= length)
            padded = np.zeros((1, bucket), np.int32)
            padded[:, :length] = tokens[:, start:stop]
            vec, mask = self._prepare_override(override, start, length, bucket)
            streams = self._embed_program(1, bucket)(
                self.params["embed"],
                jnp.asarray(padded),
                jnp.asarray(vec, self.lcfg.dtype),
                jnp.asarray(mask),
            )
            image_mask = (
                np.isin(np.arange(start, stop), override[0])
                if override is not None
                else np.zeros(length, np.bool_)
            )
            next_caches = []
            for i in range(self.cfg.n_layers):
                sample = self._start_sample("prefill", i, bucket)
                progress.update(
                    detail=f"prefill window {start}:{stop}, layer {i + 1}/{self.cfg.n_layers}"
                )
                old = None if caches is None else caches[i]
                pieces = [[] for _ in range(6)]
                for offset in range(0, length, self.prefill_chunk):
                    real_len = min(self.prefill_chunk, length - offset)
                    attention_bucket = self._bucket(real_len)
                    chunk = streams[:, offset : offset + real_len]
                    chunk = jnp.pad(
                        chunk,
                        ((0, 0), (0, attention_bucket - real_len), (0, 0), (0, 0)),
                    )
                    pre = self._prefill_pre(i, attention_bucket, old is not None)
                    residual, h, post, comb, old, idx, weights = pre(
                        self.params["layers"][i],
                        chunk,
                        jnp.int32(start + offset),
                        jnp.int32(real_len),
                        old,
                    )
                    for values, array in zip(pieces[:4], (residual, h, post, comb), strict=True):
                        values.append(array[:, :real_len])
                    pieces[4].append(idx[:real_len])
                    pieces[5].append(weights[:real_len])
                next_caches.append(old)
                joined = []
                for number, values in enumerate(pieces):
                    axis = 1 if number < 4 else 0
                    array = jnp.concatenate(values, axis=axis)
                    padding = [(0, 0)] * array.ndim
                    padding[axis] = (0, bucket - length)
                    joined.append(jnp.pad(array, padding))
                residual, h, post, comb, idx, weights = joined
                if self.cfg.mlp_types[i] == "sparse":
                    valid_idx = np.asarray(idx, np.int32)[:length]
                    self._sample_router_ready(sample)
                    packed, local, ep, _ = self._gather(
                        i, valid_idx, phase="prefill", image_mask=image_mask
                    )
                    self._sample_packed_ready(sample, packed, ep)
                    full_local = np.full((bucket, valid_idx.shape[1]), -1, np.int32)
                    full_local[:length] = local
                    rows = self._routing_plan(full_local, ep, i)
                    streams = self._post_program(
                        i,
                        1,
                        bucket,
                        ep,
                        rows.shape[1],
                        direct=len(packed) == 7,
                        packed_shapes=tuple(a.shape for a in packed),
                    )(
                        self.params["layers"][i],
                        residual,
                        h,
                        post,
                        comb,
                        weights,
                        jnp.asarray(full_local),
                        jnp.asarray(rows),
                        jnp.asarray(expert_tile_capacities(rows, self.expert_tile)),
                        *packed,
                    )
                    self._finish_sample(sample, streams)
                else:
                    streams = self._post_program(i, 1, bucket)(
                        self.params["layers"][i],
                        residual,
                        h,
                        post,
                        comb,
                        weights,
                        jnp.zeros((bucket, 1), jnp.int32),
                        jnp.zeros((1, 1), jnp.int32),
                        jnp.zeros((1,), jnp.int32),
                    )
            caches = next_caches
            logits = self._head_program(1, bucket)(
                self.params["norm"], self.params["lm_head"], streams, jnp.int32(length)
            )
        return logits, caches

    def decode_rows(self, tokens, caches, positions):
        """Decode independent streams together at their individual positions.

        ``tokens`` is ``[B]``; ``caches[b]`` is the layer-major cache list
        returned by ``prefill`` for stream ``b``; ``positions`` is ``[B]``.
        Returns ``(logits [B,V], new_caches[B][layers])``.
        """
        tokens = np.asarray(tokens, dtype=np.int32).reshape(-1)
        B = len(tokens)
        if B <= 0 or len(caches) != B:
            raise ValueError(
                "tokens, cache sets, and positions must have the same nonzero batch size"
            )
        if B not in (1, 2, 4, 8):
            raise ValueError("decode batch size must be 1, 2, 4, or 8")
        positions = np.asarray(positions, dtype=np.int64).reshape(-1)
        if len(positions) != B:
            raise ValueError(
                "tokens, cache sets, and positions must have the same nonzero batch size"
            )
        if np.any(positions < 0) or np.any(positions >= self.max_len):
            raise ValueError(f"decode positions must be in [0, {self.max_len})")
        if any(len(cache_set) != self.cfg.n_layers for cache_set in caches):
            raise ValueError("every stream must have one cache per model layer")

        pos = jax.device_put(
            positions.astype(np.int32),
            NamedSharding(self.mesh, P()),
        )
        stream = self._embed_program(B, 1)(
            self.params["embed"],
            jnp.asarray(tokens.reshape(B, 1)),
            jnp.zeros((B, 1, self.cfg.hidden), self.lcfg.dtype),
            jnp.zeros((B, 1), jnp.bool_),
        )
        new_cache_sets = [[] for _ in range(B)]

        for i in range(self.cfg.n_layers):
            sample = self._start_sample("decode", i, B)
            progress.update(detail=f"decode batch {B}, layer {i + 1}/{self.cfg.n_layers}")
            row_caches = [caches[b][i] for b in range(B)]
            pre = self._decode_pre(i, B)
            stream, h, post, comb, new_caches, idx, weights = pre(
                self.params["layers"][i], stream, pos, row_caches
            )
            for b in range(B):
                new_cache_sets[b].append(new_caches[b])

            if self.cfg.mlp_types[i] == "sparse":
                valid_idx = np.asarray(idx, dtype=np.int32)
                self._sample_router_ready(sample)
                dev_packed, local_idx, ep, _ = self._gather(i, valid_idx, phase="decode")
                self._sample_packed_ready(sample, dev_packed, ep)
                token_rows = self._routing_plan(local_idx, ep, i)
                stream = self._post_program(
                    i,
                    B,
                    1,
                    ep,
                    token_rows.shape[1],
                    direct=len(dev_packed) == 7,
                    packed_shapes=tuple(a.shape for a in dev_packed),
                )(
                    self.params["layers"][i],
                    stream,
                    h,
                    post,
                    comb,
                    weights,
                    jnp.asarray(local_idx),
                    jnp.asarray(token_rows),
                    jnp.asarray(expert_tile_capacities(token_rows, self.expert_tile)),
                    *dev_packed,
                )
                self._finish_sample(sample, stream)
            else:
                stream = self._post_program(i, B, 1)(
                    self.params["layers"][i],
                    stream,
                    h,
                    post,
                    comb,
                    weights,
                    jnp.zeros((B, 1), jnp.int32),
                    jnp.zeros((1, 1), jnp.int32),
                    jnp.zeros((1,), jnp.int32),
                )

        logits = self._head_program(B, 1)(
            self.params["norm"], self.params["lm_head"], stream, jnp.int32(1)
        )
        return logits, new_cache_sets

    def decode(self, token, caches, pos: int):
        """Single-stream convenience wrapper matching ``prefill``'s cache format."""
        logits, new_sets = self.decode_rows(np.asarray(token).reshape(1), [caches], [pos])
        return logits, new_sets[0], int(pos) + 1


__all__ = ["PackedOffloadEngine"]
