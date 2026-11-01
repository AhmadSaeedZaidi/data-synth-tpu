"""Layer-wise compiled, fully resident main model with explicit batched state."""

import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from .checkpoint import load_layer
from .config import ModelConfig
from .experts import expert_parallel
from .math import (
    choose,
    gdn_chunk,
    gdn_step,
    linear,
    ple_step,
    qsa_step,
    residual_read,
    residual_write,
    routes_and_shared,
)
from .ngram import HostNGramTable
from .parallel import expert_mesh, ordered_devices


class Engine:
    def __init__(
        self,
        config,
        mesh,
        layers,
        experts,
        expert_functions,
        final_weights,
        head,
        log=print,
        *,
        fuse_layers=True,
        gdn_prefill="serial",
        prefill_expert_functions=None,
    ):
        if gdn_prefill not in {"serial", "chunked"}:
            raise ValueError("GDN prefill must be serial or chunked")
        self.config, self.mesh = config, mesh
        self.layers, self.experts, self.expert_functions = layers, experts, expert_functions
        self.prefill_expert_functions = (
            expert_functions if prefill_expert_functions is None else prefill_expert_functions
        )
        if len(self.prefill_expert_functions) != len(expert_functions):
            raise ValueError("Every layer needs a prefill expert function")
        self.final_weights, self.head, self.log = final_weights, head, log
        self.dtype = head.dtype
        self.fuse_layers = fuse_layers
        self.gdn_prefill = gdn_prefill
        self.replica = NamedSharding(mesh, P())
        c = config
        # Caches are mutable inference state. Pin their output layout so an
        # update cannot silently replicate them, and donate the retired input
        # buffers to avoid allocating another full cache at each layer/step.
        gdn_layout = (self.replica, NamedSharding(mesh, P(None, "expert", None, None)))
        kv_layout = NamedSharding(
            mesh,
            P(None, "expert", None) if c.kv_cache_dtype == "q8" else P(None, "expert", None, None),
        )
        qsa_layout = dict(k=kv_layout, v=kv_layout)
        if c.kv_cache_dtype == "q8":
            qsa_layout.update(
                k_scale=NamedSharding(mesh, P(None, "expert")),
                v_scale=NamedSharding(mesh, P(None, "expert")),
            )
        if c.sparse_qsa:
            qsa_layout.update(
                index_keys=NamedSharding(mesh, P(None, "expert", None)),
                index_tail=self.replica,
                index_start=self.replica,
            )
        if not c.sparse_qsa and c.kv_cache_dtype != "q8":
            qsa_layout = (kv_layout, kv_layout)
        prepare_layouts = (self.replica,) * 6

        def prepare(x, w, cache, lengths, pos, active, *, qsa):
            mixed, inject = residual_read(x, w["attn_gr"], c)
            if qsa:
                y, cache = qsa_step(mixed, w["attn"], cache, lengths, pos, active, c)
            else:
                y, cache = gdn_step(mixed, w["attn"], cache, active, c)
            x = residual_write(x, y, inject, c)
            mixed, inject = residual_read(x, w["ffn_gr"], c)
            ids, weights, shared = routes_and_shared(mixed, w["mlp"], active, c)
            return x, mixed, inject, ids, weights, shared, cache

        self.gdn = jax.jit(
            lambda *args: prepare(*args, qsa=False),
            out_shardings=(*prepare_layouts, gdn_layout),
            donate_argnums=(2,),
        )
        self.qsa = jax.jit(
            lambda *args: prepare(*args, qsa=True),
            out_shardings=(*prepare_layouts, qsa_layout),
            donate_argnums=(2,),
        )
        self.ple = jax.jit(
            lambda x, features, w, history, active: ple_step(x, features, w, history, active, c),
            out_shardings=(self.replica, self.replica),
            donate_argnums=(3,),
        )
        self.write = jax.jit(lambda x, y, inject: residual_write(x, y, inject, c))
        self.final = jax.jit(
            lambda x, w, head: linear(residual_read(x, w, c)[0], head),
            out_shardings=NamedSharding(mesh, P(None, "expert")),
        )

        def chunk_prepare(x, w, cache, lengths, pos, active, *, qsa):
            batch, steps, width = x.shape
            flat = x.reshape(batch * steps, width)
            mixed, inject = residual_read(flat, w["attn_gr"], c)
            mixed = mixed.reshape(batch, steps, c.hidden)
            if qsa:
                indices = lengths[:, None] + jnp.cumsum(active, axis=1) - 1

                def attend(state, inputs):
                    token, index, position, mask = inputs
                    y, state = qsa_step(token, w["attn"], state, index, position, mask, c)
                    return state, y

                cache, attended = jax.lax.scan(
                    attend,
                    cache,
                    (mixed.swapaxes(0, 1), indices.T, pos.swapaxes(0, 1), active.T),
                )
                attended = attended.swapaxes(0, 1).reshape(batch * steps, c.hidden)
            elif gdn_prefill == "chunked":
                attended, cache = gdn_chunk(mixed, w["attn"], cache, active, c)
                attended = attended.reshape(batch * steps, c.hidden)
            else:
                # Keep the established recurrence available independently of
                # layer fusion and expert grouping. Only MoE rows are batched.
                def attend(state, inputs):
                    token, mask = inputs
                    y, state = gdn_step(token, w["attn"], state, mask, c)
                    return state, y

                cache, attended = jax.lax.scan(attend, cache, (mixed.swapaxes(0, 1), active.T))
                attended = attended.swapaxes(0, 1).reshape(batch * steps, c.hidden)
            flat = residual_write(flat, attended, inject, c)
            mixed, inject = residual_read(flat, w["ffn_gr"], c)
            ids, weights, shared = routes_and_shared(mixed, w["mlp"], active.flatten(), c)
            return flat, mixed, inject, ids, weights, shared, cache

        def chunk_ple(x, features, w, history, active):
            def step(state, inputs):
                token, feature, mask = inputs
                y, state = ple_step(token, feature, w, state, mask, c)
                return state, y

            history, x = jax.lax.scan(
                step, history, (x.swapaxes(0, 1), features.swapaxes(0, 1), active.T)
            )
            return x.swapaxes(0, 1), history

        self.chunk_gdn = jax.jit(
            lambda *args: chunk_prepare(*args, qsa=False),
            out_shardings=(*prepare_layouts, gdn_layout),
            donate_argnums=(2,),
        )
        self.chunk_qsa = jax.jit(
            lambda *args: chunk_prepare(*args, qsa=True),
            out_shardings=(*prepare_layouts, qsa_layout),
            donate_argnums=(2,),
        )
        self.chunk_ple = jax.jit(
            chunk_ple,
            out_shardings=(self.replica, self.replica),
            donate_argnums=(3,),
        )
        # One dispatch per layer, instead of prepare -> routed FFN -> residual.
        # Parameters stay dynamic inputs, so equal layer shapes share executables.
        fused = {}

        def combined(qsa, compute, chunk):
            key = qsa, compute, chunk
            if key in fused:
                return fused[key]
            before = (
                (self.chunk_qsa if qsa else self.chunk_gdn)
                if chunk
                else (self.qsa if qsa else self.gdn)
            )

            def layer(x, w, cache, lengths, pos, active, packed):
                with jax.named_scope("attention_and_routing"):
                    flat, mixed, inject, ids, weights, shared, cache = before(
                        x, w, cache, lengths, pos, active
                    )
                with jax.named_scope("routed_experts"):
                    routed = compute(
                        packed["gate"], packed["up"], packed["down"], mixed, ids, weights
                    )
                with jax.named_scope("residual_write"):
                    y = residual_write(flat, shared + routed, inject, c)
                return y.reshape(x.shape), cache

            fused[key] = jax.jit(
                layer,
                out_shardings=(self.replica, qsa_layout if qsa else gdn_layout),
                donate_argnums=(2,),
            )
            return fused[key]

        self.decode_layers = [
            combined(c.is_qsa(i), fn, False) for i, fn in enumerate(expert_functions)
        ]
        self.prefill_layers = [
            combined(c.is_qsa(i), fn, True) for i, fn in enumerate(self.prefill_expert_functions)
        ]
        self.state = None

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint,
        *,
        batch=8,
        config=ModelConfig(),
        reserve_gib=8,
        log=print,
        expert_compute="bf16",
        expert_group_size=1,
        expert_tensor_parallel=1,
        fuse_layers=True,
        gdn_prefill="serial",
        prefill_expert_group_size=None,
    ):
        devices = jax.devices()
        if len(devices) != 8 or any(device.platform != "tpu" for device in devices):
            raise RuntimeError("Full checkpoint trial requires exactly eight TPU devices")
        if any(d.device_kind.lower() not in {"tpu v5 lite", "tpu v5e"} for d in devices):
            raise RuntimeError("This memory and batching trial requires Kaggle v5e-8")
        mesh = Mesh(ordered_devices(devices), ("expert",))
        moe_mesh = expert_mesh(mesh, expert_tensor_parallel)
        capacities = [device.memory_stats()["bytes_limit"] for device in devices]
        if len(set(capacities)) != 1:
            raise RuntimeError("TPU chip capacities differ")
        plan = checkpoint.memory_plan(
            config, batch, hbm_bytes=sum(capacities), reserve_bytes=int(reserve_gib * 2**30)
        )
        log({"phase": "memory_plan", **plan})
        if not plan["fits"]:
            raise MemoryError(
                "Resident checkpoint exceeds HBM budget; select Q6/Q4 or add a spill cache"
            )
        layers, experts, functions, prefill_functions = [], [], [], []
        prefill_group = (
            expert_group_size if prefill_expert_group_size is None else prefill_expert_group_size
        )
        for i in range(config.layers):
            started = time.monotonic()
            layers.append(load_layer(checkpoint, i, config, mesh))
            packed = checkpoint.resident_experts(i, moe_mesh)
            experts.append(packed)
            for target, group in (
                (functions, expert_group_size),
                (prefill_functions, prefill_group),
            ):
                target.append(
                    expert_parallel(
                        mesh,
                        gate_type=checkpoint.tensors[f"blk.{i}.ffn_gate_exps.weight"]["type"],
                        up_type=checkpoint.tensors[f"blk.{i}.ffn_up_exps.weight"]["type"],
                        down_type=checkpoint.tensors[f"blk.{i}.ffn_down_exps.weight"]["type"],
                        hidden=config.hidden,
                        intermediate=config.intermediate,
                        compute=expert_compute,
                        group_size=group,
                        tensor_parallel=expert_tensor_parallel,
                    )
                )
            log({"phase": "layer_loaded", "layer": i, "elapsed_s": time.monotonic() - started})
            log(
                {
                    "phase": "layer_loaded",
                    "layer": i,
                    "hbm_bytes_in_use": sum(
                        d.memory_stats().get("bytes_in_use", 0) for d in devices
                    ),
                }
            )
        replica = NamedSharding(mesh, P())
        final = {}
        for key in ("norm", "down", "up"):
            matrix = checkpoint.dense(f"output_hc_{key}.weight")
            layout = NamedSharding(mesh, P("expert", None)) if matrix.ndim == 2 else replica
            final[key] = jax.device_put(matrix, layout).astype(
                jnp.float32 if key == "norm" else jnp.bfloat16
            )
        head = jax.device_put(
            checkpoint.dense("output.weight"), NamedSharding(mesh, P("expert", None))
        ).astype(jnp.bfloat16)
        return cls(
            config,
            mesh,
            layers,
            experts,
            functions,
            final,
            head,
            log,
            fuse_layers=fuse_layers,
            gdn_prefill=gdn_prefill,
            prefill_expert_functions=prefill_functions,
        )

    def release_state(self):
        """Retire completed caches before allocating a different batch shape."""
        import gc

        if self.state is not None:
            jax.block_until_ready(self.state)
        self.state = None
        self.lengths = None
        gc.collect()

    def reset(self, batch):
        if type(batch) is not int or batch < 1:
            raise ValueError("State batch size must be a positive integer")
        self.release_state()
        c = self.config

        def zeros(shape, dtype, spec):
            # Allocate from small per-device buffers, not a giant replicated
            # host state followed by resharding.
            sharding = NamedSharding(self.mesh, spec)
            return jax.make_array_from_callback(
                shape,
                sharding,
                lambda index: np.zeros(
                    tuple(
                        len(range(*part.indices(size))) if isinstance(part, slice) else 1
                        for part, size in zip(index, shape, strict=True)
                    ),
                    dtype=np.dtype(dtype),
                ),
            ).astype(dtype)

        caches = []
        for layer in range(c.layers):
            if c.is_qsa(layer):
                shape = (batch, c.max_context, c.kv_heads, c.head_dim)
                # Time sharding keeps the full 256-wide head contiguous;
                # splitting it to 32 columns incurs TPU vector tile padding.
                spec = P(None, "expert", None, None)
                dtype = jnp.int8 if c.kv_cache_dtype == "q8" else self.dtype
                if c.kv_cache_dtype == "q8":
                    shape = (batch, c.max_context, c.kv_heads * c.head_dim)
                    spec = P(None, "expert", None)
                cache = dict(k=zeros(shape, dtype, spec), v=zeros(shape, dtype, spec))
                if c.kv_cache_dtype == "q8":
                    # Store heads inside the wide last axis. [B,T,2,256]
                    # exposes a tiny penultimate axis to TPU tile padding.
                    # [B,T,512] keeps both tiled axes usefully populated.
                    scale_shape = (
                        batch * c.kv_heads * (c.head_dim // c.cache_group),
                        c.max_context,
                    )
                    for name in ("k_scale", "v_scale"):
                        cache[name] = zeros(scale_shape, jnp.bfloat16, P(None, "expert"))
                if c.sparse_qsa:
                    blocks = (c.max_context + c.indexer_compress - 1) // c.indexer_compress
                    cache.update(
                        index_keys=zeros(
                            (batch, blocks, c.indexer_dim), self.dtype, P(None, "expert", None)
                        ),
                        index_tail=zeros(
                            (batch, c.indexer_compress, c.indexer_dim), self.dtype, P()
                        ),
                        index_start=zeros((batch, 3), jnp.int32, P()),
                    )
                caches.append(
                    cache if c.sparse_qsa or c.kv_cache_dtype == "q8" else (cache["k"], cache["v"])
                )
            else:
                conv = zeros((batch, c.conv_kernel - 1, c.conv_channels), self.dtype, P())
                recurrent = zeros(
                    (batch, c.value_heads, c.state_dim, c.state_dim),
                    jnp.float32,
                    P(None, "expert", None, None),
                )
                caches.append((conv, recurrent))
        self.state = dict(
            layers=caches,
            ple=zeros(
                (batch, (c.conv_kernel - 1) * c.ngram_size, c.streams * c.hidden), self.dtype, P()
            ),
            hidden=zeros((batch, c.streams * c.hidden), self.dtype, P()),
        )
        self.lengths = np.zeros(batch, np.int32)

    def forward(self, embedding, features, positions, active):
        if self.state is None:
            raise RuntimeError("reset(batch) is required")
        c = self.config
        active = np.asarray(active, bool)
        if active.shape != self.lengths.shape:
            raise ValueError("Active mask differs from allocated state batch")
        if np.shape(positions) != (len(active), 3):
            raise ValueError("Positions require [batch, 3]")
        if np.any(self.lengths[active] >= c.max_context):
            raise ValueError("Configured context capacity exhausted")
        if np.shape(embedding) != (len(active), c.hidden) or np.shape(features) != (
            len(active),
            c.ple_dim,
        ):
            raise ValueError("Input batch/dimension differs from model")
        old_hidden = self.state["hidden"]
        x = jax.device_put(embedding, self.replica).astype(self.dtype)
        x = jnp.tile(x[:, None, :], (1, c.streams, 1)).reshape(len(active), -1)
        feature = jax.device_put(features, self.replica).astype(self.dtype)
        pos = jax.device_put(np.asarray(positions, np.int32), self.replica)
        mask = jax.device_put(active, self.replica)
        lengths = jax.device_put(self.lengths, self.replica)
        for i, (w, packed, compute) in enumerate(
            zip(self.layers, self.experts, self.expert_functions, strict=True)
        ):
            if i == c.ple_layer:
                x, self.state["ple"] = self.ple(x, feature, w["ple"], self.state["ple"], mask)
            if self.fuse_layers:
                x, self.state["layers"][i] = self.decode_layers[i](
                    x, w, self.state["layers"][i], lengths, pos, mask, packed
                )
                continue
            function = self.qsa if c.is_qsa(i) else self.gdn
            x, mixed, inject, ids, weights, shared, cache = function(
                x, w, self.state["layers"][i], lengths, pos, mask
            )
            routed = compute(packed["gate"], packed["up"], packed["down"], mixed, ids, weights)
            x = self.write(x, shared + routed, inject)
            self.state["layers"][i] = cache
        self.state["hidden"] = choose(mask, x, old_hidden)
        # device_put may borrow the NumPy buffer until dispatched work completes.
        # Replacing it preserves the positions used by asynchronous attention.
        self.lengths = self.lengths + active.astype(np.int32)

    def logits(self):
        return self.final(self.state["hidden"], self.final_weights, self.head)

    def forward_chunk(self, embedding, features, positions, active):
        """Layer-wise prompt chunking reuses each expert across B * T requests."""
        if self.state is None:
            raise RuntimeError("reset(batch) is required")
        c = self.config
        active = np.asarray(active, bool)
        if active.ndim != 2:
            raise ValueError("Chunk mask requires [batch, time]")
        batch, steps = active.shape
        if batch != len(self.lengths) or steps < 1:
            raise ValueError("Chunk batch differs from state or has no tokens")
        if np.shape(positions) != (batch, steps, 3):
            raise ValueError("Chunk positions require [batch, time, 3]")
        counts = active.sum(axis=1).astype(np.int32)
        if np.any(self.lengths + counts > c.max_context):
            raise ValueError("Chunk exceeds configured context capacity")
        if np.shape(embedding) != (batch, steps, c.hidden) or np.shape(features) != (
            batch,
            steps,
            c.ple_dim,
        ):
            raise ValueError("Chunk input dimensions differ")
        # Prompt chunks use right padding; gaps would not correspond to the
        # last-valid-row selection and are rejected rather than silently used.
        if np.any(active[:, 1:] & ~active[:, :-1]):
            raise ValueError("Chunk mask must use right padding")
        x = jax.device_put(embedding, self.replica).astype(self.dtype)
        x = jnp.tile(x[:, :, None, :], (1, 1, c.streams, 1)).reshape(batch, steps, -1)
        feature = jax.device_put(features, self.replica).astype(self.dtype)
        pos = jax.device_put(np.asarray(positions, np.int32), self.replica)
        mask = jax.device_put(active, self.replica)
        lengths = jax.device_put(self.lengths, self.replica)
        for i, (w, packed, compute) in enumerate(
            zip(self.layers, self.experts, self.prefill_expert_functions, strict=True)
        ):
            if i == c.ple_layer:
                x, self.state["ple"] = self.chunk_ple(x, feature, w["ple"], self.state["ple"], mask)
            if self.fuse_layers:
                x, self.state["layers"][i] = self.prefill_layers[i](
                    x, w, self.state["layers"][i], lengths, pos, mask, packed
                )
                continue
            function = self.chunk_qsa if c.is_qsa(i) else self.chunk_gdn
            flat, mixed, inject, ids, weights, shared, cache = function(
                x, w, self.state["layers"][i], lengths, pos, mask
            )
            routed = compute(packed["gate"], packed["up"], packed["down"], mixed, ids, weights)
            x = self.write(flat, shared + routed, inject).reshape(batch, steps, -1)
            self.state["layers"][i] = cache
        last = x[jnp.arange(batch), jnp.maximum(counts - 1, 0)]
        self.state["hidden"] = choose(jnp.asarray(counts > 0), last, self.state["hidden"])
        self.lengths = self.lengths + counts


def token_table(checkpoint, c):
    name = "token_embd.weight"
    return HostNGramTable(checkpoint.packed(name), checkpoint.tensors[name]["type"], c.hidden)
