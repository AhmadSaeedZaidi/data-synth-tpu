"""Notebook worker: preflight, immutable weight loading, vision and batch gates."""

from __future__ import annotations

import argparse
import dataclasses
import gc
import importlib.metadata
import io
import json
import re
import shutil
import time
import traceback
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from . import probe_images, progress, quality, tracing
from .diagnostics import WorkerDiagnostics, optional_device_trace
from .gguf_quant import dequantize_rows
from .weights import (
    BASE_REPO,
    BASE_REVISION,
    EXPERT_REPO,
    EXPERT_REVISION,
    BF16Reader,
    PackedExperts,
    available_ram,
    fetch_json,
    hf_url,
    open_gguf,
)


def write_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, default=str) + "\n")
    temporary.replace(path)


def log(message):
    print(time.strftime("%H:%M:%S"), message, flush=True)


@tracing.operation("decoder_gate")
def decoder_gate():
    """JIT against frozen gguf-py outputs, before any large allocation."""
    import jax

    errors = {}
    fixture = Path(__file__).parent / "data" / "decoder_reference.npz"
    with np.load(fixture, allow_pickle=False) as reference:
        for key in reference.files:
            if not key.endswith("_raw"):
                continue
            qtype = key.removesuffix("_raw")
            expected = reference[qtype + "_expected"]
            width = expected.shape[-1]
            decoded = jax.jit(lambda raw: dequantize_rows(raw, qtype, width, backend="jax"))(
                reference[key]
            )
            got = np.asarray(decoded)
            np.testing.assert_allclose(got, expected, rtol=2e-6, atol=2e-6)
            if not np.isfinite(got).all():
                raise ValueError("non-finite decoded reference weights")
            errors[qtype] = float(np.max(np.abs(got - expected)))
    return {"decoder": "PASS", "max_errors": errors, "reference": "gguf-py frozen outputs"}


@tracing.operation("vision_gate")
def vision_gate():
    """Check the sharded vision tower against a frozen independent HF model."""
    import jax
    import jax.numpy as jnp
    from glm53 import vision
    from jax.sharding import Mesh

    fixture = Path(__file__).parent / "data/vision_reference.npz"
    mesh = Mesh(np.array(jax.devices()), ("x",))
    with np.load(fixture, allow_pickle=False) as reference:
        params = vision.from_state_dict(lambda k: reference["weight/" + k], 2, 32)
        params["n_heads"] = 8
        placed = vision.to_sharded(params, mesh, dtype=jnp.float32)
        forward = vision.make_forward_sharded(placed, mesh, dtype=jnp.float32)
        grids = tuple(map(tuple, reference["grids"].tolist()))
        got = np.asarray(forward(reference["patches"], grids))
        expected = reference["expected"]
        np.testing.assert_allclose(got, expected, rtol=2e-4, atol=2e-6)
        error = float(np.max(np.abs(got - expected)))
    return {"vision": "PASS", "vision_max_error": error, "vision_reference": "Transformers 5.18.0"}


@tracing.operation("expert_cache_gate")
def expert_cache_gate():
    """Check the new cache view on the actual backend before checkpoint loading."""
    from types import SimpleNamespace

    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    from .expert_cache import ExpertCache
    from .expert_kernel import tiled_experts
    from .routing import expert_tile_capacities, expert_token_rows

    mesh = Mesh(np.array(jax.devices()), ("x",))
    sharding = NamedSharding(mesh, P("x"))
    cfg = SimpleNamespace(hidden=256, moe_inter=256, dtype=jnp.bfloat16, swiglu_limit=7.0)
    qtypes = ("Q4_K", "Q5_K", "Q6_K")
    with np.load(
        Path(__file__).parent / "data/decoder_reference.npz", allow_pickle=False
    ) as reference:
        tables = tuple(np.tile(reference[q + "_raw"], (4, 86, 1, 1))[:, :256, :1] for q in qtypes)
    tables = tuple(np.broadcast_to(a[None], (mesh.size, *a.shape)).copy() for a in tables)
    store = PackedExperts({0: tables}, {0: qtypes}, mesh.size)

    def transport(arrays):
        return tuple(jax.device_put(a.reshape(*a.shape[:3], -1), sharding) for a in arrays)

    per_expert = sum(a[:, :1].nbytes for a in tables)
    outputs = []
    for direct in (False, True):
        cache = ExpertCache(store, mesh, transport, 3 * per_expert)
        values = []
        for batch, offset in ((1, 0), (4, 0), (1, 6), (8, 0), (8, 0)):
            routes = (np.arange(batch * 2, dtype=np.int32).reshape(batch, 2) + offset) % 8
            packed, ids, ep = cache.gather(0, routes, direct=direct)
            rows = expert_token_rows(ids, ep)
            capacities = expert_tile_capacities(rows, 4)
            x = np.random.default_rng(354).normal(size=(batch, 256)).astype(np.float32)
            weights = np.full(ids.shape, 0.5, np.float32)

            def local(x, ids, weights, rows, capacities, *packed):
                y = tiled_experts(x, packed, ids, weights, cfg, qtypes, 4, rows, capacities)
                return jax.lax.psum(y, "x")

            specs = (P("x"),) * (6 if direct else 3) + ((P(),) if direct else ())
            program = jax.jit(
                jax.shard_map(
                    local,
                    mesh=mesh,
                    in_specs=(P(),) * 5 + specs,
                    out_specs=P(),
                    check_vma=False,
                )
            )
            values.append(np.asarray(program(x, ids, weights, rows, capacities, *packed)))
        outputs.append(values)
    errors = []
    for previous, current in zip(*outputs, strict=True):
        if not np.isfinite(current).all():
            raise ValueError("non-finite direct cache numerical gate")
        np.testing.assert_allclose(current, previous, rtol=2e-5, atol=5e-3)
        errors.append(float(np.max(np.abs(current - previous))))
    return {"expert_cache": "PASS", "expert_cache_max_error": max(errors)}


def environment(work):
    import jax

    devices = jax.devices()
    packages = {}
    for name in ("jax", "jaxlib", "libtpu", "numpy", "transformers", "huggingface-hub"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    disk = shutil.disk_usage(work)
    stats = []
    for device in devices:
        memory = device.memory_stats() or {}
        stats.append(
            {
                "id": device.id,
                "platform": device.platform,
                "kind": device.device_kind,
                "bytes_limit": memory.get("bytes_limit"),
                "bytes_in_use": memory.get("bytes_in_use"),
            }
        )
    return {
        "packages": packages,
        "devices": stats,
        "available_ram_bytes": available_ram(),
        "working_disk_free_bytes": disk.free,
    }


def memory_snapshot():
    import jax

    text = Path("/proc/self/status").read_text()
    host = {}
    for name in ("VmRSS", "VmHWM"):
        found = re.search(rf"{name}:\s+(\d+)", text)
        host[name + "_bytes"] = int(found[1]) * 1024 if found else None
    devices = []
    for device in jax.devices():
        stats = device.memory_stats() or {}
        devices.append({"id": device.id, **stats})
    return {"host": host, "devices": devices}


def cfg_from_json(config):
    import jax.numpy as jnp
    from glm53.model import Cfg

    text = dict(config["text_config"])
    linear = text["linear_attn_config"]
    text.update(
        linear_num_heads=linear["num_heads"],
        linear_head_dim=linear["head_dim"],
        linear_conv_kernel_dim=linear["short_conv_kernel_size"],
        linear_lower_bound=linear["gate_lower_bound"],
    )
    return Cfg.from_hf(SimpleNamespace(**text), dtype=jnp.bfloat16)


def check_hbm_budget(engine, config):
    """Conservative logical-byte plan; actual libtpu peaks remain authoritative."""
    import jax

    cfg, n = engine.cfg, engine.n
    batch, context = max(config["batch_sizes"]), config["max_context"]
    linear = cfg.layer_types.count("linear_attention")
    sparse_attention = cfg.n_layers - linear
    # Recurrent state is FP32; conv/latent/indexer caches are BF16.
    recurrent = linear * (
        cfg.kda_heads // n * cfg.kda_hd**2 * 4
        + (cfg.conv_k - 1) * 3 * cfg.kda_heads // n * cfg.kda_hd * 2
    )
    attention = sparse_attention * (
        context * cfg.kv_lora * 2
        + ((context + cfg.idx_kpool - 1) // cfg.idx_kpool + 2 * cfg.idx_kpool) * cfg.idx_hd * 2
    )
    caches = batch * (recurrent + attention)
    resident = (
        0 if engine.expert_cache is None else engine.expert_cache.stats["reserved_bytes"] // n
    )
    requested_resident = resident
    safety = config.get("cache_safety_mib_per_device", 1024) * (1 << 20)
    # Include input and output cache sets, vision/staged experts, compiler and
    # transient space. This is a rejection guard, not an OOM guarantee.
    fixed_reserve = (3 << 30) + 2 * caches + safety
    snapshots = [(device, device.memory_stats() or {}) for device in jax.devices()]
    capacities = [
        stats["bytes_limit"] - stats["bytes_in_use"] - fixed_reserve
        for _, stats in snapshots
        if stats.get("bytes_limit") is not None and stats.get("bytes_in_use") is not None
    ]
    for device, stats in snapshots:
        if (
            stats.get("bytes_limit") is not None
            and stats.get("bytes_in_use") is not None
            and stats["bytes_in_use"] + fixed_reserve > stats["bytes_limit"]
        ):
            raise MemoryError(f"device {device.id}: base, batch state and workspace exceed HBM")
    if capacities and resident > min(capacities) and engine.expert_cache is not None:
        if engine.expert_cache.stats.get("allocated_bytes", 0):
            raise MemoryError("cannot resize an allocated expert cache")
        from .expert_cache import ExpertCache

        effective = max(0, min(capacities) // (1 << 20)) * (1 << 20) * n
        engine.expert_cache = ExpertCache(engine.store, engine.mesh, engine.transport, effective)
        resident = engine.expert_cache.stats["reserved_bytes"] // n
        log(f"Expert cache reduced to {resident / 2**30:.2f} GiB/chip by the HBM guard")
    reserve = fixed_reserve + resident
    devices = []
    for device, stats in snapshots:
        limit, used = stats.get("bytes_limit"), stats.get("bytes_in_use")
        if limit is not None and used is not None and used + reserve > limit:
            raise MemoryError(
                f"device {device.id}: expert cache plus batch state and workspace reserve "
                "exceed HBM; reduce expert_cache_mib_per_device, batch or context"
            )
        devices.append({"id": device.id, "limit": limit, "used": used})
    return {
        "cache_bytes_per_device": resident,
        "requested_cache_bytes_per_device": requested_resident,
        "cache_safety_bytes_per_device": safety,
        "batch_state_bytes_per_device": caches,
        "workspace_reserve_bytes_per_device": 3 << 30,
        "devices": devices,
        "scope": "logical estimate with two state sets; validate native peak HBM on TPU",
    }


class VisionChat:
    """Native GLM template with ordered expanded image token embeddings."""

    def __init__(
        self,
        tokenizer,
        vision_forward,
        max_image_tokens=1024,
        thinking_budget=None,
        hidden_size=4096,
        reasoning_effort="max",
    ):
        self.tokenizer = tokenizer
        self.vision_forward = vision_forward
        self.max_image_tokens = max_image_tokens
        self.thinking_budget = thinking_budget
        self.reasoning_effort = reasoning_effort
        self.hidden_size = hidden_size
        self.image_cache = {}
        self.image_token = tokenizer.convert_tokens_to_ids("<|image|>")
        self.image_token_limits = {}
        if self.image_token != 154854:
            raise ValueError("unexpected GLM image token; tokenizer/model mismatch")

    @tracing.operation("vision_encode")
    def image(self, blob):
        import hashlib

        from glm53 import vision as vis
        from PIL import Image

        digest = hashlib.sha256(blob).hexdigest()
        limit = self.image_token_limits.get(digest, self.max_image_tokens)
        key = (digest, limit)
        if key not in self.image_cache:
            with Image.open(io.BytesIO(blob)) as image:
                patches, grid = vis.preprocess(image, max_tokens=limit)
            count = len(patches)
            bucket = max(256, 1 << (count - 1).bit_length())
            grids = (grid,) if count == bucket else (grid, (1, 2, (bucket - count) // 2))
            if bucket > count:
                patches = np.concatenate(
                    (patches, np.zeros((bucket - count, patches.shape[1]), patches.dtype))
                )
            vectors = np.asarray(self.vision_forward(patches, grids), np.float32)
            if vectors.shape != (bucket // 4, self.hidden_size):
                raise ValueError(
                    f"vision encoder shape {vectors.shape}; expected "
                    f"{(bucket // 4, self.hidden_size)}"
                )
            vectors = vectors[: vis.n_tokens(grid)]
            if not np.isfinite(vectors).all():
                raise ValueError("vision encoder produced non-finite embeddings")
            self.image_cache[key] = vectors
        return self.image_cache[key]

    def render(self, text, images=()):
        content = [{"type": "text", "text": text}] + [{"type": "image"} for _ in images]
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": content if images else text}],
            tokenize=False,
            add_generation_prompt=True,
            reasoning_effort=self.reasoning_effort,
            clear_thinking=True,
        )
        if self.thinking_budget == 0:
            if not prompt.endswith("<think>"):
                raise ValueError("template does not expose the expected thinking prefix")
            prompt += "</think>"
        ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
        expanded, positions, embeddings = [], [], []
        image_number = 0
        for token in ids:
            if token != self.image_token:
                expanded.append(token)
                continue
            if image_number >= len(images):
                raise ValueError("more image markers than images")
            vectors = self.image(images[image_number])
            positions.extend(range(len(expanded), len(expanded) + len(vectors)))
            embeddings.append(vectors)
            expanded.extend([self.image_token] * len(vectors))
            image_number += 1
        if image_number != len(images):
            raise ValueError("image count differs from rendered template")
        override = None
        if positions:
            override = (np.asarray(positions, np.int32), np.concatenate(embeddings))
        return np.asarray(expanded, np.int32), override

    def answer(self, output):
        return self.generation_details(output)["answer"]

    def generation_details(self, output):
        """Separate native reasoning from answers, including interrupted reasoning."""
        text = self.tokenizer.decode(output, skip_special_tokens=False)
        native = self.thinking_budget is None
        complete = not native or "</think>" in text
        reasoning_tokens = len(output) if native and not complete else 0
        answer_tokens = len(output) if not native else 0
        if "</think>" in text:
            # Locate the token boundary even for tokenizers that split the marker.
            for boundary in range(1, len(output) + 1):
                prefix = self.tokenizer.decode(output[:boundary], skip_special_tokens=False)
                if "</think>" in prefix:
                    marker_tokens = 0
                    for start in range(boundary - 1, -1, -1):
                        marker = self.tokenizer.decode(
                            output[start:boundary], skip_special_tokens=False
                        )
                        if marker == "</think>":
                            marker_tokens = boundary - start
                            break
                    reasoning_tokens = boundary - marker_tokens
                    answer_tokens = len(output) - boundary
                    break
            text = text.rsplit("</think>", 1)[1]
        elif native or "<think>" in text:
            complete = False
            reasoning_tokens = len(output)
            text = ""
            answer_tokens = 0
        return {
            "answer": re.sub(r"<\|[^>]*\|>", "", text).strip(),
            "reasoning_complete": complete,
            "reasoning_tokens": reasoning_tokens,
            "answer_tokens": answer_tokens,
        }


@dataclasses.dataclass
class PreparedBatch:
    """Immutable prompt states reused for warmup and measured decoding.

    Cached image-check states live in host RAM between measurements. Sharding
    metadata restores the exact original layout without retaining TPU buffers.
    """

    logits: np.ndarray
    caches: list
    positions: np.ndarray
    metrics: dict
    shardings: object = None

    def to_host(self):
        import jax

        shardings = jax.tree.map(lambda a: getattr(a, "sharding", None), self.caches)
        caches = jax.tree.map(lambda a: np.array(a, copy=True), jax.device_get(self.caches))
        return dataclasses.replace(self, caches=caches, shardings=shardings)

    def to_device(self):
        import jax

        if self.shardings is None:
            return self
        caches = jax.tree.map(
            lambda a, s: jax.device_put(a, s),
            self.caches,
            self.shardings,
            is_leaf=lambda a: a is None,
        )
        jax.block_until_ready(caches)
        return dataclasses.replace(self, caches=caches, shardings=None)

    @classmethod
    def combine(cls, singles):
        if len(singles) not in (1, 4, 8) or any(len(s.positions) != 1 for s in singles):
            raise ValueError("combine requires 1, 4 or 8 single-stream prompt states")
        placed = [s.to_device() for s in singles]
        metrics = {
            key: sum(s.metrics[key] for s in placed)
            for key in (
                "prefill_seconds",
                "render_and_vision_seconds",
                "language_prefill_seconds",
                "prefill_bytes_transferred",
            )
        }
        metrics["prompt_tokens"] = [int(s.positions[0]) for s in placed]
        metrics["images"] = sum(s.metrics.get("images", 0) for s in placed)
        return cls(
            np.concatenate([s.logits for s in placed]),
            [s.caches[0] for s in placed],
            np.concatenate([s.positions for s in placed]),
            metrics,
        )


@tracing.operation("prepare_requests")
def prepare_requests(engine, chat, requests, max_new):
    if max_new < 1 or len(requests) not in (1, 4, 8):
        raise ValueError("generation requires positive max_new and 1, 4 or 8 requests")
    caches, positions, first_logits = [], [], []
    render_seconds = 0.0
    language_prefill_seconds = 0.0
    prompt_lengths = []
    before = engine.store.stats["bytes_transferred"]
    started = time.perf_counter()
    for text, images in requests:
        progress.update(detail=f"rendering request {len(caches) + 1}/{len(requests)}", force=True)
        render_started = time.perf_counter()
        tokens, override = chat.render(text, images)
        render_seconds += time.perf_counter() - render_started
        prompt_lengths.append(len(tokens))
        if len(tokens) + max_new > engine.max_len:
            raise ValueError("prompt plus generation exceeds MAX_CONTEXT")
        progress.update(
            detail=f"prefill request {len(caches) + 1}: {len(tokens)} tokens", force=True
        )
        language_started = time.perf_counter()
        with tracing.span("prefill", prompt_tokens=len(tokens), images=len(images)):
            logits, cache = engine.prefill(tokens, override=override)
            # Include completion, not just asynchronous JAX dispatch, in this span.
            logits = np.asarray(logits)
        language_prefill_seconds += time.perf_counter() - language_started
        first_logits.append(np.asarray(logits).reshape(-1, engine.cfg.vocab)[-1])
        caches.append(cache)
        positions.append(len(tokens))
    prefill_seconds = time.perf_counter() - started
    return PreparedBatch(
        np.stack(first_logits),
        caches,
        np.asarray(positions, np.int32),
        {
            "prefill_seconds": prefill_seconds,
            "render_and_vision_seconds": render_seconds,
            "language_prefill_seconds": language_prefill_seconds,
            "prompt_tokens": prompt_lengths,
            "prefill_bytes_transferred": engine.store.stats["bytes_transferred"] - before,
            "images": sum(len(images) for _, images in requests),
        },
    )


@tracing.operation("decode_prepared")
def decode_prepared(engine, chat, prepared, max_new, eos):
    eos = [int(eos)] if isinstance(eos, (int, np.integer)) else list(eos)
    rows = len(prepared.positions)
    if not eos or max_new < 1 or rows not in (1, 4, 8):
        raise ValueError("generation requires EOS ids, positive max_new and 1, 4 or 8 requests")
    prepared = prepared.to_device()
    # Model caches are JAX arrays; decode creates new arrays rather than
    # donating/mutating this snapshot. Each replay starts at its own positions.
    caches = [list(c) for c in prepared.caches]
    positions = prepared.positions.copy()
    logits = prepared.logits.copy()
    outputs = [[] for _ in range(rows)]
    after_prefill = dict(engine.store.stats)
    done = np.zeros(rows, bool)
    started = time.perf_counter()
    decode_calls = 0
    decode_generated_tokens = 0
    processed_decode_row_steps = 0
    active_decode_row_steps = 0
    decode_batch_size_counts = {}
    compact = getattr(engine, "compact_decode_batches", True)
    for step in range(max_new):
        if not np.isfinite(logits).all():
            raise ValueError("non-finite generation logits")
        next_tokens = logits.argmax(axis=-1).astype(np.int32)
        for row, token in enumerate(next_tokens):
            if done[row]:
                continue
            if int(token) in eos:
                done[row] = True
            else:
                outputs[row].append(int(token))
                if step > 0:
                    decode_generated_tokens += 1
        if done.all() or step + 1 == max_new:
            break
        active = np.flatnonzero(~done)
        if compact:
            # Padding repeats a live row: finished rows don't advance state or
            # introduce unrelated experts. Four widths bound program variants.
            width = next(b for b in (1, 2, 4, 8) if b >= len(active))
            selected = np.pad(active, (0, width - len(active)), constant_values=active[0])
        else:
            width, selected = rows, np.arange(rows)
            next_tokens = np.where(done, eos[0], next_tokens).astype(np.int32)
        progress.update(detail=f"decode batch {rows} step {step + 1}/{max_new}")
        with tracing.span("decode", batch_size=width, active_rows=len(active), step=step + 1):
            logits_device, updated = engine.decode_rows(
                next_tokens[selected], [caches[b] for b in selected], positions[selected]
            )
            decoded = np.asarray(logits_device).reshape(width, -1, engine.cfg.vocab)[:, -1]
            for local, row in enumerate(selected[: len(active)] if compact else selected):
                logits[row] = decoded[local]
                caches[row] = updated[local]
                positions[row] += 1
        decode_calls += 1
        processed_decode_row_steps += width
        active_decode_row_steps += len(active)
        decode_batch_size_counts[str(width)] = decode_batch_size_counts.get(str(width), 0) + 1
    decode_seconds = time.perf_counter() - started
    details = [
        chat.generation_details(tokens)
        if hasattr(chat, "generation_details")
        else {"answer": chat.answer(tokens)}
        for tokens in outputs
    ]
    return {
        "answers": [item["answer"] for item in details],
        "reasoning_complete": [item.get("reasoning_complete") for item in details],
        "reasoning_tokens": [item.get("reasoning_tokens") for item in details],
        "answer_tokens": [item.get("answer_tokens") for item in details],
        "generation_limit_reached": (~done).tolist(),
        "output_ids": outputs,
        "stopped": done.tolist(),
        "generated_tokens": sum(map(len, outputs)),
        "decode_generated_tokens": decode_generated_tokens,
        "decode_bytes_transferred": engine.store.stats["bytes_transferred"]
        - after_prefill["bytes_transferred"],
        **prepared.metrics,
        "decode_seconds": decode_seconds,
        "decode_calls": decode_calls,
        "processed_decode_row_steps": processed_decode_row_steps,
        "active_decode_row_steps": active_decode_row_steps,
        "decode_batch_size_counts": decode_batch_size_counts,
        "compact_decode_batches": compact,
    }


@tracing.operation("generate")
def generate(engine, chat, requests, max_new, eos, *, keep_prepared=False):
    prepared = prepare_requests(engine, chat, requests, max_new)
    result = decode_prepared(engine, chat, prepared, max_new, eos)
    if keep_prepared:
        result["_prepared"] = prepared
    return result


def early_decode_probe(engine, chat, config, eos, report, save, deadline):
    """Get actual batch decode timings before spending hours on image prefills."""
    report["stage"] = "EARLY_DECODE_PROBE"
    report["early_decode_probe"] = []
    save()
    limit = min(8, config["max_new_tokens"])
    state = prepare_requests(
        engine,
        chat,
        [("Explain in eight sentences how clouds form and why rainfall varies by season.", [])],
        limit,
    )
    reference = None
    for batch in sorted(set(config["batch_sizes"])):
        deadline()
        prepared = PreparedBatch.combine([state] * batch)
        decode_prepared(engine, chat, prepared, min(2, limit), eos)
        programs_before = engine.program_count
        result = decode_prepared(engine, chat, prepared, limit, eos)
        if reference is None:
            reference = result["output_ids"][0]
        seconds = result["decode_seconds"]
        report["early_decode_probe"].append(
            {
                "batch_size": batch,
                "status": "MEASURED",
                "decode_seconds": seconds,
                "decode_calls": result["decode_calls"],
                "decode_generated_tokens": result["decode_generated_tokens"],
                "reasoning_tokens": result["reasoning_tokens"],
                "answer_tokens": result["answer_tokens"],
                "reasoning_complete": result["reasoning_complete"],
                "generation_limit_reached": result["generation_limit_reached"],
                "tokens_per_second": result["decode_generated_tokens"] / seconds if seconds else 0,
                "decode_bytes_transferred": result["decode_bytes_transferred"],
                "new_programs": engine.program_count - programs_before,
                "same_output_as_first_row": all(ids == reference for ids in result["output_ids"]),
                "scope": (
                    "repeated identical text prompt; optimistic expert reuse; "
                    "all generated decode tokens including reasoning; not a vision quality gate"
                ),
            }
        )
        save()


def execution_scope(engine, purpose, **options):
    sampling = getattr(engine, "execution_sampling", None)
    return sampling(purpose, **options) if sampling is not None else nullcontext()


def batch_benchmarks(engine, chat, config, eos, report, bank, save, deadline):
    """Replay diverse image prompt states once for warmup and once for measurement."""
    report["stage"] = "BATCH_BENCHMARKS"
    report["benchmarks"] = []
    save()
    for batch_size in config["batch_sizes"]:
        deadline()
        cases = bank[:batch_size]
        restore_started = time.perf_counter()
        prepared = PreparedBatch.combine([entry["state"] for entry in cases])
        restore_seconds = time.perf_counter() - restore_started
        warm_started = time.perf_counter()
        warm_limit = (
            config["max_new_tokens"]
            if config.get("full_decode_warmup", True)
            else min(2, config["max_new_tokens"])
        )
        with execution_scope(engine, "quality_batch_warmup", enabled=False):
            warm = decode_prepared(engine, chat, prepared, warm_limit, eos)
        warm_seconds = time.perf_counter() - warm_started
        programs_before = engine.program_count
        cache_before = dict(engine.expert_cache.stats) if engine.expert_cache else None
        with execution_scope(engine, "quality_batch_measurement", enabled=False):
            measured = decode_prepared(engine, chat, prepared, config["max_new_tokens"], eos)
        checks = []
        for case, answer, stopped in zip(
            cases, measured["answers"], measured["stopped"], strict=True
        ):
            matched, detail = quality.match(case["case"]["match"], answer)
            checks.append(
                {
                    "name": case["case"]["name"],
                    "answer": answer,
                    "status": "PASS" if matched and stopped else "FAIL",
                    "match_detail": detail,
                    "same_answer_as_single": answer == case["single_answer"],
                }
            )
        duration = measured["decode_seconds"]
        tokens = measured["generated_tokens"]
        item = {
            "batch_size": batch_size,
            "warmup_seconds": warm_seconds,
            "warmup_decode_calls": warm["decode_calls"],
            "warmup_completed": all(warm["stopped"]),
            "prompt_state_restore_seconds": restore_seconds,
            "prefill_reused": True,
            "prefill_seconds": measured["prefill_seconds"],
            "render_and_vision_seconds": measured["render_and_vision_seconds"],
            "language_prefill_seconds": measured["language_prefill_seconds"],
            "prefill_bytes_transferred": measured["prefill_bytes_transferred"],
            "decode_seconds": duration,
            "generated_tokens": tokens,
            "decode_calls": measured["decode_calls"],
            "decode_generated_tokens": measured["decode_generated_tokens"],
            "reasoning_tokens": measured["reasoning_tokens"],
            "answer_tokens": measured["answer_tokens"],
            "reasoning_complete": measured["reasoning_complete"],
            "generation_limit_reached": measured["generation_limit_reached"],
            "tokens_per_second": measured["decode_generated_tokens"] / duration if duration else 0,
            "processed_decode_tokens_per_second": measured["processed_decode_row_steps"] / duration
            if duration
            else 0,
            "processed_decode_row_steps": measured["processed_decode_row_steps"],
            "active_decode_row_steps": measured["active_decode_row_steps"],
            "decode_batch_size_counts": measured["decode_batch_size_counts"],
            "compact_decode_batches": measured["compact_decode_batches"],
            "measurement_too_short_for_steady_throughput": measured["decode_calls"] < 4,
            "end_to_end_tokens_per_second": tokens / (duration + measured["prefill_seconds"]),
            "decode_bytes_transferred": measured["decode_bytes_transferred"],
            "bytes_transferred": measured["decode_bytes_transferred"],
            "new_programs": engine.program_count - programs_before,
            "new_programs_during_measurement": engine.program_count - programs_before,
            "fully_warmed_program_variants": engine.program_count == programs_before,
            "status": "PASS"
            if tokens
            and measured["decode_calls"]
            and all(x["status"] == "PASS" and x["same_answer_as_single"] for x in checks)
            else "FAIL",
            "quality_cases": checks,
            "answers": measured["answers"],
            "prompt_tokens": measured["prompt_tokens"],
            "timing_scope": (
                "warm decode from restored diverse image prompt states; short quality answers; "
                "host expert misses and sampling included; prefill time recorded during original "
                "single-image preparation, not rerun; compilation/tracing may be included; "
                "throughput includes reasoning and answer decode tokens"
            ),
            "scaling_comparison": (
                "different case subsets; use matched_replay_benchmarks for scaling"
            ),
        }
        if cache_before is not None:
            for key in ("hits", "misses"):
                item["expert_cache_" + key] = engine.expert_cache.stats[key] - cache_before[key]
        report["benchmarks"].append(item)
        log(
            f"Batch {batch_size}: {item['tokens_per_second']:.3f} generated tokens/s; "
            f"{duration:.2f}s decode"
        )
        save()
        # Release restored snapshots before placing another batch.
        del prepared


@tracing.operation("fixed_token_replay")
def fixed_token_replay(engine, prepared, token_ids, deadline):
    """Advance all independent rows by the same fixed number of forced tokens.

    No sampling, EOS, row compaction or response-length variability. This is
    an engine throughput diagnostic, never a generated answer or quality test.
    """
    if token_ids.ndim != 2 or token_ids.shape[0] != len(prepared.positions) or not token_ids.size:
        raise ValueError("replay token IDs must have shape [batch, positive steps]")
    prepared = prepared.to_device()
    caches = [list(c) for c in prepared.caches]
    positions = prepared.positions.copy()
    if np.any(positions + token_ids.shape[1] > engine.max_len):
        raise ValueError("fixed replay exceeds model context")
    before = dict(engine.store.stats)
    cache = getattr(engine, "expert_cache", None)
    cache_before = dict(cache.stats) if cache is not None else {}
    programs = engine.program_count
    started = time.perf_counter()
    for step in range(token_ids.shape[1]):
        deadline()
        progress.update(
            detail=f"fixed replay batch {len(positions)} step {step + 1}/{token_ids.shape[1]}"
        )
        with tracing.span("fixed_replay_decode", batch_size=len(positions), step=step + 1):
            logits, caches = engine.decode_rows(token_ids[:, step], caches, positions)
            if not np.isfinite(np.asarray(logits)).all():
                raise ValueError("non-finite fixed replay logits")
        positions += 1
    seconds = time.perf_counter() - started
    tokens = int(token_ids.size)
    result = {
        "decode_seconds": seconds,
        "decode_calls": token_ids.shape[1],
        "decode_row_steps": tokens,
        "tokens_per_second": tokens / seconds if seconds else 0,
        "new_programs": engine.program_count - programs,
        "quality_gate": "NOT_RUN",
    }
    for key in ("bytes_transferred", "host_gather_seconds"):
        result[key] = engine.store.stats.get(key, 0) - before.get(key, 0)
    if cache is not None:
        for key in ("hits", "misses", "evictions", "compact_table_bytes_avoided"):
            result["expert_cache_" + key] = cache.stats.get(key, 0) - cache_before.get(key, 0)
    return result


def matched_replay_benchmarks(engine, chat, config, report, bank, save, deadline, remaining=None):
    """Same image states, forced tokens and work count at B1/B4/B8.

    Reset cache entries before each width. Warm complete workload cycles, then
    measure another cycle, so cache differences reflect processing the same
    workload in different groupings. All counters retain their lifetime values.
    """
    import hashlib

    steps = config.get("matched_replay_steps", 16)
    count = max(config["batch_sizes"])
    if not steps or len(bank) < count:
        return
    report["stage"] = "MATCHED_BATCH_REPLAY"
    report["matched_replay_benchmarks"] = []
    selected = bank[:count]
    # Identical diagnostic continuation across formats; the image prompt states
    # remain distinct. This text is NOT asserted to describe any image.
    text = "I am inspecting the image carefully and checking the requested details. " * 8
    continuation = np.asarray(chat.tokenizer(text, add_special_tokens=False).input_ids, np.int32)
    if len(continuation) < steps:
        raise ValueError("diagnostic continuation is shorter than matched_replay_steps")
    forced = np.tile(continuation[:steps], (count, 1))
    names = [entry["case"]["name"] for entry in selected]
    token_sha = hashlib.sha256(forced.astype("<i4").tobytes()).hexdigest()
    cache = getattr(engine, "expert_cache", None)
    save()
    for batch in config["batch_sizes"]:
        deadline()
        groups = [selected[start : start + batch] for start in range(0, count, batch)]
        if cache is not None:
            cache.clear_entries()
        item = {
            "batch_size": batch,
            "status": "WARMING",
            "case_names": names,
            "steps_per_case": steps,
            "forced_token_ids": continuation[:steps].tolist(),
            "forced_tokens_sha256": token_sha,
            "same_workload_rows": count,
            "cache_protocol": "clear entries; warm whole workload cycles; measure next cycle",
            "warmup_cycles": [],
            "groups": [],
            "quality_gate": "NOT_RUN",
            "scope": (
                "fixed-token engine replay; all rows live; reused image prompt states; "
                "no generated answers"
            ),
        }
        report["matched_replay_benchmarks"].append(item)
        save()

        def cycle(records, purpose):
            for number, group in enumerate(groups):
                item["active_group"] = {
                    "purpose": purpose,
                    "number": number + 1,
                    "total": len(groups),
                    "case_names": [entry["case"]["name"] for entry in group],
                }
                save()
                try:
                    deadline()
                    prepared = PreparedBatch.combine([entry["state"] for entry in group])
                    start = number * batch
                    with execution_scope(engine, purpose, enabled=False):
                        record = fixed_token_replay(
                            engine, prepared, forced[start : start + batch], deadline
                        )
                    del prepared
                except Exception as error:
                    item.update(status="INTERRUPTED", error_type=type(error).__name__)
                    save()
                    raise
                records.append(record)
                save()
            item.pop("active_group", None)

        for number in range(3):
            warm = {"status": "RUNNING", "groups": []}
            item["warmup_cycles"].append(warm)
            cycle(warm["groups"], f"matched_warmup_{number + 1}")
            warm.update(
                status="COMPLETE",
                seconds=sum(g["decode_seconds"] for g in warm["groups"]),
                new_programs=sum(g["new_programs"] for g in warm["groups"]),
            )
            save()
            if not warm["new_programs"]:
                break
        item["status"] = "MEASURING"
        save()
        cycle(item["groups"], "matched_measurement")
        seconds = sum(g["decode_seconds"] for g in item["groups"])
        tokens = sum(g["decode_row_steps"] for g in item["groups"])
        new_programs = sum(g["new_programs"] for g in item["groups"])
        item.update(
            status="MEASURED" if not new_programs else "COMPILE_CONTAMINATED",
            decode_seconds=seconds,
            decode_row_steps=tokens,
            tokens_per_second=tokens / seconds if seconds else 0,
            new_programs_during_measurement=new_programs,
            fully_warmed_program_variants=not new_programs,
        )
        for key in (
            "bytes_transferred",
            "host_gather_seconds",
            "expert_cache_hits",
            "expert_cache_misses",
            "expert_cache_evictions",
            "expert_cache_compact_table_bytes_avoided",
        ):
            item[key] = sum(g.get(key, 0) for g in item["groups"])
        item["bytes_transferred_per_row_step"] = item["bytes_transferred"] / tokens
        log(
            f"Matched replay B{batch}: {item['tokens_per_second']:.3f} row steps/s; "
            f"{tokens} fixed steps"
        )
        save()
    if config.get("profile_matched_decode", True):
        matched_decode_diagnostic(engine, selected, forced, report, save, deadline, remaining)


@tracing.operation("matched_decode_diagnostic")
def matched_decode_diagnostic(engine, selected, forced, report, save, deadline, remaining=None):
    """Profile two image steps after warmup, outside all throughput measurements."""
    report["stage"] = "MATCHED_DECODE_DIAGNOSTIC"
    width = len(selected)
    steps = min(2, forced.shape[1])
    item = {
        "status": "WARMING",
        "batch_size": width,
        "case_names": [entry["case"]["name"] for entry in selected],
        "steps": steps,
        "quality_gate": "NOT_RUN",
        "scope": "separate post-warmup image replay; extra synchronization and profiler overhead",
    }
    report["matched_decode_diagnostic"] = item
    # Reserve cleanup time and use the measured width as a conservative budget estimate.
    measured = next(
        (r for r in report.get("matched_replay_benchmarks", []) if r["batch_size"] == width), {}
    )
    calls = sum(g.get("decode_calls", 0) for g in measured.get("groups", []))
    expected = measured.get("decode_seconds", 0) / max(1, calls) * steps * 4
    if remaining is not None and remaining() < max(180, expected + 120):
        item["status"] = "SKIPPED_BUDGET"
        save()
        return
    save()
    prepared = PreparedBatch.combine([entry["state"] for entry in selected])
    try:
        with execution_scope(engine, "image_diagnostic_warmup", enabled=False):
            item["warmup"] = fixed_token_replay(engine, prepared, forced[:, :steps], deadline)
        sparse = [i for i, kind in enumerate(engine.cfg.mlp_types) if kind == "sparse"]
        layers = sorted({sparse[0], sparse[len(sparse) // 2], sparse[-1]}) if sparse else []
        item.update(status="PROFILING", sampled_layers=layers)
        save()
        directory = getattr(engine, "device_trace_dir", None)
        trace = optional_device_trace(directory) if directory is not None else nullcontext()
        with execution_scope(engine, "matched_images", layers=layers), trace:
            item["replay"] = fixed_token_replay(engine, prepared, forced[:, :steps], deadline)
        item["status"] = "COMPILE_CONTAMINATED" if item["replay"]["new_programs"] else "MEASURED"
    except Exception as error:
        item.update(status="INTERRUPTED", error_type=type(error).__name__)
        raise
    finally:
        del prepared
        save()


@tracing.operation("vision_resolution_controls")
def vision_resolution_controls(engine, chat, config, eos, images, cases, report, save, remaining):
    """Retry one failed road-text case at a larger token cap; never replace its gate."""
    import hashlib

    limit = config.get("vision_resolution_control_tokens", 1024)
    if not limit:
        return
    failed = set(report["quality"]["failed_cases"])
    candidate = next(
        (c for c in cases if c["name"] in failed and c["group"] == "beamng_visible_text"), None
    )
    if candidate is None:
        return
    original = next(c for c in report["quality"]["cases"] if c["name"] == candidate["name"])
    blobs = [images[name] for name in candidate["images"]]
    digests = [hashlib.sha256(blob).hexdigest() for blob in blobs]
    previous = {digest: chat.image_token_limits.get(digest) for digest in digests}
    old_limit = max(previous[d] or chat.max_image_tokens for d in digests)
    report["stage"] = "VISION_RESOLUTION_CONTROL"
    item = {
        "case": candidate["name"],
        "status": "RUNNING",
        "baseline_status": original["status"],
        "baseline_image_max_tokens": old_limit,
        "image_max_tokens": limit,
        "image_sha256": digests,
        "quality_gate": "NOT_RUN",
        "scope": "unchanged source image, prompt, labels and thinking; larger image token cap",
    }
    report["vision_resolution_controls"] = [item]
    expected = (
        original.get("render_and_vision_seconds", 0)
        + original.get("language_prefill_seconds", 0) * max(1, limit / old_limit)
        + original.get("decode_seconds", 0) * 1.5
    )
    item["estimated_seconds"] = expected
    if limit <= old_limit:
        item["status"] = "SKIPPED_NO_INCREASE"
        save()
        return
    if remaining() < max(600, expected * 1.5 + 180):
        item["status"] = "SKIPPED_BUDGET"
        save()
        return
    save()
    try:
        chat.image_token_limits.update(dict.fromkeys(digests, limit))
        result = generate(
            engine, chat, [(candidate["prompt"], blobs)], config["max_new_tokens"], eos
        )
        answer = result["answers"][0]
        matched, detail = quality.match(candidate["match"], answer)
        item.update(
            status="PASS" if matched and result["stopped"][0] else "FAIL",
            answer=answer,
            answer_match=matched,
            match_detail=detail,
            stopped=result["stopped"][0],
            prompt_tokens=result["prompt_tokens"],
            image_tokens=[len(chat.image(blob)) for blob in blobs],
            baseline_prompt_tokens=original["prompt_tokens"],
            seconds=result["prefill_seconds"] + result["decode_seconds"],
            reasoning_complete=result["reasoning_complete"][0],
            reasoning_tokens=result["reasoning_tokens"][0],
            answer_tokens=result["answer_tokens"][0],
        )
    except Exception as error:
        item.update(status="INTERRUPTED", error_type=type(error).__name__)
        raise
    finally:
        for digest, value in previous.items():
            if value is None:
                chat.image_token_limits.pop(digest, None)
            else:
                chat.image_token_limits[digest] = value
        save()


def long_image_benchmarks(engine, chat, config, eos, images, report, save, deadline, remaining):
    """Gather sustained image decode samples only when the observed prefill cost fits."""
    report["stage"] = "LONG_IMAGE_BENCHMARKS"
    report["long_image_benchmarks"] = []
    single_cases = [case for case in report["quality"]["cases"] if len(case["prompt_tokens"]) == 1]
    seconds_per_token = float(
        np.median(
            [
                case["language_prefill_seconds"] / max(1, case["prompt_tokens"][0])
                for case in single_cases
            ]
        )
    )
    image_cases = [case for case in quality.cases(images) if case["asserted"]]
    request = "Describe this image in detail in at least six sentences."
    bank = []
    save()
    for batch in sorted(set(config["batch_sizes"])):
        deadline()
        new_requests = [
            (request, [images[case["images"][0]]]) for case in image_cases[len(bank) : batch]
        ]
        # Images are already in VisionChat's embedding cache. This renders only
        # to estimate their exact token count; the 1.5 factor reserves variance.
        estimated = sum(len(chat.render(text, blobs)[0]) for text, blobs in new_requests)
        required = 1.5 * seconds_per_token * estimated + 120
        if remaining() < required:
            report["long_image_benchmarks"].append(
                {
                    "batch_size": batch,
                    "status": "SKIPPED_BUDGET",
                    "estimated_required_seconds": required,
                    "remaining_seconds": remaining(),
                    "scope": "long image descriptions; no throughput inferred for skipped batches",
                }
            )
            save()
            continue
        for text, blobs in new_requests:
            deadline()
            bank.append(
                prepare_requests(engine, chat, [(text, blobs)], config["max_new_tokens"]).to_host()
            )
        prepared = PreparedBatch.combine(bank[:batch])
        decode_prepared(engine, chat, prepared, min(2, config["max_new_tokens"]), eos)
        programs_before = engine.program_count
        measured = decode_prepared(engine, chat, prepared, config["max_new_tokens"], eos)
        seconds = measured["decode_seconds"]
        report["long_image_benchmarks"].append(
            {
                "batch_size": batch,
                "status": "MEASURED",
                "decode_seconds": seconds,
                "decode_calls": measured["decode_calls"],
                "decode_generated_tokens": measured["decode_generated_tokens"],
                "reasoning_tokens": measured["reasoning_tokens"],
                "answer_tokens": measured["answer_tokens"],
                "reasoning_complete": measured["reasoning_complete"],
                "generation_limit_reached": measured["generation_limit_reached"],
                "tokens_per_second": measured["decode_generated_tokens"] / seconds
                if seconds
                else 0,
                "prefill_seconds": measured["prefill_seconds"],
                "prefill_reused": True,
                "decode_bytes_transferred": measured["decode_bytes_transferred"],
                "new_programs": engine.program_count - programs_before,
                "answers": measured["answers"],
                "scope": (
                    "warm long-description decode on diverse real image prompts; "
                    "content not quality-asserted"
                ),
            }
        )
        save()
        del prepared


@tracing.operation("load_runtime")
def load_runtime(config, report, save):
    import jax
    import ml_dtypes
    from glm53 import checkpoint as checkpoint
    from glm53 import vision as vis
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    from .engine import PackedOffloadEngine

    native = fetch_json(hf_url(BASE_REPO, BASE_REVISION, "config.json"))
    cfg = cfg_from_json(native)
    report["stage"] = "GGUF_HEADERS"
    save()
    source = open_gguf(config["quant"], config.get("gguf_dirs", []), log)
    # Validate the expert formats and RAM budget before reading the BF16 base.
    tensors = {
        name: source.info(name)
        for name in source.where
        if re.fullmatch(r"blk\.\d+\.ffn_(gate|up|down)_exps\.weight", name)
        and int(name.split(".")[1]) < cfg.n_layers
    }
    if len(tensors) != sum(t == "sparse" for t in cfg.mlp_types) * 3:
        raise ValueError("GGUF lacks required routed expert tensors")
    from .gguf_quant import block_layout

    for info in tensors.values():
        block_layout(info["type"])
    required = sum(t["nbytes"] for t in tensors.values())
    if required + (48 << 30) > available_ram():
        raise MemoryError("packed experts plus base-load reserve exceed available host RAM")
    report["weights"] = {
        "packed_expert_bytes": required,
        "types": sorted({t["type"] for t in tensors.values()}),
        "base_dtype": "BF16 with float32 routers/norms/mHC",
        "expert_requantization": False,
    }
    report["stage"] = "BF16_NONEXPERT_LOAD"
    save()
    reader = BF16Reader()
    params = checkpoint.load_top(reader, dtype=ml_dtypes.bfloat16)
    params["layers"] = []
    for layer in range(cfg.n_layers):
        progress.update(detail=f"BF16 layer {layer + 1}/{cfg.n_layers}", force=True)
        with tracing.span("load_nonexpert_layer", layer=layer):
            params["layers"].append(
                checkpoint.load_layer(
                    reader,
                    layer,
                    cfg.layer_types[layer],
                    cfg.mlp_types[layer],
                    dtype=ml_dtypes.bfloat16,
                )
            )
        log(f"BF16 non-expert layer {layer + 1}/{cfg.n_layers}")
    report["stage"] = "HOST_EXPERT_LOAD"
    save()
    store = PackedExperts.from_gguf(
        source, cfg, len(jax.devices()), tile=config["expert_tile"], log=log
    )
    for shard in source.shards.values():
        shard.pf.close()
    del source
    report["stage"] = "TPU_PARAMETER_PLACEMENT"
    save()
    engine = PackedOffloadEngine(
        cfg,
        params,
        store,
        max_len=config["max_context"],
        prefill_chunk=config["prefill_chunk"],
        prefill_moe_window=config.get("prefill_moe_window", 256),
        tile_routed_experts=config.get("tile_routed_experts", True),
        expert_tile=config["expert_tile"],
        seq_shard=False,
        expert_cache_bytes=config.get("expert_cache_mib_per_device", 0)
        * (1 << 20)
        * len(jax.devices()),
        direct_expert_cache=config.get("direct_expert_cache", True),
        routing_log_path=Path(config["work_dir"]) / "routing-replay.jsonl",
        execution_log_path=Path(config["work_dir"]) / "execution-timing.jsonl",
    )
    jax.block_until_ready(engine.params)
    engine.compact_decode_batches = config.get("compact_decode_batches", True)
    engine.device_trace_dir = (
        Path(config["work_dir"]) / "device-trace"
        if config.get("capture_device_trace", False)
        else None
    )
    del params
    gc.collect()
    report["stage"] = "VISION_LOAD"
    save()
    vision_params = vis.load_vision(reader, dtype=np.float32)
    vision_params = vis.to_sharded(vision_params, engine.mesh)
    jax.block_until_ready({k: v for k, v in vision_params.items() if k != "n_heads"})
    vision_forward = vis.make_forward_sharded(vision_params, engine.mesh)
    report["weights"]["bf16_bytes_read"] = reader.bytes_read
    del reader
    gc.collect()
    token_dir = Path(config["work_dir"]) / "tokenizer"
    snapshot_download(
        BASE_REPO,
        revision=BASE_REVISION,
        local_dir=token_dir,
        allow_patterns=[
            "*config.json",
            "tokenizer.json",
            "*.jinja",
            "chat_templates/*",
            "vocab.json",
            "merges.txt",
        ],
        max_workers=4,
        token=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        token_dir, local_files_only=True, trust_remote_code=False
    )
    chat = VisionChat(
        tokenizer,
        vision_forward,
        config["vision_max_tokens"],
        config["thinking_budget"],
        reasoning_effort=config.get("reasoning_effort", "max"),
    )
    return engine, chat, native["text_config"]["eos_token_id"]


def run(config, output):
    defaults = {
        "quant": "Q6_K_XL",
        "batch_sizes": [1, 4, 8],
        "max_context": 8192,
        "max_new_tokens": 256,
        "prefill_chunk": 32,
        "prefill_moe_window": 256,
        "tile_routed_experts": True,
        "expert_tile": 4,
        "expert_cache_mib_per_device": 7168,
        "cache_safety_mib_per_device": 1024,
        "compact_decode_batches": True,
        "full_decode_warmup": True,
        "direct_expert_cache": True,
        "integer_dot_probe": True,
        "integer_projection_probe": True,
        "matched_replay_steps": 16,
        "profile_matched_decode": True,
        "vision_resolution_control_tokens": 1024,
        "vision_max_tokens": 1024,
        "thinking_budget": None,
        "reasoning_effort": "max",
        "deadline_seconds": 21600,
        "run_model": True,
        "experiment_mode": "full",
        "profile_experts": 96,
        "profile_repeats": 3,
    }
    config = {**defaults, **config}
    config.setdefault("work_dir", str(Path(output).resolve().parent))
    Path(config["work_dir"]).mkdir(parents=True, exist_ok=True)
    progress.configure(Path(output).parent / "progress.json")
    progress.update(stage="PREFLIGHT", detail="initializing JAX", force=True)
    diagnostics = WorkerDiagnostics(Path(output).parent)
    report = {
        "status": "RUNNING",
        "model_check": "NOT_RUN",
        "config": config,
        "stage": "PREFLIGHT",
        "provenance": {
            "base": f"{BASE_REPO}@{BASE_REVISION}",
            "experts": f"{EXPERT_REPO}@{EXPERT_REVISION}",
            "quantization": config["quant"],
            "thinking_budget": config["thinking_budget"],
            "reasoning_effort": config["reasoning_effort"],
            "generation_token_limit": config["max_new_tokens"],
            "transport": "host RAM -> chip-major packed device_put -> tiled JAX decode",
            "prefill_attention_chunk": config["prefill_chunk"],
            "prefill_expert_window": config["prefill_moe_window"],
            "tile_routed_experts": config["tile_routed_experts"],
            "expert_matmul": "GGUF storage -> BF16 weights/activations, FP32 accumulation",
            "integer_kernels": "isolated diagnostics; activation quantization disabled in decoder",
            "direct_expert_cache": config["direct_expert_cache"],
        },
    }
    manifest = Path(__file__).resolve().parent.parent / "bundle-manifest.json"
    if manifest.exists():
        report["provenance"]["source_manifest"] = json.loads(manifest.read_text())

    engine = None
    scene_images, scene_cases = {}, []

    def save():
        progress.update(stage=report["stage"], force=True)
        if engine is not None:
            report["expert_stats"] = dict(engine.store.stats)
            report["routing_stats"] = dict(engine.routing_stats)
            report["routing_by_layer"] = dict(engine.routing_by_layer)
            report["expert_usage_by_layer"] = getattr(engine, "expert_usage_by_layer", {})
            report["expert_usage_scope"] = (
                "original router IDs by image-embedding positions, text positions and decode; "
                "image positions are not exclusive vision experts"
            )
            report["program_count"] = engine.program_count
            report["execution_samples"] = getattr(engine, "execution_samples", [])
            if engine.expert_cache is not None:
                report["expert_cache"] = dict(engine.expert_cache.stats)
        try:
            snapshot = memory_snapshot()
        except Exception as error:
            snapshot = {"unavailable": f"{type(error).__name__}: {error}"}
        report.setdefault("memory_by_stage", {})[report["stage"]] = snapshot
        write_report(output, report)
        with (Path(output).parent / "events.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    {
                        "time": time.time(),
                        "stage": report["stage"],
                        "status": report["status"],
                        "memory": snapshot,
                    }
                )
                + "\n"
            )

    started = time.monotonic()

    def deadline():
        if remaining() <= 0:
            raise TimeoutError("notebook experiment deadline exceeded")

    def remaining():
        return config.get("worker_budget_seconds", config["deadline_seconds"]) - (
            time.monotonic() - started
        )

    try:
        if config.get("beamng_fixture_bundle"):
            from .scene_fixtures import load_scene_bundle

            scene_images, scene_cases, scene_manifest, scene_sha = load_scene_bundle(
                config["beamng_fixture_bundle"], config.get("beamng_fixture_sha256")
            )
            report["scene_fixtures"] = {
                "bundle_sha256": scene_sha,
                "label_scope": scene_manifest["label_scope"],
                "label_authority": scene_manifest["label_authority"],
                "images": scene_manifest["images"],
                "case_names": [case["name"] for case in scene_cases],
                "max_image_tokens": config.get("beamng_image_max_tokens", 384),
            }
            fixture_dir = Path(output).parent / "vision-fixtures"
            fixture_dir.mkdir(exist_ok=True)
            (fixture_dir / "manifest.json").write_text(json.dumps(scene_manifest, indent=2))
            for entry in scene_manifest["images"]:
                (fixture_dir / Path(entry["path"]).name).write_bytes(scene_images[entry["name"]])
        tracing.configure(config, Path(output).parent)
        import jax

        # Routers, mHC and recurrent state use float32 arithmetic; request full
        # float32 matmul precision on TPU while expert/large weights stay BF16.
        jax.config.update("jax_default_matmul_precision", "highest")
        jax.config.update("jax_log_compiles", config.get("log_compiles", False))
        if config["run_model"]:
            jax.config.update(
                "jax_compilation_cache_dir", str(Path(config["work_dir"]).parent / "compile-cache")
            )
            jax.config.update("jax_compilation_cache_max_size", 2 << 30)
        report["provenance"]["matmul_precision"] = "highest"
        report["provenance"]["sequence_sharded_cache"] = False
        save()
        if any(b not in (1, 4, 8) for b in config["batch_sizes"]) or not config["batch_sizes"]:
            raise ValueError("batch_sizes must contain 1, 4 or 8")
        if config["experiment_mode"] not in ("full", "expert_profile"):
            raise ValueError("experiment_mode must be full or expert_profile")
        if config["experiment_mode"] == "expert_profile":
            if (
                not isinstance(config["profile_experts"], int)
                or not 8 <= config["profile_experts"] <= 96
            ):
                raise ValueError("profile_experts must be between 8 and 96")
            if (
                not isinstance(config["profile_repeats"], int)
                or not 1 <= config["profile_repeats"] <= 5
            ):
                raise ValueError("profile_repeats must be between 1 and 5")
            if config["deadline_seconds"] > 900:
                raise ValueError("expert_profile deadline_seconds must be at most 900")
        if config["thinking_budget"] not in (0, None):
            raise ValueError("thinking_budget must be 0 (forced close) or null (native reasoning)")
        if config["reasoning_effort"] not in ("low", "high", "max"):
            raise ValueError("reasoning_effort must be low, high or max")
        selected_cases = config.get("quality_case_names")
        if selected_cases is not None:
            asserted_names = {
                case["name"] for case in quality.cases() + scene_cases if case["asserted"]
            }
            if (
                not isinstance(selected_cases, list)
                or any(not isinstance(name, str) for name in selected_cases)
                or len(selected_cases) < max(config["batch_sizes"])
                or len(set(selected_cases)) != len(selected_cases)
                or not set(selected_cases) <= asserted_names
            ):
                raise ValueError(
                    "quality_case_names must select distinct asserted cases for every batch row"
                )
        for name in (
            "max_context",
            "max_new_tokens",
            "prefill_chunk",
            "prefill_moe_window",
            "expert_tile",
            "vision_max_tokens",
            "deadline_seconds",
        ):
            if not isinstance(config[name], int) or config[name] < 1:
                raise ValueError(f"{name} must be a positive integer")
        if config["quant"] not in ("Q6_K_XL", "Q4_K_XL"):
            raise ValueError("quant must be Q6_K_XL or Q4_K_XL")
        if config["expert_tile"] not in (4, 8) or config["prefill_chunk"] not in (32, 64):
            raise ValueError("expert_tile must be 4 or 8; prefill_chunk must be 32 or 64")
        if config["prefill_moe_window"] not in (32, 64, 128, 256):
            raise ValueError("prefill_moe_window must be 32, 64, 128 or 256")
        if not isinstance(config["tile_routed_experts"], bool):
            raise ValueError("tile_routed_experts must be boolean")
        for key in (
            "direct_expert_cache",
            "integer_dot_probe",
            "integer_projection_probe",
            "profile_matched_decode",
        ):
            if not isinstance(config[key], bool):
                raise ValueError(f"{key} must be boolean")
        if (
            not isinstance(config["matched_replay_steps"], int)
            or not 0 <= config["matched_replay_steps"] <= 64
        ):
            raise ValueError("matched_replay_steps must be between 0 and 64")
        resolution_limit = config["vision_resolution_control_tokens"]
        if not isinstance(resolution_limit, int) or resolution_limit not in (0, 768, 1024):
            raise ValueError("vision_resolution_control_tokens must be 0, 768 or 1024")
        cache_mib = config["expert_cache_mib_per_device"]
        if not isinstance(cache_mib, int) or cache_mib < 0:
            raise ValueError("expert_cache_mib_per_device must be a nonnegative integer")
        safety_mib = config["cache_safety_mib_per_device"]
        if not isinstance(safety_mib, int) or safety_mib < 0:
            raise ValueError("cache_safety_mib_per_device must be a nonnegative integer")
        if config["vision_max_tokens"] < 16:
            raise ValueError("vision_max_tokens must be at least 16")
        if config["max_new_tokens"] >= config["max_context"]:
            raise ValueError("max_new_tokens must leave room for a prompt in max_context")
        report["environment"] = environment(config["work_dir"])
        if config["run_model"]:
            devices = report["environment"]["devices"]
            if len(devices) != 8 or any(d["platform"] != "tpu" for d in devices):
                raise RuntimeError(
                    "model trial requires eight TPU devices; select Kaggle TPU v5e-8"
                )
        report["validation"] = decoder_gate()
        progress.update(detail="sharded vision numerical gate", force=True)
        report["validation"].update(vision_gate())
        if config["direct_expert_cache"]:
            progress.update(detail="direct resident expert-cache numerical gate", force=True)
            report["validation"].update(expert_cache_gate())
        save()
        log("Packed decoder, sharded vision and selected cache numerical gates PASS")
        if not config["run_model"]:
            report["status"] = "PREFLIGHT_ONLY"
            return report
        if config["integer_dot_probe"]:
            from .integer_kernel import dot_probe

            report["stage"] = "INTEGER_DOT_PROBE"

            def publish_integer(value):
                report["integer_dot_probe"] = value
                save()

            dot_probe(deadline, publish_integer)
            log("Resident BF16/INT8/INT4 diagnostic complete; decoder remains BF16")
        if config["experiment_mode"] == "expert_profile":
            from .profiling import run_expert_profile

            run_expert_profile(config, report, save, deadline)
            report["status"] = "PROFILE_COMPLETE"
            report["stage"] = "COMPLETE"
            return report
        engine, chat, eos = load_runtime(config, report, save)
        if scene_images:
            import hashlib

            chat.image_token_limits = {
                hashlib.sha256(blob).hexdigest(): config.get("beamng_image_max_tokens", 384)
                for blob in scene_images.values()
            }
        report["hbm_plan"] = check_hbm_budget(engine, config)
        if config["integer_projection_probe"]:
            from .profiling import integer_projection_profile

            report["stage"] = "INTEGER_PROJECTION_PROBE"

            def publish_projection(value):
                report["integer_projection_probe"] = value
                save()

            integer_projection_profile(engine.store, engine.mesh, deadline, publish_projection)
        deadline()
        report["stage"] = "TEXT_SMOKE"
        smoke = generate(
            engine,
            chat,
            [("What is 2 + 2? Answer with a single number.", [])],
            config["max_new_tokens"],
            eos,
        )
        smoke_passed, smoke_detail = quality.match(
            {"kind": "first_int", "value": 4}, smoke["answers"][0]
        )
        report["text_smoke"] = {
            "status": "PASS" if smoke_passed and smoke["stopped"][0] else "FAIL",
            "answer": smoke["answers"][0],
            "match_detail": smoke_detail,
        }
        save()
        early_decode_probe(engine, chat, config, eos, report, save, deadline)
        images = probe_images.fixtures()
        images.update(scene_images)
        report["stage"] = "VISION_QUALITY_GATE"
        report["quality"] = {
            "status": "RUNNING",
            "asserted_cases": 0,
            "failed_cases": [],
            "cases": [],
        }
        save()
        all_cases = quality.cases(images) + scene_cases
        if selected_cases is not None:
            by_name = {case["name"]: case for case in all_cases}
            all_cases = [by_name[name] for name in selected_cases]
        report["quality"]["expected_cases"] = [case["name"] for case in all_cases]
        report["quality"]["scope"] = (
            "selected comparison cases" if selected_cases else "full battery"
        )
        priority = [case for case in all_cases if case["asserted"]][: max(config["batch_sizes"])]
        priority_names = {case["name"] for case in priority}
        ordered = priority + [case for case in all_cases if case["name"] not in priority_names]
        bank = []
        report["prompt_state_bank"] = {"host_bytes": 0, "snapshot_seconds": 0.0}
        for case in ordered:
            report["stage"] = "VISION_QUALITY_GATE"
            deadline()
            result = generate(
                engine,
                chat,
                [(case["prompt"], [images[n] for n in case["images"]])],
                config["max_new_tokens"],
                eos,
                keep_prepared=case["name"] in priority_names,
            )
            if "_prepared" in result:
                import jax

                snapshot_started = time.perf_counter()
                state = result.pop("_prepared").to_host()
                report["prompt_state_bank"]["snapshot_seconds"] += (
                    time.perf_counter() - snapshot_started
                )
                report["prompt_state_bank"]["host_bytes"] += sum(
                    a.nbytes for a in jax.tree.leaves(state.caches)
                )
                bank.append({"case": case, "state": state, "single_answer": result["answers"][0]})
                del state
            answer = result["answers"][0]
            passed, detail = quality.match(case["match"], answer)
            item = {
                "name": case["name"],
                "asserted": case["asserted"],
                "answer": answer,
                "status": "PASS" if passed and result["stopped"][0] else "FAIL",
                "answer_match": passed,
                "match_detail": detail,
                "stopped": result["stopped"][0],
                "seconds": result["prefill_seconds"] + result["decode_seconds"],
                "render_and_vision_seconds": result["render_and_vision_seconds"],
                "language_prefill_seconds": result["language_prefill_seconds"],
                "decode_seconds": result["decode_seconds"],
                "prompt_tokens": result["prompt_tokens"],
                "reasoning_complete": result["reasoning_complete"][0],
                "reasoning_tokens": result["reasoning_tokens"][0],
                "answer_tokens": result["answer_tokens"][0],
                "generated_tokens": result["generated_tokens"],
                "generation_limit_reached": result["generation_limit_reached"][0],
                "group": case["group"],
                "image_names": case["images"],
            }
            report["quality"]["cases"].append(item)
            with tracing.span(
                "quality_result",
                case=case["name"],
                asserted=case["asserted"],
                image_names=case["images"],
                group=case["group"],
            ) as check:
                # Only the built-in synthetic fixtures, never custom user requests.
                check.update(
                    answer=answer,
                    check_status=item["status"],
                    match_detail=detail,
                    stopped=item["stopped"],
                )
            if case["asserted"]:
                report["quality"]["asserted_cases"] += 1
                if not passed or not item["stopped"]:
                    report["quality"]["failed_cases"].append(case["name"])
            log(f"{case['name']}: {item['status']} {answer!r}")
            save()
            if bank and len(bank) == len(priority):
                batch_benchmarks(engine, chat, config, eos, report, bank, save, deadline)
                matched_replay_benchmarks(
                    engine, chat, config, report, bank, save, deadline, remaining
                )
                bank.clear()
                gc.collect()
        report["quality"]["status"] = "FAIL" if report["quality"]["failed_cases"] else "PASS"
        vision_resolution_controls(
            engine, chat, config, eos, images, all_cases, report, save, remaining
        )
        if config.get("long_image_benchmarks", False):
            long_image_benchmarks(
                engine, chat, config, eos, images, report, save, deadline, remaining
            )
        if config.get("inputs"):
            report["custom_outputs"] = []
            inputs = config["inputs"]
            if len(inputs) not in (1, 4, 8):
                raise ValueError("inputs must contain 1, 4 or 8 prompts")
            requests = [
                (x["text"], [Path(p).read_bytes() for p in x.get("images", [])]) for x in inputs
            ]
            report["custom_outputs"] = generate(
                engine, chat, requests, config["max_new_tokens"], eos
            )
        report["environment_after"] = environment(config["work_dir"])
        report["expert_stats"] = engine.store.stats
        passed = (
            report["text_smoke"]["status"] == "PASS"
            and report["quality"]["status"] == "PASS"
            and all(x["status"] == "PASS" for x in report["benchmarks"])
        )
        report["model_check"] = "PASS" if passed else "FAIL"
        report["status"] = report["model_check"]
        report["stage"] = "COMPLETE"
    except Exception as error:
        report["status"] = "ERROR"
        report["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        log(f"ERROR during {report['stage']}: {type(error).__name__}: {error}")
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        save()
        progress.configure(None)
        tracing.finish(report["status"])
        diagnostics.close()
    return report


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = run(json.loads(Path(args.config).read_text()), args.output)
    log(f"Experiment status: {report['status']}; report: {args.output}")
    return 0 if report["status"] in {"PASS", "PREFLIGHT_ONLY", "PROFILE_COMPLETE"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
