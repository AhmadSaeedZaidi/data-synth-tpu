"""Bounded, observable vision/batching trial. Invoked by the notebook watchdog."""

import argparse
import hashlib
import io
import json
import time
import traceback
from contextlib import nullcontext
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from glm53_offload import probe_images, progress, quality, tracing
from glm53_offload.diagnostics import WorkerDiagnostics, atomic_json, optional_device_trace
from glm53_offload.scene_fixtures import load_scene_bundle
from PIL import Image

from .checkpoint import Checkpoint
from .compute import choose_policy, native_dot_proof
from .config import MODEL_REPO, MODEL_REVISION, ModelConfig
from .engine import Engine, token_table
from .evaluation.bundle import load_bundle
from .evaluation.policy import QUALITY_POLICY, AnswerReserve, quality_case
from .evaluation.scoring import match as match_answer
from .expert_calibration import calibrate_experts
from .experts import EXPERT_ROW_TILE
from .gates import run_gates
from .math import GDN_CHUNK
from .numerics_validation import validate_numerics
from .parallel import ordered_devices
from .routing_validation import validate_expert_routing
from .trial import (
    MemoryTracker,
    evaluation_sanity,
    replay_workload,
    update_summaries,
    validate_batches,
)
from .validation import validate, validate_long_context, validate_q8_codec
from .vision import VisionEncoder, load_vision, preprocess, text_positions


@jax.jit
def checked_argmax(logits):
    """One small host result contains token ids and invalid-logit counts.

    argmax alone silently maps an all-NaN vocabulary to token zero. Reducing
    across the vocabulary here catches it without transferring full logits.
    """
    return jnp.stack(
        (jnp.argmax(logits, axis=-1), jnp.sum(~jnp.isfinite(logits), axis=-1)), axis=-1
    )


class InvalidLogitsError(FloatingPointError):
    def __init__(self, step, counts, prompts):
        self.diagnostics = dict(
            stage="prefill" if step == 0 else "decode",
            decode_step=step,
            rows=[
                dict(
                    row=int(row),
                    case=prompts[row]["case"]["name"],
                    nonfinite_logits=int(counts[row]),
                )
                for row in np.flatnonzero(counts)
            ],
        )
        super().__init__("Non-finite model logits: " + json.dumps(self.diagnostics))


PREFILL_WINDOW = 128
PREFILL_ROWS = 1024


def prefill_window(batch):
    """Keep chunked-prefill rows near PREFILL_ROWS at every batch shape.

    Chunk-parallel GDN transients scale with batch * window, so smaller batches
    use the longest window and larger batches shorten it instead of doubling
    the live state. The floor is one GDN chunk.
    """
    return min(PREFILL_WINDOW, max(GDN_CHUNK, PREFILL_ROWS // batch))


def warmup_prompt(prototype):
    """Warm kernels without applying a request's long final-answer reserve."""
    positions = prototype["positions"][:16]
    return {
        **prototype,
        "case": {**prototype["case"], "answer_reserve_tokens": 0},
        "ids": prototype["ids"][:16],
        "positions": positions,
        "next_position": int(positions.max()) + 1,
    }


def final_answer(text):
    if "</think>" not in text:
        return "", "THINKING_TRUNCATED"
    return text.split("</think>", 1)[1].strip(), "FINAL"


def prepare_prompt(
    tokenizer, case, images, encoder, c, *, max_new_tokens, effort, vision_cache=None
):
    """Preserve ordered frame/text inputs; encoder=None performs token preflight."""
    content, grids, embeddings = [], [], []
    cache = {} if vision_cache is None else vision_cache
    for name in case["images"]:
        key = (
            hashlib.sha256(images[name]).hexdigest(),
            case.get("min_pixels", 65536),
            case.get("max_pixels", 589824),
            encoder is not None,
        )
        if key not in cache:
            patches, grid = preprocess(
                Image.open(io.BytesIO(images[name])), min_pixels=key[1], max_pixels=key[2]
            )
            encoded = None
            if encoder is not None:
                encoded = encoder.encode_patches(patches, grid)
                encoded.block_until_ready()
                encoded = np.asarray(encoded, np.float32)
                if not np.isfinite(encoded).all():
                    raise FloatingPointError(f"Non-finite vision features for case {case['name']}")
            cache[key] = (grid, encoded)
        grid, encoded = cache[key]
        grids.append(grid)
        embeddings.append(encoded)
    ordered = case.get(
        "content",
        [
            *[{"type": "image", "image": n} for n in case["images"]],
            {"type": "text", "text": case["prompt"]},
        ],
    )
    for item in ordered:
        content.append(
            {"type": "image"} if item["type"] == "image" else {"type": "text", "text": item["text"]}
        )
    messages = [{"role": "user", "content": content}]
    if case.get("system_prompt"):
        messages.insert(0, {"role": "system", "content": case["system_prompt"]})
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
        reasoning_effort=effort,
    )
    if not prompt.endswith("<think>\n"):
        raise ValueError("Tokenizer template did not enable thinking")
    if effort == "xhigh" and "Reasoning effort is set to xhigh." not in prompt:
        raise ValueError("Tokenizer did not apply xhigh reasoning instruction")
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    expanded, image = [], 0
    for token in ids:
        if token == c.image_token:
            count = int(np.prod(grids[image])) // 4
            if embeddings[image] is not None and len(embeddings[image]) != count:
                raise ValueError("Vision merge count differs from prompt expansion")
            expanded.extend([token] * count)
            image += 1
        else:
            expanded.append(token)
    if image != len(embeddings):
        raise ValueError("Tokenizer did not produce every image placeholder")
    output_budget = case.get("max_new_tokens", max_new_tokens)
    if len(expanded) + output_budget > c.max_context:
        raise ValueError("Prompt + requested output exceeds configured context capacity")
    positions, next_position = text_positions(expanded, grids, c)
    visual = {}
    spans = np.flatnonzero(np.asarray(expanded) == c.image_token)
    if embeddings and encoder is not None:
        for index, value in zip(spans, np.concatenate(embeddings), strict=True):
            visual[int(index)] = value
    return dict(
        ids=np.asarray(expanded, np.int64),
        positions=positions,
        next_position=next_position,
        visual=visual,
        case=case,
        max_new_tokens=output_budget,
        image_grids=grids,
        vision_encoded=encoder is not None,
    )


def prefill(engine, prompts, hasher, ngrams, tokens, emit, memory=None):
    batch, c = len(prompts), engine.config
    start = time.perf_counter()
    engine.reset(batch)
    if memory:
        memory.sample("state_allocated", batch)
    history = hasher.initial_history(batch)
    with tracing.span(
        "prefill", batch=batch, prompt_tokens=sum(len(p["ids"]) for p in prompts)
    ) as metrics:
        window = prefill_window(batch)
        for first in range(0, max(len(p["ids"]) for p in prompts), window):
            token_ids, hash_ids, positions, masks = [], [], [], []
            for step in range(first, first + window):
                active = np.asarray([step < len(p["ids"]) for p in prompts])
                ids = np.asarray(
                    [p["ids"][step] if active[row] else c.eos for row, p in enumerate(prompts)]
                )
                hashes, history = hasher.step(ids, history, active)
                token_ids.append(ids)
                hash_ids.append(hashes)
                positions.append(
                    np.asarray(
                        [
                            p["positions"][step] if active[row] else [0, 0, 0]
                            for row, p in enumerate(prompts)
                        ]
                    )
                )
                masks.append(active)
            mask = np.stack(masks, 1)
            embeddings = tokens.gather(np.stack(token_ids, 1)[..., None], active=mask)
            features = ngrams.gather(np.stack(hash_ids, 1), active=mask)
            for row, prompt in enumerate(prompts):
                for step in range(first, min(first + window, len(prompt["ids"]))):
                    if step in prompt["visual"]:
                        embeddings[row, step - first] = prompt["visual"][step]
            engine.forward_chunk(
                embeddings,
                features,
                np.stack(positions, 1),
                mask,
            )
            engine.state["hidden"].block_until_ready()
            if memory and first % 64 == 0:
                memory.sample("prefill", batch)
            emit(
                stage="prefill",
                detail=f"batch={batch} step={first + window}",
                prefill_steps=first + window,
            )
        logits = engine.logits()
        logits.block_until_ready()
        metrics["seconds"] = time.perf_counter() - start
    return logits, history, time.perf_counter() - start


def generate(
    engine,
    prompts,
    logits,
    history,
    hasher,
    ngrams,
    tokens,
    tokenizer,
    max_tokens,
    emit,
    memory=None,
    record_cases=True,
    profile_directory=None,
):
    batch, c = len(prompts), engine.config
    if profile_directory is not None:
        # Metadata is written before optional_device_trace enters, and an
        # early EOS writes a skip status without ever entering that context.
        # Both paths therefore need the destination before decoding starts.
        Path(profile_directory).mkdir(parents=True, exist_ok=True)
    output = [[] for _ in prompts]
    active = np.ones(batch, bool)
    limits = np.asarray([min(max_tokens, p.get("max_new_tokens", max_tokens)) for p in prompts])
    reserve = (
        AnswerReserve(tokenizer, prompts, limits, c.generation_eos)
        if any(p["case"].get("answer_reserve_tokens", 0) for p in prompts)
        else None
    )
    exhausted = np.zeros(batch, bool)
    decode_seconds, forward_tokens, allocated_row_steps = 0.0, 0, 0
    host_lookup_seconds = model_step_seconds = selection_seconds = 0.0
    profiled = False
    decode_started = time.perf_counter()
    for step in range(int(limits.max())):
        started = time.perf_counter()
        selected = np.asarray(checked_argmax(logits), np.int64)
        ids, invalid = selected[:, 0], np.where(active, selected[:, 1], 0)
        if invalid.any():
            error = InvalidLogitsError(step, invalid, prompts)
            with tracing.span("invalid_logits") as diagnostic:
                diagnostic.update(status="FAIL", **error.diagnostics)
            raise error
        if reserve is not None:
            ids = reserve.select(ids, active, output)
        for row in np.flatnonzero(active):
            if int(ids[row]) in c.generation_eos:
                active[row] = False
            else:
                output[row].append(int(ids[row]))
                if len(output[row]) >= limits[row]:
                    exhausted[row] = True
                    active[row] = False
        selection_seconds += time.perf_counter() - started
        if not active.any():
            break
        with tracing.span(
            "decode_step", batch=batch, active_rows=int(active.sum()), step=step
        ) as metrics:
            gather_started = time.perf_counter()
            hashes, history = hasher.step(ids, history, active)
            pos = np.asarray([[p["next_position"] + step] * 3 for p in prompts])
            embedding = tokens.gather(ids[:, None], active=active)
            features = ngrams.gather(hashes, active=active)
            metrics["host_lookup_seconds"] = time.perf_counter() - gather_started
            host_lookup_seconds += metrics["host_lookup_seconds"]
            model_started = time.perf_counter()
            # The first warm-up step compiles. Capture only the next step,
            # outside measured benchmarks, including the final vocabulary dot.
            if profile_directory and step == 1:
                atomic_json(
                    Path(profile_directory).parent / "profile-context.json",
                    dict(
                        label=Path(profile_directory).parent.name,
                        batch=batch,
                        active_rows=int(active.sum()),
                        prompt_tokens=[len(p["ids"]) for p in prompts],
                        decode_step=step,
                        scope="synchronized model forward including vocabulary projection",
                        timed_benchmark=False,
                    ),
                )
            trace = (
                optional_device_trace(profile_directory)
                if profile_directory and step == 1
                else nullcontext()
            )
            with trace:
                engine.forward(embedding, features, pos, active)
                logits = engine.logits()
                logits.block_until_ready()
            profiled |= bool(profile_directory and step == 1)
            metrics["model_step_seconds"] = time.perf_counter() - model_started
            model_step_seconds += metrics["model_step_seconds"]
            seconds = time.perf_counter() - started
            decode_seconds += seconds
            forward_tokens += int(active.sum())
            allocated_row_steps += batch
            metrics.update(seconds=seconds, generated_tokens=int(active.sum()))
        if memory and step % 16 == 0:
            memory.sample("decode", batch)
        emit(
            stage="decode",
            detail=f"batch={batch} step={step + 1} active={active.sum()}",
            decode_steps=step + 1,
            decode_tokens=forward_tokens,
            decode_seconds=time.perf_counter() - decode_started,
            ngram_cache_hits=ngrams.hits,
            ngram_cache_misses=ngrams.misses,
        )
    decode_wall_seconds = time.perf_counter() - decode_started
    if profile_directory and not profiled:
        atomic_json(
            Path(profile_directory).parent / "device-trace-status.json",
            dict(status="SKIPPED", reason="Warm-up ended before the second decode step"),
        )
    rows = []
    for row, prompt in enumerate(prompts):
        text = tokenizer.decode(output[row], skip_special_tokens=False)
        answer, phase = final_answer(text)
        passed, detail = match_answer(prompt["case"]["match"], answer)
        asserted = prompt["case"].get("asserted", True)
        rows.append(
            dict(
                name=prompt["case"]["name"],
                request_id=prompt.get("request_id"),
                replay_index=prompt.get("replay_index", 0),
                batch=batch,
                group=prompt["case"].get("group", "unknown"),
                asserted=asserted,
                provenance=prompt["case"].get("provenance"),
                prompt_tokens=len(prompt["ids"]),
                image_grids=prompt.get("image_grids"),
                max_new_tokens=int(limits[row]),
                generated_tokens=len(output[row]),
                reasoning_and_answer=text,
                answer=answer,
                answer_phase=phase,
                passed=(passed and phase == "FINAL" and not exhausted[row]) if asserted else None,
                diagnostic_match=passed if not asserted else None,
                matcher=detail,
                truncated=bool(exhausted[row]),
                reasoning_budget_forced=bool(reserve and reserve.forced[row]),
                final_answer_reserve_tokens=reserve.reserves[row] if reserve else 0,
                reasoning_budget_tokens=reserve.budgets[row] if reserve else None,
                runtime_policy=prompt["case"].get("runtime_policy"),
            )
        )
        if record_cases:
            with tracing.span(
                "case_result",
                case=prompt["case"]["name"],
                batch=batch,
                request_id=prompt.get("request_id"),
            ) as result:
                result.update(
                    passed=rows[-1]["passed"],
                    asserted=asserted,
                    group=rows[-1]["group"],
                    truncated=bool(exhausted[row]),
                    matcher=detail,
                    answer=answer,
                    answer_phase=phase,
                    generated_tokens=len(output[row]),
                    reasoning_budget_forced=rows[-1]["reasoning_budget_forced"],
                    final_answer_reserve_tokens=rows[-1]["final_answer_reserve_tokens"],
                    reasoning_budget_tokens=rows[-1]["reasoning_budget_tokens"],
                )
    if memory:
        memory.sample("decode_complete", batch)
    return rows, dict(
        decode_seconds=decode_wall_seconds,
        decode_active_seconds=decode_seconds,
        host_lookup_seconds=host_lookup_seconds,
        model_step_seconds=model_step_seconds,
        selection_seconds=selection_seconds,
        decode_overhead_seconds=max(
            0.0, decode_wall_seconds - host_lookup_seconds - model_step_seconds - selection_seconds
        ),
        forward_tokens=forward_tokens,
        allocated_row_steps=allocated_row_steps,
        aggregate_tokens_per_second=forward_tokens / decode_wall_seconds
        if decode_wall_seconds
        else None,
        latency_includes_host_ngram_lookup=True,
        latency_includes_tracking_and_progress=True,
        throughput_clock="decode_wall_time",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    validate_batches(config["batch_sizes"], config.get("requests"))
    validate_batches([config.get("evaluation_batch_size", 8)])
    jax.config.update("jax_default_matmul_precision", "highest")
    directory = Path(config["output_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    diagnostics = WorkerDiagnostics(directory)
    report = {
        "status": "RUNNING",
        "model": MODEL_REPO,
        "quant": config["quant"],
        "batch_results": [],
        "batch_trials": [],
        "evaluation_results": [],
    }
    memory = None
    progress.configure(directory / "progress.json")
    tracing.configure(config, directory)

    def emit(*, stage=None, detail=None, **fields):
        if stage is not None:
            fields["stage"] = stage
        if detail is not None:
            fields["detail"] = detail
        progress.update(force=True, **fields)

    def log(item):
        if isinstance(item, dict):
            stage = item.pop("phase", "load")
            emit(stage=stage, **item)
        else:
            emit(detail=str(item))
        print(json.dumps(item) if isinstance(item, dict) else item, flush=True)

    try:
        from jax.sharding import Mesh

        devices = jax.devices()
        if len(devices) != 8 or any(d.platform != "tpu" for d in devices):
            raise RuntimeError("Model trial requires eight TPU devices")
        if any(d.device_kind.lower() not in {"tpu v5 lite", "tpu v5e"} for d in devices):
            raise RuntimeError("Select the v5e-8 Kaggle accelerator for this trial")
        report["device_kinds"] = [d.device_kind for d in devices]
        report["trial_settings"] = {
            key: config.get(key)
            for key in (
                "max_context",
                "kv_cache_dtype",
                "kv_scale_gather",
                "max_new_tokens",
                "evaluation_max_new_tokens",
                "reasoning_effort",
                "expert_compute",
                "expert_parallel",
                "fuse_layers",
                "gdn_prefill",
                "stop_on_evaluation_collapse",
                "deadline_seconds",
                "quality_first",
            )
        }
        preflight = directory / "optimization-preflight.json"
        if preflight.exists():
            report["optimization_preflight"] = json.loads(preflight.read_text())

        evaluation_images, evaluation_cases = {}, []
        if config.get("evaluation_bundle"):
            evaluation_images, evaluation_cases, evaluation_manifest, sha = load_bundle(
                config["evaluation_bundle"], config["evaluation_sha256"]
            )
            report["evaluation_contract"] = dict(
                sha256=sha,
                expected_cases=len(evaluation_cases),
                expected_asserted=sum(case["asserted"] for case in evaluation_cases),
                coverage=evaluation_manifest["coverage"],
            )
            if config.get("quality_first", False):
                evaluation_cases = [quality_case(case) for case in evaluation_cases]
                report["evaluation_contract"]["runtime_policy"] = QUALITY_POLICY
                report["evaluation_contract"]["comparison"] = (
                    "Same source cases/images/matchers; larger budgets, explicit final-output "
                    "instructions and high-resolution OCR. Compare separately with v11."
                )
        mesh = Mesh(ordered_devices(devices), ("expert",))
        c = ModelConfig(
            max_context=config["max_context"],
            kv_cache_dtype=config.get("kv_cache_dtype", "bf16"),
            kv_scale_gather=config.get("kv_scale_gather", "indexed"),
        )
        if config["reasoning_effort"] != "xhigh":
            raise ValueError("This trial requires the highest supported xhigh thinking effort")
        policy = config.get("expert_compute", "bf16")
        if policy not in {"bf16", "int8_q8", "auto_int8"}:
            raise ValueError("Invalid expert compute policy")
        gdn_prefill = config.get("gdn_prefill", "serial")
        if gdn_prefill not in {"serial", "chunked"}:
            raise ValueError("GDN prefill must be serial or chunked")
        long_gate = partial(
            validate_long_context,
            gdn_prefill=gdn_prefill,
            fuse_layers=config.get("fuse_layers", True),
            kv_scale_gather=c.kv_scale_gather,
        )
        parallel_policy = config.get("expert_parallel", "serial")
        if parallel_policy not in {"serial", "auto"}:
            raise ValueError("Invalid expert parallel selection policy")
        if parallel_policy == "auto" and config["quant"] != "q6":
            raise ValueError("Expert calibration currently matches the pinned Q6 checkpoint")
        gates = [
            ("elementary_numeric_gate", lambda m: validate_numerics(mesh, diagnostics=m)),
            ("q8_codec_gate", lambda m: validate_q8_codec(mesh, diagnostics=m)),
            ("numeric_gate", lambda m: validate(mesh, batch_sizes=config["batch_sizes"])),
            (
                "unquantized_long_numeric_gate",
                lambda m: long_gate(mesh, "bf16", diagnostics=m),
            ),
            (
                "long_context_numeric_gate",
                lambda m: long_gate(mesh, c.kv_cache_dtype, diagnostics=m),
            ),
            (
                "bf16_numeric_gate",
                lambda m: long_gate(mesh, "q8", dtype=jnp.bfloat16, diagnostics=m),
            ),
            ("native_compute_proof", lambda m: native_dot_proof()),
            ("expert_routing_gate", lambda m: validate_expert_routing(mesh, diagnostics=m)),
        ]
        if policy in {"auto_int8", "int8_q8"}:
            gates.append(
                (
                    "integer_numeric_gate",
                    lambda m: long_gate(mesh, "q8", expert_compute="int8_q8", diagnostics=m),
                )
            )
        if parallel_policy == "auto":
            for tp in (1, 2):
                gates.append(
                    (
                        f"grouped_tp{tp}_routing_gate",
                        lambda m, tp=tp: validate_expert_routing(
                            mesh, diagnostics=m, group_size=4, tensor_parallel=tp
                        ),
                    )
                )
                for gate_policy, dtype, label in (
                    ("bf16", jnp.float32, "fp32"),
                    ("bf16", jnp.bfloat16, "bf16"),
                    *(
                        [("int8_q8", jnp.float32, "int8")]
                        if policy in {"auto_int8", "int8_q8"}
                        else []
                    ),
                ):
                    gates.append(
                        (
                            f"grouped_tp{tp}_{label}_gate",
                            lambda m, tp=tp, p=gate_policy, d=dtype: long_gate(
                                mesh,
                                "q8",
                                expert_compute=p,
                                dtype=d,
                                diagnostics=m,
                                expert_group_size=4,
                                expert_tensor_parallel=tp,
                            ),
                        )
                    )
        # Independent tiny gates all finish and persist evidence even if one
        # rejects the implementation. Full weights require every gate PASS.
        run_gates(gates, report, directory, emit)
        if policy == "auto_int8":
            with tracing.span("compute_calibration") as metrics:
                batches = list(config["batch_sizes"])
                if evaluation_cases:
                    evaluation_batch = config.get("evaluation_batch_size", 8)
                    batches.append(evaluation_batch)
                    if len(evaluation_cases) % evaluation_batch:
                        batches.append(len(evaluation_cases) % evaluation_batch)
                rows = batches + [min(b * prefill_window(b), EXPERT_ROW_TILE) for b in batches]
                metrics.update(choose_policy(rows, emit))
                report["compute_calibration"] = dict(metrics)
                policy = metrics["selected"]
        if policy not in {"bf16", "int8_q8"}:
            raise ValueError("Invalid expert compute policy")
        expert_options = dict(group_size=1, tensor_parallel=1, prefill_group_size=1)
        if parallel_policy == "auto":
            emit(stage="expert_calibration")
            with tracing.span("expert_calibration") as metrics:
                calibration_batches = list(config["batch_sizes"])
                if evaluation_cases:
                    eb = config.get("evaluation_batch_size", 8)
                    calibration_batches.append(eb)
                    if len(evaluation_cases) % eb:
                        calibration_batches.append(len(evaluation_cases) % eb)
                calibration_rows = sorted(
                    set(calibration_batches + [b * prefill_window(b) for b in calibration_batches])
                )
                metrics.update(
                    calibrate_experts(
                        mesh,
                        c,
                        compute=policy,
                        rows=calibration_rows,
                        decode_rows=calibration_batches,
                        emit=emit,
                    )
                )
                report["expert_calibration"] = dict(metrics)
                atomic_json(directory / "report.json", report)
                if metrics["status"] != "PASS":
                    raise RuntimeError(f"Expert calibration rejected: {metrics['selection_error']}")
                expert_options = {key: metrics["selected"][key] for key in expert_options}
            atomic_json(directory / "report.json", report)
        if parallel_policy == "auto":
            # Validate the actual selected placement and phase composition,
            # including new group sizes, against independent tiny references
            # on this TPU/compiler before admitting the full checkpoint.
            dtypes = (jnp.float32, jnp.bfloat16) if policy == "bf16" else (jnp.float32,)
            selected_gates = [
                (
                    f"selected_group{group}_routing_gate",
                    lambda m, g=group: validate_expert_routing(
                        mesh,
                        diagnostics=m,
                        group_size=g,
                        tensor_parallel=expert_options["tensor_parallel"],
                    ),
                )
                for group in sorted(
                    {expert_options["group_size"], expert_options["prefill_group_size"]}
                )
            ]
            run_gates(
                selected_gates
                + [
                    (
                        f"selected_phase_{jnp.dtype(dtype).name}_gate",
                        lambda m, d=dtype: long_gate(
                            mesh,
                            "q8",
                            expert_compute=policy,
                            dtype=d,
                            diagnostics=m,
                            expert_group_size=expert_options["group_size"],
                            expert_tensor_parallel=expert_options["tensor_parallel"],
                            prefill_expert_group_size=expert_options["prefill_group_size"],
                        ),
                    )
                    for dtype in dtypes
                ],
                report,
                directory,
                emit,
            )
        report["compute_policy"] = dict(
            experts=policy,
            projections="bf16_fp32_accumulator",
            attention="bf16_fp32_accumulator",
            kv_cache=c.kv_cache_dtype,
            kv_scale_gather=c.kv_scale_gather,
            router_and_index_scores="fp32",
            recurrent_state="fp32",
            expert_row_tile=EXPERT_ROW_TILE,
            prefill_window=PREFILL_WINDOW,
            prefill_rows=PREFILL_ROWS,
            gdn_prefill=gdn_prefill,
            state_buffers="donated_with_fixed_sharding",
            expert_parallel_size=8 // expert_options["tensor_parallel"],
            expert_tensor_parallel_size=expert_options["tensor_parallel"],
            expert_group_size=expert_options["group_size"],
            prefill_expert_group_size=expert_options["prefill_group_size"],
            layer_execution="fused" if config.get("fuse_layers", True) else "separate",
        )
        checkpoint = Checkpoint(
            Path(__file__).with_name("checkpoints") / (config["quant"] + ".json"),
            local_dir=config.get("checkpoint_dir"),
            log=log,
        )
        report["checkpoint_schema_tensors"] = checkpoint.validate(c)

        def relay_memory(snapshot):
            if snapshot["counters_available"]:
                emit(
                    hbm_bytes_in_use=snapshot["total_bytes_in_use"],
                    hbm_minimum_free_bytes_per_chip=snapshot["minimum_free_bytes"],
                )

        memory = MemoryTracker(devices, directory / "memory.jsonl", observer=relay_memory)
        capacity = sum(d.memory_stats()["bytes_limit"] for d in devices)
        plans = {
            batch: checkpoint.memory_plan(c, batch, hbm_bytes=capacity)
            for batch in config["batch_sizes"]
        }
        plan = plans[min(config["batch_sizes"])]
        report["hbm_plan"] = plan
        report["hbm_plans"] = plans
        atomic_json(directory / "report.json", report)
        if not plan["fits"]:
            raise MemoryError("Quantization exceeds resident HBM budget")
        # Host allocation is explicit and checked against the cgroup-aware
        # value, rather than Google's physical host RAM specification.
        from glm53_offload.weights import available_ram

        available = available_ram()
        needed = plan["host_ngram_bytes"] + plan["host_token_embedding_bytes"] + (16 << 30)
        if available < needed:
            raise MemoryError("Not enough available host RAM for packed n-grams + staging reserve")
        from transformers import AutoTokenizer

        emit(stage="tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_REPO, revision=MODEL_REVISION, trust_remote_code=False
        )
        mesh = Mesh(ordered_devices(devices), ("expert",))
        images = probe_images.fixtures()
        selected = {"colour_red", "count_circles_7", "count_circles_3", "read_text_stop_7"}
        cases = [case for case in quality.cases() if case["name"] in selected]
        if config.get("beamng_fixture_bundle"):
            real_images, real_cases, _, _ = load_scene_bundle(
                config["beamng_fixture_bundle"], config["beamng_fixture_sha256"]
            )
            images.update(real_images)
            cases.extend(real_cases)
        images.update(evaluation_images)
        if config.get("quality_first", False):
            # Replay requests must use the same input/completion policy as
            # evaluation, rather than silently reverting to low-resolution OCR.
            cases = [quality_case(case) for case in cases]
        # Tokenize every requested case before loading vision or full model weights.
        preflight_cases = evaluation_cases or cases
        report["token_preflight"] = []
        for case in preflight_cases:
            planned = prepare_prompt(
                tokenizer,
                case,
                images,
                None,
                c,
                max_new_tokens=config["max_new_tokens"],
                effort=config["reasoning_effort"],
            )
            report["token_preflight"].append(
                dict(
                    name=case["name"],
                    prompt_tokens=len(planned["ids"]),
                    max_new_tokens=planned["max_new_tokens"],
                    image_grids=planned["image_grids"],
                    max_pixels=case.get("max_pixels", 589824),
                    runtime_policy=case.get("runtime_policy"),
                )
            )
        atomic_json(directory / "report.json", report)
        emit(stage="vision_load")
        with tracing.span("vision_load"):
            encoder = VisionEncoder(load_vision(checkpoint, mesh))
        vision_cache = {}
        prompts = []
        for case in cases:
            emit(stage="vision_encode", detail=case["name"])
            with tracing.span("vision_encode", case=case["name"]) as metrics:
                prompt = prepare_prompt(
                    tokenizer,
                    case,
                    images,
                    encoder,
                    c,
                    max_new_tokens=config["max_new_tokens"],
                    effort=config["reasoning_effort"],
                    vision_cache=vision_cache,
                )
                metrics["prompt_tokens"] = len(prompt["ids"])
                prompts.append(prompt)
        evaluation_prompts = []
        for case in evaluation_cases:
            emit(stage="evaluation_vision_encode", detail=case["name"])
            with tracing.span("vision_encode", case=case["name"]):
                prompt = prepare_prompt(
                    tokenizer,
                    case,
                    images,
                    encoder,
                    c,
                    max_new_tokens=config["max_new_tokens"],
                    effort=config["reasoning_effort"],
                    vision_cache=vision_cache,
                )
                evaluation_prompts.append(
                    {**prompt, "request_id": f"evaluation-{case['name']}", "replay_index": 0}
                )
        prompts = replay_workload(prompts, config.get("requests"), config["batch_sizes"])
        report["workload"] = dict(
            requests_per_batch_size=len(prompts),
            unique_labeled_cases=len(cases),
            repeats_per_case=len(prompts) // len(cases),
            kind="labeled_cohort_replay" if len(prompts) > len(cases) else "labeled_cohort",
            limitation=(
                "Repeated images share prompts and expert routes; "
                "throughput is not a diverse-production estimate"
            ),
            preparation=(
                "Unique images encoded once; all measured requests reuse prepared vision features"
            ),
        )
        emit(stage="ngram_ram_load")
        with tracing.span("ngram_ram_load"):
            ngrams, hasher = checkpoint.host_ngram(c)
            tokens = token_table(checkpoint, c)
        emit(stage="model_load")
        with tracing.span("model_load"):
            engine = Engine.from_checkpoint(
                checkpoint,
                batch=min(config["batch_sizes"]),
                config=c,
                log=log,
                expert_compute=policy,
                expert_group_size=expert_options["group_size"],
                expert_tensor_parallel=expert_options["tensor_parallel"],
                fuse_layers=config.get("fuse_layers", True),
                gdn_prefill=gdn_prefill,
                prefill_expert_group_size=expert_options["prefill_group_size"],
            )
        jax.block_until_ready((engine.layers, engine.experts, engine.head, engine.final_weights))
        memory.sample("resident_model_loaded")
        report["placement"] = {
            "ngram": "packed_host_ram",
            "token_embedding": "packed_host_ram",
            "experts": "resident_expert_parallel",
            "dense": "bf16_sharded",
            "vision": "native_jax",
            "thinking": True,
            "reasoning_effort": config["reasoning_effort"],
            "kv_cache": c.kv_cache_dtype,
            "expert_compute": policy,
            "expert_parallel_size": 8 // expert_options["tensor_parallel"],
            "expert_tensor_parallel_size": expert_options["tensor_parallel"],
            "expert_group_size": expert_options["group_size"],
            "prefill_expert_group_size": expert_options["prefill_group_size"],
            "gdn_prefill": gdn_prefill,
        }
        if evaluation_prompts:
            evaluation_batch = config.get("evaluation_batch_size", 8)
            engine.release_state()
            admission = memory.admit(
                evaluation_batch, checkpoint.memory_plan(c, evaluation_batch, hbm_bytes=capacity)
            )
            if not admission["admitted"]:
                raise MemoryError("Full evaluation batch denied by per-chip HBM gate")
            baseline_index = len(memory.records) - 1
            for first in range(0, len(evaluation_prompts), evaluation_batch):
                chunk = evaluation_prompts[first : first + evaluation_batch]
                emit(stage="evaluation_batch", detail=f"cases {first + 1}..{first + len(chunk)}")
                logits, history, seconds = prefill(
                    engine, chunk, hasher, ngrams, tokens, emit, memory
                )
                rows, metrics = generate(
                    engine,
                    chunk,
                    logits,
                    history,
                    hasher,
                    ngrams,
                    tokens,
                    tokenizer,
                    config.get("evaluation_max_new_tokens", 4096),
                    emit,
                    memory,
                    profile_directory=(directory / "profiles" / "evaluation-first" / "device-trace")
                    if config.get("profile_decode", False) and first == 0
                    else None,
                )
                del logits, history
                report["evaluation_results"].append(
                    dict(batch=len(chunk), prefill_seconds=seconds, cases=rows, **metrics)
                )
                update_summaries(report)
                report["memory"] = memory.summary()
                atomic_json(directory / "report.json", report)
                if first == 0:
                    sanity = evaluation_sanity(rows)
                    report["evaluation_sanity"] = sanity
                    if (
                        config.get("stop_on_evaluation_collapse", True)
                        and sanity["status"] == "FAIL"
                    ):
                        report.update(
                            status="QUALITY_FAILED",
                            evaluation_coverage="PARTIAL",
                            benchmark_coverage="SKIPPED",
                            stopped_after="first_evaluation_cohort",
                        )
                        emit(stage="evaluation_rejected", detail=sanity["reason"])
                        engine.release_state()
                        return
            memory.complete(evaluation_batch, baseline_index)
            engine.release_state()
            report["evaluation_coverage"] = "COMPLETE"
            atomic_json(directory / "report.json", report)
        for batch in config["batch_sizes"]:
            engine.release_state()
            with tracing.span("batch_memory_admission", batch=batch) as metrics:
                admission = memory.admit(batch, plans[batch])
                metrics.update(admission)
            baseline_index = len(memory.records) - 1
            trial = dict(
                batch=batch,
                status="RUNNING" if admission["admitted"] else "SKIPPED",
                memory_admission=admission,
            )
            report["batch_trials"].append(trial)
            report["memory"] = memory.summary()
            atomic_json(directory / "report.json", report)
            if not admission["admitted"]:
                emit(stage="batch_skipped", detail=f"batch={batch} {admission['reason']}")
                continue
            emit(stage="batch_warmup", detail=f"batch={batch}")
            # A small prefill/decode warms each batch shape before measurements.
            with tracing.span("batch_warmup", batch=batch):
                prototype = prompts[0]
                warm = [warmup_prompt(prototype)] * batch
                logits, history, _ = prefill(engine, warm, hasher, ngrams, tokens, emit, memory)
                generate(
                    engine,
                    warm,
                    logits,
                    history,
                    hasher,
                    ngrams,
                    tokens,
                    tokenizer,
                    3,
                    emit,
                    memory,
                    record_cases=False,
                    profile_directory=(directory / "profiles" / f"batch-{batch}" / "device-trace")
                    if config.get("profile_decode", False)
                    else None,
                )
                del logits, history
            for first in range(0, len(prompts), batch):
                chunk = prompts[first : first + batch]
                if len(chunk) != batch:
                    raise ValueError("Measured batch must match requested batch exactly")
                logits, history, prefill_s = prefill(
                    engine, chunk, hasher, ngrams, tokens, emit, memory
                )
                rows, metrics = generate(
                    engine,
                    chunk,
                    logits,
                    history,
                    hasher,
                    ngrams,
                    tokens,
                    tokenizer,
                    config["max_new_tokens"],
                    emit,
                    memory,
                )
                del logits, history
                report["batch_results"].append(
                    dict(
                        batch=len(chunk),
                        requested_batch=batch,
                        prefill_seconds=prefill_s,
                        prompt_tokens=sum(len(p["ids"]) for p in chunk),
                        **metrics,
                        cases=rows,
                    )
                )
                update_summaries(report)
                report["memory"] = memory.summary()
                atomic_json(directory / "report.json", report)
            memory.complete(batch, baseline_index)
            trial["status"] = "COMPLETE"
            engine.release_state()
            memory.sample("batch_released")
            report["memory"] = memory.summary()
            atomic_json(directory / "report.json", report)
        report["benchmark_coverage"] = (
            "COMPLETE"
            if all(trial["status"] == "COMPLETE" for trial in report["batch_trials"])
            else "PARTIAL"
        )
        if not report["batch_results"]:
            raise MemoryError("No requested batch admitted by measured per-chip memory gate")
        report["status"] = "PASS" if not report["quality"]["failed_cases"] else "QUALITY_FAILED"
    except BaseException as error:
        report.update(
            status="ERROR", error_type=type(error).__name__, error_message=str(error)[:6000]
        )
        if isinstance(error, InvalidLogitsError):
            report["invalid_logits"] = error.diagnostics
        (directory / "worker-traceback.txt").write_text(traceback.format_exc())
        raise
    finally:
        update_summaries(report)
        if memory:
            report["memory"] = memory.summary()
        atomic_json(directory / "report.json", report)
        emit(stage="complete", detail=report["status"])
        tracing.finish(report["status"])
        diagnostics.close()


if __name__ == "__main__":
    main()
