"""Bounded NIM comparison with unchanged Kaggle vision fixtures and text controls.

Use --dry-run to validate fixtures without calling APIs. Live runs use at most two
requests concurrently, no automatic retries, and stop each model after two
consecutive endpoint errors. Finished cases resume from disk, while the repeated
red-image case makes a separate request so local caching cannot affect latency.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kaggle"))

from glm53_offload import probe_images, quality  # noqa: E402
from glm53_offload.scene_fixtures import load_scene_bundle  # noqa: E402

from data_synth.config import Settings  # noqa: E402

MODELS = ("z-ai/glm-5.3-flash", "deepseek-ai/deepseek-v4.1-flash")
BUNDLE_SHA = "129352904eed775b2ea9820245cb1613b36d67ccf5ed31b75062e7f342629840"
BUNDLE = ROOT / "output/current/inputs/qwen38-beamng-fixtures.bundle"
COMPLETE = {"PASS", "FAIL", "CONTROL"}


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def fixtures():
    images = probe_images.fixtures()
    cases = quality.cases(images)
    real_images, real_cases, manifest, digest = load_scene_bundle(BUNDLE, BUNDLE_SHA)
    images.update(real_images)
    road = next(c for c in real_cases if c["name"] == "beamng_city_road_text")
    cases = [cases[0], road, *cases[1:], *[c for c in real_cases if c != road]]
    for name, prompt, expected in (
        (
            "collision_time",
            "A lead car is 25 metres ahead. Our closing speed is 5 metres per second "
            "and remains constant. How many seconds until collision?",
            5,
        ),
        (
            "braking_distance",
            "A car travels at 20 metres per second and brakes with constant deceleration "
            "of 5 metres per second squared. Ignore reaction time. "
            "What is the stopping distance in metres?",
            40,
        ),
        (
            "telemetry_order",
            "Three telemetry samples have timestamps in milliseconds and speeds "
            "in metres per second: (300, 12), (100, 8), (200, 10). "
            "What is the speed in the chronologically latest sample?",
            12,
        ),
        (
            "frame_sampling",
            "Exactly 20 seconds of video contain 30 frames per second. Keep every tenth "
            "frame starting with frame zero, excluding a frame at the 20-second boundary. "
            "How many frames are kept?",
            60,
        ),
    ):
        cases.append(
            dict(
                name="text_" + name,
                group="text_control",
                images=[],
                prompt=prompt + " Answer with a single number.",
                match={"kind": "first_int", "value": expected},
                asserted=True,
            )
        )
    return images, cases, manifest, digest


def payload_for(model, case, images, effort="low", max_tokens=8192):
    content = [{"type": "text", "text": case["prompt"]}]
    for name in case["images"]:
        data = images[name]
        mime = "image/png" if data.startswith(b"\x89PNG") else "image/jpeg"
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:" + mime + ";base64," + base64.b64encode(data).decode("ascii")
                },
            }
        )
    payload = dict(
        model=model,
        messages=[{"role": "user", "content": content}],
        temperature=0,
        top_p=1,
        max_tokens=max_tokens,
        stream=False,
        reasoning_effort=effort,
    )
    # These controls match the working Windows teacher-comparison examples.
    # Accepted effort names do not imply equal internal budgets across models.
    return payload


def run_case(client, payload, case, path, timeout, deadline):
    start = time.perf_counter()
    result = dict(
        case=case["name"],
        group=case["group"],
        asserted=case["asserted"],
        requested_model=payload["model"],
        started_at_utc=now(),
        status="API_ERROR",
        request_sha256=fingerprint(payload),
    )
    save(path.with_name(path.stem + "-request.json"), payload)
    try:
        remaining = max(1, min(timeout, deadline - start))
        completion = client.chat.completions.create(**payload, timeout=remaining)
        response = completion.model_dump(mode="json")
        save(path.with_name(path.stem + "-response.json"), response)
        choice = response["choices"][0]
        message = choice["message"]
        answer = message.get("content") or ""
        reasoning = message.get("reasoning_content") or ""
        if "</think>" in answer:
            head, answer = answer.split("</think>", 1)
            reasoning += head.split("<think>", 1)[-1]
        result.update(
            http_status=200,
            response_model=response.get("model"),
            request_id=getattr(completion, "_request_id", None),
            answer=answer.strip(),
            reasoning_characters=len(reasoning),
            usage=response.get("usage"),
            finish_reason=choice.get("finish_reason"),
        )
        if result["finish_reason"] == "length":
            result["status"] = "INCOMPLETE"
        elif (
            result["finish_reason"] != "stop"
            or not answer.strip()
            or message.get("refusal")
            or message.get("tool_calls")
            or "<think>" in answer
        ):
            result["error"] = "No complete nonempty final answer"
        elif not case["asserted"]:
            result["status"] = "CONTROL"
        else:
            passed, detail = quality.match(case["match"], answer.strip())
            result.update(status="PASS" if passed else "FAIL", match_detail=detail)
    except (
        APIConnectionError,
        APIStatusError,
        APITimeoutError,
        ValueError,
        KeyError,
        TypeError,
        IndexError,
    ) as error:
        result["error_type"] = type(error).__name__
        status = getattr(error, "status_code", None)
        if isinstance(status, int) and 100 <= status <= 599:
            result["http_status"] = status
    finally:
        result.update(wall_seconds=time.perf_counter() - start, completed_at_utc=now())
        save(path, result)
    return result


def run_model(model, cases, images, output, endpoint, key, timeout, deadline, effort, max_tokens):
    errors = 0
    with OpenAI(api_key=key, base_url=endpoint, timeout=timeout, max_retries=0) as client:
        for case in cases:
            if time.perf_counter() >= deadline:
                break
            path = output / "results" / model.split("/")[0] / (case["name"] + ".json")
            payload = payload_for(model, case, images, effort, max_tokens)
            if path.exists():
                existing = json.loads(path.read_text())
                if existing.get("request_sha256") != fingerprint(payload):
                    raise ValueError("Saved request differs; choose a new output directory")
                if existing["status"] in COMPLETE:
                    continue
            result = run_case(client, payload, case, path, timeout, deadline)
            print(
                json.dumps(
                    {k: result[k] for k in ("requested_model", "case", "status", "wall_seconds")}
                ),
                flush=True,
            )
            errors = errors + 1 if result["status"] == "API_ERROR" else 0
            if errors >= 2:
                print(model + ": stopped after two consecutive endpoint errors", flush=True)
                break


def summarize(output, cases):
    summaries, by_model = {}, {}
    for model in MODELS:
        records = {}
        for case in cases:
            path = output / "results" / model.split("/")[0] / (case["name"] + ".json")
            if path.exists():
                records[case["name"]] = json.loads(path.read_text())
        by_model[model] = records
        complete = [r for r in records.values() if r["status"] in COMPLETE]
        groups = {}
        for label, selected in (
            (
                "synthetic_vision",
                [c for c in cases if c["images"] and not c["name"].startswith("beamng_")],
            ),
            ("beamng_vision", [c for c in cases if c["name"].startswith("beamng_")]),
            ("text_controls", [c for c in cases if c["group"] == "text_control"]),
        ):
            names = {c["name"] for c in selected}
            completed = [r for r in complete if r["case"] in names]
            groups[label] = dict(
                planned=len(selected),
                complete=len(completed),
                passed=sum(r["status"] == "PASS" for r in completed),
            )
        summaries[model] = dict(
            groups=groups,
            attempted=len(records),
            api_errors=sum(r["status"] == "API_ERROR" for r in records.values()),
            incomplete=sum(r["status"] == "INCOMPLETE" for r in records.values()),
            median_completed_request_seconds=statistics.median(r["wall_seconds"] for r in complete)
            if complete
            else None,
            reasoning_observed_cases=sum(r.get("reasoning_characters", 0) > 0 for r in complete),
        )
    paired = [
        c["name"]
        for c in cases
        if c["asserted"]
        and all(by_model[m].get(c["name"], {}).get("status") in {"PASS", "FAIL"} for m in MODELS)
    ]
    unique_vision = [
        c["name"] for c in cases if c["asserted"] and c["images"] and c["group"] != "repeat_cost"
    ]
    paired_vision = [name for name in unique_vision if name in paired]
    vision_metrics = {
        model: dict(
            passed=sum(by_model[model][name]["status"] == "PASS" for name in paired_vision),
            median_request_seconds=statistics.median(
                by_model[model][name]["wall_seconds"] for name in paired_vision
            )
            if paired_vision
            else None,
        )
        for model in MODELS
    }
    status = "COMPLETE" if len(paired) == sum(c["asserted"] for c in cases) else "PARTIAL"
    if not paired and all(
        s["api_errors"] > 0 and s["median_completed_request_seconds"] is None
        for s in summaries.values()
    ):
        status = "ENDPOINT_BLOCKED"
    report = dict(
        status=status,
        updated_at_utc=now(),
        paired_complete_asserted_cases=paired,
        paired_unique_vision=dict(
            planned=len(unique_vision),
            complete=len(paired_vision),
            cases=paired_vision,
            models=vision_metrics,
        ),
        models=summaries,
        limits=[
            "One observation per case: application smoke comparison, "
            "not a general capability benchmark.",
            "The repeated red-image request is not an independent quality example.",
            "Gate scores use the existing notebook matchers, not literal format validation; "
            "first_int can reject a correct final count preceded by another integer.",
            "Both models receive the same requested effort name; "
            "internal thinking budgets may differ.",
            "Image bytes and prompts match; hosted preprocessing and hardware may differ.",
            "Latency includes service queue, networking, image prefill, "
            "reasoning and final answer.",
            "NIM latency does not predict Kaggle TPU decode speed.",
        ],
    )
    save(output / "comparison.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "output/glm53-deepseek41-api-comparison-20261008/sdk-low-20261008",
    )
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--budget-seconds", type=float, default=3600)
    parser.add_argument("--effort", choices=("low", "high", "max"), default="low")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.timeout <= 600 or not 1 <= args.budget_seconds <= 7200:
        parser.error("Transport timeout must be 1..600 seconds and run budget 1..7200 seconds")
    if not 256 <= args.max_tokens <= 16384:
        parser.error("Output token cap must be 256..16384")
    images, cases, manifest, digest = fixtures()
    assets = args.output / "fixtures"
    assets.mkdir(parents=True, exist_ok=True)
    for name, data in images.items():
        (assets / (name + (".png" if data.startswith(b"\x89PNG") else ".jpg"))).write_bytes(data)
    save(
        args.output / "plan.json",
        dict(
            models=MODELS,
            fixture_bundle_sha256=digest,
            cases=cases,
            beamng_manifest=manifest,
            asset_sha256={name: hashlib.sha256(data).hexdigest() for name, data in images.items()},
            parameters={
                m: {
                    k: v
                    for k, v in payload_for(
                        m, cases[0], images, args.effort, args.max_tokens
                    ).items()
                    if k != "messages"
                }
                for m in MODELS
            },
            timeout_seconds=args.timeout,
            budget_seconds=args.budget_seconds,
            concurrency=2,
            automatic_retries=0,
            transport="official OpenAI SDK, non-streaming",
            consecutive_api_errors_before_stop=2,
        ),
    )
    print(
        f"Validated {len(cases)} cases ({sum(c['asserted'] for c in cases)} asserted), "
        f"{len(images)} unchanged images.",
        flush=True,
    )
    if args.dry_run:
        return
    settings = Settings.load(ROOT / ".env")
    if (
        settings.nvidia_base_url != "https://integrate.api.nvidia.com/v1"
        or not settings.nvidia_api_key
    ):
        raise ValueError("Comparison requires the configured NVIDIA NIM endpoint and key")
    deadline = time.perf_counter() + args.budget_seconds
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = {
            pool.submit(
                run_model,
                m,
                cases,
                images,
                args.output,
                settings.nvidia_base_url,
                settings.nvidia_api_key.get_secret_value(),
                args.timeout,
                deadline,
                args.effort,
                args.max_tokens,
            )
            for m in MODELS
        }
        while pending:
            completed, pending = wait(pending, timeout=20)
            for future in completed:
                future.result()
            report = summarize(args.output, cases)
            if pending:
                print(
                    json.dumps(
                        dict(
                            heartbeat_utc=now(),
                            active_models=len(pending),
                            paired_complete=len(report["paired_complete_asserted_cases"]),
                        )
                    ),
                    flush=True,
                )
    print(json.dumps(summarize(args.output, cases), indent=2), flush=True)


if __name__ == "__main__":
    main()
