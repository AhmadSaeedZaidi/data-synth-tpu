"""Check the comparison against the official SDK's actual JSON transport."""

import importlib.util
import json
import time
from pathlib import Path

import httpx
import pytest
from openai import OpenAI

SPEC = importlib.util.spec_from_file_location(
    "nim_comparison",
    Path(__file__).resolve().parents[1] / "scripts/compare_glm53_deepseek41_nim.py",
)
comparison = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparison)

CASE = dict(
    name="count_circles_7",
    group="counting",
    images=["fixture"],
    prompt="How many circles? Answer with a single number.",
    match={"kind": "first_int", "value": 7},
    asserted=True,
)


def response(answer="7", reasoning="The wrong answer would be 8.", finish="stop"):
    return dict(
        id="chatcmpl-test",
        object="chat.completion",
        created=1,
        model=comparison.MODELS[0],
        choices=[
            dict(
                index=0,
                message=dict(role="assistant", content=answer, reasoning_content=reasoning),
                finish_reason=finish,
            )
        ],
        usage=dict(prompt_tokens=100, completion_tokens=10, total_tokens=110),
    )


@pytest.mark.parametrize("model", comparison.MODELS)
def test_working_profile_uses_same_image_and_low_effort_through_sdk(tmp_path, model):
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        assert str(request.url) == "https://example.invalid/v1/chat/completions"
        return httpx.Response(200, json=response())

    payload = comparison.payload_for(model, CASE, {"fixture": b"\x89PNG-original"})
    with OpenAI(
        api_key="test-key",
        base_url="https://example.invalid/v1",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    ) as client:
        result = comparison.run_case(
            client, payload, CASE, tmp_path / "result.json", 600, time.perf_counter() + 900
        )
    assert seen == [payload]
    assert payload["reasoning_effort"] == "low"
    assert payload["max_tokens"] == 8192
    assert payload["stream"] is False
    assert result["status"] == "PASS"
    assert result["answer"] == "7"
    assert result["reasoning_characters"] > 0


@pytest.mark.parametrize(
    ("answer", "finish", "status"),
    [
        ("8", "stop", "FAIL"),
        ("7", "length", "INCOMPLETE"),
        ("", "stop", "API_ERROR"),
        ("<think>8</think>7", "stop", "PASS"),
        ("<think>7", "stop", "API_ERROR"),
    ],
)
def test_incomplete_or_hidden_reasoning_cannot_be_scored_as_answer(
    tmp_path, answer, finish, status
):
    with OpenAI(
        api_key="test-key",
        base_url="https://example.invalid/v1",
        max_retries=0,
        http_client=httpx.Client(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(200, json=response(answer=answer, finish=finish))
            )
        ),
    ) as client:
        result = comparison.run_case(
            client,
            comparison.payload_for(comparison.MODELS[0], CASE, {"fixture": b"\x89PNG-original"}),
            CASE,
            tmp_path / "result.json",
            600,
            time.perf_counter() + 900,
        )
    assert result["status"] == status


def test_http_failure_retains_status_without_sdk_retry_or_error_body(tmp_path):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(502, json={"error": {"message": "private-request-body"}})

    with OpenAI(
        api_key="test-key",
        base_url="https://example.invalid/v1",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    ) as client:
        result = comparison.run_case(
            client,
            comparison.payload_for(comparison.MODELS[0], CASE, {"fixture": b"\x89PNG-original"}),
            CASE,
            tmp_path / "result.json",
            600,
            time.perf_counter() + 900,
        )
    assert len(seen) == 1
    assert result["status"] == "API_ERROR"
    assert result["http_status"] == 502
    assert "private-request-body" not in json.dumps(result)


def test_paired_vision_metrics_exclude_repeat_and_unpaired_endpoint_errors(tmp_path):
    cases = [
        dict(CASE, name="paired"),
        dict(CASE, name="unpaired"),
        dict(CASE, name="repeat", group="repeat_cost"),
        dict(CASE, name="text", group="text_control", images=[]),
    ]
    for model in comparison.MODELS:
        for case in cases:
            status = "PASS"
            if case["name"] == "paired" and model == comparison.MODELS[1]:
                status = "FAIL"
            if case["name"] == "unpaired" and model == comparison.MODELS[1]:
                status = "API_ERROR"
            comparison.save(
                tmp_path / "results" / model.split("/")[0] / (case["name"] + ".json"),
                dict(
                    case=case["name"],
                    status=status,
                    wall_seconds=10 if case["name"] == "paired" else 1000,
                ),
            )
    report = comparison.summarize(tmp_path, cases)
    vision = report["paired_unique_vision"]
    assert report["status"] == "PARTIAL"
    assert vision["planned"] == 2
    assert vision["complete"] == 1
    assert vision["cases"] == ["paired"]
    assert vision["models"][comparison.MODELS[0]] == dict(passed=1, median_request_seconds=10)
    assert vision["models"][comparison.MODELS[1]] == dict(passed=0, median_request_seconds=10)
