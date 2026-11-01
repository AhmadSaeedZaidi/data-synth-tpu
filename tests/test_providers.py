import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest
from openai import OpenAI

from data_synth.cache import CacheCorruptionError, ResponseCache, image_to_data_url
from data_synth.providers import (
    DEFAULT_ENDPOINT,
    ModelRequest,
    OpenAIChatProvider,
    ProviderError,
)


def request(**overrides):
    values = {
        "provider": "NVIDIA",
        "endpoint": DEFAULT_ENDPOINT + "/",
        "model": "z-ai/glm-5.3",
        "prompt_version": "traffic-v1",
        "messages": [{"role": "user", "content": "Describe this scene."}],
        "parameters": {"temperature": 0, "max_tokens": 32},
    }
    values.update(overrides)
    return ModelRequest(**values)


def completion_json(content="a clear road"):
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1,
        "model": "z-ai/glm-5.3",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }


def sdk_client(handler):
    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=transport, base_url=DEFAULT_ENDPOINT)
    return OpenAI(
        api_key="fake-token",
        base_url=DEFAULT_ENDPOINT,
        http_client=http_client,
        max_retries=0,
    )


def test_openai_chat_payload_and_auth_are_sent_to_configured_endpoint(tmp_path):
    seen = {}

    def handler(http_request):
        seen["path"] = str(http_request.url)
        seen["auth"] = http_request.headers.get("authorization")
        seen["body"] = json.loads(http_request.content)
        return httpx.Response(200, json=completion_json())

    provider = OpenAIChatProvider(
        api_key=None,
        cache=ResponseCache(tmp_path),
        client=sdk_client(handler),
    )
    result = provider.complete(request())

    assert result.cache_hit is False
    assert seen["path"].endswith("/v1/chat/completions")
    assert seen["auth"] == "Bearer fake-token"
    assert seen["body"]["stream"] is False
    assert seen["body"]["model"] == "z-ai/glm-5.3"
    assert seen["body"]["messages"][0]["content"] == "Describe this scene."
    assert seen["body"]["max_tokens"] == 32


def test_multimodal_message_is_preserved(tmp_path):
    seen = {}

    def handler(http_request):
        seen["body"] = json.loads(http_request.content)
        return httpx.Response(200, json=completion_json("looks normal"))

    multimodal = request(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is visible?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
                ],
            }
        ],
        input_assets=[{"id": "frame-1", "sha256": "a" * 64}],
    )
    OpenAIChatProvider(
        api_key=None,
        cache=ResponseCache(tmp_path),
        client=sdk_client(handler),
    ).complete(multimodal)
    assert seen["body"]["messages"] == multimodal.messages


def test_cache_hit_replays_without_key_or_network(tmp_path):
    calls = 0

    def handler(http_request):
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=completion_json())

    cache = ResponseCache(tmp_path)
    first = OpenAIChatProvider(
        api_key=None,
        cache=cache,
        client=sdk_client(handler),
    ).complete(request())
    second = OpenAIChatProvider(api_key=None, cache=cache).complete(request())
    assert first.response == second.response
    assert second.cache_hit is True
    assert calls == 1


def test_cache_key_changes_with_request_content_but_not_api_key(tmp_path):
    cache = ResponseCache(tmp_path)
    first_request = request()
    changed_request = request(messages=[{"role": "user", "content": "A different scene."}])
    assert cache.request_hash(first_request) != cache.request_hash(changed_request)

    calls = 0

    def handler(http_request):
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=completion_json())

    provider = OpenAIChatProvider(api_key="first-key", cache=cache, client=sdk_client(handler))
    provider.complete(first_request)
    OpenAIChatProvider(api_key="second-key", cache=cache).complete(first_request)
    assert calls == 1


def test_provider_errors_are_sanitized_and_not_cached(tmp_path):
    def handler(http_request):
        return httpx.Response(401, json={"error": {"message": "secret-token-should-not-leak"}})

    cache = ResponseCache(tmp_path)
    provider = OpenAIChatProvider(api_key=None, cache=cache, client=sdk_client(handler))
    with pytest.raises(ProviderError, match="model completion request failed") as error:
        provider.complete(request())
    assert "secret-token" not in str(error.value)
    assert list(tmp_path.rglob("*.json")) == []


def test_corrupt_cache_fails_without_recalling(tmp_path):
    cache = ResponseCache(tmp_path)
    req = request()
    path = cache.response_path(req)
    path.write_text("not json")
    provider = OpenAIChatProvider(api_key=None, cache=cache)
    with pytest.raises(CacheCorruptionError):
        provider.complete(req)


def test_same_key_concurrent_requests_call_server_once(tmp_path):
    calls = 0
    calls_lock = threading.Lock()

    def handler(http_request):
        nonlocal calls
        with calls_lock:
            calls += 1
        time.sleep(0.05)
        return httpx.Response(200, json=completion_json())

    cache = ResponseCache(tmp_path)
    provider = OpenAIChatProvider(api_key=None, cache=cache, client=sdk_client(handler))
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(provider.complete, [request()] * 4))
    assert calls == 1
    assert sum(result.cache_hit for result in results) == 3


def test_request_validation_and_image_helper(tmp_path):
    with pytest.raises(ValueError):
        request(parameters={"stream": True})
    with pytest.raises(ValueError):
        request(parameters={"extra_body": {"model": "attacker-model"}})
    with pytest.raises(ValueError):
        request(endpoint="https://user:pass@example.com/v1")

    image = tmp_path / "frame.png"
    image.write_bytes(b"\x89PNG\r\nfirst")
    first = image_to_data_url(image)
    image.write_bytes(b"\x89PNG\r\nsecond")
    assert first != image_to_data_url(image)
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    with pytest.raises(ValueError):
        image_to_data_url(video)


def test_provider_rejects_request_endpoint_mismatch(tmp_path):
    mismatched = request(endpoint="https://other.example/v1")
    provider = OpenAIChatProvider(api_key=None, cache=ResponseCache(tmp_path))
    with pytest.raises(ProviderError, match="endpoint"):
        provider.complete(mismatched)


@pytest.mark.parametrize(
    "field,value",
    [
        ("provider", "other"),
        ("endpoint", "https://other.example/v1"),
        ("model", "moonshotai/kimi-k3"),
        ("prompt_version", "v2"),
        ("parameters", {"temperature": 0.5}),
        ("input_assets", [{"id": "frame", "sha256": "c" * 64}]),
    ],
)
def test_all_provenance_dimensions_change_cache_key(tmp_path, field, value):
    cache = ResponseCache(tmp_path)
    assert cache.request_hash(request()) != cache.request_hash(request(**{field: value}))


@pytest.mark.parametrize("reason", ["length", "tool_calls", "content_filter", None, "unknown"])
def test_incomplete_responses_not_cached(tmp_path, reason):
    response = completion_json()
    response["choices"][0]["finish_reason"] = reason
    provider = OpenAIChatProvider(
        None,
        ResponseCache(tmp_path),
        client=sdk_client(lambda _: httpx.Response(200, json=response)),
    )
    with pytest.raises(ProviderError):
        provider.complete(request())
    assert list(tmp_path.rglob("*.json")) == []


def test_request_mutation_and_mutable_remote_image_rejected(tmp_path):
    req = request()
    req.parameters["extra_headers"] = {"Authorization": "bad"}
    with pytest.raises(ValueError):
        OpenAIChatProvider(None, ResponseCache(tmp_path)).complete(req)
    with pytest.raises(ValueError, match="embed their bytes"):
        request(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.com/mutable.png"},
                        }
                    ],
                }
            ]
        )
