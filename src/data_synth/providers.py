"""Provider-neutral model request types and an OpenAI-compatible chat adapter."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .cache import ResponseCache, normalize_endpoint

try:  # Keep imports of request/cache utilities usable without the optional SDK.
    from openai import OpenAI
except ImportError:  # pragma: no cover - exercised only in minimal installations.
    OpenAI = None  # type: ignore[assignment,misc]


DEFAULT_PROVIDER = "nvidia"
DEFAULT_ENDPOINT = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "z-ai/glm-5.3"

_ALLOWED_PARAMETER_NAMES = frozenset(
    {
        "temperature",
        "top_p",
        "max_tokens",
        "max_completion_tokens",
        "seed",
        "stop",
        "presence_penalty",
        "frequency_penalty",
        "response_format",
        "n",
        "logprobs",
        "top_logprobs",
        "user",
        "reasoning_effort",
    }
)
_UNUSABLE_FINISH_REASONS = frozenset({"length", "tool_calls", "function_call", "content_filter"})


class ModelRequest(BaseModel):
    """A fully specified, cacheable chat completion request."""

    model_config = ConfigDict(extra="forbid", strict=True)

    provider: str
    endpoint: str
    model: str
    prompt_version: str
    messages: list[dict[str, Any]]
    parameters: dict[str, Any] = Field(default_factory=dict)
    input_assets: list[dict[str, str]] = Field(default_factory=list)

    @field_validator("provider", "model", "prompt_version")
    @classmethod
    def _non_empty_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must not be empty")
        return value.strip()

    @field_validator("provider")
    @classmethod
    def _normalise_provider(cls, value: str) -> str:
        return value.lower()

    @field_validator("endpoint")
    @classmethod
    def _validate_endpoint(cls, value: str) -> str:
        return normalize_endpoint(value)

    @field_validator("messages")
    @classmethod
    def _validate_messages(cls, value: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not value:
            raise ValueError("messages must contain at least one message")
        for message in value:
            if not isinstance(message, dict) or not message:
                raise ValueError("each message must be a non-empty object")
            role = message.get("role")
            if role not in {"system", "developer", "user", "assistant", "tool"}:
                raise ValueError("message role must be a supported chat role")
            content = message.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        image = part.get("image_url", {})
                        url = image.get("url", "") if isinstance(image, dict) else ""
                        if not isinstance(url, str) or not url.startswith("data:image/"):
                            raise ValueError("cacheable images must embed their bytes as data URLs")
        return value

    @field_validator("parameters")
    @classmethod
    def _validate_parameters(cls, value: dict[str, Any]) -> dict[str, Any]:
        unsupported = sorted(key for key in value if key not in _ALLOWED_PARAMETER_NAMES)
        if unsupported:
            raise ValueError(f"unsupported completion parameter: {unsupported[0]}")
        _ensure_json(value, "parameters")
        return value

    @field_validator("input_assets")
    @classmethod
    def _validate_assets(cls, value: list[dict[str, str]]) -> list[dict[str, str]]:
        for asset in value:
            if set(asset) != {"id", "sha256"}:
                raise ValueError("each input asset must contain only id and sha256")
            if not asset["id"].strip():
                raise ValueError("input asset id must not be empty")
            checksum = asset["sha256"].lower()
            if len(checksum) != 64 or any(char not in "0123456789abcdef" for char in checksum):
                raise ValueError("input asset sha256 must be a 64-character hexadecimal digest")
            asset["sha256"] = checksum
        _ensure_json(value, "input_assets")
        return value

    @field_validator("messages")
    @classmethod
    def _validate_message_json(cls, value: list[dict[str, Any]]) -> list[dict[str, Any]]:
        _ensure_json(value, "messages")
        return value


class ModelResult(BaseModel):
    """Successful model output together with cache provenance."""

    model_config = ConfigDict(extra="forbid", strict=True)

    response: dict[str, Any]
    request_hash: str
    cache_hit: bool


class ProviderError(RuntimeError):
    """A safe, intentionally non-sensitive provider failure."""


class ModelProvider(Protocol):
    def complete(self, request: ModelRequest) -> ModelResult: ...


def _ensure_json(value: Any, label: str) -> None:
    try:
        json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must contain JSON-compatible values") from exc


def _response_as_dict(response: Any) -> dict[str, Any]:
    try:
        if isinstance(response, dict):
            result = response
        elif hasattr(response, "model_dump"):
            try:
                result = response.model_dump(mode="json")
            except TypeError:
                result = response.model_dump()
        elif hasattr(response, "to_dict"):
            result = response.to_dict()
        else:
            raise ProviderError("model returned an invalid completion")
    except Exception:
        raise ProviderError("model returned an invalid completion") from None
    if not isinstance(result, dict):
        raise ProviderError("model returned an invalid completion")
    _ensure_json(result, "model response")
    return result


def _validate_successful_response(response: dict[str, Any]) -> None:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ProviderError("model returned an invalid completion")
    for choice in choices:
        if not isinstance(choice, dict):
            raise ProviderError("model returned an invalid completion")
        reason = choice.get("finish_reason")
        if isinstance(reason, str) and reason.lower() in _UNUSABLE_FINISH_REASONS:
            raise ProviderError("model completion was incomplete or requested a tool call")
        if reason != "stop":
            raise ProviderError("model completion has no supported successful finish reason")
        message = choice.get("message")
        if not isinstance(message, dict):
            raise ProviderError("model returned an invalid completion")
        if message.get("refusal"):
            raise ProviderError("model refused the completion")
        if message.get("tool_calls") or message.get("function_call"):
            raise ProviderError("model completion requested a tool call")
        content = message.get("content")
        if isinstance(content, str):
            has_content = bool(content.strip())
        elif isinstance(content, list):
            has_content = any(
                isinstance(part, dict)
                and isinstance(part.get("text"), str)
                and part["text"].strip()
                for part in content
            )
        else:
            has_content = False
        if not has_content:
            raise ProviderError("model returned an empty completion")


class OpenAIChatProvider:
    """OpenAI Chat Completions adapter, suitable for NVIDIA's OpenAI endpoint."""

    def __init__(
        self,
        api_key: str | None,
        cache: ResponseCache,
        timeout: float = 60,
        client: Any | None = None,
        endpoint: str = DEFAULT_ENDPOINT,
        provider: str = DEFAULT_PROVIDER,
    ):
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.api_key = api_key
        self.cache = cache
        self.timeout = timeout
        self.endpoint = normalize_endpoint(endpoint)
        self.provider = provider.strip().lower()
        if not self.provider:
            raise ValueError("provider must not be empty")
        self._injected_client = client
        self._client = client
        self._clients: dict[str, Any] = {}
        if client is not None:
            self._clients[self.endpoint] = client

    def _client_for(self, endpoint: str) -> Any:
        existing = self._clients.get(endpoint)
        if existing is not None:
            return existing
        if self.api_key is None or not str(self.api_key).strip():
            raise ProviderError("an API key is required for an uncached completion")
        if OpenAI is None:
            raise ProviderError("the OpenAI SDK is unavailable")
        try:
            client = OpenAI(
                api_key=self.api_key,
                base_url=endpoint,
                timeout=self.timeout,
                max_retries=0,
            )
        except Exception:
            raise ProviderError("unable to initialize the model provider") from None
        self._clients[endpoint] = client
        self._client = client
        return client

    def complete(self, request: ModelRequest) -> ModelResult:
        if not isinstance(request, ModelRequest):
            raise TypeError("request must be a ModelRequest")
        request = ModelRequest.model_validate(deepcopy(request.model_dump()))
        if request.provider != self.provider:
            raise ProviderError("request provider does not match the configured provider")
        if request.endpoint != self.endpoint:
            raise ProviderError("request endpoint does not match the configured provider")
        request_hash = self.cache.request_hash(request)
        # The lock spans lookup and the billable call, so concurrent workers
        # sharing this directory reuse one successful response.
        with self.cache.locked(request_hash):
            cached = self.cache.get(request)
            if cached is not None:
                _validate_successful_response(cached)
                return ModelResult(response=cached, request_hash=request_hash, cache_hit=True)

            client = self._client_for(request.endpoint)
            payload = dict(request.parameters)
            payload.update(model=request.model, messages=request.messages, stream=False)
            try:
                completion = client.chat.completions.create(**payload)
            except Exception:
                raise ProviderError("model completion request failed") from None

            response = _response_as_dict(completion)
            _validate_successful_response(response)
            self.cache.put(request, response)
            return ModelResult(response=response, request_hash=request_hash, cache_hit=False)
