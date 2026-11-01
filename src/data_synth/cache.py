"""A small, content addressed cache for successful model responses.

The cache deliberately knows nothing about provider credentials.  Its key is
the complete logical request, while the response file includes an integrity
digest so a damaged local file cannot silently cause a second billable call.
"""

from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator
from urllib.parse import urlsplit, urlunsplit

from filelock import FileLock

if TYPE_CHECKING:
    from .providers import ModelRequest


CACHE_VERSION = "responses-v1"


class CacheError(RuntimeError):
    """Base class for cache failures."""


class CacheCorruptionError(CacheError):
    """Raised when an existing cache entry cannot be trusted."""


def normalize_endpoint(endpoint: str) -> str:
    """Return a canonical endpoint URL and reject credential-bearing URLs."""

    if not isinstance(endpoint, str) or not endpoint.strip():
        raise ValueError("endpoint must be a non-empty URL")
    endpoint = endpoint.strip()
    parsed = urlsplit(endpoint)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise ValueError("endpoint must use an HTTP or HTTPS URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("endpoint must not contain credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("endpoint must not contain a query or fragment")

    scheme = parsed.scheme.lower()
    hostname = parsed.hostname.lower()
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("endpoint has an invalid port") from exc
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    netloc = hostname
    default_port = (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
    if port is not None and not default_port:
        netloc = f"{netloc}:{port}"

    path = parsed.path.rstrip("/")
    return urlunsplit((scheme, netloc, path, "", ""))


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("request or response contains a non-JSON value") from exc


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def image_to_data_url(path: str | os.PathLike[str]) -> str:
    """Read a local image and return a byte-preserving ``data:`` URL.

    MIME detection is intentionally limited to image types.  In particular,
    a video file is never guessed as an image merely because it has a path.
    """

    image_path = Path(path)
    data = image_path.read_bytes()
    mime, _ = mimetypes.guess_type(image_path.name)
    if not mime or not mime.startswith("image/"):
        raise ValueError("path does not have a recognized image type")
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{encoded}"


# Descriptive aliases make the helper convenient for callers without adding a
# second implementation that could accidentally handle video inputs.
local_image_to_data_url = image_to_data_url
image_file_to_data_url = image_to_data_url


class ResponseCache:
    """Content-addressed JSON response cache with per-request file locks."""

    def __init__(self, directory: Path | str):
        self.directory = Path(directory)
        self.namespace_directory = self.directory / CACHE_VERSION
        self.namespace_directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def canonical_request(request: "ModelRequest") -> dict[str, Any]:
        """Build the exact logical request material used by the cache key."""

        return {
            "cache_version": CACHE_VERSION,
            "provider": str(request.provider).strip().lower(),
            "endpoint": normalize_endpoint(request.endpoint),
            "model": request.model,
            "prompt_version": request.prompt_version,
            "messages": request.messages,
            "parameters": request.parameters,
            "input_assets": request.input_assets,
        }

    def request_hash(self, request: "ModelRequest") -> str:
        return _sha256_json(self.canonical_request(request))

    # Short aliases are useful to callers that treat the hash as an address.
    key_for = request_hash

    def response_path(self, request_or_hash: "ModelRequest | str") -> Path:
        request_hash = (
            self.request_hash(request_or_hash)
            if not isinstance(request_or_hash, str)
            else request_or_hash
        )
        if len(request_hash) != 64 or any(char not in "0123456789abcdef" for char in request_hash):
            raise ValueError("request hash must be a SHA-256 hexadecimal digest")
        return self.namespace_directory / f"{request_hash}.json"

    def lock_path(self, request_or_hash: "ModelRequest | str") -> Path:
        return self.response_path(request_or_hash).with_suffix(".lock")

    @contextmanager
    def locked(self, request_or_hash: "ModelRequest | str") -> Iterator[None]:
        """Hold the per-key lock across cache lookup, call, and cache write."""

        lock = FileLock(str(self.lock_path(request_or_hash)))
        with lock:
            yield

    def get(self, request: "ModelRequest") -> dict[str, Any] | None:
        """Return a verified response, or ``None`` when no entry exists."""

        request_hash = self.request_hash(request)
        path = self.response_path(request_hash)
        if not path.exists():
            return None
        try:
            raw = path.read_bytes()
            record = json.loads(raw.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            raise CacheCorruptionError(f"cache entry {request_hash} is unreadable") from None
        if not isinstance(record, dict):
            raise CacheCorruptionError(f"cache entry {request_hash} is invalid")
        if record.get("cache_version") != CACHE_VERSION:
            raise CacheCorruptionError(f"cache entry {request_hash} has an unsupported version")
        if record.get("request_hash") != request_hash:
            raise CacheCorruptionError(f"cache entry {request_hash} has a mismatched key")
        response = record.get("response")
        if not isinstance(response, dict):
            raise CacheCorruptionError(f"cache entry {request_hash} has an invalid response")
        if record.get("response_sha256") != _sha256_json(response):
            raise CacheCorruptionError(f"cache entry {request_hash} failed integrity verification")
        return response

    def put(self, request: "ModelRequest", response: dict[str, Any]) -> str:
        """Atomically persist a successful response and return its request hash."""

        if not isinstance(response, dict):
            raise ValueError("cached response must be a JSON object")
        request_hash = self.request_hash(request)
        record = {
            "cache_version": CACHE_VERSION,
            "request_hash": request_hash,
            "response_sha256": _sha256_json(response),
            "response": response,
        }
        payload = _canonical_bytes(record)
        destination = self.response_path(request_hash)
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=self.namespace_directory,
                prefix=f".{request_hash}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_name = temporary.name
                temporary.write(payload)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_name, destination)
        except OSError:
            raise CacheError("unable to persist model response") from None
        finally:
            if temporary_name:
                try:
                    os.unlink(temporary_name)
                except FileNotFoundError:
                    pass
        return request_hash
