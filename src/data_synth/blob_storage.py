"""Content-addressed local and Hugging Face Storage Bucket media stores."""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_CHUNK_SIZE = 1024 * 1024
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_BUCKET_PART_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_KEY_RE = re.compile(r"^media/sha256/[0-9a-f]{2}/([0-9a-f]{64})$")


class StorageError(RuntimeError):
    """A safe storage failure that never includes credentials or SDK details."""


def _validate_digest(value: str) -> str:
    if not isinstance(value, str) or not _DIGEST_RE.fullmatch(value):
        raise ValueError("sha256 must be a 64-character lowercase hexadecimal digest")
    return value


def _validate_bucket_id(bucket_id: str) -> str:
    if not isinstance(bucket_id, str):
        raise ValueError("bucket_id must be a namespace/name string")
    parts = bucket_id.split("/")
    if len(parts) != 2 or any(not _BUCKET_PART_RE.fullmatch(part) for part in parts):
        raise ValueError("bucket_id must be a valid namespace/name string")
    return bucket_id


def _validate_key(key: str) -> str:
    if "\\" in key or any(part in {"", ".", ".."} for part in key.split("/")):
        raise ValueError("storage key contains invalid path components")
    match = _KEY_RE.fullmatch(key)
    if match is None or match.group(1)[:2] != key.split("/")[-2]:
        raise ValueError("storage key is not a canonical media SHA-256 key")
    return key


def _key_for_digest(digest: str) -> str:
    _validate_digest(digest)
    return f"media/sha256/{digest[:2]}/{digest}"


def _parse_uri(uri: str) -> tuple[str, str | None, str]:
    if not isinstance(uri, str) or not uri:
        raise ValueError("storage URI must be non-empty")
    parsed = urlsplit(uri)
    if parsed.query or parsed.fragment or parsed.username or parsed.password:
        raise ValueError("storage URI must not contain query, fragment, or credentials")
    if parsed.scheme == "local":
        if not parsed.netloc:
            raise ValueError("local URI is missing its key")
        key = f"{parsed.netloc}{parsed.path}"
        return "local", None, _validate_key(key.lstrip("/"))
    if parsed.scheme == "hf":
        if parsed.netloc != "buckets":
            raise ValueError("Hugging Face URI must use hf://buckets")
        parts = parsed.path.lstrip("/").split("/")
        if len(parts) < 3:
            raise ValueError("Hugging Face URI is missing its bucket or key")
        bucket_id = _validate_bucket_id("/".join(parts[:2]))
        return "hf", bucket_id, _validate_key("/".join(parts[2:]))
    raise ValueError("storage URI must use local:// or hf://buckets/")


class BlobRef(BaseModel):
    """Immutable reference to one content-addressed media object."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    uri: str
    sha256: str
    size_bytes: int = Field(ge=0)

    @field_validator("sha256")
    @classmethod
    def _sha256(cls, value: str) -> str:
        return _validate_digest(value)

    @model_validator(mode="after")
    def _uri_matches_digest(self) -> "BlobRef":
        scheme, bucket, key = _parse_uri(self.uri)
        if key.rsplit("/", 1)[-1] != self.sha256:
            raise ValueError("storage URI digest does not match sha256")
        canonical = f"local://{key}" if scheme == "local" else f"hf://buckets/{bucket}/{key}"
        if self.uri != canonical:
            raise ValueError("storage URI is not canonical")
        return self


class BlobStore(Protocol):
    def put(self, path: Path, expected_sha256: str | None = None) -> BlobRef: ...

    def get(self, ref: BlobRef, destination: Path) -> Path: ...


def _hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(_CHUNK_SIZE), b""):
                digest.update(chunk)
                size += len(chunk)
    except OSError:
        raise StorageError("unable to read media file") from None
    return digest.hexdigest(), size


def _stage_file(path: Path, directory: Path) -> tuple[Path, str, int]:
    """Copy and hash a source once, producing the stable upload snapshot."""

    temporary_name: str | None = None
    try:
        with (
            path.open("rb") as source,
            tempfile.NamedTemporaryFile(
                mode="wb", dir=directory, prefix=".blob-stage-", delete=False
            ) as temporary,
        ):
            temporary_name = temporary.name
            digest = hashlib.sha256()
            size = 0
            for chunk in iter(lambda: source.read(_CHUNK_SIZE), b""):
                temporary.write(chunk)
                digest.update(chunk)
                size += len(chunk)
    except OSError:
        if temporary_name:
            try:
                os.unlink(temporary_name)
            except OSError:
                pass
        raise StorageError("unable to stage media file") from None
    return Path(temporary_name), digest.hexdigest(), size


def _verify_file(path: Path, expected_digest: str, expected_size: int) -> None:
    digest, size = _hash_file(path)
    if digest != expected_digest or size != expected_size:
        raise StorageError("media integrity verification failed")


def _existing_destination(destination: Path, ref: BlobRef) -> Path | None:
    if not destination.exists():
        return None
    if not destination.is_file():
        raise StorageError("destination already exists and is not a file")
    try:
        _verify_file(destination, ref.sha256, ref.size_bytes)
    except StorageError:
        raise StorageError("destination already exists with different content") from None
    return destination


def _install_new_file(temporary: Path, destination: Path, ref: BlobRef) -> Path:
    """Install without replacing a file another writer created meanwhile."""

    try:
        os.link(temporary, destination)
        temporary.unlink()
    except FileExistsError:
        _existing_destination(destination, ref)
        try:
            temporary.unlink()
        except OSError:
            pass
    except OSError:
        raise StorageError("unable to install media file") from None
    return destination


class LocalBlobStore:
    """Offline content-addressed store rooted at one directory."""

    def __init__(self, root: Path | str):
        self.root = Path(root).resolve()
        try:
            self.root.mkdir(parents=True, exist_ok=True)
        except OSError:
            raise StorageError("unable to initialize local blob store") from None

    def _path_for_key(self, key: str) -> Path:
        _validate_key(key)
        path = (self.root / key).resolve()
        try:
            path.relative_to(self.root)
        except ValueError:
            raise StorageError("storage key escapes local store") from None
        return path

    def put(self, path: Path, expected_sha256: str | None = None) -> BlobRef:
        source = Path(path)
        if expected_sha256 is not None:
            _validate_digest(expected_sha256)
        stage, digest, size = _stage_file(source, self.root)
        try:
            if expected_sha256 is not None and digest != expected_sha256:
                raise StorageError("source checksum does not match expected sha256")
            key = _key_for_digest(digest)
            destination = self._path_for_key(key)
            destination.parent.mkdir(parents=True, exist_ok=True)
            ref = BlobRef(uri=f"local://{key}", sha256=digest, size_bytes=size)
            existing = _existing_destination(destination, ref)
            if existing is not None:
                return ref
            _install_new_file(stage, destination, ref)
            return ref
        finally:
            try:
                stage.unlink()
            except OSError:
                pass

    def get(self, ref: BlobRef, destination: Path) -> Path:
        if not isinstance(ref, BlobRef):
            raise TypeError("ref must be a BlobRef")
        scheme, bucket, key = _parse_uri(ref.uri)
        if scheme != "local" or bucket is not None:
            raise StorageError("blob reference does not belong to local storage")
        source = self._path_for_key(key)
        if not source.is_file():
            raise StorageError("local blob does not exist")
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        if _existing_destination(target, ref) is not None:
            return target
        stage_name, digest, size = _stage_file(source, target.parent)
        try:
            if digest != ref.sha256 or size != ref.size_bytes:
                raise StorageError("stored blob failed integrity verification")
            return _install_new_file(stage_name, target, ref)
        finally:
            try:
                stage_name.unlink()
            except OSError:
                pass


class HFBucketBlobStore:
    """Content-addressed storage backed by a native Hugging Face Bucket."""

    def __init__(self, bucket_id: str, token: str | None = None, client: Any | None = None):
        self.bucket_id = _validate_bucket_id(bucket_id)
        if token is not None and not isinstance(token, str):
            raise TypeError("token must be a string or None")
        self._token = token
        self._client = client

    def __repr__(self) -> str:
        return (
            f"HFBucketBlobStore(bucket_id={self.bucket_id!r}, token_configured={bool(self._token)})"
        )

    def _client_for_remote(self) -> Any:
        if self._client is not None:
            return self._client
        if not self._token:
            raise StorageError("Hugging Face token is required for remote storage")
        try:
            from huggingface_hub import HfApi

            self._client = HfApi(token=self._token)
        except Exception:
            raise StorageError("unable to initialize Hugging Face storage") from None
        return self._client

    def create_bucket(self) -> Any:
        client = self._client_for_remote()
        try:
            return client.create_bucket(self.bucket_id, private=True, token=self._token)
        except Exception:
            raise StorageError("unable to create Hugging Face bucket") from None

    def put(self, path: Path, expected_sha256: str | None = None) -> BlobRef:
        if expected_sha256 is not None:
            _validate_digest(expected_sha256)
        stage, digest, size = _stage_file(Path(path), Path(tempfile.gettempdir()))
        try:
            if expected_sha256 is not None and digest != expected_sha256:
                raise StorageError("source checksum does not match expected sha256")
            key = _key_for_digest(digest)
            ref = BlobRef(
                uri=f"hf://buckets/{self.bucket_id}/{key}",
                sha256=digest,
                size_bytes=size,
            )
            client = self._client_for_remote()
            try:
                client.batch_bucket_files(
                    self.bucket_id,
                    add=[(stage, key)],
                    token=self._token,
                )
            except Exception:
                raise StorageError("unable to upload media to Hugging Face") from None
            return ref
        finally:
            try:
                stage.unlink()
            except OSError:
                pass

    def get(self, ref: BlobRef, destination: Path) -> Path:
        if not isinstance(ref, BlobRef):
            raise TypeError("ref must be a BlobRef")
        scheme, bucket, key = _parse_uri(ref.uri)
        if scheme != "hf" or bucket != self.bucket_id:
            raise StorageError("blob reference does not belong to this Hugging Face bucket")
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        if _existing_destination(target, ref) is not None:
            return target
        stage_name: str | None = None
        try:
            client = self._client_for_remote()
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=target.parent, prefix=f".{target.name}.", delete=False
            ) as stage:
                stage_name = stage.name
            try:
                client.download_bucket_files(
                    self.bucket_id,
                    files=[(key, Path(stage_name))],
                    raise_on_missing_files=True,
                    token=self._token,
                )
            except Exception:
                raise StorageError("unable to download media from Hugging Face") from None
            _verify_file(Path(stage_name), ref.sha256, ref.size_bytes)
            return _install_new_file(Path(stage_name), target, ref)
        except StorageError:
            raise
        except OSError:
            raise StorageError("unable to prepare media download") from None
        finally:
            if stage_name:
                try:
                    os.unlink(stage_name)
                except OSError:
                    pass


__all__ = ["BlobRef", "BlobStore", "HFBucketBlobStore", "LocalBlobStore", "StorageError"]
