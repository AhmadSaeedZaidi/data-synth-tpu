from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import create_autospec

import pytest
from huggingface_hub import HfApi

from data_synth.blob_storage import (
    BlobRef,
    HFBucketBlobStore,
    LocalBlobStore,
    StorageError,
)


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_local_roundtrip_and_content_addressed_dedup(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"fixture-media")
    store = LocalBlobStore(tmp_path / "blobs")

    first = store.put(source)
    second = store.put(source)
    destination = tmp_path / "out" / "copy.bin"
    store.get(first, destination)

    assert first == second
    assert first.uri.startswith("local://media/sha256/")
    assert destination.read_bytes() == b"fixture-media"
    assert len(list((tmp_path / "blobs").rglob("*"))) == 4


def test_local_large_file_streaming_and_expected_checksum(tmp_path: Path) -> None:
    data = (bytes(range(256)) * 8192) + b"tail"
    source = tmp_path / "large.bin"
    source.write_bytes(data)
    digest = _digest(data)

    ref = LocalBlobStore(tmp_path / "store").put(source, expected_sha256=digest)

    assert ref.sha256 == digest
    assert ref.size_bytes == len(data)


def test_expected_checksum_fails_before_hf_call(tmp_path: Path) -> None:
    client = create_autospec(HfApi, instance=True)
    source = tmp_path / "media.bin"
    source.write_bytes(b"bytes")
    store = HFBucketBlobStore("org/bucket", client=client)

    with pytest.raises(StorageError, match="checksum"):
        store.put(source, expected_sha256="0" * 64)

    client.batch_bucket_files.assert_not_called()


def test_hf_bucket_is_private_only_when_explicitly_created(tmp_path: Path) -> None:
    client = create_autospec(HfApi, instance=True)
    store = HFBucketBlobStore("org/bucket", client=client)

    result = store.create_bucket()

    assert result is client.create_bucket.return_value
    client.create_bucket.assert_called_once_with("org/bucket", private=True, token=None)
    assert "secret" not in repr(HFBucketBlobStore("org/bucket", token="secret"))


def test_hf_upload_uses_stable_path_and_reuses_logical_key(tmp_path: Path) -> None:
    client = create_autospec(HfApi, instance=True)
    source = tmp_path / "media.bin"
    payload = b"remote-bytes"
    source.write_bytes(payload)
    store = HFBucketBlobStore("org/bucket", token="fake-token", client=client)
    staged_payloads: list[bytes] = []

    def upload(bucket_id, *, add, token):
        staged_path, _ = add[0]
        staged_payloads.append(Path(staged_path).read_bytes())

    client.batch_bucket_files.side_effect = upload

    first = store.put(source)
    second = store.put(source)

    assert first == second
    assert client.batch_bucket_files.call_count == 2
    assert staged_payloads == [payload, payload]
    args, kwargs = client.batch_bucket_files.call_args
    staged_path, key = kwargs["add"][0]
    assert args == ("org/bucket",)
    assert key == "media/sha256/" + first.sha256[:2] + "/" + first.sha256
    assert not Path(staged_path).exists()
    assert kwargs["token"] == "fake-token"


def test_hf_upload_snapshot_survives_source_mutation_during_transfer(tmp_path: Path) -> None:
    client = create_autospec(HfApi, instance=True)
    source = tmp_path / "media.bin"
    payload = b"original-content"
    source.write_bytes(payload)
    staged_payloads: list[bytes] = []

    def mutating_upload(bucket_id, *, add, token):
        staged_payloads.append(Path(add[0][0]).read_bytes())
        source.write_bytes(b"changed-after-staging")

    client.batch_bucket_files.side_effect = mutating_upload
    ref = HFBucketBlobStore("org/bucket", token="fake-token", client=client).put(source)

    assert ref.sha256 == _digest(payload)
    assert staged_payloads == [payload]


def test_hf_upload_failure_is_sanitized_and_staging_is_cleaned(tmp_path: Path) -> None:
    client = create_autospec(HfApi, instance=True)
    source = tmp_path / "media.bin"
    source.write_bytes(b"remote-bytes")
    staged_paths: list[Path] = []

    def failed_upload(bucket_id, *, add, token):
        staged_paths.append(Path(add[0][0]))
        raise RuntimeError("request leaked fake-token")

    client.batch_bucket_files.side_effect = failed_upload
    with pytest.raises(StorageError, match="unable to upload") as error:
        HFBucketBlobStore("org/bucket", token="fake-token", client=client).put(source)

    assert "fake-token" not in str(error.value)
    assert not staged_paths[0].exists()


def test_hf_download_verifies_and_preserves_destination_on_corruption(tmp_path: Path) -> None:
    client = create_autospec(HfApi, instance=True)
    payload = b"correct-content"
    digest = _digest(payload)
    ref = BlobRef(
        uri=f"hf://buckets/org/bucket/media/sha256/{digest[:2]}/{digest}",
        sha256=digest,
        size_bytes=len(payload),
    )

    def corrupt_download(bucket_id, files, **kwargs):
        files[0][1].write_bytes(b"corrupt")

    client.download_bucket_files.side_effect = corrupt_download
    store = HFBucketBlobStore("org/bucket", client=client)
    destination = tmp_path / "download.bin"

    with pytest.raises(StorageError, match="integrity"):
        store.get(ref, destination)

    assert not destination.exists()
    assert list(tmp_path.glob(".*download.bin.*")) == []


def test_hf_download_roundtrip_uses_raise_on_missing_and_atomic_destination(tmp_path: Path) -> None:
    client = create_autospec(HfApi, instance=True)
    payload = b"remote-content"
    digest = _digest(payload)
    ref = BlobRef(
        uri=f"hf://buckets/org/bucket/media/sha256/{digest[:2]}/{digest}",
        sha256=digest,
        size_bytes=len(payload),
    )

    def download(bucket_id, files, **kwargs):
        files[0][1].write_bytes(payload)

    client.download_bucket_files.side_effect = download
    destination = tmp_path / "nested" / "download.bin"
    result = HFBucketBlobStore("org/bucket", client=client).get(ref, destination)

    assert result == destination
    assert destination.read_bytes() == payload
    _, kwargs = client.download_bucket_files.call_args
    assert kwargs["raise_on_missing_files"] is True


def test_hf_download_failure_is_sanitized_and_staging_is_cleaned(tmp_path: Path) -> None:
    client = create_autospec(HfApi, instance=True)
    payload = b"remote-content"
    digest = _digest(payload)
    ref = BlobRef(
        uri=f"hf://buckets/org/bucket/media/sha256/{digest[:2]}/{digest}",
        sha256=digest,
        size_bytes=len(payload),
    )
    staged_paths: list[Path] = []

    def failed_download(bucket_id, files, **kwargs):
        staged_paths.append(Path(files[0][1]))
        raise RuntimeError("response leaked fake-token")

    client.download_bucket_files.side_effect = failed_download
    destination = tmp_path / "download.bin"
    with pytest.raises(StorageError, match="unable to download") as error:
        HFBucketBlobStore("org/bucket", client=client).get(ref, destination)

    assert "fake-token" not in str(error.value)
    assert not destination.exists()
    assert not staged_paths[0].exists()


def test_missing_hf_credentials_and_bucket_mismatch_fail_without_network(tmp_path: Path) -> None:
    source = tmp_path / "media.bin"
    source.write_bytes(b"bytes")
    store = HFBucketBlobStore("org/bucket")
    with pytest.raises(StorageError, match="token"):
        store.put(source)

    digest = _digest(b"other")
    ref = BlobRef(
        uri=f"hf://buckets/other/bucket/media/sha256/{digest[:2]}/{digest}",
        sha256=digest,
        size_bytes=5,
    )
    with pytest.raises(StorageError, match="does not belong"):
        HFBucketBlobStore("org/bucket", client=create_autospec(HfApi, instance=True)).get(
            ref, tmp_path / "destination"
        )


@pytest.mark.parametrize(
    "uri",
    [
        "local://media/sha256/aa/" + "a" * 64 + "/../escape",
        "local://media/sha256/aa/" + "b" * 64,
        "local://media/sha256/bb/" + "a" * 64,
        "HF://buckets/org/bucket/media/sha256/aa/" + "a" * 64,
        "hf://buckets//org/bucket/media/sha256/aa/" + "a" * 64,
        "hf://buckets/org/bucket/media/sha256/aa/" + "a" * 64 + "?token=secret",
        "hf://buckets/org/../bucket/media/sha256/aa/" + "a" * 64,
    ],
)
def test_blob_ref_rejects_malformed_or_traversal_uris(uri: str) -> None:
    with pytest.raises(ValueError):
        BlobRef(uri=uri, sha256="a" * 64, size_bytes=1)


def test_existing_conflicting_destination_is_preserved(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"source")
    store = LocalBlobStore(tmp_path / "store")
    ref = store.put(source)
    destination = tmp_path / "destination"
    destination.write_bytes(b"keep-me")

    with pytest.raises(StorageError, match="different content"):
        store.get(ref, destination)
    assert destination.read_bytes() == b"keep-me"
