"""Link successful blob transfers to SQL assets without changing research evidence."""

from pathlib import Path

from sqlalchemy.orm import Session

from .blob_storage import BlobRef, BlobStore, StorageError
from .models import Asset


def upload_asset(session: Session, asset_id: str, store: BlobStore) -> BlobRef:
    """Upload an existing, checksummed asset; caller owns the SQL transaction.

    SQL and object storage cannot commit atomically. A successful upload followed
    by a SQL rollback may leave an unreferenced object; retry uses the same key.
    """
    asset = session.get(Asset, asset_id)
    if asset is None or not asset.local_path or not asset.sha256:
        raise StorageError("Asset requires a local file and a recorded SHA-256.")
    ref = store.put(Path(asset.local_path), expected_sha256=asset.sha256)
    asset.remote_uri = ref.uri
    asset.metadata_json = {
        **asset.metadata_json,
        "blob": {"sha256": ref.sha256, "size_bytes": ref.size_bytes},
    }
    session.flush()
    return ref


def download_asset(session: Session, asset_id: str, destination: Path, store: BlobStore) -> Path:
    """Verify downloaded bytes before recording a usable local path."""
    asset = session.get(Asset, asset_id)
    if asset is None or not asset.remote_uri or not asset.sha256:
        raise StorageError("Asset requires a stored blob URI and SHA-256.")
    blob = asset.metadata_json.get("blob", {})
    if not isinstance(blob, dict) or blob.get("sha256") != asset.sha256:
        raise StorageError("Asset blob metadata is missing or inconsistent.")
    size = blob.get("size_bytes")
    if type(size) is not int or size < 0:
        raise StorageError("Asset blob size is missing or invalid.")
    ref = BlobRef(uri=asset.remote_uri, sha256=asset.sha256, size_bytes=size)
    path = store.get(ref, destination)
    asset.local_path = str(path.resolve())
    session.flush()
    return path
