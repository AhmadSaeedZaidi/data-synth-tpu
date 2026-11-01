#!/usr/bin/env python3
"""Inject a local W&B key into a temporary private notebook and submit to Kaggle."""

import argparse
import ast
import copy
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOKEN = re.compile(r"(?:wandb_v1_|KGAT_|nvapi-)[A-Za-z0-9_-]{24,}")
AUTH_TAG = "kaggle-upload-auth"


class PublishError(RuntimeError):
    pass


def literal(node):
    return isinstance(node, ast.Constant) and isinstance(node.value, str) and bool(node.value)


def check_notebook(document):
    """Reject known token patterns and literal W&B credentials, including outputs."""
    if not isinstance(document, dict):
        raise PublishError("Expected a Jupyter notebook object")
    if TOKEN.search(json.dumps(document)):
        raise PublishError("Notebook contains a credential pattern; provide a key-free source")
    if document.get("nbformat") != 4 or not isinstance(document.get("cells"), list):
        raise PublishError("Expected a version-4 Jupyter notebook")
    if not isinstance(document.get("metadata", {}), dict):
        raise PublishError("Invalid notebook metadata")
    for cell in document["cells"]:
        if not isinstance(cell, dict) or cell.get("cell_type") not in {"code", "markdown", "raw"}:
            raise PublishError("Invalid notebook cell")
        metadata = cell.get("metadata", {})
        if not isinstance(metadata, dict) or not isinstance(metadata.get("tags", []), list):
            raise PublishError("Invalid cell metadata")
        if AUTH_TAG in metadata.get("tags", []):
            raise PublishError("An upload copy cannot be used as the safe source notebook")
        source = cell.get("source", "")
        if isinstance(source, list) and all(isinstance(line, str) for line in source):
            source = "".join(source)
        if not isinstance(source, str):
            raise PublishError("Invalid cell source")
        if cell["cell_type"] != "code":
            continue
        # IPython magics may prevent AST parsing. This also covers ordinary
        # environment assignments in cells that contain a magic command.
        if re.search(r"WANDB_API_KEY['\"]?\]?\s*[:=]\s*['\"][^'\"\r\n]+['\"]", source):
            raise PublishError("Notebook contains a literal W&B key assignment")
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign)) and literal(node.value):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    named = isinstance(target, ast.Name) and target.id == "WANDB_API_KEY"
                    indexed = (
                        isinstance(target, ast.Subscript)
                        and isinstance(target.slice, ast.Constant)
                        and target.slice.value == "WANDB_API_KEY"
                    )
                    if named or indexed:
                        raise PublishError("Notebook contains a literal W&B key assignment")
            if isinstance(node, ast.Call):
                for keyword in node.keywords:
                    if keyword.arg == "WANDB_API_KEY" and literal(keyword.value):
                        raise PublishError("Notebook contains a literal W&B key")
                    if (
                        keyword.arg == "key"
                        and literal(keyword.value)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "login"
                    ):
                        raise PublishError("Notebook contains a literal login key")
                if (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr == "setdefault"
                    and len(node.args) > 1
                    and isinstance(node.args[0], ast.Constant)
                    and node.args[0].value == "WANDB_API_KEY"
                    and literal(node.args[1])
                ):
                    raise PublishError("Notebook contains a literal W&B default key")
            if isinstance(node, ast.Dict):
                for key, value in zip(node.keys, node.values, strict=True):
                    if (
                        isinstance(key, ast.Constant)
                        and key.value == "WANDB_API_KEY"
                        and literal(value)
                    ):
                        raise PublishError("Notebook contains a literal W&B key dictionary")


def check_repo(root=ROOT):
    names = (
        subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=root
        )
        .decode()
        .split("\0")
    )
    checked, notebooks = 0, 0
    binaries = {".pdf", ".npz", ".png", ".jpg", ".jpeg", ".gif", ".zip", ".ico"}
    for name in sorted(set(names) - {""}):
        path = root / name
        if not path.is_file() or path.is_symlink() or path.suffix.lower() in binaries:
            continue
        if path.name == "kaggle.json":
            raise PublishError(f"Local Kaggle credentials must not be committed: {name}")
        if path.stat().st_size > 8_000_000:
            raise PublishError(f"Source file exceeds the credential-check size limit: {name}")
        text = path.read_bytes().decode("utf-8", errors="replace")
        if TOKEN.search(text):
            raise PublishError(f"Credential pattern found in {name}")
        if path.suffix == ".ipynb":
            try:
                check_notebook(json.loads(text))
            except PublishError:
                raise PublishError(f"Notebook is not key-free: {name}") from None
            notebooks += 1
        checked += 1
    return {"checked_source_files": checked, "checked_notebooks": notebooks, "status": "PASS"}


def inject(document, key):
    check_notebook(document)
    if not key or any(c in key for c in "\n\r\0"):
        raise PublishError("W&B key must be a nonempty single-line value")
    if key in json.dumps(document):
        raise PublishError("The provided W&B key already appears in the source notebook")
    result = copy.deepcopy(document)
    result.setdefault("metadata", {}).pop("widgets", None)
    for cell in result["cells"]:
        if cell["cell_type"] == "code":
            cell.update(outputs=[], execution_count=None)
    source = (
        "import os as _kaggle_upload_os\n"
        + '_kaggle_upload_os.environ["WANDB_API_KEY"] = '
        + json.dumps(key)
        + "\n"
        + '_kaggle_upload_os.environ["WANDB_DISABLE_CODE"] = "true"\n'
        + '_kaggle_upload_os.environ["WANDB_DISABLE_GIT"] = "true"\n'
    )
    result["cells"].insert(
        0,
        {
            "id": AUTH_TAG,
            "cell_type": "code",
            "execution_count": None,
            "outputs": [],
            "metadata": {"tags": [AUTH_TAG], "jupyter": {"source_hidden": True}},
            "source": source.splitlines(keepends=True),
        },
    )
    return result


def private_write(path, value):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def owned_target_visibility(api, ref):
    """Resolve Kaggle's ambiguous 403/404 using the authenticated owner's list."""
    owner, _ = ref.split("/")
    if api.get_config_value("username") != owner:
        raise PublishError("Cannot establish a missing target outside the authenticated account")
    for page in range(1, 6):
        rows = api.kernels_list(page=page, page_size=100, mine=True, user=owner)
        if rows is None or any(row is None for row in rows):
            raise PublishError(
                "Owned notebook inventory is unavailable; inspect Kaggle before retrying"
            )
        for row in rows:
            if row.ref == ref:
                return getattr(row, "is_private", False) is True
        if len(rows) < 100:
            return None
    raise PublishError(
        "Owned notebook inventory exceeds the lookup bound; inspect Kaggle before retrying"
    )


class KaggleClient:
    def __init__(self):
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
        except ImportError:
            raise PublishError(
                "Install the publisher dependency: python -m pip install -e '.[kaggle]'"
            ) from None
        self.api = KaggleApi()
        self.api.authenticate()
        original = self.api.build_kaggle_client

        @contextmanager
        def bounded_client():
            with original() as client:
                http = client.http_client()
                http._init_session()
                send = http._session.send

                def bounded(request, **kwargs):
                    kwargs.setdefault("timeout", (10, 45))
                    return send(request, **kwargs)

                http._session.send = bounded
                yield client

        self.api.build_kaggle_client = bounded_client

    def is_private(self, ref):
        from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest

        owner, slug = ref.split("/")
        request = ApiGetKernelRequest()
        request.user_name, request.kernel_slug = owner, slug
        try:
            with self.api.build_kaggle_client() as client:
                response = client.kernels.kernels_api_client.get_kernel(request)
        except Exception as error:
            if getattr(getattr(error, "response", None), "status_code", None) in (403, 404):
                return owned_target_visibility(self.api, ref)
            raise
        return getattr(response.metadata, "is_private", False) is True

    def push(self, folder, timeout, accelerator):
        return self.api.kernels_push(str(folder), timeout=str(timeout), acc=accelerator)


def publish(
    notebook, metadata, key, *, dry_run=False, accelerator=None, timeout=21600, client=None
):
    original = notebook.read_bytes()
    document = json.loads(original)
    meta = json.loads(metadata.read_text())
    if not isinstance(meta, dict):
        raise PublishError("Expected a notebook metadata object")
    if (
        notebook.suffix != ".ipynb"
        or meta.get("language") != "python"
        or meta.get("kernel_type") != "notebook"
    ):
        raise PublishError("Expected a Python .ipynb and Python notebook metadata")
    if meta.get("is_private", True) is not True:
        raise PublishError("Embedded keys require is_private: true in metadata")
    ref = meta.get("id", "")
    if not isinstance(ref, str) or not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+", ref):
        raise PublishError("Metadata id must be owner/notebook-slug")
    if type(timeout) is not int or not 1 <= timeout <= 21600:
        raise PublishError("Session timeout must be between 1 and 21600 seconds")
    if key and key in json.dumps(meta):
        raise PublishError("The provided W&B key appears in notebook metadata")
    if TOKEN.search(json.dumps(meta)):
        raise PublishError("Notebook metadata contains a credential pattern")
    meta.update(is_private=True, code_file=notebook.name)
    # The SDK prefers id_no over id. Use the same slug for privacy checks and
    # saving so a stale numeric ID cannot redirect a credential-bearing upload.
    meta.pop("id_no", None)
    upload_document = inject(document, key)
    result = {
        "submitted": False,
        "dry_run": dry_run,
        "notebook": ref,
        "source_sha256": hashlib.sha256(original).hexdigest(),
        "session_timeout_seconds": timeout,
        "accelerator": accelerator or meta.get("machine_shape"),
    }
    phase = "authentication"
    try:
        # SDK errors may include request bodies. Neither its output nor its
        # raw exceptions are safe to forward from a credential-bearing call.
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            if not dry_run:
                client = client or KaggleClient()
                phase = "target privacy check"
                if client.is_private(ref) is False:
                    raise PublishError(
                        "Target is public or privacy is unconfirmed; no notebook was sent"
                    )
            with tempfile.TemporaryDirectory(prefix="kaggle-private-upload-") as temporary:
                folder = Path(temporary)
                if folder.resolve().is_relative_to(ROOT.resolve()):
                    raise PublishError("Set TMPDIR/TEMP to a location outside the repository")
                private_write(folder / notebook.name, upload_document)
                private_write(folder / "kernel-metadata.json", meta)
                if not dry_run:
                    phase = "upload"
                    response = client.push(folder, timeout, accelerator)
                    if response is None or getattr(response, "error", None):
                        raise PublishError(
                            "Kaggle rejected the upload; inspect its notebook before retrying"
                        )
                    if any(
                        getattr(response, "invalid_" + name, None)
                        for name in (
                            "tags",
                            "dataset_sources",
                            "competition_sources",
                            "kernel_sources",
                            "model_sources",
                        )
                    ):
                        raise PublishError(
                            "Kaggle returned invalid sources/tags; "
                            "inspect the saved version before retrying"
                        )
                    version = getattr(response, "version_number", None)
                    if not isinstance(version, int) or version < 1:
                        raise PublishError(
                            "Kaggle returned no saved version; inspect the notebook before retrying"
                        )
                    phase = "private readback"
                    if client.is_private(ref) is not True:
                        raise PublishError(
                            "Upload was accepted but privacy readback failed; "
                            "inspect Kaggle before retrying"
                        )
                    result.update(
                        submitted=True, version=version, url=f"https://www.kaggle.com/code/{ref}"
                    )
    except PublishError:
        raise
    except (Exception, SystemExit) as error:
        status = getattr(getattr(error, "response", None), "status_code", None)
        http = f", HTTP {status}" if isinstance(status, int) else ""
        raise PublishError(
            f"Kaggle {phase} failed ({type(error).__name__}{http}); inspect Kaggle before retrying"
        ) from None
    result["temporary_upload_deleted"] = True
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", nargs="?", type=Path)
    parser.add_argument("--metadata", type=Path, help="Defaults to adjacent kernel-metadata.json")
    parser.add_argument(
        "--wandb-key-file", type=Path, help="Otherwise read WANDB_API_KEY from environment"
    )
    parser.add_argument("--kaggle-token-file", type=Path, help="Optional Kaggle access-token file")
    parser.add_argument("--accelerator", help="Override metadata machine_shape, e.g. TpuV5E8")
    parser.add_argument(
        "--timeout", type=int, default=21600, help="Active session cap in seconds (max 21600)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate/inject locally, then delete; no Kaggle calls",
    )
    parser.add_argument(
        "--check-repo", action="store_true", help="Offline check of nonignored source and notebooks"
    )
    args = parser.parse_args(argv)
    try:
        if args.check_repo:
            result = check_repo()
        else:
            if args.notebook is None:
                parser.error("Provide a notebook, or use --check-repo")
            key = (
                args.wandb_key_file.expanduser().read_text().strip()
                if args.wandb_key_file
                else os.environ.get("WANDB_API_KEY", "").strip()
            )
            if args.kaggle_token_file:
                token_path = args.kaggle_token_file.expanduser().resolve()
                if not token_path.is_file():
                    raise PublishError("Kaggle access-token file does not exist")
                os.environ["KAGGLE_API_TOKEN"] = str(token_path)
            notebook = args.notebook.expanduser()
            metadata = (
                args.metadata.expanduser()
                if args.metadata
                else notebook.with_name("kernel-metadata.json")
            )
            result = publish(
                notebook,
                metadata,
                key,
                dry_run=args.dry_run,
                accelerator=args.accelerator,
                timeout=args.timeout,
            )
    except (PublishError, OSError, ValueError) as error:
        message = (
            str(error)
            if isinstance(error, PublishError)
            else f"Invalid/unreadable input ({type(error).__name__})"
        )
        print(TOKEN.sub("[redacted]", message), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
