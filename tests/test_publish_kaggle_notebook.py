"""Publishing credentials must never modify source, enter logs or survive staging."""

import importlib.util
import json
import os
import stat
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    "publish_kaggle_notebook", Path(__file__).parents[1] / "scripts/publish_kaggle_notebook.py"
)
publisher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publisher)
KEY = "wandb_v1_" + "fakeForOfflineTest" * 4


@pytest.fixture
def source(tmp_path):
    notebook = tmp_path / "safe.ipynb"
    document = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {"widgets": {"transient_state": "old"}},
        "cells": [
            {
                "cell_type": "code",
                "metadata": {},
                "source": ["print('hello')\n"],
                "outputs": [{"output_type": "stream", "text": "old output"}],
                "execution_count": 7,
            }
        ],
    }
    notebook.write_text(json.dumps(document))
    metadata = tmp_path / "kernel-metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "id": "test-owner/private-notebook",
                "id_no": 999,
                "title": "Private Notebook",
                "code_file": "old.ipynb",
                "is_private": True,
                "language": "python",
                "kernel_type": "notebook",
                "dataset_sources": ["test-owner/engine/11"],
                "machine_shape": "TpuV5E8",
                "docker_image": "pinned-image",
            }
        )
    )
    return notebook, metadata


class FakeClient:
    def __init__(self, *, privacy=(True, True), fail=False, response=None):
        self.privacy = iter(privacy)
        self.fail = fail
        self.response = response or SimpleNamespace(error="", version_number=11)
        self.pushes = 0
        self.folder = None

    def is_private(self, ref):
        assert ref == "test-owner/private-notebook"
        return next(self.privacy)

    def push(self, folder, timeout, accelerator):
        self.pushes += 1
        self.folder = folder
        self.notebook = json.loads((folder / "safe.ipynb").read_text())
        self.metadata = json.loads((folder / "kernel-metadata.json").read_text())
        if os.name == "posix":
            assert stat.S_IMODE(folder.stat().st_mode) == 0o700
            assert stat.S_IMODE((folder / "safe.ipynb").stat().st_mode) == 0o600
        assert timeout == 21600 and accelerator == "TpuV5E8"
        print("SDK body could contain " + KEY)
        if self.fail:
            raise RuntimeError("Request body: " + KEY)
        return self.response


def test_private_upload_preserves_source_and_cleans_credentials(source, capsys):
    notebook, metadata = source
    originals = notebook.read_bytes(), metadata.read_bytes()
    client = FakeClient()
    result = publisher.publish(notebook, metadata, KEY, client=client, accelerator="TpuV5E8")
    assert result["submitted"] and result["version"] == 11 and result["temporary_upload_deleted"]
    assert not client.folder.exists()
    assert (notebook.read_bytes(), metadata.read_bytes()) == originals
    assert KEY in "".join(client.notebook["cells"][0]["source"])
    assert client.notebook["cells"][1]["outputs"] == []
    assert client.notebook["cells"][1]["execution_count"] is None
    assert "widgets" not in client.notebook["metadata"]
    assert client.metadata["is_private"] is True
    assert "id_no" not in client.metadata
    assert client.metadata["code_file"] == "safe.ipynb"
    assert client.metadata["dataset_sources"] == ["test-owner/engine/11"]
    assert client.metadata["docker_image"] == "pinned-image"
    assert KEY not in capsys.readouterr().out + json.dumps(result)


def test_failure_never_echoes_sdk_body_and_deletes_upload(source, capsys):
    client = FakeClient(fail=True)
    with pytest.raises(publisher.PublishError) as error:
        publisher.publish(*source, KEY, client=client, accelerator="TpuV5E8")
    captured = capsys.readouterr()
    assert KEY not in str(error.value) + captured.out + captured.err
    assert not client.folder.exists() and client.pushes == 1


@pytest.mark.parametrize("where", ["metadata", "server"])
def test_refuses_public_notebooks_before_upload(source, where):
    notebook, metadata = source
    client = FakeClient(privacy=(False,))
    if where == "metadata":
        data = json.loads(metadata.read_text())
        data["is_private"] = False
        metadata.write_text(json.dumps(data))
    with pytest.raises(publisher.PublishError):
        publisher.publish(notebook, metadata, KEY, client=client)
    assert client.pushes == 0


def test_new_private_target_and_failed_post_upload_readback(source):
    result = publisher.publish(
        *source, KEY, client=FakeClient(privacy=(None, True)), accelerator="TpuV5E8"
    )
    assert result["submitted"]
    client = FakeClient(privacy=(True, False))
    with pytest.raises(publisher.PublishError, match="accepted"):
        publisher.publish(*source, KEY, client=client, accelerator="TpuV5E8")
    assert client.pushes == 1 and not client.folder.exists()


@pytest.mark.parametrize("field", ["error", "invalid_dataset_sources"])
def test_response_errors_are_sanitized_and_not_retried(source, field):
    response = SimpleNamespace(version_number=11, **{field: [KEY]})
    client = FakeClient(response=response)
    with pytest.raises(publisher.PublishError) as error:
        publisher.publish(*source, KEY, client=client, accelerator="TpuV5E8")
    assert KEY not in str(error.value)
    assert client.pushes == 1 and not client.folder.exists()


@pytest.mark.parametrize(
    "code",
    [
        "import os\nos.environ['WANDB_API_KEY'] = 'old-legacy-key'",
        "WANDB_API_KEY = 'old-legacy-key'",
        "wandb.login(key='old-legacy-key')",
        "os.environ.setdefault('WANDB_API_KEY', 'old-legacy-key')",
        "config = {'WANDB_API_KEY': 'old-legacy-key'}",
        "%pip install wandb\nos.environ['WANDB_API_KEY'] = 'old-legacy-key'",
    ],
)
def test_rejects_existing_literal_authentication(source, code):
    document = json.loads(source[0].read_text())
    document["cells"][0]["source"] = code
    with pytest.raises(publisher.PublishError):
        publisher.inject(document, KEY)


def test_rejects_credentials_in_saved_outputs(source):
    document = json.loads(source[0].read_text())
    document["cells"][0]["outputs"][0]["text"] = KEY
    with pytest.raises(publisher.PublishError):
        publisher.inject(document, KEY)


def test_key_is_python_data_not_executable_source(source, monkeypatch):
    key = 'fake-key"; raise RuntimeError("must not execute"); #'
    for name in ("WANDB_API_KEY", "WANDB_DISABLE_CODE", "WANDB_DISABLE_GIT"):
        monkeypatch.setenv(name, "old-value")
    cell = publisher.inject(json.loads(source[0].read_text()), key)["cells"][0]
    exec("".join(cell["source"]), {})
    assert os.environ["WANDB_API_KEY"] == key


def test_dry_run_does_not_import_sdk_or_modify_source(source, monkeypatch, capsys):
    original = source[0].read_bytes()
    monkeypatch.setattr(publisher, "KaggleClient", lambda: pytest.fail("SDK used in dry run"))
    monkeypatch.setenv("WANDB_API_KEY", KEY)
    assert publisher.main([str(source[0]), "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert KEY not in output and json.loads(output)["submitted"] is False
    assert source[0].read_bytes() == original


def test_repo_check_ignores_private_artifacts_but_catches_source_leaks(tmp_path):
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text("output/\n")
    (tmp_path / "source.py").write_text("print('safe')\n")
    (tmp_path / "output").mkdir()
    (tmp_path / "output/private.txt").write_text(KEY)
    assert publisher.check_repo(tmp_path)["status"] == "PASS"
    (tmp_path / "source.py").write_text(KEY)
    with pytest.raises(publisher.PublishError) as error:
        publisher.check_repo(tmp_path)
    assert KEY not in str(error.value)


def test_session_budget_and_empty_key_fail_before_any_upload(source):
    client = FakeClient()
    with pytest.raises(publisher.PublishError):
        publisher.publish(*source, KEY, timeout=21601, client=client)
    with pytest.raises(publisher.PublishError):
        publisher.publish(*source, "", client=client)
    assert client.pushes == 0


@pytest.mark.parametrize("document", [[], {"nbformat": 4, "cells": [], "metadata": []}])
def test_invalid_notebook_structure_has_a_safe_error(document):
    with pytest.raises(publisher.PublishError):
        publisher.inject(document, KEY)


def test_cli_file_key_path_and_error_do_not_echo_secrets(source, tmp_path, capsys):
    key_file = tmp_path / "key"
    key_file.write_text(KEY)
    assert publisher.main([str(source[0]), "--wandb-key-file", str(key_file), "--dry-run"]) == 0
    assert KEY not in capsys.readouterr().out
    key_file.write_text("")
    assert publisher.main([str(source[0]), "--wandb-key-file", str(key_file), "--dry-run"]) == 1
    assert "nonempty" in capsys.readouterr().err


def test_configured_temporary_directory_cannot_put_keys_in_checkout(source, monkeypatch):
    monkeypatch.setattr(publisher, "ROOT", Path(tempfile.gettempdir()))
    with pytest.raises(publisher.PublishError, match="outside"):
        publisher.publish(*source, KEY, dry_run=True)


@pytest.mark.parametrize("private", [None, False, True])
def test_new_target_lookup_distinguishes_absence_from_existing_public_target(private):
    class Inventory:
        def get_config_value(self, key):
            assert key == "username"
            return "test-owner"

        def kernels_list(self, **kwargs):
            assert kwargs == {"page": 1, "page_size": 100, "mine": True, "user": "test-owner"}
            return (
                []
                if private is None
                else [SimpleNamespace(ref="test-owner/private-notebook", is_private=private)]
            )

    assert publisher.owned_target_visibility(Inventory(), "test-owner/private-notebook") is private


def test_missing_target_lookup_refuses_other_owners_and_unknown_inventory():
    api = SimpleNamespace(get_config_value=lambda key: "different-owner")
    with pytest.raises(publisher.PublishError, match="authenticated account"):
        publisher.owned_target_visibility(api, "test-owner/private-notebook")
    api = SimpleNamespace(get_config_value=lambda key: "test-owner", kernels_list=lambda **kw: None)
    with pytest.raises(publisher.PublishError, match="unavailable"):
        publisher.owned_target_visibility(api, "test-owner/private-notebook")
