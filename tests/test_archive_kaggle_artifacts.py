"""Historical output pruning requires a verified, recoverable archive."""

import importlib.util
import tarfile
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "archive_artifacts", Path(__file__).parents[1] / "scripts/archive_kaggle_artifacts.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_deduplicated_archive_restores_both_files_and_prunes_only_retired_output(tmp_path):
    old = tmp_path / "output/old"
    old.mkdir(parents=True)
    (old / "a.txt").write_text("original evidence")
    (old / "b.txt").write_text("original evidence")
    (old / "__pycache__").mkdir()
    (old / "__pycache__/generated.pyc").write_bytes(b"disposable")
    active = tmp_path / "output/current"
    active.mkdir()
    archive = tmp_path / "output/archives/history.tar.gz"
    result = module.archive([old], archive, prune=True, root=tmp_path)
    assert result["verified"] and result["pruned"] and result["deduplicated_files"] == 1
    assert not old.exists() and active.is_dir()
    restore = tmp_path / "restore"
    restore.mkdir()
    with tarfile.open(archive) as bundle:
        bundle.extractall(restore, filter="data")
    assert (restore / "output/old/a.txt").read_text() == "original evidence"
    assert (restore / "output/old/b.txt").read_text() == "original evidence"
    assert not (restore / "output/old/__pycache__").exists()


def test_active_and_outside_directories_cannot_be_archived(tmp_path):
    for name in ["output/current", "output/.task-secrets", "source"]:
        path = tmp_path / name
        path.mkdir(parents=True, exist_ok=True)
        with pytest.raises(ValueError):
            module.archive([path], tmp_path / "bad.tar.gz", prune=True, root=tmp_path)


def test_relative_command_line_paths_are_supported(tmp_path, monkeypatch):
    old = tmp_path / "output/old"
    old.mkdir(parents=True)
    (old / "evidence.txt").write_text("preserved")
    monkeypatch.chdir(tmp_path)
    result = module.archive(
        [Path("output/old")], Path("output/archives/history.tar.gz"), root=tmp_path
    )
    assert result["verified"] and result["archive"] == "output/archives/history.tar.gz"
    assert old.is_dir()
