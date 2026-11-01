#!/usr/bin/env python3
"""Archive retired output directories, verify contents, then optionally prune originals."""

import argparse
import gzip
import hashlib
import json
import os
import shutil
import stat
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTECTED = {"current", "tooling", "reviews", "archives", ".task-secrets"}
CACHES = {"__pycache__", ".pytest_cache", ".ruff_cache"}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def entries(top):
    for directory, dirs, files in os.walk(top, followlinks=False):
        dirs[:] = sorted(name for name in dirs if name not in CACHES)
        path = Path(directory)
        yield path
        for name in dirs:
            if (path / name).is_symlink():
                yield path / name
        for name in sorted(files):
            if not name.endswith((".pyc", ".pyo")):
                yield path / name


def archive(paths, target, *, prune=False, root=ROOT):
    root, target = Path(root).absolute(), Path(target).absolute()
    output = root / "output"
    selected = [path.absolute() for path in paths]
    for path in selected:
        if (
            path.parent != output
            or path.name in PROTECTED
            or not path.is_dir()
            or path.is_symlink()
        ):
            raise ValueError(
                f"Only retired top-level output directories may be archived: {path.name}"
            )
    if target.exists():
        raise FileExistsError("Archive already exists; do not overwrite historical evidence")
    target.parent.mkdir(parents=True, exist_ok=True)
    records, identical = {}, {}
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with (
        os.fdopen(descriptor, "wb") as raw,
        gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=3, mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w|") as bundle,
    ):
        for top in selected:
            print(f"Archiving {top.name}", flush=True)
            for path in entries(top):
                name = path.relative_to(root).as_posix()
                info = bundle.gettarinfo(str(path), arcname=name)
                record = {"mode": stat.S_IMODE(path.lstat().st_mode)}
                if path.is_symlink():
                    record.update(kind="symlink", link=os.readlink(path))
                elif path.is_file():
                    sha = digest(path)
                    record.update(kind="file", bytes=path.stat().st_size, sha256=sha)
                    key = (sha, record["bytes"], record["mode"])
                    if key in identical:
                        info.type, info.linkname, info.size = tarfile.LNKTYPE, identical[key], 0
                        record["duplicate_of"] = identical[key]
                    else:
                        identical[key] = name
                    if info.isreg():
                        with path.open("rb") as stream:
                            bundle.addfile(info, stream)
                    else:
                        bundle.addfile(info)
                else:
                    record["kind"] = "directory"
                if record["kind"] != "file":
                    bundle.addfile(info)
                records[name] = record
    seen = set()
    with tarfile.open(target, "r|gz") as bundle:
        for member in bundle:
            record = records[member.name]
            if member.isreg():
                with bundle.extractfile(member) as stream:
                    actual = hashlib.file_digest(stream, "sha256").hexdigest()
                if actual != record["sha256"]:
                    raise ValueError("Archive file hash mismatch; originals retained")
            elif member.islnk():
                if (
                    member.linkname not in seen
                    or records[member.linkname].get("sha256") != record["sha256"]
                ):
                    raise ValueError("Archive duplicate target mismatch; originals retained")
            elif member.issym() and member.linkname != record["link"]:
                raise ValueError("Archive symlink mismatch; originals retained")
            seen.add(member.name)
    if seen != records.keys():
        raise ValueError("Archive membership mismatch; originals retained")
    result = {
        "archive": str(target.relative_to(root)),
        "archive_bytes": target.stat().st_size,
        "sha256": digest(target),
        "verified": True,
        "original_directories": [str(p.relative_to(root)) for p in selected],
        "logical_file_bytes": sum(r.get("bytes", 0) for r in records.values()),
        "deduplicated_files": sum("duplicate_of" in r for r in records.values()),
        "excluded_regenerable_caches": sorted(CACHES),
        "pruned": False,
        "entries": records,
    }
    index = target.with_suffix(target.suffix + ".index.json")
    index.write_text(json.dumps(result, indent=2) + "\n")
    index.chmod(0o600)
    if prune:
        current = {p.relative_to(root).as_posix(): p for top in selected for p in entries(top)}
        if current.keys() != records.keys():
            raise ValueError("Source membership changed during archiving; originals retained")
        for name, path in current.items():
            r = records[name]
            if r["kind"] == "file" and digest(path) != r["sha256"]:
                raise ValueError("Source changed during archiving; originals retained")
            if r["kind"] == "symlink" and os.readlink(path) != r["link"]:
                raise ValueError("Source link changed during archiving; originals retained")
        for path in selected:
            shutil.rmtree(path)
        result["pruned"] = True
        index.write_text(json.dumps(result, indent=2) + "\n")
    return {key: value for key, value in result.items() if key != "entries"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", type=Path, nargs="+")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument(
        "--prune",
        action="store_true",
        help="Remove originals only after round-trip/hash verification",
    )
    args = parser.parse_args()
    print(json.dumps(archive(args.paths, args.archive, prune=args.prune), indent=2))
