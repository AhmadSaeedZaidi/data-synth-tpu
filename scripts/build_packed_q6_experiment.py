#!/usr/bin/env python3
"""Stage a portable source bundle and bounded notebook; never submit a run."""

import argparse
import base64
import hashlib
import io
import json
import textwrap
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULES = ("__init__.py", "layout.py", "reference.py", "kernel.py", "baselines.py", "benchmark.py")
TESTS = ("test_packed_q6.py", "test_packed_q6_benchmark.py")


def stage(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    files = {
        f"packed_q6/{name}": (ROOT / "kaggle/packed_q6" / name).read_bytes() for name in MODULES
    }
    files.update({f"tests/{name}": (ROOT / "kaggle/tests" / name).read_bytes() for name in TESTS})
    files["README.md"] = (ROOT / "kaggle/packed_q6/README.md").read_bytes()
    manifest = {name: hashlib.sha256(data).hexdigest() for name, data in sorted(files.items())}
    files["manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            entry = zipfile.ZipInfo(name, date_time=(2026, 10, 22, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, data)
    payload = buffer.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    (destination / "packed-q6-source.zip").write_bytes(payload)

    setup = textwrap.dedent(f"""\
        import base64, hashlib, importlib.metadata, io, json, os, sys, zipfile
        from pathlib import Path

        # Metadata lookup does not initialize a TPU client in this parent.
        for package in ('jax', 'jaxlib'):
            version = importlib.metadata.version(package)
            assert version == '0.11.2', f'Expected {{package}} 0.11.2; found {{version}}'
        payload = base64.b64decode({base64.b64encode(payload).decode()!r})
        assert hashlib.sha256(payload).hexdigest() == {digest!r}
        working = Path('/kaggle/working') if Path('/kaggle/working').is_dir() else Path.cwd()
        source = working / 'packed-q6-{digest[:12]}'
        source.mkdir(exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            archive.extractall(source)
        manifest = json.loads((source / 'manifest.json').read_text())
        for name, expected in manifest.items():
            assert hashlib.sha256((source / name).read_bytes()).hexdigest() == expected, name
        print('Verified source bundle:', {digest!r})
    """)
    run = textwrap.dedent("""\
        import signal, subprocess

        environment = dict(os.environ, JAX_PLATFORMS='tpu', PYTHONPATH=str(source),
                           OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2')
        result_path = working / 'packed-q6-results.json'
        command = [sys.executable, '-m', 'packed_q6.benchmark',
                   '--output', str(result_path), '--timeout-seconds', '900']
        process = subprocess.Popen(command, env=environment, start_new_session=True)
        try:
            return_code = process.wait(timeout=930)
        except BaseException:
            # The outer deadline/interactive interrupt also kills descendants.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            raise
        result = json.loads(result_path.read_text())
        print(json.dumps(result, indent=2))
        assert return_code == 0 and result['status'] == 'passed', str(result_path)
    """)

    def cell(kind, text):
        value = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
        if kind == "code":
            value.update(execution_count=None, outputs=[])
        return value

    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
        },
        "cells": [
            cell(
                "markdown",
                "# Packed Q6 TPU projection experiment\n\n"
                "Requires a TPU runtime with JAX/jaxlib 0.11.2. Uses one chip of the allocation. "
                "A 15-minute watchdog bounds the worker. No model, key or dataset is needed. "
                "This is an unvalidated hardware prototype, not the Qwen evaluation notebook. "
                "Results are saved incrementally to `packed-q6-results.json`.\n",
            ),
            cell("code", setup),
            cell("code", run),
        ],
    }
    for i, value in enumerate(notebook["cells"]):
        value["id"] = f"packed-q6-{i}"
    (destination / "packed-q6-microbenchmark.ipynb").write_text(
        json.dumps(notebook, indent=2) + "\n"
    )
    summary = {"source_sha256": digest, "files": manifest, "uploaded": False, "queued": False}
    (destination / "staging.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=ROOT / "output/current/packed-q6")
    args = parser.parse_args()
    result = stage(args.destination)
    print(
        json.dumps({"destination": str(args.destination), "source_sha256": result["source_sha256"]})
    )
