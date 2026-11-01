#!/usr/bin/env python3
"""Build a credential-free Kaggle CPU notebook for remote tracking verification."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build(output, key_file=None):
    files = {
        f"glm53_offload/{name}": (ROOT / "kaggle/glm53_offload" / name).read_text()
        for name in (
            "__init__.py",
            "diagnostics.py",
            "monitoring.py",
            "tracing.py",
            "tracking_check.py",
        )
    }
    files["tracking_cpu_probe.py"] = (ROOT / "kaggle/tracking_cpu_probe.py").read_text()
    hashes = {name: hashlib.sha256(text.encode()).hexdigest() for name, text in files.items()}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    setup = """import os, json, subprocess, sys, time
from pathlib import Path
ROOT = Path("/kaggle/working/tracking-cpu-smoke")
ROOT.mkdir(parents=True, exist_ok=True)
os.environ["WANDB_CONSOLE"] = "off"
os.environ["WANDB_DISABLE_CODE"] = "true"
os.environ["WANDB_DISABLE_GIT"] = "true"
subprocess.run([
    sys.executable, "-m", "pip", "install", "--quiet",
    "wandb==0.30.0", "weave==0.53.11"
], check=True, timeout=240)
"""
    sources = (
        "sources = "
        + repr(files)
        + """
for name, source in sources.items():
    path = ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
sys.path.insert(0, str(ROOT))
"""
    )
    run = f"""from tracking_cpu_probe import run
CONFIG = {{
    "wandb_mode": "online", "wandb_project": "glm53-kaggle-tpu",
    "wandb_entity": "your-wandb-entity",
    "weave_enabled": True, "local_traces": True, "allow_tracking_fallback": False,
    "run_model": False, "quant": "CPU_TRACKING_PROBE", "batch_sizes": [],
    "comparison_id": "kaggle-cpu-" + str(time.time_ns()), "tracking_source_sha256": {digest!r},
}}
proof = run(ROOT / "evidence", CONFIG)
"""

    def cell(source):
        compile(source, "tracking-cpu-notebook", "exec")
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source,
        }

    credentials = []
    if key_file is not None:
        # This is an explicitly authorized private upload artifact. Never write
        # credentials into tracked source files or the shared engine dataset.
        output = output.resolve()
        if not output.is_relative_to(ROOT / "output"):
            raise ValueError("Credential-bearing notebooks must stay under ignored output/")
        key = key_file.read_text().strip()
        if not key:
            raise ValueError("Task tracking key is empty")
        credentials = [cell("import os\nos.environ['WANDB_API_KEY'] = " + repr(key) + "\n")]
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "tracking_source_sha256": digest,
            "tracking_source_files": hashes,
        },
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": "# GLM tracking CPU smoke test\n\n"
                + (
                    "This private upload uses the task tracking key. "
                    if key_file is not None
                    else "Enable `WANDB_API_KEY` in **Add-ons → Secrets** for this notebook. "
                )
                + "This CPU-only test loads no model. PASS requires remote readback of active "
                "and completed Weave spans, nested calls, a deliberate error, W&B heartbeats "
                "and final run state. No offline fallback is allowed.",
            },
            *credentials,
            cell(setup),
            cell(sources),
            cell(run),
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    path = output / "glm53-tracking-cpu-smoke.ipynb"
    if key_file is not None:
        path.touch(mode=0o600, exist_ok=True)
        path.chmod(0o600)
    path.write_text(json.dumps(notebook, indent=2) + "\n")
    metadata = {
        "id": "your-kaggle-username/glm53-tracking-cpu-smoke",
        "title": "GLM53 tracking CPU smoke",
        "code_file": "glm53-tracking-cpu-smoke.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
    }
    (output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return {
        "source_sha256": digest,
        "upload_bytes": len(json.dumps(notebook).encode()),
        "directory": str(output),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=ROOT / "output/kaggle-notebooks/glm53-tracking-cpu-smoke"
    )
    parser.add_argument("--embed-key-file", type=Path)
    arguments = parser.parse_args()
    print(json.dumps(build(arguments.output, arguments.embed_key_file), indent=2))
