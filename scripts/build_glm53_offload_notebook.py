#!/usr/bin/env python3
"""Build the self-contained GLM Flash RAM-offload Kaggle experiment notebook.

Builds every cell and a compressed, hashed source bundle from the checkout.
No existing notebook or external scaffold is required.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import textwrap
import zipfile
import zlib
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "kaggle/glm53-flash-ram-offload.ipynb"
APP_SOURCE_ROOT = REPO_ROOT / "kaggle/glm53_offload"
VENDOR_SOURCE_ROOT = REPO_ROOT / "kaggle/vendor/glm53"
APP_ENTRYPOINT = APP_SOURCE_ROOT / "run.py"
KAGGLE_SOURCE_LIMIT = 1_000_000


def kaggle_source_bytes(notebook: dict[str, Any]) -> int:
    """Size of the source string sent by Kaggle's SDK, including JSON escaping."""
    cells = []
    for original in notebook["cells"]:
        cell = dict(original)
        if isinstance(cell.get("source"), list):
            cell["source"] = "".join(cell["source"])
        if cell["cell_type"] == "code" and "outputs" in cell:
            cell["outputs"] = []
        cells.append(cell)
    return len(json.dumps({**notebook, "cells": cells}).encode("utf-8"))


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _archive_sources() -> tuple[bytes, list[dict[str, Any]]]:
    """Create a deterministic archive and a source manifest from this checkout."""

    if not APP_ENTRYPOINT.is_file():
        raise SystemExit(f"Runtime is not ready; expected CLI entrypoint {APP_ENTRYPOINT}")
    if not APP_SOURCE_ROOT.is_dir() or not VENDOR_SOURCE_ROOT.is_dir():
        raise SystemExit("Runtime package and kaggle/vendor/glm53 source tree must exist")

    files: dict[str, Path] = {}
    for path in sorted(APP_SOURCE_ROOT.glob("*.py")):
        files[f"glm53_offload/{path.name}"] = path
    for path in sorted((APP_SOURCE_ROOT / "data").rglob("*.npz")):
        if path.is_file() and not path.is_symlink():
            relative = path.relative_to(APP_SOURCE_ROOT).as_posix()
            files[f"glm53_offload/{relative}"] = path

    for path in sorted(VENDOR_SOURCE_ROOT.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(VENDOR_SOURCE_ROOT).as_posix()
        lower_name = path.name.lower()
        if path.suffix == ".py":
            archive_name = f"vendor/glm53/{relative}"
        elif path.suffix == ".npz":
            archive_name = f"vendor/glm53/{relative}"
        elif any(token in lower_name for token in ("license", "copying", "notice", "provenance")):
            archive_name = f"vendor/glm53/{relative}"
        else:
            continue
        files[archive_name] = path

    kaggle_root = REPO_ROOT / "kaggle"
    for path in sorted(kaggle_root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        lower_name = path.name.lower()
        if not any(
            token in lower_name
            for token in ("license", "copying", "notice", "provenance", "source_manifest")
        ):
            continue
        try:
            relative = path.relative_to(kaggle_root).as_posix()
        except ValueError:
            continue
        if path.is_relative_to(VENDOR_SOURCE_ROOT):
            continue
        archive_name = (
            "SOURCE_MANIFEST.json"
            if relative.lower() == "source_manifest.json"
            else f"provenance/{relative}"
        )
        files[archive_name] = path

    app_modules = [name for name in files if name.startswith("glm53_offload/")]
    vendor_modules = [
        name for name in files if name.startswith("vendor/glm53/") and name.endswith(".py")
    ]
    data_files = [name for name in files if name.endswith(".npz")]
    notice_files = [
        name
        for name in files
        if any(
            token in PurePosixPath(name).name.lower()
            for token in ("license", "copying", "notice", "provenance")
        )
    ]
    if not app_modules or not vendor_modules or not data_files or not notice_files:
        missing = []
        if not app_modules:
            missing.append("kaggle/glm53_offload/*.py")
        if not vendor_modules:
            missing.append("kaggle/vendor/glm53/**/*.py")
        if not data_files:
            missing.append("kaggle/vendor/glm53/**/*.npz")
        if not notice_files:
            missing.append("vendor LICENSE/provenance files")
        raise SystemExit("Incomplete runtime bundle; missing " + ", ".join(missing))

    records = []
    archive_payload: dict[str, bytes] = {}
    for archive_name, source_path in sorted(files.items()):
        payload = source_path.read_bytes()
        archive_payload[archive_name] = payload
        records.append(
            {
                "path": archive_name,
                "size": len(payload),
                "sha256": _sha256(payload),
            }
        )

    manifest = {
        "format": 1,
        "source_files": records,
    }
    archive_payload["bundle-manifest.json"] = (
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")

    archive_buffer = BytesIO()
    with zipfile.ZipFile(
        archive_buffer,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for archive_name, payload in sorted(archive_payload.items()):
            info = zipfile.ZipInfo(archive_name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, payload, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)

    return archive_buffer.getvalue(), records


def _markdown(source: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def _code(source: str, *, hidden: bool = False) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if hidden:
        metadata = {
            "collapsed": True,
            "jupyter": {"source_hidden": True, "outputs_hidden": True},
        }
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": metadata,
        "outputs": [],
        "source": source.rstrip().splitlines(keepends=True),
    }


def _configuration_cell(
    run_model: bool, experiment_mode: str = "full", quant: str = "Q6_K_XL"
) -> str:
    return f"""# Human-editable experiment settings.
QUANT = {quant!r}
EXPERIMENT_MODE = {experiment_mode!r}  # comparison, full or expert_profile
QUANTIZATIONS = ["Q4_K_XL", "Q6_K_XL"]  # Comparison order; one shared session budget.
BATCH_SIZES = [1, 4, 8]
MAX_CONTEXT = 8192
MAX_NEW_TOKENS = 256  # Includes reasoning and answer; an unfinished response fails quality.
PREFILL_CHUNK = 32
PREFILL_MOE_WINDOW = 256  # Reuse experts across causal attention chunks in each window.
TILE_ROUTED_EXPERTS = True  # A busy expert enlarges only its own small compute tile.
EXPERT_TILE = 4
EXPERT_CACHE_MIB_PER_DEVICE = 7168  # Q4 and Q6 target 7 GiB/chip; startup guard can reduce it.
CACHE_SAFETY_MIB_PER_DEVICE = 1024  # Extra HBM headroom beyond state/workspace reserves.
COMPACT_DECODE_BATCHES = True  # Shrink to 8/4/2/1; padding duplicates live rows.
FULL_DECODE_WARMUP = True  # Replay the whole response before measuring each vision batch.
DIRECT_EXPERT_CACHE = True  # Read resident expert tiles without a full compact table copy.
INTEGER_DOT_PROBE = True  # Bounded resident BF16/INT8/INT4 capability and timing check.
INTEGER_PROJECTION_PROBE = True  # Isolated Q4_K projection; decoder activations remain BF16.
MATCHED_REPLAY_STEPS = 16  # Same eight image states and forced tokens at B1/B4/B8; no EOS tail.
PROFILE_MATCHED_DECODE = True  # Separate two-step image diagnostic after warmup, outside timing.
VISION_RESOLUTION_CONTROL_TOKENS = 1024  # Retry one failed BeamNG OCR case if the budget fits.
VISION_MAX_TOKENS = 1024
DEADLINE_SECONDS = {900 if experiment_mode == "expert_profile" else 21600}
MAX_NO_PROGRESS_SECONDS = {300 if experiment_mode == "expert_profile" else 1800}
PROFILE_EXPERTS = 96  # Profile mode loads only these experts from one sparse layer.
PROFILE_REPEATS = 3
THINKING_BUDGET = None  # Native thinking; 0 is an explicit forced-close diagnostic.
REASONING_EFFORT = "max"  # low, high or max; same setting for quality and batch checks.
GGUF_DIRS = []  # Optional mounted Kaggle input paths containing GGUF shards.
INPUTS = []  # Optional: [{{"text": "...", "images": ["/kaggle/input/.../image.jpg"]}}]
RUN_MODEL = {run_model}
WANDB_MODE = {("online" if run_model else "disabled")!r}  # online, offline or disabled
WANDB_PROJECT = "glm53-kaggle-tpu"
WANDB_ENTITY = "your-wandb-entity"
WEAVE_ENABLED = {run_model}  # Fine-grained traces; requires online W&B authentication.
ALLOW_TRACKING_FALLBACK = False  # Tracking must be online before loading model weights.
REQUIRE_LIVE_TRACKING = {run_model}  # Verify server-visible Weave spans and W&B heartbeats.
LOCAL_TRACES = True
LOG_COMPILES = True
CAPTURE_DEVICE_TRACE = True  # Capture the separate post-warmup image diagnostic.
LONG_IMAGE_BENCHMARKS = {experiment_mode != "comparison"}  # Full mode, if budget permits.

CONFIG = {{
    "quant": QUANT,
    "experiment_mode": EXPERIMENT_MODE,
    "profile_experts": PROFILE_EXPERTS,
    "profile_repeats": PROFILE_REPEATS,
    "batch_sizes": BATCH_SIZES,
    "max_context": MAX_CONTEXT,
    "max_new_tokens": MAX_NEW_TOKENS,
    "prefill_chunk": PREFILL_CHUNK,
    "prefill_moe_window": PREFILL_MOE_WINDOW,
    "tile_routed_experts": TILE_ROUTED_EXPERTS,
    "expert_tile": EXPERT_TILE,
    "expert_cache_mib_per_device": EXPERT_CACHE_MIB_PER_DEVICE,
    "cache_safety_mib_per_device": CACHE_SAFETY_MIB_PER_DEVICE,
    "compact_decode_batches": COMPACT_DECODE_BATCHES,
    "full_decode_warmup": FULL_DECODE_WARMUP,
    "direct_expert_cache": DIRECT_EXPERT_CACHE,
    "integer_dot_probe": INTEGER_DOT_PROBE,
    "integer_projection_probe": INTEGER_PROJECTION_PROBE,
    "matched_replay_steps": MATCHED_REPLAY_STEPS,
    "profile_matched_decode": PROFILE_MATCHED_DECODE,
    "vision_resolution_control_tokens": VISION_RESOLUTION_CONTROL_TOKENS,
    "vision_max_tokens": VISION_MAX_TOKENS,
    "deadline_seconds": DEADLINE_SECONDS,
    "max_no_progress_seconds": MAX_NO_PROGRESS_SECONDS,
    "thinking_budget": THINKING_BUDGET,
    "reasoning_effort": REASONING_EFFORT,
    "gguf_dirs": GGUF_DIRS,
    "inputs": INPUTS,
    "run_model": RUN_MODEL,
    "wandb_mode": WANDB_MODE,
    "wandb_project": WANDB_PROJECT,
    "wandb_entity": WANDB_ENTITY,
    "weave_enabled": WEAVE_ENABLED,
    "allow_tracking_fallback": ALLOW_TRACKING_FALLBACK,
    "require_live_tracking": REQUIRE_LIVE_TRACKING,
    "local_traces": LOCAL_TRACES,
    "log_compiles": LOG_COMPILES,
    "capture_device_trace": CAPTURE_DEVICE_TRACE,
    "long_image_benchmarks": LONG_IMAGE_BENCHMARKS,
}}
CONFIG
"""


def _comparison_run_cell() -> str:
    source = _run_cell()
    return f"""import json
from glm53_offload.comparison import run_comparison

CONFIG["quantizations"] = QUANTIZATIONS
comparison_report, COMPARISON_REPORT_PATH = run_comparison(
    CONFIG, runtime_root=_runtime_root, runtime_dir=_runtime_dir,
    bundle_sha256=_EXPECTED_ARCHIVE_SHA256,
    session_started_at=_experiment_started_at,
    launcher_source={source!r},
)
print("Comparison summary:", json.dumps(comparison_report, indent=2))
if comparison_report["status"] not in ("PASS", "PREFLIGHT_ONLY"):
    raise RuntimeError(
        "Comparison ended with " + comparison_report["status"]
        + "; saved reports and evidence: " + str(COMPARISON_REPORT_PATH)
    )
"""


def _bootstrap_cell(encoded_payload: str, payload_sha256: str, archive_sha256: str) -> str:
    # Short literal lines add enough whitespace/JSON escaping to exceed Kaggle's
    # source limit. This hidden payload stays byte-identical after concatenation.
    wrapped_payload = "\n".join(
        f'    "{part}"' for part in textwrap.wrap(encoded_payload, width=4096)
    )
    return f'''# Source payload: zlib-compressed ZIP; the visible experiment cells stay small.
import base64
import hashlib
import io
import json
import tempfile
import zipfile
import zlib
from pathlib import Path, PurePosixPath

_PAYLOAD_B85 = (\n{wrapped_payload}\n)
_EXPECTED_ARCHIVE_SHA256 = "{archive_sha256}"
_EXPECTED_SOURCE_MANIFEST_SHA256 = (
    "{payload_sha256}"
)
_archive_bytes = zlib.decompress(base64.b85decode(_PAYLOAD_B85))
if hashlib.sha256(_archive_bytes).hexdigest() != _EXPECTED_ARCHIVE_SHA256:
    raise ValueError("Embedded runtime bundle failed its archive SHA-256 check")

# Support both Kaggle input mount layouts. An attached source dataset must
# agree with this notebook before any dependency install or model download.
_dataset_bundle_name = "glm53-offload-runtime.bundle"
_dataset_roots = (
    Path("/kaggle/input/glm53-spill-engine"),
    Path("/kaggle/input/datasets/your-kaggle-username/glm53-spill-engine"),
)
_dataset_bundles = [root / _dataset_bundle_name for root in _dataset_roots
                    if (root / _dataset_bundle_name).is_file()]
for _dataset_bundle in _dataset_bundles:
    _dataset_archive = _dataset_bundle.read_bytes()
    if hashlib.sha256(_dataset_archive).hexdigest() != _EXPECTED_ARCHIVE_SHA256:
        raise ValueError(f"Attached engine dataset differs from this notebook: {{_dataset_bundle}}")
    _archive_bytes = _dataset_archive
    print(f"Attached engine dataset verified: {{_dataset_bundle}}")
if not _dataset_bundles:
    print("Source dataset bundle absent; using the verified embedded runtime.")

_working = Path("/kaggle/working")
_runtime_base = _working if _working.is_dir() else Path(tempfile.gettempdir())
_runtime_root = _runtime_base / "glm53-offload-runtime"
_runtime_dir = _runtime_root / "sources" / _EXPECTED_ARCHIVE_SHA256
_runtime_dir.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(_archive_bytes)) as _bundle:
    _manifest_bytes = _bundle.read("bundle-manifest.json")
    if hashlib.sha256(_manifest_bytes).hexdigest() != _EXPECTED_SOURCE_MANIFEST_SHA256:
        raise ValueError("Embedded runtime bundle failed its source-manifest SHA-256 check")
    _manifest = json.loads(_manifest_bytes)
    _expected = {{item["path"]: item for item in _manifest["source_files"]}}
    _members = set(_bundle.namelist())
    if set(_expected) | {{"bundle-manifest.json"}} != _members:
        raise ValueError("Embedded runtime bundle has unexpected or missing files")
    for _relative, _record in _expected.items():
        _path = PurePosixPath(_relative)
        if _path.is_absolute() or ".." in _path.parts:
            raise ValueError(f"Unsafe path in source manifest: {{_relative}}")
        _data = _bundle.read(_relative)
        if len(_data) != _record["size"] or hashlib.sha256(_data).hexdigest() != _record["sha256"]:
            raise ValueError(f"Embedded file failed integrity check: {{_relative}}")
        _target = _runtime_dir.joinpath(*_path.parts)
        _target.parent.mkdir(parents=True, exist_ok=True)
        _target.write_bytes(_data)
    (_runtime_dir / "bundle-manifest.json").write_bytes(_manifest_bytes)

print(f"Runtime source ready: {{len(_expected)}} verified files at {{_runtime_dir}}")
'''


def _dependency_cell() -> str:
    return """import subprocess
import sys
import os
import time
import importlib.metadata
from collections import deque

_experiment_started_at = time.monotonic()
sys.path.insert(0, str(_runtime_dir))
from glm53_offload.diagnostics import bounded_call
if CONFIG.get("wandb_mode") == "online" and not os.environ.get("WANDB_API_KEY"):
    try:
        from glm53_offload.tracking_check import load_kaggle_key
        load_kaggle_key()
    except Exception as error:
        if not CONFIG.get("allow_tracking_fallback"):
            raise RuntimeError("Online tracking required: " + str(error)) from None
        CONFIG["wandb_mode"] = "offline"
        CONFIG["weave_enabled"] = False
        CONFIG["tracking_warning"] = "WANDB_API_KEY unavailable; using local/offline tracking"
        print(CONFIG["tracking_warning"])
if not CONFIG["run_model"]:
    print("RUN_MODEL=False: dependency installation skipped for local validation.")
else:
    dependency_command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "jax[tpu]==0.11.2",
        "transformers==5.18.0",
        "huggingface-hub==1.33.0",
        "pillow==12.3.0",
        "ml-dtypes==0.6.0",
        "numpy==2.5.3",
        "wandb==0.30.0",
        "weave==0.53.11",
    ]
    dependency_log_path = _runtime_dir / "dependency-setup.log"
    with dependency_log_path.open("w", encoding="utf-8") as dependency_log:
        dependency_result = subprocess.run(
            dependency_command,
            stdout=dependency_log,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=max(1, min(1800, CONFIG["deadline_seconds"] - (
                time.monotonic() - _experiment_started_at
            ))),
            check=False,
        )
    tail = deque(maxlen=30)
    with dependency_log_path.open(encoding="utf-8", errors="replace") as dependency_log:
        for line in dependency_log:
            tail.append(line.rstrip())
    print("Dependency setup log:", dependency_log_path)
    for line in tail:
        print(line[:300])
    if dependency_result.returncode != 0:
        raise RuntimeError(
            f"Dependency setup failed with return code {dependency_result.returncode}"
        )
    expected_versions = {
        "jax": "0.11.2", "transformers": "5.18.0", "huggingface-hub": "1.33.0",
        "pillow": "12.3.0", "ml-dtypes": "0.6.0", "numpy": "2.5.3",
        "wandb": "0.30.0", "weave": "0.53.11",
    }
    actual_versions = {name: importlib.metadata.version(name) for name in expected_versions}
    (_runtime_dir / "dependency-summary.json").write_text(
        json.dumps({"expected": expected_versions, "actual": actual_versions}, indent=2) + "\\n"
    )
    if actual_versions != expected_versions:
        raise RuntimeError("Installed package versions differ from the pinned experiment versions")
"""


def _run_cell() -> str:
    return """import json
import os
import subprocess
import sys
import time
import traceback
from collections import deque

RUN_ROOT = _runtime_root / "experiment-run"
RUN_ROOT.mkdir(parents=True, exist_ok=True)
RUN_DIR = Path(tempfile.mkdtemp(prefix=CONFIG["quant"] + "-", dir=RUN_ROOT))
CONFIG_PATH = RUN_DIR / "config.json"
REPORT_PATH = RUN_DIR / "report.json"
LOG_PATH = RUN_DIR / "worker.log"
TRACEBACK_PATH = RUN_DIR / "worker-traceback.txt"
CONFIG["runtime_bundle_sha256"] = _EXPECTED_ARCHIVE_SHA256
sys.path.insert(0, str(_runtime_dir))
from glm53_offload.monitoring import RunMonitor
from glm53_offload.supervisor import supervise
from glm53_offload.diagnostics import atomic_json, bounded_call, package_evidence
monitor = RunMonitor(
    RUN_DIR, CONFIG, mode=CONFIG.get("wandb_mode", "disabled"),
    project=CONFIG.get("wandb_project", "glm53-kaggle-tpu"),
    entity=CONFIG.get("wandb_entity"),
)
if monitor.state["mode"] != "online":
    CONFIG["weave_enabled"] = False
if CONFIG.get("require_live_tracking"):
    from glm53_offload.tracking_check import require_live_tracking
    try:
        tracking_proof = bounded_call(
            lambda: require_live_tracking(monitor, CONFIG, RUN_DIR), timeout=150
        )
        print("Tracking server readback PASS:", tracking_proof["weave_url"])
    except Exception as error:
        atomic_json(RUN_DIR / "tracking-readback.json", {
            "status": "FAIL", "error_type": type(error).__name__,
        })
        try:
            bounded_call(lambda: monitor.finish(1), timeout=25)
        except Exception:
            pass
        package_evidence(RUN_DIR, _runtime_root / ("tracking-startup-" + RUN_DIR.name + ".zip"))
        raise RuntimeError(
            "Online tracking failed before model work; see tracking-readback.json"
        ) from None
CONFIG["worker_budget_seconds"] = max(0, int(CONFIG["deadline_seconds"] - (
    time.monotonic() - _experiment_started_at
)))
CONFIG_PATH.write_text(json.dumps(CONFIG, indent=2) + "\\n", encoding="utf-8")
REPORT_PATH.unlink(missing_ok=True)
TRACEBACK_PATH.unlink(missing_ok=True)

worker_env = os.environ.copy()
worker_env["PYTHONUNBUFFERED"] = "1"
worker_env["PYTHONPATH"] = os.pathsep.join(
    [str(_runtime_dir), str(_runtime_dir / "vendor"), worker_env.get("PYTHONPATH", "")]
).rstrip(os.pathsep)
command = [
    sys.executable,
    "-m",
    "glm53_offload.run",
    "--config",
    str(CONFIG_PATH),
    "--output",
    str(REPORT_PATH),
]
print("Worker config:", CONFIG_PATH)
print("Worker report:", REPORT_PATH)
print("Run enabled:", CONFIG["run_model"])

timeout_seconds = CONFIG["worker_budget_seconds"]
timed_out = False
progress_timed_out = False
spawn_error = None
started_at = time.monotonic()
process = None
supervision = {}
EVIDENCE_PATH = _runtime_root / ("evidence-" + RUN_DIR.name + ".zip")

def save_evidence():
    try:
        package_evidence(
            RUN_DIR, EVIDENCE_PATH, extra_files=[
                _runtime_dir / "dependency-summary.json",
                _runtime_dir / "dependency-setup.log",
                _runtime_dir / "bundle-manifest.json",
            ],
        )
        print("Diagnostic evidence archive:", EVIDENCE_PATH)
    except Exception as exc:
        print("Evidence packaging failed:", type(exc).__name__)

try:
    with LOG_PATH.open("w", encoding="utf-8") as worker_log:
        process = subprocess.Popen(
            command,
            cwd=_runtime_dir,
            env=worker_env,
            stdout=worker_log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        supervision = supervise(
            process, RUN_DIR, monitor, deadline_seconds=timeout_seconds,
            idle_seconds=int(CONFIG.get("max_no_progress_seconds", 0)),
        )
        timed_out = supervision["status"] == "TIMEOUT"
        progress_timed_out = supervision["status"] == "STALLED"
except OSError as exc:
    spawn_error = repr(exc)
    TRACEBACK_PATH.write_text(
        f"Could not start the runtime worker.\\n{exc!r}\\n",
        encoding="utf-8",
    )
except KeyboardInterrupt:
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    raise
finally:
    worker_exit = process.returncode if process is not None else None
    if not (RUN_DIR / "supervisor.json").exists():
        atomic_json(RUN_DIR / "supervisor.json", {
            "status": "SPAWN_FAILED", "returncode": worker_exit,
        })
    try:
        if supervision.get("monitor_thread_stopped", True):
            finish_code = (
                1 if timed_out or progress_timed_out or worker_exit is None else worker_exit
            )
            try:
                bounded_call(lambda: monitor.finish(finish_code), timeout=25)
            except Exception as exc:
                print("Tracking finalization failed:", type(exc).__name__)
        else:
            print("Tracking thread did not stop; supervisor evidence remains available.")
    finally:
        save_evidence()
if timed_out:
    TRACEBACK_PATH.write_text(
        f"Worker exceeded the {timeout_seconds}-second notebook timeout.\\n",
        encoding="utf-8",
    )
if progress_timed_out:
    TRACEBACK_PATH.write_text(
        "Worker stopped after the configured interval without reported progress.\\n",
        encoding="utf-8",
    )
completed = process

tail = deque(maxlen=20)
if LOG_PATH.exists():
    with LOG_PATH.open(encoding="utf-8", errors="replace") as worker_log:
        for line in worker_log:
            tail.append(line.rstrip())
if tail:
    print("\\nRecent worker log lines:")
    for line in tail:
        print(line[:240])

if not REPORT_PATH.is_file():
    if TRACEBACK_PATH.exists():
        detail = TRACEBACK_PATH.read_text(encoding="utf-8")
    else:
        detail = "No report was written."
    raise RuntimeError(f"Worker did not produce report.json. Details: {detail}")

try:
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
except Exception:
    TRACEBACK_PATH.write_text(traceback.format_exc(), encoding="utf-8")
    raise RuntimeError(f"Could not parse worker report; traceback: {TRACEBACK_PATH}")
if not isinstance(report, dict):
    TRACEBACK_PATH.write_text(
        "Worker report must be a JSON object, not " + type(report).__name__,
        encoding="utf-8",
    )
    raise TypeError(f"Worker report must be a JSON object; details: {TRACEBACK_PATH}")
status = str(report.get("status", "unknown")).lower()
model_check = str(report.get("model_check", "unknown")).upper()
validation = report.get("validation")
if not isinstance(validation, dict):
    validation = {}
decoder_status = str(validation.get("decoder", "unknown")).upper()
vision_status = str(validation.get("vision", "unknown")).upper()
quality = report.get("quality")
if not isinstance(quality, dict):
    quality = {}
benchmarks = report.get("benchmarks")
if not isinstance(benchmarks, list):
    benchmarks = []
errors = report.get("errors")
error_message = report.get("error")
if completed is not None and completed.returncode != 0 and not error_message:
    error_message = f"Worker exited with return code {completed.returncode}"
if spawn_error and not error_message:
    error_message = spawn_error
has_error = (
    timed_out
    or progress_timed_out
    or status in {"error", "fail", "failed", "failure"}
    or bool(errors)
    or bool(error_message)
)
if decoder_status != "PASS" or vision_status != "PASS":
    has_error = True
if CONFIG["run_model"] and CONFIG.get("experiment_mode") == "expert_profile":
    profile_cases = report.get("expert_profile", {}).get("cases", [])
    expected_cases = len(set([*CONFIG["batch_sizes"], CONFIG["prefill_chunk"]])) + 1
    if (
        status != "profile_complete" or model_check != "NOT_RUN"
        or len(profile_cases) != expected_cases
        or any(case.get("status") != "PASS" for case in profile_cases)
    ):
        has_error = True
elif CONFIG["run_model"]:
    text_smoke = report.get("text_smoke")
    if not isinstance(text_smoke, dict) or str(text_smoke.get("status", "")).upper() != "PASS":
        has_error = True
    quality_status = str(quality.get("status", "")).upper()
    if status != "pass" or model_check != "PASS" or quality_status != "PASS":
        has_error = True
    failed_cases = quality.get("failed_cases")
    if (
        int(quality.get("asserted_cases", 0)) <= 0
        or not isinstance(failed_cases, list)
        or failed_cases
    ):
        has_error = True
    measured = {
        int(item["batch_size"])
        for item in benchmarks
        if isinstance(item, dict)
        and str(item.get("status", "")).upper() == "PASS"
        and "batch_size" in item
    }
    if not set(int(size) for size in CONFIG["batch_sizes"]).issubset(measured):
        has_error = True
else:
    if status != "preflight_only" or model_check != "NOT_RUN":
        has_error = True
traceback_text = report.get("traceback")
if not traceback_text and isinstance(report.get("error"), dict):
    traceback_text = report["error"].get("traceback")
if traceback_text:
    TRACEBACK_PATH.write_text(str(traceback_text), encoding="utf-8")
if has_error and not TRACEBACK_PATH.exists():
    failure_details = {
        "status": report.get("status"),
        "model_check": report.get("model_check"),
        "validation": report.get("validation"),
        "text_smoke": report.get("text_smoke"),
        "quality": report.get("quality"),
        "benchmarks": report.get("benchmarks"),
        "errors": errors,
        "error": error_message,
    }
    TRACEBACK_PATH.write_text(json.dumps(failure_details, indent=2, default=str), encoding="utf-8")

provenance = report.get("provenance")
if not isinstance(provenance, dict):
    provenance = {}
report_config = report.get("config")
if not isinstance(report_config, dict):
    report_config = {}
provenance_keys = (
    "base", "experts", "quantization", "thinking_budget", "reasoning_effort",
    "generation_token_limit", "transport",
)
quality_summary = {
    key: quality[key] for key in ("status", "asserted_cases", "failed_cases") if key in quality
}
text_smoke = report.get("text_smoke")
if not isinstance(text_smoke, dict):
    text_smoke = {}
text_smoke_summary = {
    key: text_smoke[key] for key in ("status", "answer", "match_detail") if key in text_smoke
}
benchmark_summary = [
    {
        key: item[key]
        for key in (
            "batch_size",
            "status",
            "prefill_seconds",
            "decode_seconds",
            "generated_tokens",
            "decode_generated_tokens",
            "tokens_per_second",
            "end_to_end_tokens_per_second",
            "bytes_transferred",
            "decode_bytes_transferred",
            "new_programs",
            "expert_cache_hits",
            "expert_cache_misses",
        )
        if key in item
    }
    for item in benchmarks
    if isinstance(item, dict)
]
memory_summary = {}
for stage, snapshot in report.get("memory_by_stage", {}).items():
    host_peak = snapshot.get("host", {}).get("VmHWM_bytes")
    chips = snapshot.get("devices", [])
    memory_summary[stage] = {
        "host_peak_GiB": round(host_peak / 2**30, 2) if host_peak is not None else None,
        "device_used_GiB": [
            round(chip["bytes_in_use"] / 2**30, 2) if "bytes_in_use" in chip else None
            for chip in chips
        ],
    }
summary = {
    "status": report.get("status"),
    "model_check": report.get("model_check"),
    "quant": report_config.get("quant", provenance.get("quantization")),
    "batch_sizes": report_config.get("batch_sizes"),
    "max_context": report_config.get("max_context"),
    "validation": validation,
    "expert_cache": report.get("expert_cache"),
    "routing_stats": report.get("routing_stats"),
    "expert_profile": report.get("expert_profile"),
    "early_decode_probe": report.get("early_decode_probe"),
    "long_image_benchmarks": report.get("long_image_benchmarks"),
    "matched_replay_benchmarks": report.get("matched_replay_benchmarks"),
    "matched_decode_diagnostic": report.get("matched_decode_diagnostic"),
    "vision_resolution_controls": report.get("vision_resolution_controls"),
    "integer_dot_probe": report.get("integer_dot_probe"),
    "integer_projection_probe": report.get("integer_projection_probe"),
    "hbm_plan": report.get("hbm_plan"),
    "text_smoke": text_smoke_summary,
    "quality": quality_summary,
    "benchmarks": benchmark_summary,
    "memory_by_stage": memory_summary,
    "provenance": {key: provenance[key] for key in provenance_keys if key in provenance},
    "error": str(error_message)[:500] if error_message else None,
}
print("\\nReport summary:", json.dumps(summary, indent=2, default=str))
save_evidence()
if errors:
    print("Reported errors:", json.dumps(errors, indent=2, default=str)[:1200])
if has_error:
    trace_note = f" Full traceback: {TRACEBACK_PATH}" if TRACEBACK_PATH.exists() else ""
    reason = "Worker timed out" if timed_out else f"Worker reported status={status!r} or an error"
    raise RuntimeError(f"{reason}.{trace_note}")
"""


def build_notebook(
    *,
    out_path: Path,
    run_model: bool,
    experiment_mode: str = "full",
    quant: str = "Q6_K_XL",
    beamng_bundle: Path | None = None,
) -> dict[str, Any]:
    if experiment_mode not in ("full", "expert_profile", "comparison"):
        raise ValueError("unknown experiment mode")
    if quant not in ("Q6_K_XL", "Q4_K_XL"):
        raise ValueError("quant must be Q6_K_XL or Q4_K_XL")
    if experiment_mode == "comparison":
        quant = "Q4_K_XL"
    archive, source_records = _archive_sources()
    compressed = zlib.compress(archive, level=9)
    encoded_payload = base64.b85encode(compressed).decode("ascii")
    source_manifest = {"format": 1, "source_files": source_records}
    source_manifest_json = json.dumps(
        source_manifest,
        sort_keys=True,
        separators=(",", ":"),
    )
    source_manifest_bytes = (source_manifest_json + "\n").encode()

    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
    }

    cells = [
        _markdown(
            "# GLM-5.3-Flash: vision with experts in host RAM\n\n"
            "Test Unsloth Q6 or Q4 expert weights on Kaggle TPU v5e-8, "
            "using a custom packed expert offload path and native vision. "
            "Compare image accuracy and measured decode batches of 1, 4 and 8. "
            "This notebook is ready for a live hardware trial; full-model "
            "accuracy and TPU performance remain to be measured.\n"
        ),
        _markdown(
            """## Objective and success criteria

Measure the host RAM-offload runtime for GLM-5.3-Flash across the requested
batch sizes. The worker records load status, memory and timing measurements, and
per-batch results in a JSON report. A usable result requires an explicit
successful status and completed measurements; a launched process alone is not
evidence of inference success.

The configuration cell selects the Q6/Q4 trials, uses
`unsloth/GLM-5.3-Flash-GGUF` revision
`621d456e93e926e4b52f85cff5f634358c1828f9`, and compares batch sizes 1, 4,
and 8. The BF16 reference is `zai-org/GLM-5.3-Flash-BF16` revision
`a5b45eb41df6402735dedc900be14a42e8d5e538`.
"""
        ),
        _markdown(
            """## Experiment settings

Edit the next cell before running. Attaching a Kaggle input dataset with GGUF
shards is optional. If provided, list its mounted directory in `GGUF_DIRS`,
such as `/kaggle/input/my-glm53-q6-weights`; the source-only
`glm53-spill-engine` dataset does not contain model shards. With `GGUF_DIRS=[]`,
the runtime streams pinned GGUF data directly into host RAM, not
`/kaggle/working`. The notebook bundle contains the loader code and its small
support data, not the multi-hundred-gigabyte model shards. Loading or network
transfer may take hours. The six-hour budget covers dependency setup and all
model trials, including loading. Shutdown and diagnostic packaging can add seconds.
The worker saves its log and report under the runtime directory.

`Q6_K_XL` routes 275,452,526,592 bytes (256.535 GiB) of expert weights;
`Q4_K_XL` routes 185,478,414,336 bytes (172.740 GiB). The full Q6 checkpoint
is about 292 GB before runtime buffers, and loading or HTTP streaming can take
hours. Host RAM retains all packed experts; a bounded packed TPU LRU cache
serves repeated routes without retransferring them. The default reserves at most
7168 MiB per chip (56 GiB across eight chips), populated on demand. The startup
guard can reduce this target to fit the tightest chip, keeping batch-state,
workspace and an extra 1 GiB/chip safety allowance. Native peaks still require
validation. Set
`EXPERT_CACHE_MIB_PER_DEVICE=0` for the all-host baseline. BF16 non-expert
weights and vision weights run on the TPU. Check host capacity before setting
`RUN_MODEL=True`. Native maximum-effort thinking is enabled by default, with a
256-token generation limit including reasoning and the answer. Unfinished
reasoning returns no final answer and fails quality. Reports separate reasoning
and answer token counts; decode tokens/second counts both. `THINKING_BUDGET=0`
is an explicit forced-close diagnostic, not the default quality setting.
`INPUTS` adds custom prompts to the built-in quality and benchmark
battery; it does not replace those checks.

`DIRECT_EXPERT_CACHE=True` passes resident and miss tables to the expert kernel,
which reads only its current tile. This removes the explicit full compact-table
copy on cache hits. `False` selects the previous lossless path for comparison.
Before model loading, `INTEGER_DOT_PROBE` checks resident BF16, INT8 and INT4
dots at 1/8/128 rows on all chips, with blocked timings and HLO evidence.
`INTEGER_PROJECTION_PROBE` then compares one actual Q4_K expert projection with
W4A8 INT8, W4A8 using two INT4 dots and W4A4 using one INT4 dot. Original GGUF
scales and offsets are preserved; activation quantization error is reported.
These are isolated diagnostics. Model generation still uses BF16 activations;
a successful integer dot does not establish vision quality or peak throughput.

Batch 4 and 8 are jointly decoded, while each request is prefilled in sequence.
Decode tokens/second excludes the first token produced during prefill;
end-to-end tokens/second includes prefill time. Vision batches replay the full
response for warmup, recorded separately; later compilation is still reported as
`new_programs`. Finished rows are removed and batches shrink to widths 8/4/2/1;
unused slots duplicate a live row, reducing unrelated expert traffic. Reports
record active and processed row steps. Stage memory snapshots include host `VmHWM` and native memory
statistics from all eight TPU devices. Peak HBM is available only when the
installed libtpu reports it; otherwise snapshots are point-in-time readings.

An early repeated-text batch probe gives optimistic decode diagnostics before
the image suite. After the first eight asserted image checks, batches 1/4/8
replay their original prompt states, without additional image prefills. Saved
states reside in host RAM between checks. Short answers test batch correctness;
their throughput can be too brief to represent sustained generation. In full
single-model mode, long-description batch samples follow the image checks if
their estimated prefill cost fits the remaining budget; otherwise they record
`SKIPPED_BUDGET`. Comparison mode omits these additional samples.

`MATCHED_REPLAY_STEPS=16` additionally runs the same eight image prompt states
and fixed diagnostic token continuation in sequential, four-row and eight-row
groups. All rows advance equally, without sampling or EOS tails. Cache entries
are cleared before each width; complete workload cycles are warmed, then another
cycle is measured. Token hashes, row steps, host gather time, cache hits/misses
and bytes per step are recorded in `matched_replay_benchmarks`. This measures
engine work, not generated answers. Original short-answer batches use different
case subsets and should not be used alone to infer batching speedup.
"""
        ),
        _code(_configuration_cell(run_model, experiment_mode, quant)),
        _markdown(
            """## Prepare the bundled runtime

The hidden bootstrap cell below contains one compressed, integrity-checked
source bundle. It extracts the runtime and vendor package under
`/kaggle/working/glm53-offload-runtime` (or the system temporary directory
outside Kaggle). No GGUF checkpoint is downloaded by the bootstrap.
When `glm53-spill-engine` is attached, its runtime bundle must match this
notebook's SHA-256 before it is used. A mismatched version stops immediately.
"""
        ),
        _code(
            _bootstrap_cell(
                encoded_payload,
                _sha256(source_manifest_bytes),
                _sha256(archive),
            ),
            hidden=True,
        ),
        _markdown(
            """## Run the experiment

The dependency setup cell runs in a separate process before the worker imports
JAX. It is skipped when `RUN_MODEL=False`. Enable the Kaggle secret
`WANDB_API_KEY` for online tracking; credentials never enter the JSON config.
Online tracking is required by default. Before model work, the launcher reads
an unfinished Weave span and a W&B heartbeat back from the remote services.
Failed authentication or remote readback stops before loading model weights.
The parent relays worker logs about every two seconds and logs stage, progress age,
download bytes/rate, worker CPU/RAM, validation and benchmark metrics every
15 seconds. TPU memory readings are stage snapshots. Weave adds operation
traces. Tracking runs separately from JAX so a blocked worker remains visible.
Local metrics remain in `metrics.jsonl`. Thirty minutes without a reported
worker progress event triggers cancellation; this is a quota guard, not proof
of a deadlock. Set `MAX_NO_PROGRESS_SECONDS=0` to disable that guard.
The watchdog reads worker progress independently of the tracking thread.
Before stopping a stalled or timed-out worker, it requests Python stack frames.
Set `REQUIRE_LIVE_TRACKING=False`, `WANDB_MODE="offline"` and
`WEAVE_ENABLED=False` to record W&B locally without a dashboard. The runtime
and parent enforce the remaining six-hour setup-plus-worker budget.
Compile logs, local operation spans and bounded expert-route records are enabled.
A separate two-step replay of the same diverse images follows matched-batch
warmup and measurement. It records dependent wall timings on the first, middle
and last sparse layers and attempts a native JAX trace. These extra
synchronizations and profiler overhead do not enter throughput measurements.
Completed replay groups are checkpointed individually; an interrupted cycle
retains its completed groups and the active group. Profiler failures are
reported separately and do not repeat inference.
If the original BeamNG road-text check fails, one unchanged image is retried
with a 1024-token cap when the remaining worker budget fits its estimated cost.
Its result is a separate diagnostic; the original quality result is retained.
"""
        ),
        _code(_dependency_cell()),
        _code(_comparison_run_cell() if experiment_mode == "comparison" else _run_cell()),
        _markdown(
            """## Review and record results

Review the worker report and log paths printed above. Compare completed batch
prefill and decode timings, decode and end-to-end throughput, transfer volume,
new compiled programs, quality checks, text smoke, and per-stage host/device
memory. Stage memory snapshots include host `VmHWM` and native statistics for
all eight TPU devices; peak HBM is shown only if libtpu supplies a true peak.
Keep failed or skipped runs visible in the report; do not interpret a
configuration request as a successful batch measurement.

The printed evidence ZIP includes reports, supervisor state, log tails, stacks,
local spans, routing records and small native trace files. Its manifest records
captured hashes, omissions and truncation; model weights are excluded. The
archive captures at most 24 MiB plus its manifest. Raw files remain in the run
directory. `supervisor.json` is authoritative if the worker was killed while
its last report still said `RUNNING`.

Record the Kaggle accelerator/host configuration, mounted shard revision,
runtime source bundle hash, warm or cold state, and any OOM or timeout. The
default context and output limits are experiment controls, not a fit or
throughput guarantee.
"""
        ),
    ]
    if experiment_mode == "comparison":
        cells.insert(
            2,
            _markdown(
                "## Q4 and Q6 in one TPU session\n\n"
                "Q4 runs first, then Q6 in a fresh worker process, under one six-hour cap "
                "including setup. The first trial receives half the remaining budget; "
                "unused time passes to Q6. Each uses native maximum-effort thinking, "
                "the same selected single-image checks (colour, counting, OCR and side), "
                "and actual decode batches 1/4/8. This matched comparison does not "
                "run the full 13-case battery or sustained description benchmark. "
                "Per-trial reports and evidence survive errors and timeouts; a failed "
                "first trial does not prevent the second. The comparison index records "
                "incomplete/skipped trials explicitly. Models are released between trials "
                "by process exit; Q4 and Q6 weights are never loaded together.\n"
            ),
        )
    if experiment_mode == "expert_profile":
        cells[0] = _markdown("# GLM-5.3-Flash: bounded TPU expert kernel probe")
        cells[1] = _markdown("""## Scope and completion criteria

This copy selects `EXPERIMENT_MODE='expert_profile'`. It downloads at most
96 experts from the first sparse layer, runs the packed-decoder and tiny vision
numerical gates, then compares the original all-token expert kernel with routed
token matmuls using global and per-tile capacity for 1, 4, 8 and 32 tokens.
A hot-expert case checks occupancy padding. Compilation, host gather, blocked
transfer and warm kernel
execution are measured separately. Dependency setup and the worker share a
**15-minute hard limit**, followed by bounded shutdown. A completed probe records
`PROFILE_COMPLETE` and `model_check=NOT_RUN`. It does not load the full vision
tower or establish model accuracy or model tokens/second. Use the full notebook
for those checks after assessing these results.
""")
        cells[2] = _markdown(f"""## Experiment settings

The next cell selects `{quant}` and at most 96 real experts from the first
sparse layer. It uses the same pinned checkpoint revisions and verified runtime
bundle as the full vision notebook. `GGUF_DIRS=[]` reads bounded HTTP ranges
directly into RAM. It skips the full BF16 base and other expert layers. Full
generation, context, image-token and expert-cache settings are unused in this
probe. Edit `PROFILE_EXPERTS`, `PROFILE_REPEATS`, `PREFILL_CHUNK` or
`BATCH_SIZES` to control the measurements; the worker rejects limits above
96 experts, five repetitions or 900 seconds.
""")
        cells[6] = _markdown("""## Run and monitor the probe

Enable the Kaggle secret `WANDB_API_KEY` for this notebook. Online W&B metrics
and Weave traces identify loading, compilation and measurement progress. Tracking
startup must pass remote readback before model work. To explicitly choose local
tracking, set `REQUIRE_LIVE_TRACKING=False`, `WANDB_MODE='offline'` and
`WEAVE_ENABLED=False`. Dependencies are
installed before the worker starts and count toward the 900-second budget.
The parent stops the worker when that budget expires, or after 300 seconds
without reported progress. Each compilation and measurement phase is saved
before the next begins, so incomplete cases retain useful evidence.
""")
        cells[-1] = _markdown("""## Assess the measurements

Compare `old_kernel_seconds`, `routed_kernel_seconds`, `tile_routed_kernel_seconds`,
`kernel_speedup`, `tile_kernel_speedup`, output
agreement and occupancy counters for each case. Compilation, host gather and
blocked transfer have separate timings. Compiler cost and memory estimates are
recorded when the backend exposes them; these are not measured peak HBM.
Kernel timing reuses device buffers;
it does not include host misses, attention, vision or model generation. The
hot-expert result exposes padding overhead. Logical arithmetic savings do not
guarantee a hardware speedup. A successful result is `PROFILE_COMPLETE` with
all five default cases passing; `model_check` remains `NOT_RUN`.

The integer diagnostics also run in this mode. Q4_K checkpoints compare a
single real packed projection with W4A8/W4A4 alternatives; Q6 records a skip
instead of requantizing its weights. Compiler support, speed and activation
error are separate results, and the full decoder remains BF16.
""")
    scene_metadata = {}
    if beamng_bundle is not None:
        if experiment_mode == "expert_profile":
            raise ValueError("expert_profile does not run scene fixtures")
        with zipfile.ZipFile(beamng_bundle) as fixture_archive:
            fixture_manifest = json.loads(fixture_archive.read("manifest.json"))
        scene_sha = _sha256(beamng_bundle.read_bytes())
        scene_names = [case["name"] for case in fixture_manifest["cases"]]
        selected = ["colour_red", "count_circles_7", "count_circles_3", "read_text_stop_7"]
        scene_config = {"beamng_fixture_sha256": scene_sha, "beamng_image_max_tokens": 384}
        if experiment_mode == "comparison":
            if len(scene_names) != 4:
                raise ValueError("comparison requires four real-scene cases")
            scene_config["quality_case_names"] = [
                name for pair in zip(selected, scene_names, strict=True) for name in pair
            ]
        scene_cell = _code(f"""# Real captures are dataset assets, not embedded notebook payloads.
_scene_candidates = [root / "glm53-beamng-fixtures.bundle"
                     for root in (*_dataset_roots, _runtime_root)]
_scene_bundle = next((path for path in _scene_candidates if path.is_file()), None)
if _scene_bundle is None:
    raise FileNotFoundError("Attach the matching engine dataset with BeamNG fixtures")
if hashlib.sha256(_scene_bundle.read_bytes()).hexdigest() != {scene_sha!r}:
    raise ValueError("Attached BeamNG fixtures differ from this notebook")
CONFIG.update({scene_config!r})
CONFIG["beamng_fixture_bundle"] = str(_scene_bundle)
print("BeamNG fixtures verified:", {scene_names!r})
""")
        bootstrap_index = next(
            i for i, cell in enumerate(cells) if "_PAYLOAD_B85 =" in "".join(cell["source"])
        )
        cells.insert(bootstrap_index + 1, scene_cell)
        cells.insert(
            bootstrap_index,
            _markdown(
                "## Real BeamNG capture checks\n\n"
                "Four source frames test visible car counts, relative position and road text. "
                "Labels come from inspection of these exact pixels; scenario names and "
                "teacher predictions do not establish collision or traffic-event truth. "
                "Original images and hashes are kept in the evidence. Real frames use "
                "at most 384 image tokens to keep this comparison within its session budget. "
                "The comparison retains four synthetic anchors including the seven-circle "
                "regression. Both quants use the same eight requests.\n"
            ),
        )
        scene_metadata = {"beamng_fixture_sha256": scene_sha, "beamng_case_names": scene_names}
    for index, cell in enumerate(cells):
        cell["id"] = f"glm53-{index:02d}"
    notebook["cells"] = cells
    metadata = notebook.setdefault("metadata", {})
    metadata["glm53_offload"] = {
        "bundle_sha256": _sha256(archive),
        "source_manifest_sha256": _sha256(source_manifest_bytes),
        "source_file_count": len(source_records),
        "run_model_default": run_model,
        "experiment_mode": experiment_mode,
        "quant": quant,
        "quantizations": ["Q4_K_XL", "Q6_K_XL"] if experiment_mode == "comparison" else [quant],
        **scene_metadata,
    }

    source_bytes = kaggle_source_bytes(notebook)
    if source_bytes >= KAGGLE_SOURCE_LIMIT:
        raise ValueError(
            f"Kaggle notebook source is {source_bytes:,} bytes; "
            f"it must be below {KAGGLE_SOURCE_LIMIT:,} bytes"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as output:
        json.dump(notebook, output, indent=2, ensure_ascii=False)
        output.write("\n")
    return notebook


def embed_tracking_key(notebook: dict[str, Any], out_path: Path, key_file: Path) -> None:
    """Inject task authentication only into the authorized private upload copy."""
    if not out_path.resolve().is_relative_to(REPO_ROOT / "output"):
        raise ValueError("Credential-bearing notebooks must stay under ignored output/")
    key = key_file.read_text().strip()
    if not key:
        raise ValueError("Task tracking key is empty")
    credentials = _code("import os\nos.environ['WANDB_API_KEY'] = " + repr(key), hidden=True)
    credentials["id"] = "glm53-task-auth"
    notebook["cells"].insert(0, credentials)
    notebook["metadata"]["glm53_offload"]["private_task_authentication"] = True
    if kaggle_source_bytes(notebook) >= KAGGLE_SOURCE_LIMIT:
        raise ValueError("Private authenticated notebook exceeds Kaggle's source limit")
    out_path.chmod(0o600)
    out_path.write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=NOTEBOOK_PATH, help="Notebook output path")
    parser.add_argument("--quant", choices=("Q6_K_XL", "Q4_K_XL"))
    parser.add_argument("--beamng-bundle", type=Path, help="Checksummed real-scene dataset asset")
    parser.add_argument(
        "--embed-key-file",
        type=Path,
        help="Embed the authorized task tracking key in a private upload under ignored output/.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--expert-profile",
        action="store_true",
        help="Build the bounded one-layer expert kernel probe instead of the full trial.",
    )
    mode.add_argument("--compare-quants", action="store_true", help="Q4/Q6 comparison (default).")
    mode.add_argument(
        "--single-quant", action="store_true", help="Run the full single-model battery."
    )
    parser.add_argument(
        "--local-validation",
        action="store_true",
        help="Build a validation copy with RUN_MODEL=False; never starts model loading.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    notebook = build_notebook(
        out_path=out_path,
        run_model=not args.local_validation,
        experiment_mode=(
            "expert_profile"
            if args.expert_profile
            else "full"
            if args.single_quant or (args.quant and not args.compare_quants)
            else "comparison"
        ),
        quant=args.quant or "Q6_K_XL",
        beamng_bundle=args.beamng_bundle,
    )
    if args.embed_key_file is not None:
        embed_tracking_key(notebook, out_path, args.embed_key_file)
    metadata = notebook["metadata"]["glm53_offload"]
    print(
        f"Wrote {out_path} with {metadata['source_file_count']} source files; "
        f"bundle SHA-256 {metadata['bundle_sha256']}"
    )


if __name__ == "__main__":
    main()
