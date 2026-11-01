"""Stage a private Kaggle runtime dataset and notebook without publishing or queueing."""

import argparse
import hashlib
import io
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHARED = (
    "__init__.py",
    "weights.py",
    "gguf_quant.py",
    "routing.py",
    "progress.py",
    "tracing.py",
    "diagnostics.py",
    "supervisor.py",
    "monitoring.py",
    "tracking_check.py",
    "quality.py",
    "probe_images.py",
    "scene_fixtures.py",
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def cell(source, *, markdown=False):
    out = dict(
        id=digest(source.encode())[:12],
        cell_type="markdown" if markdown else "code",
        metadata={},
        source=source.splitlines(keepends=True),
    )
    if not markdown:
        compile(source, "qwen-notebook-cell", "exec")
        out.update(execution_count=None, outputs=[])
    return out


def embed_tracking_key(document, notebook_path, key_file):
    """Use the previously authorized task key only in the private upload copy."""
    notebook_path = Path(notebook_path)
    if not notebook_path.resolve().is_relative_to(ROOT / "output"):
        raise ValueError("Credential-bearing notebooks must stay under ignored output/")
    metadata_path = notebook_path.with_name("kernel-metadata.json")
    if json.loads(metadata_path.read_text()).get("is_private") is not True:
        raise ValueError("Task authentication requires a private notebook")
    key = Path(key_file).read_text().strip()
    if not key:
        raise ValueError("Task tracking key is empty")
    authentication = cell("import os\nos.environ['WANDB_API_KEY'] = " + repr(key) + "\n")
    authentication["metadata"] = {"jupyter": {"source_hidden": True}}
    document["cells"].insert(0, authentication)
    document["metadata"]["qwen38_tpu"] = {"private_task_authentication": True}
    notebook_path.chmod(0o600)
    notebook_path.write_text(json.dumps(document, indent=2) + "\n")


def build(
    destination,
    *,
    beamng_bundle=None,
    evaluation_bundle=None,
    embed_key_file=None,
    batch_sizes=None,
    requests=None,
    expert_compute="bf16",
    gdn_prefill="serial",
    optimization_probes=False,
    rounding_probe=False,
    quality_first=False,
):
    from sys import path

    path.insert(0, str(ROOT / "kaggle"))
    from qwen38_tpu.trial import validate_batches

    if expert_compute not in {"bf16", "auto_int8"} or gdn_prefill not in {"serial", "chunked"}:
        raise ValueError("Unsupported expert compute or GDN prefill policy")
    if quality_first and not evaluation_bundle:
        raise ValueError("Quality-first mode requires the complete evaluation bundle")
    batch_sizes = (
        ([1] if quality_first else [1, 2, 4, 8]) if batch_sizes is None else list(batch_sizes)
    )
    validate_batches(batch_sizes, requests)
    destination = Path(destination)
    dataset, notebook = destination / "dataset", destination / "notebook"
    dataset.mkdir(parents=True, exist_ok=True)
    notebook.mkdir(parents=True, exist_ok=True)
    files = {}
    for p in (ROOT / "kaggle/qwen38_tpu").rglob("*"):
        if (
            p.is_file()
            and p.suffix in (".py", ".json", ".npz", ".md")
            and "__pycache__" not in p.parts
        ):
            files[p.relative_to(ROOT / "kaggle").as_posix()] = p.read_bytes()
    for name in SHARED:
        files["glm53_offload/" + name] = (ROOT / "kaggle/glm53_offload" / name).read_bytes()
    if optimization_probes:
        files["experiments/benchmark_qwen38_kv_gather.py"] = (
            ROOT / "scripts/benchmark_qwen38_kv_gather.py"
        ).read_bytes()
        for path in (ROOT / "kaggle/packed_q6").glob("*.py"):
            files["packed_q6/" + path.name] = path.read_bytes()
    files["licenses/vendor-glm53.txt"] = (ROOT / "kaggle/vendor/LICENSE").read_bytes()
    manifest = {name: digest(data) for name, data in files.items()}
    files["runtime-manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 10, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, data)
    runtime = buffer.getvalue()
    (dataset / "qwen38-runtime.bundle").write_bytes(runtime)
    config = dict(
        quant="q6",
        batch_sizes=batch_sizes,
        requests=requests,
        max_context=16384 if quality_first else 8192,
        kv_cache_dtype="q8",
        max_new_tokens=4096 if quality_first else 2048,
        evaluation_max_new_tokens=8192 if quality_first else 4096,
        reasoning_effort="xhigh",
        # Native BF16 baseline until BF16-activation INT8 end-to-end quality
        # is established. The optional auto_int8 calibration path is retained.
        expert_compute=expert_compute,
        expert_parallel="auto",
        fuse_layers=True,
        gdn_prefill=gdn_prefill,
        stop_on_evaluation_collapse=not quality_first,
        profile_decode=True,
        deadline_seconds=21600,
        max_no_progress_seconds=900,
        wandb_mode="online",
        wandb_project="qwen38-kaggle-tpu",
        wandb_entity="your-wandb-entity",
        weave_enabled=True,
        trace_prefix="qwen38",
        allow_tracking_fallback=False,
        require_live_tracking=True,
        local_traces=True,
        optimization_probes=optimization_probes,
        rounding_probe=rounding_probe,
        packed_q6_probe=optimization_probes,
        kv_scale_gather="indexed",
        quality_first=quality_first,
        runtime_bundle_sha256=digest(runtime),
    )
    if beamng_bundle:
        from sys import path

        path.insert(0, str(ROOT / "kaggle"))
        from glm53_offload.scene_fixtures import load_scene_bundle

        _, _, _, sha = load_scene_bundle(beamng_bundle)
        shutil.copyfile(beamng_bundle, dataset / "qwen38-beamng-fixtures.bundle")
        config["beamng_fixture_sha256"] = sha
    if evaluation_bundle:
        from qwen38_tpu.evaluation.bundle import load_bundle

        _, cases, _, sha = load_bundle(evaluation_bundle)
        shutil.copyfile(evaluation_bundle, dataset / "qwen38-evaluation.bundle")
        config.update(
            evaluation_sha256=sha,
            evaluation_batch_size=1 if quality_first else 8,
            evaluation_expected_cases=len(cases),
        )
    (dataset / "dataset-metadata.json").write_text(
        json.dumps(
            dict(
                id="your-kaggle-username/qwen38-flash-next-engine",
                title="Qwen3.8 Flash Next JAX TPU engine",
                licenses=[dict(name="other")],
            ),
            indent=2,
        )
        + "\n"
    )
    cells = [
        cell(
            f"""Qwen3.8-Flash-Next on Kaggle v5e-8.
Native JAX vision and expert parallelism; packed Q8
n-grams in host RAM. Q6 trial, thinking enabled, batches {"/".join(map(str, batch_sizes))},
{requests or "one labeled cohort of"} requests per batch size,
maximum {config["max_context"]} total tokens.
Q8 KV cache, native BF16 attention, xhigh thinking. Expert compute policy: {expert_compute}.
The default uses BF16 expert dots with FP32 accumulation. The optional auto_int8
policy requires independent correctness and on-device accuracy/speed calibration gates.
Grouped experts and 4-way expert / 2-way tensor parallelism are compared with
the serial 8-way expert baseline on full-size synthetic FFNs before loading weights.
EP4/TP2 compares groups of 1, 2 and 4; the single-expert path slices contiguous
packed bytes and avoids padded expert slots. The selected phase composition
must also pass independent routing and full-model gates on this TPU.
Packed weights are partitioned without replication. Decode and prefill kernels
are selected separately within one fixed expert/tensor placement. Unsafe memory
fallbacks are rejected. The serial expert baseline visits only active experts;
grouped alternatives retain every route. These are native JAX kernels, not a
vLLM runtime or a fused packed-Q6 Pallas kernel.
Prefill uses batch-dependent token windows and GDN policy: {gdn_prefill}.
The default keeps the serial recurrent rule for prefill and decode. Experimental
chunked GDN uses 64-token chunks with a direct FP32 triangular solve; the unstable
repeated-squaring inverse was removed. Independent numerical gates run before
full weights load, and the first evaluation cohort includes a warmed device trace.
Non-finite vision features or active-row logits stop the worker with diagnostics.
Default replay mode stops remaining work after complete first-cohort thinking collapse.
Quality-first mode records failures and attempts the complete evaluation suite.
Q8 codes use the exact stored BF16 scale, with explicit rounding. Independent
FP32 elementary arithmetic, cache-code, unquantized, Q8, BF16 and integer checks
preserve diagnostics before full weights are admitted; a failed gate prevents model loading.
Accurate FP32 nonlinear operations and corrected division protect quantization
boundaries; failure reports retain token errors and the actual cache window.
Each batch shape is warmed separately, then runs the same labeled workload.
Every admitted batch has a separate labeled warm decode profile, outside timed
benchmarks; an early EOS can skip capture and is reported explicitly.
Finished rows are masked; this is static batching without request refilling.
Larger workloads replay labeled cases; repeated images do not expand quality coverage.
An attached full evaluation runs once before the replay benchmark, using the complete
GLM suite, eight-frame BeamNG pipeline reviews, spatial and strict SFT format tests.
Per-chip allocator measurements gate each batch shape before allocation. Skipped shapes
are reported and completed smaller-batch results are preserved.
CPU/TPU golden gate and live tracking readback run before full weights are loaded. Six-
hour active-session watchdog. Full Q8 is rejected unless its memory plan fits.
""",
            markdown=True,
        )
    ]
    if optimization_probes:
        cells[0]["source"] += [
            "\nBefore model loading: isolate a <=15-minute eight-chip KV gather A/B probe, "
            "then a <=15-minute packed-Q6 projection probe. The KV candidate must pass "
            "exact correctness at every shape, >=1.10x geometric full-read speedup, "
            "<=15% worst-case regression and a 64 MiB temporary-memory growth cap. "
            "Otherwise indexed gathering remains selected. Independent full-model gates "
            "exercise the selected path before checkpoint loading. The Q6 probe is "
            "diagnostic only; its kernel is not enabled in inference. Both probe budgets "
            "count against the shared six-hour session cap.\n"
        ]
    if quality_first:
        cells[0]["source"] += [
            "\nQuality-first protocol: evaluate all original 49 cases sequentially at B1, "
            "with the same images, source prompts and strict matchers. A separate system "
            "message reinforces the requested final-answer format. Annotation confidence "
            "is explicitly a top-level field. Pipeline requests receive 8192 generated "
            "tokens (2048 reserved for the answer); other cases receive 4096 (1024 "
            "reserved). At the reasoning budget the normal model input receives the "
            "closing thinking delimiter; no answer content is inserted. xhigh remains "
            "enabled. OCR and vision-formatting inputs use at least 1,048,576 pixels. "
            "Every policy change and forced reasoning boundary is recorded; this is "
            "a new protocol, not a re-score of v11. The context is 16384 with Q8 KV. "
            "Enabled diagnostic budgets count against this same session. All numerical gates and "
            "the shared six-hour cap remain active.\n"
        ]
    if rounding_probe:
        cells[0]["source"] += [
            "\nAn optional BF16-rounding comparison runs for at most ten minutes before model "
            "loading, reserving at least four hours for subsequent work. Its failures are "
            "recorded and the baseline model trial continues after clean process shutdown. "
            "The probe does not change model quantization or inference precision. The full "
            "quality suite, expert calibration and batch benchmarks remain enabled.\n"
        ]
    cells.append(
        cell(
            """import json, os, sys, time, hashlib, zipfile, subprocess
from pathlib import Path
_started = time.monotonic()
CONFIG = """
            + repr(config)
            + """
_assets = next(Path('/kaggle/input').rglob('qwen38-runtime.bundle'))
if hashlib.sha256(_assets.read_bytes()).hexdigest() != CONFIG['runtime_bundle_sha256']:
    raise ValueError('Runtime bundle checksum differs')
_runtime = Path('/kaggle/working/qwen38-runtime')
_runtime.mkdir(exist_ok=True)
with zipfile.ZipFile(_assets) as archive:
    if sum(x.file_size for x in archive.infolist()) > 32 << 20:
        raise ValueError('Runtime exceeds expansion limit')
    expected = json.loads(archive.read('runtime-manifest.json'))
    if set(archive.namelist()) != set(expected) | {'runtime-manifest.json'}:
        raise ValueError('Runtime members differ from manifest')
    for name, sha in expected.items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('Unsafe runtime path')
        data = archive.read(name)
        if hashlib.sha256(data).hexdigest() != sha:
            raise ValueError('Runtime member checksum differs')
        target = _runtime / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
sys.path.insert(0, str(_runtime))
CONFIG['output_dir'] = '/kaggle/working/qwen38-trial'
_run = Path(CONFIG['output_dir'])
_run.mkdir(parents=True, exist_ok=True)
if CONFIG.get('beamng_fixture_sha256'):
    CONFIG['beamng_fixture_bundle'] = str(_assets.parent / 'qwen38-beamng-fixtures.bundle')
if CONFIG.get('evaluation_sha256'):
    CONFIG['evaluation_bundle'] = str(_assets.parent / 'qwen38-evaluation.bundle')
(_run / 'config.json').write_text(json.dumps(CONFIG, indent=2))
print('Verified runtime', CONFIG['runtime_bundle_sha256'])
"""
        )
    )
    cells.append(
        cell(
            """from glm53_offload.diagnostics import atomic_json, bounded_call, package_evidence
_monitor = None

def _finish_tracking(exit_code):
    if _monitor is not None:
        try:
            bounded_call(lambda: _monitor.finish(exit_code), timeout=25)
        except Exception as error:
            atomic_json(_run / 'tracking-cleanup.json', {
                'status': 'ERROR', 'error_type': type(error).__name__,
            })
            print('Tracking cleanup:', type(error).__name__)

def _save_evidence():
    package_evidence(
        _run, Path('/kaggle/working/qwen38-evidence.zip'),
        extra_files=[_runtime / 'dependency-setup.log', _run / 'tracking-cleanup.json',
                     _run / 'memory.jsonl', _run / 'kv-gather.json',
                     _run / 'packed-q6.json', _run / 'optimization-preflight.json',
                     _run / 'kv-gather.log', _run / 'packed-q6.log',
                     _run / 'rounding-probe.json', _run / 'rounding-probe.log'],
    )

_dependencies = [
    'jax[tpu]==0.11.2', 'transformers==5.18.0', 'huggingface-hub==1.33.0',
    'numpy==2.5.3', 'ml-dtypes==0.6.0', 'pillow==12.3.0',
    'wandb==0.30.0', 'weave==0.53.11',
    'pydantic==2.13.5',
]
try:
    from glm53_offload.tracking_check import load_kaggle_key
    load_kaggle_key()
    with (_runtime / 'dependency-setup.log').open('w') as log:
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check', *_dependencies],
            stdout=log, stderr=subprocess.STDOUT,
            timeout=max(1, min(1800, CONFIG['deadline_seconds'] - (time.monotonic() - _started))),
        )
    if result.returncode:
        raise RuntimeError('Dependency install failed; inspect dependency-setup.log')
    from glm53_offload.monitoring import RunMonitor
    from glm53_offload.tracking_check import require_live_tracking
    _monitor = RunMonitor(
        CONFIG['output_dir'], CONFIG, mode='online',
        project=CONFIG['wandb_project'], entity=CONFIG['wandb_entity'],
    )
    print(bounded_call(
        lambda: require_live_tracking(_monitor, CONFIG, CONFIG['output_dir']), timeout=150,
    ))
except BaseException as error:
    atomic_json(_run / 'report.json', {
        'status': 'ERROR', 'phase': 'startup', 'error_type': type(error).__name__,
    })
    _finish_tracking(1)
    _save_evidence()
    raise
"""
        )
    )
    cells.append(
        cell(
            """from glm53_offload.supervisor import finalize_report, stop_worker, supervise
_config = _run / 'config.json'
_env = os.environ.copy()
_env['PYTHONPATH'] = str(_runtime)
_env['PYTHONUNBUFFERED'] = '1'
_process = None
_result = None
try:
    if CONFIG.get('optimization_probes', False):
        from qwen38_tpu.experiments import run_optimization_probes
        print(run_optimization_probes(
            CONFIG, _run, _monitor,
            remaining_seconds=CONFIG['deadline_seconds'] - (time.monotonic() - _started),
        ))
    if CONFIG.get('rounding_probe', False):
        from qwen38_tpu.experiments import run_rounding_probe
        _rounding = run_rounding_probe(
            CONFIG, _run, _monitor,
            remaining_seconds=CONFIG['deadline_seconds'] - (time.monotonic() - _started),
        )
        print({'rounding_probe': _rounding.get('status'), 'production_enabled': False})
    _config.write_text(json.dumps(CONFIG, indent=2))
    # Leave a minute within the six-hour budget for termination and diagnostics.
    _remaining = CONFIG['deadline_seconds'] - (time.monotonic() - _started) - 60
    if _remaining <= 0:
        raise TimeoutError('Active-session budget exhausted during setup')
    with (_run / 'worker.log').open('w') as log:
        _process = subprocess.Popen(
            [sys.executable, '-m', 'qwen38_tpu.run', '--config', str(_config)],
            stdout=log, stderr=subprocess.STDOUT, env=_env,
        )
        _result = supervise(
            _process, _run, _monitor, deadline_seconds=_remaining,
            idle_seconds=CONFIG['max_no_progress_seconds'],
        )
except BaseException as error:
    if not (_run / 'report.json').exists():
        atomic_json(_run / 'report.json', {
            'status': 'ERROR', 'phase': 'launch', 'error_type': type(error).__name__,
        })
    raise
finally:
    if _process is not None and _process.poll() is None:
        _cleanup = stop_worker(_process, grace_seconds=2, kill_seconds=5)
        _supervisor_path = _run / 'supervisor.json'
        _result = _result or (
            json.loads(_supervisor_path.read_text()) if _supervisor_path.exists()
            else {'status': 'INTERRUPTED'}
        )
        _result['termination'] = _cleanup
        atomic_json(_supervisor_path, _result)
    _report = finalize_report(
        _run, _result, _process.returncode if _process is not None else None,
    )
    if CONFIG.get('rounding_probe') and (_run / 'rounding-probe.json').is_file():
        _report['rounding_preflight'] = json.loads((_run / 'rounding-probe.json').read_text())
        atomic_json(_run / 'report.json', _report)
    _failed = _report['status'] in {
        'ERROR', 'TIMEOUT', 'STALLED', 'INTERRUPTED',
    }
    _finish_tracking((_process.returncode or int(_failed)) if _process is not None else 1)
    _save_evidence()
print({key: _report.get(key) for key in (
    'status', 'evaluation_coverage', 'benchmark_coverage', 'benchmarks', 'compute_policy',
)})
if _process.returncode or _failed:
    raise RuntimeError('Qwen trial failed; report, supervisor and worker log are preserved')
"""
        )
    )
    document = dict(
        cells=cells,
        metadata=dict(
            kernelspec=dict(display_name="Python 3", language="python", name="python3"),
            language_info=dict(name="python", version="3.12"),
        ),
        nbformat=4,
        nbformat_minor=5,
    )
    (notebook / "qwen38-v5e8.ipynb").write_text(json.dumps(document, indent=2) + "\n")
    (notebook / "kernel-metadata.json").write_text(
        json.dumps(
            dict(
                id="your-kaggle-username/qwen38-flash-next-v5e8",
                title="Qwen38 Flash Next V5E8",
                code_file="qwen38-v5e8.ipynb",
                language="python",
                kernel_type="notebook",
                is_private=True,
                enable_gpu=False,
                enable_tpu=True,
                enable_internet=True,
                machine_shape="TpuV5E8",
                docker_image_pinning_type="original",
                docker_image=(
                    "gcr.io/kaggle-private-byod/python-tpuvm@sha256:"
                    "a2111cb9be558ea4bc187754bb95d7b65e90d8259434f1eb0e0ab1193ff498c0"
                ),
                dataset_sources=["your-kaggle-username/qwen38-flash-next-engine"],
                kernel_sources=[],
                competition_sources=[],
            ),
            indent=2,
        )
        + "\n"
    )
    if embed_key_file:
        embed_tracking_key(document, notebook / "qwen38-v5e8.ipynb", embed_key_file)
    (destination / "build.json").write_text(
        json.dumps(
            dict(
                runtime_sha256=digest(runtime),
                file_count=len(manifest),
                runtime_bytes=len(runtime),
                config=config,
            ),
            indent=2,
        )
        + "\n"
    )
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "output/current/qwen38-staging")
    parser.add_argument("--optimization-probes", action="store_true")
    parser.add_argument(
        "--rounding-probe",
        action="store_true",
        help="Include the bounded rounding comparison in the full six-hour trial",
    )
    parser.add_argument("--quality-first", action="store_true")
    parser.add_argument("--beamng-bundle", type=Path)
    parser.add_argument("--evaluation-bundle", type=Path)
    parser.add_argument("--batch-sizes", type=int, nargs="+")
    parser.add_argument("--expert-compute", choices=("bf16", "auto_int8"), default="bf16")
    parser.add_argument("--gdn-prefill", choices=("serial", "chunked"), default="serial")
    parser.add_argument(
        "--requests",
        type=int,
        help="Fixed request count per batch size; must fill every batch (max 256).",
    )
    parser.add_argument(
        "--embed-key-file",
        type=Path,
        help="Embed the authorized task key only in the private notebook under ignored output/.",
    )
    args = parser.parse_args()
    print(
        build(
            args.output,
            beamng_bundle=args.beamng_bundle,
            evaluation_bundle=args.evaluation_bundle,
            embed_key_file=args.embed_key_file,
            batch_sizes=args.batch_sizes,
            requests=args.requests,
            expert_compute=args.expert_compute,
            gdn_prefill=args.gdn_prefill,
            optimization_probes=args.optimization_probes,
            rounding_probe=args.rounding_probe,
            quality_first=args.quality_first,
        )
    )


if __name__ == "__main__":
    main()
