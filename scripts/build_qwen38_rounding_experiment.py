#!/usr/bin/env python3
"""Stage a self-contained ten-minute BF16 probe; never upload or queue it."""

import argparse
import base64
import hashlib
import io
import json
import textwrap
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULES = (
    "__init__.py",
    "config.py",
    "precision.py",
    "rounding_candidate.py",
    "rounding_benchmark.py",
)


def stage(destination, *, tracking=False):
    destination.mkdir(parents=True, exist_ok=True)
    files = {
        f"qwen38_tpu/{name}": (ROOT / "kaggle/qwen38_tpu" / name).read_bytes() for name in MODULES
    }
    if tracking:
        for name in ("experiments.py", "rounding_session.py"):
            files[f"qwen38_tpu/{name}"] = (ROOT / "kaggle/qwen38_tpu" / name).read_bytes()
        for name in (
            "__init__.py",
            "diagnostics.py",
            "monitoring.py",
            "tracing.py",
            "tracking_check.py",
        ):
            files[f"glm53_offload/{name}"] = (ROOT / "kaggle/glm53_offload" / name).read_bytes()
    manifest = {name: hashlib.sha256(data).hexdigest() for name, data in sorted(files.items())}
    files["manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 30, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    payload = buffer.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    (destination / "rounding-source.zip").write_bytes(payload)
    setup = textwrap.dedent(f"""\
        import base64, hashlib, importlib.metadata, io, json, os, sys, zipfile
        from pathlib import Path

        for package in {() if tracking else ("jax", "jaxlib")!r}:
            assert importlib.metadata.version(package) == '0.11.2', package
        payload = base64.b64decode({base64.b64encode(payload).decode()!r})
        assert hashlib.sha256(payload).hexdigest() == {digest!r}
        working = Path('/kaggle/working') if Path('/kaggle/working').is_dir() else Path.cwd()
        source = working / 'rounding-{digest[:12]}'
        source.mkdir(exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            archive.extractall(source)
        manifest = json.loads((source / 'manifest.json').read_text())
        for name, expected in manifest.items():
            assert hashlib.sha256((source / name).read_bytes()).hexdigest() == expected, name
        print('Verified rounding probe:', {digest!r}, flush=True)
        sys.path.insert(0, str(source))
    """)
    run = textwrap.dedent("""\
        import signal, subprocess

        environment = dict(os.environ, JAX_PLATFORMS='tpu', PYTHONPATH=str(source),
                           OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2')
        result_path = working / 'rounding-results.json'
        command = [sys.executable, '-m', 'qwen38_tpu.rounding_benchmark',
                   '--output', str(result_path), '--timeout-seconds', '600', '--capture-trace']
        process = subprocess.Popen(command, env=environment, start_new_session=True)
        try:
            code = process.wait(timeout=630)
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            raise
        report = json.loads(result_path.read_text())
        print(json.dumps(report, indent=2))
        assert code == 0 and report['status'] == 'passed', str(result_path)
    """)
    if tracking:
        run = textwrap.dedent(f"""\
            from qwen38_tpu.rounding_session import run

            report = run(working / 'qwen38-trial', {digest!r})
            print(json.dumps({{'status': report['status'], 'cases': len(report.get('cases', []))}}))
            assert report['status'] == 'passed', 'Inspect qwen38-trial/report.json'
        """)
    cells = []
    for kind, source in (
        (
            "markdown",
            "# BF16 rounding TPU comparison\n\n"
            "Requires eight TPU devices and JAX/jaxlib 0.11.2. No checkpoint. "
            "A ten-minute worker deadline and outer cleanup bound execution. Tests B1/B2/B4/B8, "
            "saves incremental results, optimized HLO and device traces. This experiment does "
            "not enable the candidate in model inference or establish model throughput.\n",
        ),
        ("code", setup),
        ("code", run),
    ):
        cell = dict(
            cell_type=kind,
            id=f"rounding-{len(cells)}",
            metadata={},
            source=source.splitlines(keepends=True),
        )
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    notebook = dict(
        nbformat=4,
        nbformat_minor=5,
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
        },
    )
    if tracking:
        notebook["cells"][0]["source"].append(
            "\nOnline W&B/Weave is required and verified before the probe. Use the safe publisher "
            "to inject the task key into the private upload. Session cap: 20 minutes; "
            "benchmark cap: 10 minutes. Standard report/log files support completion tracking.\n"
        )
    (destination / "bf16-rounding-probe.ipynb").write_text(json.dumps(notebook, indent=2) + "\n")
    result = dict(
        source_sha256=digest, files=manifest, tracking=tracking, uploaded=False, queued=False
    )
    (destination / "staging.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination", type=Path, default=ROOT / "output/current/qwen38-bf16-rounding/staged"
    )
    parser.add_argument("--tracking", action="store_true", help="Require live W&B/Weave readback")
    args = parser.parse_args()
    print(json.dumps(stage(args.destination, tracking=args.tracking)))
