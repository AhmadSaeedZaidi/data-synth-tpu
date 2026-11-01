# Data synthesis and JAX TPU inference

Open sourced after one month of development. Some information and assets are
not being shared at this time.

This is a clean source snapshot of an experimental vision-language inference
runtime for Kaggle TPU v5e-8 and a traffic annotation/data synthesis application.
The public snapshot has a fresh Git history. Private captures, evaluation bundles,
run outputs, internal development notes, model weights and credentials are not
part of this release.

## What is included

- Native JAX Qwen3.8-Flash-Next inference: packed Q6 expert weights in HBM,
  host RAM n-gram tables, native vision, Q8 KV cache and batched generation.
- Expert and tensor parallel layouts, calibration, numerical correctness gates,
  isolated compiler experiments and JAX profiling tools for eight TPU chips.
- Earlier GLM inference/offloading code and an experimental packed-Q6 Pallas kernel.
- W&B/Weave monitoring, a private Kaggle notebook publisher and a completion watcher.
- Traffic-media import, model response caching, evidence verification and JSONL export.
- CPU regression tests and small synthetic numerical reference fixtures.

## Status and limits

This is research code. CPU tests establish numerical and orchestration properties;
they do not establish target-TPU speed or full-checkpoint quality. Some experimental
paths have known compiler/performance limitations and are disabled by default.
The packed-Q6 Pallas prototype is not the production inference kernel. Generation
uses static batches; completed requests are not continuously refilled. A schema
pass is not evidence that an annotation is factually correct or ready for training.
Private BeamNG evaluation tests skip when their local input bundles are absent.

Q6 describes checkpoint storage. Native BF16 matrix multiplication and FP32
accumulation are used in the baseline; packed weights are decoded during execution.
The BF16 rounding candidate is opt-in and is not enabled in production inference.

## Local setup

Python 3.12 or newer is required. The application and ordinary tests do not need
TPU access, simulator access or API credentials.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -c requirements-dev.lock -e '.[dev,models,hf]'
pytest tests -q
python -m data_synth demo data/demo
```

For the JAX CPU tests, install the TPU-test extra and run a separate pytest process:

```bash
python -m pip install -e '.[dev,models,hf,tpu-test]'
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 pytest kaggle/tests -q
```

The runtime tests emulate eight CPU devices. They do not submit any Kaggle jobs.
Use `.env.example` for optional configuration; put real values in an ignored
`.env` file. Never commit credentials or upload a credential-bearing notebook publicly.

## Build a Kaggle trial

```bash
python scripts/build_qwen38_notebook.py --output output/qwen38-staging
```

This generates a source notebook and runtime dataset locally. Configure your
Kaggle owner, notebook, dataset and W&B entity in the generated configuration.
The checked-in examples use placeholders. You need access to the pinned model
checkpoint and enough host memory and HBM; checkpoint and model licenses apply
separately. Upload the matching runtime dataset before submitting its notebook.

The full quality protocol accepts compatible BeamNG/evaluation bundles. Private
bundles are not supplied. When building replacements, adapt the fixture plan and
expected provenance hashes to your own captures and independently verified labels:

```bash
python scripts/build_qwen38_notebook.py --quality-first --rounding-probe \
  --batch-sizes 1 2 4 8 --output output/full-trial \
  --beamng-bundle /path/to/beamng.bundle \
  --evaluation-bundle /path/to/evaluation.bundle
```

Use a substantial evaluation plan for a long TPU queue. Bundle bounded diagnostics
with useful model experiments under the six-hour active cap. An isolated probe
can fail while independent model work continues after device cleanup; numerical
model correctness gates remain mandatory.

See [publishing and monitoring](docs/operations.md) for private notebook upload.

## Source map

| Path | Purpose |
| --- | --- |
| `kaggle/qwen38_tpu/` | Qwen runtime, checkpoints metadata, gates and evaluation contracts |
| `kaggle/glm53_offload/` | GLM runtime and shared monitoring/supervision utilities |
| `kaggle/packed_q6/` | Experimental Pallas kernel and isolated benchmarks |
| `kaggle/vendor/` | Attributed third-party GLM source and license |
| `src/data_synth/` | Traffic annotation, verification, storage and import application |
| `scripts/` | Builders, publisher, watcher, profiling and reference-generation tools |
| `tests/`, `kaggle/tests/` | Offline application and CPU runtime regression tests |

## License

Original project code is released under the [MIT license](LICENSE). Retained
third-party licenses and provenance are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
Model weights, external packages, simulator software and private media are not
relicensed or distributed by this repository.
