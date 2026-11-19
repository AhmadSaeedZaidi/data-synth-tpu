# Getting started

[Repository overview](../README.md) · [Experiment timeline](experiments.md)

## Run the application locally

Use Python 3.12 or newer. The demo and application tests run without a TPU,
simulator installation, or API credentials.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -c requirements-dev.lock -e '.[dev,models,hf]'
pytest tests -q
python -m data_synth demo data/demo
```

Copy [the environment example](../.env.example) when you need optional
configuration. Keep real credentials in an ignored `.env` file.

## Run the runtime checks on CPU

```bash
python -m pip install -e '.[dev,models,hf,tpu-test]'
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 pytest kaggle/tests -q
```

The runtime tests emulate eight devices on CPU to check numerical behavior
and orchestration. Use the [experiment timeline](experiments.md) for TPU
measurements. Tests requiring private BeamNG bundles skip when the files are
absent.

## Build a Kaggle trial

```bash
python scripts/build_qwen38_notebook.py --output output/qwen38-staging
```

The builder stages a notebook and matching runtime dataset. Set the Kaggle
owner, notebook, dataset, and W&B entity for your account. You need access to
the pinned checkpoint and sufficient host RAM and HBM. Publish the dataset
before submitting the notebook. [Publishing and monitoring](operations.md)

To build the full quality trial with your own compatible BeamNG and evaluation
bundles, adapt the fixture plan and expected provenance hashes to your captures
and independently verified labels.

```bash
python scripts/build_qwen38_notebook.py --quality-first --rounding-probe \
  --batch-sizes 1 2 4 8 --output output/full-trial \
  --beamng-bundle /path/to/beamng.bundle \
  --evaluation-bundle /path/to/evaluation.bundle
```

Group diagnostics with substantial model evaluation to make the most of a
slot. Probes have their own deadlines within the six-hour session cap. The
model's numerical gates are mandatory.

## Runtime details

The baseline decodes Q6 checkpoint weights for BF16 matrix products with
FP32 accumulation. Generation uses static batches. The packed-Q6 Pallas
prototype and BF16 rounding candidate are separate experiments. A schema
check validates output structure. Training data also needs independent checks
of the underlying events and labels.
