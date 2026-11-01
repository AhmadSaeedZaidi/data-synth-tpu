# Private Kaggle publishing and monitoring

Build notebooks locally and configure their metadata for your own account.
Publish the matching runtime dataset first and use its exact version in
`dataset_sources`. The notebook verifies the attached runtime checksum.

Install `pip install -e '.[kaggle]'`. Keep a W&B key in a local ignored file or
the `WANDB_API_KEY` environment variable. Validate the upload without a network call:

```bash
python scripts/publish_kaggle_notebook.py output/qwen38-staging/notebook/qwen38-v5e8.ipynb \
  --wandb-key-file /path/to/local/key --accelerator TpuV5E8 --dry-run
```

Removing `--dry-run` uploads the notebook and requests execution. The publisher
requires private visibility, injects the key into a temporary upload copy and
deletes that copy. The public source stays credential-free. Do not publish the
private upload copy or copy its source back into Git. A returned version confirms
submission; a queue or running state needs separate status verification.

The full model trial requires online W&B and Weave readback before loading
weights. JAX device traces, incremental JSON reports and logs preserve local
evidence. Probe and model deadlines share one six-hour budget.

`scripts/watch_kaggle_run.py --help` describes a bounded local watcher for an
exact saved version. Provide a completion callback command of your choice. It
polls read-only APIs and retrieves bounded report/log outputs; it does not launch
another TPU session. Keep the watcher host awake and online. It is not a cloud
scheduler and does not survive a reboot.

Before committing new source, run:

```bash
python scripts/publish_kaggle_notebook.py --check-repo
```

The check covers known token formats and credential-bearing notebook cells.
Review all newly added assets and history as well; pattern matching alone is
not a complete credential audit.
