# Data Synth TPU

Experimental vision-language inference on **eight Kaggle TPU v5e chips** for
generating and checking traffic annotations. The JAX runtime runs
Qwen3.8-Flash-Next with packed Q6 expert weights, n-grams in host RAM, native
vision, and a Q8 KV cache.

> [!IMPORTANT]
> **About 36 hours in the Kaggle TPU queue for one run.** Two recent notebooks
> took **40 h 03 min** and **31 h 29 min** to reach their first observed running
> state. Once a slot opens, the trial has a **six-hour active cap**. A startup
> bug can cost a day and a half before the next hardware check.
> [Measured queue waits](docs/experiments.md#queue-wait)

[Experiment timeline](docs/experiments.md) ·
[Detailed results and run counts](docs/experiment-results.md) ·
[Getting started](docs/getting-started.md) ·
[Kaggle publishing and monitoring](docs/operations.md)

## Results so far

- **Quality.** Qwen v14, v15 and v17 passed all 48 asserted sequential checks,
  including four schema checks. The same OCR case failed at batch 4.
- **Decode.** Qwen v17 measured **8.08 / 8.30 / 10.22 tokens/s** at batch
  **1 / 2 / 4**, respectively, across all active requests. The batch 8 memory
  check skipped allocation.
- **TorchTPU.** Native vision and eight-rank expert/tensor parallel checks
  passed. The timing and BF16×6 probes stopped at synchronization.
  [Detailed v17 report](docs/qwen38-v17.md)

## Repository map

| Path | Purpose |
| --- | --- |
| [kaggle/qwen38_tpu](kaggle/qwen38_tpu/) | JAX vision-language runtime, expert and tensor parallelism, and numerical gates |
| [kaggle/qwen38_torch](kaggle/qwen38_torch/) | TorchTPU vision, packed expert and hybrid component experiments |
| [kaggle/glm53_offload](kaggle/glm53_offload/) | Earlier GLM inference, expert offload, and shared monitoring |
| [kaggle/packed_q6](kaggle/packed_q6/) | Experimental Pallas kernel and benchmarks |
| [src/data_synth](src/data_synth/) | Traffic annotation, evidence verification, and JSONL export |
| [scripts](scripts/) | Notebook and dataset builders, publisher, watcher, and profiling tools |
| [tests](tests/) and [kaggle/tests](kaggle/tests/) | Offline application and CPU runtime checks |

## Public release

The code was open sourced after one month of development. This public source
snapshot includes synthetic reference fixtures. Private captures, evaluation
bundles, raw run outputs, internal notes, model weights, and credentials are
withheld. The experiment timeline includes aggregate results from later trials.

Original code is available under the [MIT license](LICENSE). See
[third-party notices](THIRD_PARTY_NOTICES.md) for upstream code and licenses.
