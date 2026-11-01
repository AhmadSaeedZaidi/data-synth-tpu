# Experimental packed Q6_K projection

This package contains an isolated TPU Pallas prototype, a host repacker,
independent reference arithmetic and a bounded benchmark harness. It is disabled
in model inference. CPU interpretation does not establish TPU compilation or speed.

The intended operation is BF16 activations multiplied by decoded Q6_K weights,
with FP32 accumulation. The repacker preserves original quantized values.
The benchmark compares the Pallas path with XLA decoding and predecoded BF16
matrix multiplication; these have different resident-memory requirements.

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 pytest \
  kaggle/tests/test_packed_q6.py kaggle/tests/test_packed_q6_benchmark.py -q
JAX_PLATFORMS=cpu PYTHONPATH=kaggle python -m packed_q6.benchmark --interpret \
  --output output/packed-q6/correctness.json --timeout-seconds 90
```

CPU mode reports correctness, without TPU timing claims. On real TPU, inspect
compiler legality, memory and numerical results before interpreting any timing.
Expert row counts are not request batch sizes. A single projection's latency
does not establish full-model throughput or communication cost.

`scripts/build_packed_q6_experiment.py` can generate a hardware notebook locally.
For a long TPU queue, combine useful diagnostics and model trials under a shared
budget rather than spending an allocation on an isolated minor check.

Layout and numerical references include
[GGML Q6_K](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c)
and the [JAX Pallas TPU documentation](https://docs.jax.dev/en/latest/pallas/tpu/details.html).
See the root third-party notices for attribution.
