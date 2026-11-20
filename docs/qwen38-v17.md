# Qwen v17 results

Kaggle marked v17 **COMPLETE** on November 20, 2026. The model report recorded
**QUALITY_FAILED** because one B4 OCR replay failed. The JAX worker exited
normally and all 49 sequential evaluation cases ran.

## Queue and experiment scope

| Event | UTC |
| --- | --- |
| Submission accepted | Nov 19, 09:19:44 |
| First observed running | Nov 20, 07:22:45 |
| First observed complete | Nov 20, 09:34:18 |

The observed queue wait was **22 h 03 min**. The tracker observed **2 h 12 min**
between running and completion. The supervised model worker used **2 h 06 min**.
We budget **36 hours** for a TPU queue and cap active time at six hours.

The run used Qwen3.8-Flash-Next with Q6 checkpoint weights, native vision,
host-RAM n-grams, a 16K Q8 KV cache, and xhigh thinking. Expert sharding used
EP4 × TP2 with four experts per group in prefill and decode. The main
projections used BF16 inputs and FP32 accumulation. JAX requested its highest
precision for FP32 dots in routing, sparse-attention indexing and the GDN
recurrent state. [JAX matmul precision](https://docs.jax.dev/en/latest/201/precision.html)

## Quality and decode

All **17 numerical and routing gates** passed. The sequential suite passed
**48/48 assertions**, including **44 answer/content checks** and **four
schema-only checks**. One control was ungraded. The suite covered native
vision, BeamNG, spatial reasoning and SFT formatting. The schema-only cases
still need independent event labels to measure event accuracy.

| Replay | Checks passed | Decode tokens/s | Active row occupancy | Prefill / decode |
| --- | ---: | ---: | ---: | ---: |
| B1 | 8/8 | 8.08 | 100.0% | 163.81 s / 95.96 s |
| B2 | 8/8 | 8.30 | 64.4% | 114.76 s / 108.51 s |
| B4 | 7/8 | 10.22 | 56.3% | 109.82 s / 73.08 s |
| B8 | Skipped | — | — | — |

Rates count reasoning tokens across active rows and exclude prefill, weight
loading, vision preparation and warmup. These replays generated different
token counts, so the B4 rate is a workload measurement. B8 projected **−180
MiB** of free HBM on the tightest chip against a required **512 MiB** margin.
The admission gate skipped allocation.

## The B4 OCR regression

B1 and B2 answered `STOP 7`. B4 answered `STOP ?`, repeating the v14 and v15
outcomes. The v17 diagnostic replayed the same target and forced its B1 token
history into both four identical prompts and a mixed B4 batch. The reported
target logit summaries agreed across those two B4 cohorts at all 47 steps.

The first post-prefill logits differed from B1 by **4.1094** at the largest
coordinate and **30.0%** in relative L2. At zero-based generation step **12**,
B1 preferred token 9077 by **0.500** logit points, while B4 preferred token
18126 by **0.125**. This places the initial drift at or before the first
output projection, while the prompt and decoded history are still matched.
The v17 trace ends at the vocabulary logits. Layer outputs and a same-shape
projection control are needed to localize the drift.

The earlier v10/v11 324-token OCR prompt also varied by batch size. B4 passed
while B1/B2 failed. The current higher-resolution 1,148-token prompt first
showed its B4 failure in v14. V12 evaluated the updated prompt at B1 and v13
stopped in profiling before replay, so the run history cannot isolate a
particular code change.

## TorchTPU, hybrid, and BF16×6

| Experiment | Outcome on v5e-8 |
| --- | --- |
| TorchTPU environment | Installed successfully in an isolated environment. |
| Torch vision | Passed the frozen FP32 and BF16 references. |
| Eight-rank EP8/TP1 | Passed the small distributed FFN reference. |
| Eight-rank EP4/TP2 | Passed the same FFN reference. |
| Torch eager, compiled and hybrid packed projection | All stopped at the first synchronization call. |
| Torch native-highest and explicit six-product BF16 | Both stopped at synchronization on their first router case. Neither admitted latency or FP64 accuracy results. |
| JAX single-chip projection control | Failed its first Q6 projection reference. 302 of 320 outputs missed the tolerance, with about 0.0146 maximum absolute error. |

The Torch worker called the internal `torch.tpu.synchronize()` entry point.
TorchTPU directs external code to `torch.accelerator.synchronize()`. The
successful vision and distributed checks establish component parity on this
fixture. Full-model Torch speed, checkpoint loading and HBM fit remain open.
[TorchTPU public API](https://github.com/google-pytorch/torch_tpu/blob/b945ee1d9cf2ec0bcd2a27c32ad026d5c08142c1/docs/public_apis.md)

The explicit BF16×6 prototype splits each FP32 operand into three BF16
components. It retains hh, hm, mh, hl, mm and lh, summing small terms first.
V17 shipped it as a separate experiment. The production JAX policy continued
to use its existing precision controls.

## Observability

Weave startup readback passed. W&B startup readback was unverified within its
window. The notebook continued under its fallback and preserved local reports
and progress. The exact-version tracker observed start, hourly progress, and
completion and recovered the final report.
