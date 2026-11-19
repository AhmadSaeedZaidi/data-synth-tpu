# TPU experiment timeline

[Repository overview](../README.md) · [Detailed results](experiment-results.md) ·
[Version ledger (CSV)](experiment-ledger.csv)

The full audit covers **38 saved versions across seven TPU notebook series**
and **two CPU smoke-test versions**. This page highlights milestones from that
history. The detailed results describe what each run measured and where it
stopped.

## Queue wait

**Budget about 36 hours before a Kaggle TPU v5e-8 trial begins.** These waits
come from this project's submissions. Measure from Kaggle accepting the
submission to the first explicit `RUNNING` response. The tracker polls every
three minutes.

| Qwen notebook | Accepted (UTC) | Running first observed (UTC) | Wait | Outcome |
| --- | --- | --- | ---: | --- |
| v15 | Nov 15, 09:30 | Nov 17, 01:32 | **40 h 03 min** | Full JAX results recovered after evidence export failed |
| v16 | Nov 17, 18:59 | Nov 19, 02:27 | **31 h 29 min** | Startup stopped on W&B heartbeat readback |

Times are rounded independently to the nearest minute. The two waits average
**35 h 46 min**. Each hardware trial has a separate six-hour active cap. The
long wait favors checking the exact upload bundle on CPU and grouping hardware
questions into one session.

## Milestones

Dates mark when the result was recorded or the trial was submitted, as noted.
Notebook versions belong to their own model series. Some trials use code newer
than the public source snapshot.

| Date (2026) | Trial | Result and what changed |
| --- | --- | --- |
| Nov 19 | Qwen v17 · submitted | Startup now records W&B and Weave independently and can continue with local diagnostics. A CPU probe reproduced delayed W&B visibility and verified Weave. Hardware results pending at submission. |
| Nov 19 | Qwen v16 · startup error | Weave received its startup trace. The W&B heartbeat was missing during the 45-second readback window and appeared after the notebook stopped. The run ended before model tests. |
| Nov 17 | Qwen v15 · evaluation completed | 48/48 sequential assertions passed. The B4 OCR error recurred. Torch vision passed FP32/BF16 reference checks. A missing module blocked the other backend comparisons, and duplicate archive paths broke the evidence ZIP. |
| Oct 29 | Qwen v14 · evaluation completed | 48/48 sequential assertions passed. B1/B2 passed the eight-request replay, while B4 misread one OCR case. The B8 memory gate skipped allocation. |
| Oct 28 | Qwen v13 · profiling error | All 17 numerical and routing gates passed. Calibration kept four-way expert parallelism and two-way tensor parallelism with expert groups of four. A missing output directory stopped the trial at its first decode trace. |
| Oct 27 | Qwen v12 · evaluation completed | 46/48 assertions passed. Both misses were formatting checks. In the sampled traces, packed expert selection and decoding accounted for roughly 52–53% of covered device time. |
| Oct 24–25 | Qwen v10–v11 · evaluation completed | Both scored 40/48 assertions. The v11 alternate KV gather passed correctness checks with a 1.005× full-read speedup, so the existing gather stayed in use. |
| Oct 21 | Qwen v9 · active time cap | All 31 image answers failed. The worker was still decoding at B8 when the six-hour cap ended the run. |
| Oct 17 | Qwen v6 · evaluation completed | All nine startup gates passed with Q8 KV. All 49 cases ran and 42/48 assertions passed. |
| Mid-Oct | Qwen v3–v5 · numerical gates | Long-context Q8 checks exposed a scale-rounding mismatch. The codec fix passed its own gate. Later gates exposed remaining FP32 and INT8 compute differences before full weight loading. |
| Oct 12 | Qwen v1 · first full run | Ran native vision and batched Q6 inference with n-grams in RAM. Seven of eight unique cases passed across three batch sizes. The same OCR case failed at each size. |
| Oct 6–7 | GLM v13–v14 · host expert offload | Ran Q4 and Q6 with native reasoning. Warm B8 decode reached 2.42 tokens/s with Q4 and 1.49 with Q6 in v14. Both variants failed the road-sign OCR case. |

## Decode measurements

Aggregate tokens per second across active batch rows. Rates count reasoning
tokens and exclude prefill, weight loading, vision preparation, and warmup.

| Qwen trial | B1 | B2 | B4 | B8 |
| --- | ---: | ---: | ---: | ---: |
| v1 · early replay | 6.53 | — | 11.24 | 13.89 |
| v11 · 8K Q8 trial | 8.12 | 8.36 | 9.28 | Memory gate |
| v14 · quality-first replay | 7.99 | 8.18 | 10.18 | Memory gate |
| v15 · quality-first replay | 8.02 | 8.10 | 10.19 | Memory gate |

Compare rates within a trial. Prompts, token budgets, or context policies can
differ across trials. A dash means that batch size was absent from this
measurement. Sequential quality checks and batch replay are scored separately.
The v14/v15 sequential denominator covers 44 answer/content checks and four
schema checks. One additional control is ungraded.

## Next trial

Qwen v17 was accepted on **November 19 at 09:20 UTC**, with a six-hour active
cap. The initial API response left queue status unconfirmed. The plan compares
built-in and explicit BF16×6 matmul, probes JAX, native TorchTPU and hybrid
operators on matched inputs, and checks eight-rank expert/tensor parallel
execution. The full JAX trial covers native vision, BeamNG, spatial reasoning,
SFT formatting, and batching. A separate OCR diagnostic follows the B1 token
history at larger batch sizes to see where predictions diverge.

Update this log when a new trial yields evidence. Record the version, observed
queue delay if available, experiment outcome, and a measurement or blocker
that informs the following trial. The presentation takes inspiration from
[AnyPS5’s progress history](https://boykopovar.github.io/AnyPS5/history.html).
