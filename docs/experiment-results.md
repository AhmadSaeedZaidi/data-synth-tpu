# TPU experiment results

[Experiment timeline](experiments.md) · [Version ledger (CSV)](experiment-ledger.csv) · [Repository overview](../README.md)

## How much did we run?

The November 19, 2026 audit found **38 saved versions across seven TPU notebook
series**, plus **two CPU smoke-test versions**. The main Qwen and GLM series alone
account for **32 versions**. The shorter timeline is a selection of milestones.

| Series | Saved versions | Kaggle COMPLETE | ERROR | Canceled | Unconfirmed |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.8-Flash-Next | 17 | 7 | 7 | 3 | 0 |
| GLM-5.3-Flash Q4/Q6 | 15 | 8 | 3 | 4 | 0 |
| TPU environment, vision, runtime, and BF16 probes | 6 | 6 | 0 | 0 | 0 |
| **TPU-series total** | **38** | **21** | **10** | **7** | **0** |
| CPU tracking and publisher smoke tests | 2 | 2 | 0 | 0 | 0 |

The unit is a saved notebook version with a version-specific Kaggle status.
One version can contain several experiments. GLM v13–v15, for example, each
ran separate Q4 and Q6 workers. A canceled version can have partial results.
The API and retained evidence do not establish an exact lifetime count of
physical sessions. Two predecessor GLM notebook references no longer resolve
through the API and are outside the count. Local checks, API calls, and dataset
revisions are also outside it.

**Three versions have comparable queue timestamps.** V15, v16 and v17
waited **40 h 03 min**, **31 h 29 min** and **22 h 03 min**. We budget **36 hours**.
Queue time and the six-hour active limit are separate. [Queue measurements](experiments.md#queue-wait)

Kaggle's status describes notebook execution. The application report can
record a quality failure when Kaggle says `COMPLETE`, or preserve a completed
evaluation when an export error leaves Kaggle at `ERROR`.

## Qwen3.8-Flash-Next

| Version | Kaggle | What we tried and what happened |
| --- | --- | --- |
| v1 | COMPLETE | Ran native vision, resident Q6 experts, RAM n-grams, and B1/B4/B8. Seven of eight unique cases passed. The same OCR case failed at all three sizes. |
| v2 | Canceled | Staged 64 requests per size for B8/B16/B32, with per-chip memory admission. The cancellation left no throughput result in the reviewed evidence. |
| v3 | ERROR | Vision and the original numerical reference passed. The new Q8 long-context gate failed at prefill token zero before weight loading. |
| v4 | Canceled | Prepared a Q8 change to quantize against the rounded BF16 scale that the cache actually stores. No completed model trial. |
| v5 | ERROR | The corrected Q8 codec passed exact checks and the BF16 model passed its gate. FP32 long-context Q8 and INT8 model gates still failed after several tokens. |
| v6 | COMPLETE | Higher-accuracy FP32 nonlinear operations brought all nine startup gates within tolerance. The full 8K Q8 trial scored **42/48** and measured B8/B16. B32 was skipped by the memory gate. |
| v7 | Canceled | Staged grouped expert and tensor parallel layouts. Kaggle showed a queue entry in the version view while its activity list omitted it. The exact-version cancellation recorded zero TPU time reserved. |
| v8 | ERROR | Grouped-expert restart passed BF16 checks but failed the serial, grouped TP1, and grouped TP2 INT8 numerical gates. It downloaded zero model weights. |
| v9 | ERROR | Text-only cases passed **17/17**. All **31** image-bearing cases emitted repeated punctuation and failed. B8 produced partial timing data before the six-hour cap. |
| v10 | COMPLETE | Numerical fixes restored usable vision outputs. Full evaluation scored **40/48**, including **23/31** image-bearing cases. B8 ran the evaluation, then the later replay memory check skipped B8. |
| v11 | COMPLETE | Full evaluation held at **40/48**. The alternate Q8 gather passed exact checks with a **1.0049×** full-read speedup, so the existing gather stayed in use. The packed-Q6 Pallas canary failed to compile. |
| v12 | COMPLETE | Sequential 16K Q8 evaluation scored **46/48**. All 44 answer/content checks passed. Two pipeline schema failures came from an ASCII arrow and exceeding the permitted number of reasons. |
| v13 | ERROR | All 17 startup gates passed. Expert group sizes 1 and 2 did not beat the selected group size 4. A missing profile directory stopped the first evaluation trace. |
| v14 | COMPLETE | Profile export worked. With the updated formatting rules and configured final-answer reserve, the sequential suite scored **48/48**. B1/B2 replay passed. B4 OCR returned `STOP ?` where B1/B2 returned `STOP 7`. |
| v15 | ERROR | Again **48/48** sequential, with the same B4 OCR error. Native Torch vision passed. A missing packed-Q6 module blocked the backend and EP/TP comparisons. An archive filename collision failed export after the JAX worker exited. |
| v16 | ERROR | The notebook reached W&B and Weave. The W&B heartbeat appeared after the startup deadline, so the notebook exited before the TPU comparisons and model evaluation. |
| v17 | COMPLETE | **48/48** sequential, with B4 OCR drift before generation. Torch vision and eight-rank EP/TP passed. Synchronization blocked the Torch timing and BF16×6 probes. [Detailed breakdown](qwen38-v17.md) |

The full sequential suite has 49 cases, including one ungraded control. Its 48
assertions include four schema-only BeamNG reviews. Passing those four means
the output followed the required format. Independent event labels are still
needed to assess the claims for SFT.

### Decode measurements

These are aggregate decode rates across active requests. They count reasoning
tokens. Prefill, model loading, vision preparation, and warmup are outside the
rate. Prompt mixes, generated lengths, memory policies, and quality vary across
versions.

| Qwen version | B1 | B2 | B4 | B8 | B16 | Context for these rates |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| v1 | 6.53 | — | 11.24 | 13.89 | — | Eight prepared requests per size |
| v6 | — | — | — | 10.34 | 13.34 | 64 requests per size using repeats of eight prompts |
| v9 | — | — | — | 12.22 | — | Partial B8 run during the vision regression |
| v10 | 8.07 | 8.32 | 9.26 | Skipped | — | Eight-request replay |
| v11 | 8.12 | 8.36 | 9.28 | Skipped | — | Eight-request replay |
| v12 | 8.09 | — | — | — | — | B1 measurement with 16K Q8 |
| v14 | 7.99 | 8.18 | 10.18 | Skipped | — | Eight-request replay with B4 OCR failure |
| v15 | 8.02 | 8.10 | 10.19 | Skipped | — | Eight-request replay with B4 OCR failure |
| v17 | 8.08 | 8.30 | 10.22 | Skipped | — | Eight-request replay with teacher-forced B4 OCR diagnostics |

A dash means no comparable measurement appears in the reviewed record.
Skipped B8 replays were denied by the per-chip memory gate. The v14 gate
projected **−180 MiB** of free HBM on the tightest chip against a required
**512 MiB** free margin. The worker skipped B8 before allocating its state.

### Optimization decisions

| Experiment | What the evidence supports |
| --- | --- |
| N-grams in host RAM | The model fit while keeping expert weights resident. In v6, host n-gram lookup consumed about 0.2% of measured decode time. Packed expert movement and preparation are different costs. |
| Packed expert decode and KV reads | In v14's one-step warm profiles, expert selection/layout/unpack was about **53% at B1** and **41% at B4** of covered device time. Q8 KV scale gathers rose from **11% to 22%**. These percentages describe the captured steps. |
| Smaller expert groups | On the v13 synthetic full-width FFN, four-way expert and two-way tensor parallelism with groups of four reached **1.45×** weighted decode speed versus serial eight-way expert parallelism. Groups of one and two reached **1.03×** and **1.08×**. The selected layout kept groups of four. |
| Native INT8 expert compute | v6 passed the INT8 model reference after the numerical fix. Nine of ten calibration shapes met the speed threshold, so the all-shape selection rule kept BF16. v8 exposed separate INT8 parity failures in the grouped trial. |
| Packed-Q6 Pallas kernel | The CPU reference passed. The v11 TPU canary failed in Mosaic on a vector reshape, so there is no admitted TPU timing for that kernel. |
| Explicit BF16 rounding | The standalone TPU comparison passed all **25** pairs exactly. The candidate measured **0.9986×** geometric speedup on model-width cases and kept the same collective patterns. Full-model profiles had extra rounding-related gathers that this probe did not reproduce. |
| Static batching | Active rows averaged **56.3%** at B4 in v14 and **41.7%** at B16 in v6. Short requests leave idle rows while longer requests finish. Continuous refill and length-aware scheduling remain unmeasured. |
| TorchTPU and BF16×6 | V17 passed native vision and both eight-rank EP/TP layouts. The Torch synchronization error prevented speed and BF16×6 results. A separate JAX Q6 projection reference mismatch is still open. |

## GLM-5.3-Flash

| Version | Kaggle | What we tried and what happened |
| --- | --- | --- |
| v1–v2 | COMPLETE | The retained console output confirms dataset mounts. It does not establish model execution or quality. |
| v3 | COMPLETE | Ran the first spill-engine diagnostics. Engine, quantization, and Q6 plane checks failed. The host-to-chip bandwidth probe ran. |
| v4 | COMPLETE | Quantization and bit-exact Q6 plane checks passed. The broader engine suite and resident-Q6 test invocation remained unsuccessful. |
| v5 | COMPLETE | Routing, quantization, and plane checks passed. The broader suite and resident invocation still failed. Eight-chip host transfer measured **37.32 GB/s aggregate** in the bandwidth probe. |
| v6 | COMPLETE | A missing log-parsing helper stopped the launcher during the first suite. The application reported failure. |
| v7 | COMPLETE | Routing, quantization, and plane checks passed. The batched MoE and broader engine tests failed. The bandwidth probe measured **37.03 GB/s aggregate**. |
| v8–v10 | Canceled | The version-specific output listings returned no logs or files. |
| v11 | Canceled | Loaded the full routed expert store and passed packed decoder and vision reference checks. It passed **9/10** completed image assertions before cancellation. Expert cache hits were **26.6%**, with about **12.5 TiB** of reported host-to-device expert payload. |
| v12 | ERROR | Completed **11/12** image assertions. Seven circles were counted as eight in both single and batched execution. Median language prefill accounted for **99.63%** of the measured single-image case time. |
| v13 | COMPLETE | Q4 and Q6 each passed all **8** selected vision checks and **13** batch answer checks, with native reasoning enabled. The short benchmark included compilation of new batch programs. |
| v14 | ERROR | Both worker trials completed with a larger expert cache and passed **7/8** vision checks. Both read an upside-down road marking as `SLOW` where the expected text was `STOP`. |
| v15 | ERROR | Both workers again passed **7/8**. The larger-resolution road control still returned `SLOW`. The resident cache numerical gate passed, and a toy INT4 dot compiled and ran on TPU. |

### GLM batch-8 results

| Version and workload | Q4 tokens/s | Q6 tokens/s |
| --- | ---: | ---: |
| v13 short batch quality runs with compilation | 0.643 | 0.334 |
| v14 warmed full responses | 2.417 | 1.490 |
| v15 diverse full responses | 1.711 | 1.119 |
| v15 matched short decode window | 12.140 | 4.900 |

Repeated or fixed-window work can reuse routes and avoid the long tail of a full
response. Use the workload labels when comparing speeds. Hosted GLM also read the
same road marking as `SLOW`, so the available comparisons do not isolate the
error to TPU quantization.

## Standalone probes and hosted comparisons

| Experiment | Result |
| --- | --- |
| TPU environment probe | Saw eight devices and completed a sharded operation. Established the allocation used for later work. |
| GLM image-input gate v1–v2 | Model server and image delivery passed startup. Quality scored **10/11** and **11/12** respectively. Seven circles were counted as eight in both versions. |
| Qwen 27B image-input gate | Model server and image delivery worked. **10/11** asserted cases passed, with a failure on `GO 3`. |
| Qwen traffic runtime gate | Environment and processor checks passed. Native multimodal RoPE and tiny-model checks failed on the experimental dependency stack. |
| BF16 rounding on TPU | All 25 equivalence pairs passed. The candidate stayed experimental after the timing comparison. |
| CPU tracking and publisher smoke tests | Verified private submission, startup traces, and report retrieval in bounded CPU checks. Later v16 showed that live W&B summary readback can lag independently of Weave. |
| NVIDIA NIM low-effort GLM versus DeepSeek | On 15 matched vision cases, GLM scored **13/15** and DeepSeek **12/15**. Both passed **3/4** real frames and **4/4** text controls. Median hosted vision request time was **144.65 s** for GLM and **15.84 s** for DeepSeek on this sample. |
| NVIDIA NIM maximum-effort follow-up | DeepSeek passed three selected controls. GLM passed both completed controls, with one HTTP 504. These follow-ups used cases selected after the low-effort comparison. |
| DeepSeek TPU fit study | The inspected checkpoint required about **475 GiB**. Its measured host/HBM budget was about **445 GiB** before buffers. This was a storage feasibility check. |

## Open questions carried into v17

- **Torch performance.** Compare the built-in and explicit BF16×6 products
  against independent FP64 references before timing. Measure matched JAX,
  native TorchTPU, and hybrid packed projections and KV reads. Confirm eight-rank
  EP/TP correctness. The full language model still runs in JAX for this trial.
- **Batch-sensitive OCR.** Compare B1, identical B4, and mixed B4 with a matched
  token prefix to identify the first prediction difference. Keep the ordinary
  strictly scored replay alongside the diagnostic.
- **Cost to generate usable answers.** Report quality, preparation time,
  decode rates, and memory admission together. Prioritize the packed expert and
  KV paths in the next device profiles, then test batch occupancy.

This public record uses exact-version Kaggle status, output logs, application
reports, profiler summaries, and the earlier run archive. It publishes aggregate
findings and synthetic-case examples. The version ledger captures the audit
date, platform status, and evidence used for each version.
