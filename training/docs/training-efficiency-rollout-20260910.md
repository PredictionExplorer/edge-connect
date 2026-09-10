# Training efficiency rollout — September 10, 2026

## Current deployment — completed

The H100 training server now runs production source **`e8e877cd7292dd44558e280c8117e6242820c2ce`** from `/home/ubuntu/edgeconnect-releases/variant-training-followup-e8e877c`, using frozen profile `/home/ubuntu/edgeconnect-runs/variant-network/profile-training-optimized-20260910.yaml`.

All three validated options are enabled: shared homogeneous training geometry, smaller CUDA-graph inference buckets, and preservation of computed policy targets during clean actor interruptions. The compact-gather experiment remains disabled because its earlier H100 comparison showed no material gain. Search budgets, model architecture, BF16 mixed precision, UTD target, optimizer/LR/EMA clocks, and evaluation settings retain their prior values.

The final activation stopped at **171,673** with **87,896,576 examples consumed**, preserving checkpoint SHA-256 `405dbeb16e7aa3a93fe0c08d81f0ef43d5c0634dc3a6b815b77e31ead3a7cdd8`. Migration discarded zero optimizer steps, preserved all frozen learner/arena control hashes, and made no UTD transition. Training resumed from that exact checkpoint and advanced to **171,674**. A sustained readiness check and an independent final check passed with ten healthy workers, zero restarts or inference failures, active monitoring/backup/report timers, and stopped temporary recovery guards. Verified backups preceded both source migration and activation; normal disaster backups again use the regular namespace.

Final profile SHA-256: `1db1a8b29a177e9b92a455d1468247e88e5fcfc5e74ce0aef1d9867a09265d62`. Final materialized configuration SHA-256: `6af7df9bf422e2befbcf3c4a19a9090271996cb4183c80e466aaac0c33ce57b1`.

### Validated execution gains

| H100 comparison | Baseline | Enabled option | Scope |
| --- | ---: | ---: | --- |
| Ring-10 B512 training step | 527.975 ms | 420.530 ms | 1.2555× throughput; resident synthetic batch, verified EMA checkpoint, fresh production optimizer |
| B512 peak allocated memory | 70.27 GiB | 62.13 GiB | 8.14 GiB less allocated; paired-run reserved memory stayed similar |
| Ring-10 B128 training step | 186.340 ms | 159.501 ms | 1.1683× throughput |
| Inference, 3 valid rows | 4 physical rows | 3 physical rows | 1.0554× adapter throughput |
| Inference, 5 valid rows | 8 physical rows | 6 physical rows | 1.0584× adapter throughput |
| Inference, 9 valid rows | 16 physical rows | 12 physical rows | 1.0983× adapter throughput |
| Inference, 17 valid rows | 32 physical rows | 24 physical rows | 1.1700× adapter throughput |
| Inference, 33 valid rows | 64 physical rows | 48 physical rows | 1.2034× adapter throughput |
| Inference control, 65 valid rows | 96 physical rows | 96 physical rows | 0.9995×, effectively unchanged |

The geometry benchmark used production static compilation and BF16/TF32 settings, while its FP32 oracle disabled TF32. All valid output heads matched bitwise between baseline and sharing in both FP32 and BF16. BF16 differences were 0.0089% across gradients and 0.114% across relation-bias gradients; all numerical and finite-optimizer checks passed. Isolation held across 122 observations plus the final check, and the complete run took 270 seconds. H100 numerical evidence covers ring 10 at B16, with B128/B512 timing; CPU tests cover other rings. These timings exclude loader, EMA and checkpoint I/O and are not an Elo measurement.

The first bucket benchmark could not verify GPU ownership because PyTorch omitted the `GPU-` UUID prefix required by nvidia-smi. Its timings were rejected. The correction is committed as **`e6c883c`** and deployed separately as a pinned tool at `/home/ubuntu/edgeconnect-rollouts/training-followup-20260910/benchmark_graph_buckets-e6c883c.py`; the immutable production release remains unchanged. A fresh isolated comparison passed all output tolerances and graph-replay checks. The changed-size geometric mean was 1.1155×; this is not a fleet-throughput forecast.

### Live data-path evidence

A real live selection at step 171,673 contained **1,000,000 ring-10 rows**, evenly divided across all six modes to within one row. **999,936** rows can fill complete batches, versus **124,416** admitted by the former per-file chunk rule. This is approximately **8.04× usable data availability** for that selection. The actual sampler produced 8,192 distinct rows, including 6,662 from short files previously excluded.

The earlier source-only release's stable 11:08–15:20 UTC period observed 939 learner steps/hour versus 839 in the earlier 04:50–07:15 period. Its key-construction timers fell substantially, but useful neural rows/second increased only 1.9% observationally and UTD wait remained about 85%. Different models, trajectories and arena occupancy prevent causal throughput or Elo attribution.

The follow-up passed **512 tests on the server, with five skips**, source verification for all 500 tracked artifact files, and Rust checks. The UUID correction passed 30 focused tests; the geometry benchmark passed 34 controller/numerical tests. Detailed receipts and measurements are in [the evidence summary](training-efficiency-h100-evidence-20260910.json) and the server rollout directory.

### Limits of this rollout

No 10× Elo/hour gain or global optimum has been established. Cheaper actors, richer action-value learning, regret-guided restarts, reanalysis and search-light alternatives remain research directions requiring controlled learning trials; they were not switched into this training run. Current execution gains and recovered replay coverage must not be multiplied into an Elo forecast.

## First deployed release

Commit `ef7f9ea443d6844b88caa23af579f3b5b0f89c18` is deployed in the immutable release `/home/ubuntu/edgeconnect-releases/variant-training-efficiency-ef7f9ea` on the H100 training host.

The release repairs streamed replay consumption, aligns readiness and capacity with the actual sampler, retains the eligible sample window rather than relying solely on a file count, and replaces native inference keys with exact compact state descriptors and lazy feature construction. It also adds memoized alpha-beta exact endgames, source-role metrics, and explicit strength-epoch accounting.

The sampler preserves short spans and tails, packs homogeneous batches across arbitrary files, avoids duplicate rows within a window, and partitions distributed batches without overlap. The retention selector preserves the same age-eligible sample window used by training.

## Continuity and protection

Training stopped at optimizer step **165,847**, with **84,913,664 examples consumed**. The exact recovery checkpoint SHA-256 was `a2e5d2cd8cb5ae913011a8c67727bcd211866245609dd6674d1bab33156d99ea`.

Both local replay backup and an independently verified off-host disaster-recovery snapshot completed before migration. The snapshot covers 12,169 objects and 17,603,340,005 catalog bytes; its manifest SHA-256 is `94aa36175d210fbb0fbc672c28ddb86b30f51b98a263f8e9aea846676aea52ba`.

The source-only migration discarded **zero optimizer steps**. Profile bytes, recorded configuration identity, UTD segment, checkpoint controls, optimizer/LR state, and arena evidence were preserved. Profile SHA-256 remains `8dc815f2f25790d882a690313857694b3c3c9d27ea58b5dcef64dd951e6a8a07`; recorded configuration SHA-256 remains `cef8e21f826199dfb00f06c292c10fb09d2e4f874743c73d2de4b8407310c78b`.

A sustained readiness gate passed. Independent post-deployment verification observed step **165,857**, all ten workers healthy with zero restarts, zero actor inference failures, and the expected source and process identities. The main service, monitor, report timer, local backup timer, and disaster backup timer were active. The temporary recovery guard was stopped. The pre-existing global continuity timer was left disabled.

Evidence is retained under `/home/ubuntu/edgeconnect-rollouts/training-efficiency-20260910`, including source checksums, build/test logs, frozen unit files, stop-boundary checkpoint, before/after control hashes, migration receipts, backup verification, readiness observations, and `final-health.json`.

## Live replay evidence

At the new release's first selected window, ring 10 contained **709,238 selected rows**, of which **709,120** could fill complete batches. The previous per-file chunk rule admitted only **71,680** of these rows. That is approximately **9.89× greater packable data availability**, not an Elo or update-throughput multiplier.

A read-only probe ran the actual sampler against live selected metadata: **8,192 distinct sampled rows**, including **6,839 from short files/spans previously excluded**. No replay payload files were opened by this probe. The learner itself logged the repaired `selected-span-packing-v2` selection and a 1,000-batch window.

The live window also exposed a follow-up freshness issue: after 1,307 seconds, only 11 allocated batches had been consumed under the existing UTD allowance, while 2,787 new committed rows were outside the immutable selection. The deployed follow-up adds a bounded refresh for usable new data. Enlarging the usable pool must not postpone fresh-data admission for the entire larger window.

## Performance controls and limitations

Native compact keys reduced a local ring-10 descriptor from 54,682 to 389 bytes. A small CPU network comparison against the retained generic path improved warm request latency from 2.273 to 1.021 ms. This is not an H100 throughput comparison against the previous binary.

The exact-endgame oracle sweep reduced visited nodes from 19,536 to 9,474 while preserving exact results. One six-empty ring-10 case fell from 1,957 to 285 nodes. Endgames represent only part of overall runtime.

The optional compact inference gather produced exact six-head outputs on H100 for tested batches, but showed no material speed improvement (latency ratios 1.007, 1.021, and 1.000). It remains disabled. Smaller CUDA-graph buckets and shared training geometry require isolated H100 evidence before activation.

Training remains BF16 mixed precision with FP32 parameters and sensitive operations. Search budgets, network size, EMA/LR clocks, UTD target, and evaluation strength settings were preserved. No EdgeConnect Elo/hour multiplier has yet been demonstrated; the new strength epoch measures the realized champion frontier and does not establish causal release effects.

## Validation

The first release passed 264 selected tests on the server, a 12-case H100 native/search smoke, Rust workspace tests and strict Clippy, focused rebuilt-native/replay/model checks, and frontend/WASM checks. Local broad-suite failures from additive configuration representations and actor test doubles were corrected and their affected suites rerun successfully. The initial H100 smoke retained the production BF16 mode and verified actual native search requests and outputs.

The completed follow-up and activation are recorded above.

## Follow-up implementation

The follow-up repairs two additional data-consumption issues without changing the profile schema or training allowance:

- An active selection older than five minutes is refreshed when at least a full global batch of usable new rows can enter the successor selection. Rank zero makes and broadcasts the decision; unchanged, future-model, and zero-quota data cannot trigger repeated rebuilds. The existing loader drain and watermark cleanup path retires unconsumed work without crediting it as training.
- The `ring10_priority` objective now splits handicap and pie replay quotas equally between classic and double, matching the existing validated six-mode contract. Mode shortages spill within the aggregate segment. The same selector governs retention, preserving available coverage under GC.

A real-store fixture with equally available modes previously selected no handicap-classic or pie-classic rows. The repaired selector and distributed sampler consume ten distinct rows from each of the six modes. The combined freshness, replay, six-mode, pipeline and shutdown validation passed 127 tests, including persistent-loader quiescence and unchanged UTD state. An independent integration review reran 101 tests successfully.

Two additional controls are implemented with strict default-False compatibility and reversible profile migration:

- `selfplay.preserve_interrupted_policy` persists validated policy targets from cleanly interrupted trajectories. Missing outcome, score and spatial supervision stays masked. Salvaged rows carry separate provenance and accounting; completed-game counts remain factual. Real FP32/BF16 consumer tests validate masked losses and preservation of normal value gradients in mixed batches.
- `train.share_homogeneous_geometry` reuses validated homogeneous geometry and trainable bias computation across a batch. CPU topology proof survives worker IPC and transfers, invalidates on mutation, and falls back for noncanonical inputs. Activation requires full-size compiled H100 numerical and performance evidence.

Bounded benchmark tools compare smaller CUDA-graph buckets and shared training geometry against the current execution paths using verified immutable EMA manifests. Benchmark descendants run in a separate transient service and must be stopped before production resumes. Incomplete, shared-GPU, numerically invalid or out-of-memory comparisons do not authorize activation.

The broad Python suite completed with one obsolete exact-configuration test expectation; its hardcoded historical hash was retained and the test was corrected to compose the new explicit default-omission epoch. Its 59-test suite passed afterward. Production code passed Ruff and Pyright with the project's interpreter.
