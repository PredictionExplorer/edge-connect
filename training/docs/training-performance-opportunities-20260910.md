# Training performance opportunities

## Assessment

There is a concrete, high-impact problem in the data path: streamed self-play produces small replay shards, but the learner only admits complete 512-row chunks from individual shards. In a read-only snapshot at learner step 164,379, the selected ring-10 window contained 745,314 positions, yet only 91,136 survived this chunk filter. **654,178 positions, or 87.77%, were unavailable to the sampler.** The replay counter nevertheless credits committed positions toward the update-to-data allowance.

This is the first intervention to make. It can expand the available ring-10 training pool by approximately **8.18× without generating additional positions**. That is a data-availability ratio, not a demonstrated Elo/hour multiplier. Repairing the consumer does not itself increase the rate at which actors generate positions or remove the configured UTD limit.

A second incompatible assumption compounds it: retention counts files, while streaming changed how many positions each file contains. A nominal 3,000-file history can represent far fewer positions than before streaming. There is also substantial per-window subsampling of the chunks that do survive.

The strongest subsequent opportunities are cheaper neural search, compact semantic cache keys, lower-bandwidth actor inference, and extracting more learning signal from existing search. New 2026 research supports testing regret-guided restarts and action-value-based learning with much less search. None of those papers establishes a transferable 10× Elo/hour gain for this system.

The correct sequence is **restore fresh-data consumption, establish a clean learning baseline, then pursue the larger computational and algorithmic changes**. Raising UTD first could intensify repeated training on the restricted subset.

## Measured baseline

The main steady-state window is September 10, 04:50–07:15 UTC: 1,741 existing five-second observations, covering 8,700 seconds. It predates the 07:30 shutdown and uses release f414829 with the same training settings as the current dced953 release. These observations are not a matched benchmark of the newly deployed bookkeeping changes.

| Measurement | Observation | Interpretation |
| --- | ---: | --- |
| Learner phase: UTD wait | 7,470 seconds / 85.86% | Fresh committed data limits permitted updates |
| Learner phase: training | 1,160 seconds / 13.33% | Much learner capacity is unused |
| Learner GPU utilization | 11.49% mean | Consistent with data-supply throttling |
| Learner GPU memory used | 76,523 MiB mean | Idle compute does not imply free memory |
| Actor GPUs 1–6 utilization | 50.72–55.31% mean | Requires kernel/CPU-path profiling; utilization is not FLOP efficiency |
| Learner progress | 162,157 → 164,184 | Approximately 839 optimizer updates/hour |
| Median device step duration | 0.502 seconds | Logged warm device timing, not full wall time |
| Median logged per-step data-wait average | 0.00098 seconds | Ordinary loader wait was not the main observed bottleneck |
| Actor prediction-cache hits | 0.84–1.49% | Low hit rates despite substantial key construction |
| Padded neural rows | 13.03–17.10% | Paid neural work that produces no additional search result |
| Mean physical neural batch | 63–78 rows | Below the configured 256-row broker maximum |
| CUDA graph evictions/fallbacks/validation failures | Zero during this window | Reintroducing graph caching is not an opportunity; it already works |

Physical inference metrics were differenced only across matching worker PIDs. Preparation, key construction, neural round-trip and other timers overlap or nest; they must not be added into a fictitious critical-path total.

The newly restarted run was observed separately. Its initial scarcity of completed games is a startup effect, not a steady-state throughput estimate.

## 1. Repair the streamed-replay consumer

**Confidence: confirmed code incompatibility, reproduced independently and quantified on live metadata. Priority: first.**

The producer and consumer disagree about storage granularity:

1. Self-play finalizes completed games and flushes them immediately when streaming is enabled. A flush may contain one or several games; it does not wait for the configured 4,096-row shard target. See [streaming publication](/Users/tarasbobrovytsky/Dev/EdgeConnect/training/startrain/selfplay.py:946).
2. The learner creates chunks using the integer quotient of each selected span's row count and the full batch size. A span below 512 contributes zero chunks. See [chunk construction](/Users/tarasbobrovytsky/Dev/EdgeConnect/training/startrain/learner.py:305).
3. The sampler passes batch size 512 to that routine, even though it subsequently takes only 128 rows from each of four chunks. Both persistent and ordinary loaders use this sampler. See [sampler construction](/Users/tarasbobrovytsky/Dev/EdgeConnect/training/startrain/learner.py:448) and [row selection](/Users/tarasbobrovytsky/Dev/EdgeConnect/training/startrain/learner.py:515).
4. All committed rows still increase the UTD allowance, whether or not the sampler can select them. See [allowance calculation](/Users/tarasbobrovytsky/Dev/EdgeConnect/training/startrain/learner.py:3577).

The read-only selection snapshot produced:

| Board | Selected rows | Rows in complete 512-row chunks | Excluded rows | Excluded fraction |
| --- | ---: | ---: | ---: | ---: |
| Ring 4 | 393,844 | 102,400 | 291,444 | 74.00% |
| Ring 6 | 1,000,000 | 850,432 | 149,568 | 14.96% |
| Ring 8 | 1,000,000 | 846,336 | 153,664 | 15.37% |
| Ring 10 | 745,314 | 91,136 | 654,178 | 87.77% |

For ring 10, 636,232 rows belonged to 2,877 whole spans smaller than 512. Another 17,946 rows were incomplete tails of larger spans. These positions are excluded by the current indexing logic, not merely given a low sampling probability.

At the same cutoff, all 13 ring-10 shards created since the latest restart—1,617 positions—and all 31 ring-4 shards—1,447 positions—were below 512. This does not mean all future streamed data is excluded: several games completing together can produce a sufficiently large shard.

A metadata-only reproduction demonstrates the mechanism without reading or writing replay payloads:

| Available spans | Production sampler behavior |
| --- | --- |
| Four 128-row spans | 512 available rows, zero chunks, sampler rejects |
| Four older 512-row spans plus four newer 128-row spans | Across 100 epochs, older rows are selected; no newer short-span row is selected |

The 91,136 surviving ring-10 rows form 178 full chunks. Four-chunk grouping can produce at most 44 batches, or 22,528 unique rows, in one window. This additional omission is different: rows within accepted chunks can be selected in later windows. The short-span exclusion persists until the consumer or physical packing changes.

**Proposed repair:** preserve durable streaming and assemble homogeneous-ring batches across arbitrary short spans and partial tails. Treat shard diversity as a goal or minimum where feasible, rather than requiring exactly four files. Four ring-4 games cannot fill a 512-position batch. Changing the threshold from 512 to 128 would therefore leave another exclusion problem.

Unify readiness, capacity calculation and actual sampling. Currently, [readiness](/Users/tarasbobrovytsky/Dev/EdgeConnect/training/startrain/learner.py:3150) can count aggregate rows while [capacity](/Users/tarasbobrovytsky/Dev/EdgeConnect/training/startrain/learner.py:3045) floors each span separately.

**Acceptance evidence:** run real streamed game shards through the actual 512-row loader; cover all-short, mixed-size, partially selected and multi-ring data; verify row identity, uniqueness, model-age filters, mode exposure, distributed partitioning and GC protection. Measure which game IDs and model versions are actually consumed. Existing streaming tests and sampler tests each covered their own contracts but missed this combination.

This finding changes how other diagnostics should be interpreted. Nominal UTD, target freshness, losses and mode balance can all describe repeated training on an unexpectedly restricted subset.

## 2. Make replay retention independent of file size

**Confidence: confirmed retention policy; capacity estimates are conditional.**

[Garbage collection](/Users/tarasbobrovytsky/Dev/EdgeConnect/training/startrain/replay_store.py:887) retains a number of files per ring. It does not enforce a minimum retained position count or consult the learner's million-position target. The live setting remains 3,000 files per ring.

| Average positions/file | Nominal positions in 3,000 files |
| --- | ---: |
| Former 4,096-row shard target | 12,288,000 |
| 200-row streamed files | 600,000 |
| Observed selected ring-10 mean, 244.21 | Approximately 732,615 |

The observed-size comparison is approximately **16.8× less nominal capacity** than full 4,096-row files. It is not evidence of a 16.8× strength loss, nor a measurement that old files always reached 4,096 rows.

Active selections have protected ID ranges, so the cap can temporarily be exceeded. The learner [clears its watermark before periodic GC](/Users/tarasbobrovytsky/Dev/EdgeConnect/training/startrain/learner.py:2360). These pins protect in-flight reads; they do not guarantee a persistent million-position history.

**Proposed repair:** retain coverage by eligible positions and relevant mode/ring cells, subject to a separate byte budget. Optional immutable compaction can reduce file overhead, but must preserve logical sample identities, timestamps, checksums and model provenance. Compaction must not re-credit old positions as newly generated experience.

This is a second producer–consumer contract issue to resolve alongside sampling, before drawing conclusions about the value of a smaller replay window.

## 3. Replace expensive cache keys and move lookup earlier

**Confidence: measured waste and concrete code path; whole-system speedup unmeasured.**

Every native request is scored and feature-packed before the prediction cache is consulted. Python then validates, clones and serializes the complete inputs into a bytes key. Static topology appears repeatedly inside those keys. See [native packing](/Users/tarasbobrovytsky/Dev/EdgeConnect/training/crates/star-py/src/lib.rs:3066), [preparation](/Users/tarasbobrovytsky/Dev/EdgeConnect/training/startrain/inference.py:515), [serialization](/Users/tarasbobrovytsky/Dev/EdgeConnect/training/startrain/inference.py:638), and [lookup](/Users/tarasbobrovytsky/Dev/EdgeConnect/training/startrain/inference.py:891).

A local ring-10 dimension inspection produced a **54,629-byte key with a short dummy namespace**, for a **2,320-byte prediction payload**. The actual namespace adds a small amount. Much of the storage and copying describes the input rather than the answer.

In the steady window, key construction consumed approximately 998–1,334 producer CPU-seconds per actor GPU over 8,700 wall seconds. It is included in preparation time and may overlap GPU work. Cache eviction counts were nearly as large as miss counts, while hits were only about 1%.

**Proposed design:** an exact compact semantic descriptor for native requests, including stones, all observable history, turn state, rules/features, PDA, immutable model identity and value semantics. Look up predictions before expensive feature construction. Share topology by immutable ring/schema identity. Retain complete equality checking rather than trusting a lossy hash alone; keep the general full-input path for non-native callers.

Then test cache admission and retention policies suited to search: protect likely next roots and recently useful descendants, and compare against a dedup-only control. Blindly increasing cache size or disabling caching while leaving full-key deduplication enabled would miss much of the cost.

D5-canonical caching is a later extension. The architecture has tested symmetry, but policies must be inverse-mapped and finite-precision differences measured. Cross-game reuse of later positions is likely limited and should be measured by phase; within-search transpositions already exist. This is not a universal 10× cache multiplier.

## 4. Reduce padding and improve compatible request packing

**Confidence: padding measured; optimal batching policy unknown.**

The current [CUDA bucket rule](/Users/tarasbobrovytsky/Dev/EdgeConnect/training/startrain/inference.py:780) uses powers of two, with intermediate buckets above 64 for graph inference. Missing rows are padded with repeated inputs. They do not become extra search results.

The observed 13–17% padding corresponds to a theoretical 1.15–1.21× reduction in physical row work if removed without any other cost. Actual speed depends on kernel shapes, graph residency and queue delay. That ceiling is not a whole-training speedup prediction.

Screen workload-specific buckets—especially the commonly used smaller sizes—and scheduling that combines compatible cohorts without extending game tails excessively. Mean physical batches of 63–78 against a maximum of 256 do not prove that increasing the number of cohorts is beneficial: more simultaneous games can lengthen time to terminal labels and model refresh. Previous cohort expansions already showed that tradeoff.

The freshly installed first-visit batching is useful for testing this interaction. It is already implemented, so it should not be counted as a newly discovered optimization. Across a full 640-simulation search, it affects only the first candidate visits.

## 5. Audit actual activation precision and fuse graph message passing

**Confidence: eager dtype behavior confirmed; compiled H100 materialization needs profiling.**

BF16 autocast applies to eligible operations, not every tensor. The [initial node projection](/Users/tarasbobrovytsky/Dev/EdgeConnect/training/startrain/model.py:707) is multiplied by an FP32 mask. FP32 embeddings and residual layer scales promote later values. A small CPU autocast inspection confirmed FP32 normalization/gather inputs and BF16 projection outputs.

At batch 128, ring 10 and width 384, a fully materialized FP32 neighbor tensor with padded degree seven is approximately **360.94 MiB**. There are sixteen local blocks. Compilers can fuse away some intermediates, so this is an eager tensor-size calculation, not a claim that all those buffers coexist on the H100.

The promising target is fused normalization/projection/gather/nonlinearity/reduction, with deliberate lower-precision storage where validated and safe accumulation where needed. [Graphiler](https://proceedings.mlsys.org/paper_files/paper/2022/hash/a1126573153ad7e9f44ba80e99316482-Abstract.html) provides relevant compiler precedent, but its gains against generic GNN frameworks do not transfer directly to this compiled model.

An important rejected shortcut: projecting before gathering was already tested. Its compiled ring-10 local-block result regressed, with an old/new latency ratio of 0.71 under contention. Do not redeploy that rewrite based on algebra alone. See [prior experiment](/Users/tarasbobrovytsky/Dev/EdgeConnect/training/docs/elo-per-hour-audit-2026-09-05.md:185).

Selective FP8 hidden linears are also worth profiling after this audit. H100 supports ordinary FP8, but conversion/scaling overhead and relatively small matrix shapes can erase the benefit. NVIDIA documents this shape dependence in [Transformer Engine performance guidance](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/speedups.html). Keep sensitive normalization and prediction operations at appropriate precision, and charge quantization overhead and search-quality changes.

## 6. Find the search-budget/strength frontier

**Confidence: arithmetic is exact for nominal budgets; learning consequences require experiments.**

Ring-10 scaling gives 53 fast simulations, 640 full simulations, and 53 considered candidates. With 65% fast and 35% full searches, nominal work before PDA is:

**0.65 × 53 + 0.35 × 640 = 258.45 simulations per searched atomic decision.**

Full searches consume approximately **86.7%** of this work. See [budget scaling](/Users/tarasbobrovytsky/Dev/EdgeConnect/training/startrain/selfplay.py:458).

| Effective ring-10 fast/full budgets | Nominal mean simulations | Reduction in simulation work |
| --- | ---: | ---: |
| 53 / 640, current | 258.45 | 1× |
| 16 / 128 | 55.20 | 4.68× |
| 8 / 64 | 27.60 | 9.36× |

These are experimental effective budgets, not suggested unreviewed YAML edits and not Elo forecasts. Candidate count, target weighting, PDA floors and ratios must be adjusted coherently.

Gumbel planning was designed to improve policy learning with small search budgets. Its policy-improvement guarantee still depends on correctly evaluated action values; it does not guarantee that eight noisy simulations train this game as well as hundreds. [Gumbel AlphaZero paper](https://openreview.net/pdf?id=bERaNdoegnO).

First establish fixed-position disagreement and equal-time playing strength; then compare learning curves from common checkpoints. Test whether reduced full-search frequency or a smaller full budget produces more useful supervision per hour.

Replace entropy-only allocation, eventually, with an estimate of benefit from additional search: predicted action regret, disagreement with deeper search, or instability of the selected move. A confidently wrong model can have low entropy; many equally good actions can have high entropy.

## 7. Distill search into a cheaper actor and richer action-value targets

**Confidence: plausible larger redesign, not a low-risk switch.**

The current system applies the same 17.4-million-parameter model to enormous numbers of search leaves. A strong learner paired with a smaller distilled actor could reduce the cost of those evaluations. Reserve the large model or deeper search for difficult positions, with a measured fallback policy.

For illustration, reducing width from 384 to 256 and groups from eight to four gives a rough quadratic dense-compute ratio of 2/9; width 192 and four groups gives roughly 1/8. Those are architecture arithmetic scenarios. They do not establish 4.5× or 8× latency or retained strength.

The existing search computes action Q values, but [replay fields](/Users/tarasbobrovytsky/Dev/EdgeConnect/training/startrain/replay.py:226) and [decision recording](/Users/tarasbobrovytsky/Dev/EdgeConnect/training/startrain/selfplay.py:1317) do not preserve action-specific Q supervision. Distilling visited Q values with visit/confidence information could teach an inexpensive per-action value head and reduce future leaf evaluations.

For non-PDA states with at least 53 legal placements, fast search spends its 53 visits on 53 candidates, leaving no budget for explicit deeper exploration of those candidates. Smaller legal sets and PDA-adjusted budgets can permit deeper exploration. An action-value head is especially relevant to the one-visit case, although eliminating fast-search work alone removes only 13.3% of nominal work.

Domain-aligned representations can produce large gains: [Chessformer, ICLR 2026](https://arxiv.org/abs/2605.19091) reports stronger chess play using square tokens, geometric attention bias and an appropriate action head. This model already has graph-relative geometry and rule conditioning, so “add positional bias” is not a missing feature. The useful lesson is to compare specialized representations and actor sizes under a shared compute budget.

## 8. Restart from valuable positions and reanalyse existing data

**Confidence: strong research directions after the data-path repair.**

Self-play [starts from empty boards](/Users/tarasbobrovytsky/Dev/EdgeConnect/training/startrain/selfplay.py:908). Surprise weighting affects training weights, but does not start new games from important intermediate positions.

[Regret-Guided Search Control, ICLR 2026](https://arxiv.org/abs/2602.20809) collects trajectory and search-tree states, estimates which states contain learning opportunities, and revisits them as new starting positions. It reports mean gains of 77 Elo over AlphaZero across Go, Othello and Hex. These are different games, limited repetitions and a different compute profile; its additional inference cost must be charged.

A practical progression is a small fraction of uniformly selected intermediate starts, followed by disagreement/regret-based prioritization. Keep ordinary opening games for distribution coverage. Star's exact semantic state and history must be preserved; turn ownership, handicap and pie state cannot be approximated.

Selective reanalysis can refresh stale policy targets with the current model while preserving factual terminal outcomes separately. [MuZero Unplugged](https://arxiv.org/abs/2104.06294) established this broader idea; [ReZero, 2024](https://arxiv.org/abs/2404.16364) specifically addresses reanalysis cost through backward reuse and periodic sweeps.

Use idle learner capacity through bounded, coordinated work rather than launching an unrelated process into a GPU already holding about 75 GiB. Most importantly, store reanalysed targets as derived versions of existing samples. Relabelled copies must not increment the fresh-self-play counter or manufacture additional UTD allowance.

## 9. Explore much less search during training

**Confidence: high-upside research branch.**

[KLENT, ICML 2026, revised May 21](https://arxiv.org/html/2602.10894v2) revisits KL- and entropy-regularized policy optimization with action-value learning and lambda returns. It eliminates training-time look-ahead search. The paper reports up to fourfold training efficiency primarily in simulator-evaluation comparisons; a wall-clock Go example reports a smaller saving. Superior asymptotic performance on larger boards is not established.

This is a legitimate alternative to assuming every useful training decision needs hundreds of full-network evaluations. A hybrid path could learn action values from the search data already available, use cheap policies for some training games, and preserve search for evaluation and selected difficult states.

Adapting temporal-difference returns requires care: consecutive atomic placements in Double Star can belong to the same player. Blindly alternating the sign every step would introduce an error. Classic, handicap and pie transitions must retain their exact semantics.

The broader compute-optimal question is whether more cheap, diverse experience can outperform fewer expensive searches after equal wall time. It cannot be answered from simulator counts alone.

## 10. Repair exposure and feedback clocks before tuning them

The sampled steady-window ring-10 training diagnostics contained 88,576 rows. Pie-double received 7.51%, while handicap-classic received 22.83%, against an equal-six-mode evaluation objective. Diagnostics are sampled, and chunk exclusion is a confound, so this is not a complete all-batch exposure census.

[Replay selection](/Users/tarasbobrovytsky/Dev/EdgeConnect/training/startrain/replay_store.py:1579) enforces four aggregate segments, not six modes. Its shortfall redistribution is greedy rather than proportional as the docstring describes. After fixing packing, measure complete consumed distributions and allocate generation/sampling to deficient cells. A curriculum should optimize learning progress while preserving evaluation coverage.

Historical-model work is also different from opponent diversity. A selected older evaluator plays both sides of a cohort. These are older-model self-play games, not current-versus-history matches. In the approximately one-day finished-task sample, history and champion cohorts consumed 42.7% of neural rows. In the shorter steady publication window, candidate-generated positions dominated. Planned role probabilities must not substitute for measured shares, and age filtering is already implemented.

At 839 learner steps/hour, EMA decay 0.9999 implies approximately an 8.26-hour half-life and 11.92-hour mean weight age. Snapshot and promotion-candidate cadences imply approximately 3.49 and 17.46 hours respectively at that rate. These are cadence calculations, not proof that raw weights are better.

Earlier EMA/freshness/clinch screens did not establish promoted gains, and a corrected held-out screen favored EMA over raw weights. See [experiment history](/Users/tarasbobrovytsky/Dev/EdgeConnect/training/docs/model-improvement-roadmap.md:78) and [EMA comparison evidence](/Users/tarasbobrovytsky/Dev/EdgeConnect/training/docs/gradient-clipping-rollout-20260908.md:113). Revisit them only after fresh-data flow is working and with current common-checkpoint comparisons.

Higher UTD—such as 1.5 versus 3 or 6—could use idle learner capacity afterward. It means sample presentations per fresh committed position, not optimizer steps per transition. Preserve publication cadence per fresh data and explicitly define EMA/LR clocks. More updates on unchanged, stale targets can worsen overfitting.

## 11. Preserve supervision across interruptions

In 625 finished GPU-task records over approximately one day, 552,990 of 7,260,915 attempted decisions were dropped: **7.62%**. All 511 cleanly completed task records had zero dropped decisions. These aggregates describe recorded completed tasks, not a stationary all-work loss rate.

The latest shutdown alone dropped 145,852 decisions across 1,644 unfinished games. Graceful deployment preserved learner checkpoints, but the actor's in-flight trajectories existed only in memory. Perfect recovery of the whole-day dropped-decision fraction corresponds to approximately 1.082× decision yield, not 10×.

The shortest salvage path is to persist already-computed policy supervision with unavailable outcomes masked. [Replay already supports policy-only samples](/Users/tarasbobrovytsky/Dev/EdgeConnect/training/startrain/replay.py:565). A stronger design checkpoints per-game state, completed-move trajectory, immutable model/configuration identity and random streams, then resumes to obtain exact terminal outcomes.

Preserve model artifacts needed by resumable games, avoid duplicate publication, and retain distinct provenance for policy-only salvage. Pin expiry itself is not a discard deadline: it stops refills and drains games.

## 12. Additional opportunities with narrower scope

| Opportunity | Why investigate it | Important limit |
| --- | --- | --- |
| Exact clinch/solved-state propagation inside search | Actor-level proofs are not generally reused as solved interior tree results | Benefit depends on how often searched leaves are provably decided |
| Memoized/pruned exact endgames | Current bounded solver enumerates action orders without full transposition memoization or alpha-beta pruning | A large solver speedup may affect little total runtime |
| Learned pair/afterstate proposals | Could avoid some full-network evaluations between same-player placements | Atomic factorization and completed-pair transpositions already exist; enumerating all pairs is expensive |
| Calibrated resignation with an audit continuation subset | May shorten hopeless trajectories before an exact clinch proof is available | This trades exactness for learned judgments; score/spatial labels and false resignations need explicit treatment |
| Outcome-only versus synthetic clinch auxiliaries | Exact binary outcomes coexist with synthetic loser-filled score/ownership labels used by score-aware search | Intentional existing option, not an established bug; earlier screens were inconclusive |
| Fused trainable attention-bias backward | Current workaround constructs dense FP32 attention quantities | Secondary while the learner waits; preserve the known gradient-correctness regression tests |
| Better replay chunk utilization | Current four-shard selection consumes only one quarter of each admitted chunk per window | Solve arbitrary short-span packing as part of the same design |
| Reconsider redundant D5 augmentation | Architecture already has tested D5 equivariance, while augmentation copies/reconstructs samples | Compiled finite-precision loss/gradient parity must be checked |
| Persistent arena evaluators | Avoid repeated model/compile preparation around evaluation leases | Measure lease overhead; do not deprive self-play of its shared GPU |

For the attention workaround, [FlexAttention's trainable-bias support](https://pytorch.org/blog/flexattention-for-inference/) is relevant. Do not infer that every Flash backend supports the required backward path, or import LLM decoding speedups into this square-attention workload.

## Measurement and experiment order

There is not yet a clean measured 10× target baseline. Current reports contain historical workers and multiple search/evaluation epochs. A rating epoch's new-anchor gain divided by whole-run elapsed time is not the current release's Elo/hour. Also, [balanced-strength contract reconstruction](/Users/tarasbobrovytsky/Dev/EdgeConnect/training/startrain/balanced_strength.py:48) omits the new optional execution settings and rejects nondefault execution contracts. The default live setting is unaffected; experimental evidence needs compatible accounting before those settings are enabled.

The current balanced promotion test also has limited power for small improvements. At six cells times 40 pairs, its implemented boundary needs an observed score near 59.94%, approximately +70 Elo. An expected +35-Elo score does not pass that deterministic boundary calculation. This is not a probability-of-promotion calculation. Better calibrated paired/stratified sequential inference may improve feedback latency without lowering evidence standards; more frequent promotions alone would not establish stronger learning.

Use time to a fixed, independently measured strength target from common checkpoints. Keep rules, six-mode weighting, openings, color reversals and evaluation time budgets fixed. Charge compilation, evaluation, restarts and failed treatments. Use cheap one-seed screens to reject weak candidates and multiple seeds for the selected comparison.

| Order | Proposed experiment | Main success criterion |
| --- | --- | --- |
| 1 | Arbitrary short-span packing, capacity/readiness alignment | Fresh committed game IDs actually reach training; no row loss or duplicates |
| 2 | Sample-aware retention and complete consumed-data accounting | Intended history, freshness and mode coverage survive GC |
| 3 | Re-establish control learning curves | Valid current-epoch strength/time denominator |
| 4 | Compact native keys, cache admission and bucket scheduling | More useful evaluations/second at equal outputs and memory limits |
| 5 | Selective activation precision / fused message passing | Whole-model and whole-search speed with acceptable numerical/strength behavior |
| 6 | Search-budget frontier | Better equal-wall-time learning, not merely fewer simulations |
| 7 | Restart/reanalysis and richer Q supervision | More improvement per original environment position and per total GPU hour |
| 8 | Cheap actor / search-light learner alternatives | Stable learning and a superior strength-versus-time frontier |

Two credible routes toward a large gain are now identifiable. The first is recovering the large fraction of already-paid-for data that cannot currently be sampled. The second is a compound redesign using cheaper actors and substantially less search while preserving or improving supervision. Their factors cannot be multiplied into an Elo forecast: they interact through target quality, data distribution and learning dynamics.

## Evidence scope and sources

The code audit covered the Rust search/engine and Python inference, actor, self-play, replay, sampler, learner, model, optimizer, loss, promotion and strength-accounting paths, together with prior experiment reports. Current production source is dced953118ac6e6e4a5c5cc88551a41e2979c00e; training settings were read from its frozen live profile. No source, configuration, service, checkpoint or replay mutation was performed for this audit.

Read-only evidence included 2,146 monitor observations, 981 finished actor-task records including CPU records, 28,585 publication records, 1,998 learner metric records, and consistent SQLite read snapshots. The 625-task GPU subset and the 1,741-record steady window are explicitly separated above. Small local metadata/dtype inspections were diagnostic reproductions, not implementation or H100 performance benchmarks. A full GPU kernel trace and controlled learning trials remain outstanding.

The literature inventory emphasizes primary sources:

1. Ota et al. **Revisiting Regularized Policy Optimization for Stable and Efficient Reinforcement Learning in Two-Player Games**. ICML 2026, revision May 21, 2026. [KLENT](https://arxiv.org/html/2602.10894v2).
2. Tsai et al. **Regret-Guided Search Control for Efficient Learning in AlphaZero**. ICLR 2026. [Paper](https://arxiv.org/abs/2602.20809).
3. **ReZero: Boosting MCTS-based Algorithms by Backward-view and Entire-buffer Reanalyze**. 2024. [Paper](https://arxiv.org/abs/2404.16364).
4. Schrittwieser et al. **Online and Offline Reinforcement Learning by Planning with a Learned Model**. 2021. [MuZero Unplugged](https://arxiv.org/abs/2104.06294).
5. Danihelka et al. **Policy Improvement by Planning with Gumbel**. ICLR 2022. [Paper](https://openreview.net/pdf?id=bERaNdoegnO).
6. Monroe et al. **Chessformer: A Unified Architecture for Chess Modeling**. ICLR 2026. [Paper](https://arxiv.org/abs/2605.19091).
7. Neumann and Gros. **AlphaZero Neural Scaling and Zipf's Law: a Tale of Board Games and Power Laws**. Revised October 2025. [Paper](https://arxiv.org/abs/2412.11979). Supports investigating state-frequency/phase imbalance; does not prove that larger models or endgame data are harmful here.
8. Neumann and Gros. **Scaling Laws for a Multi-Agent Reinforcement Learning Model**. 2022. [Paper](https://arxiv.org/abs/2210.00849). Larger networks can be more sample-efficient; optimal size depends on compute.
9. Wu. **Accelerating Self-Play Learning in Go**. 2019–2020. [KataGo paper](https://arxiv.org/abs/1902.10565). Its reported 50× comparison is historical; many contributing techniques are already present here.
10. **Graphiler: Optimizing Graph Neural Networks with Message Passing Data Flow Graph**. MLSys 2022. [Paper](https://proceedings.mlsys.org/paper_files/paper/2022/hash/a1126573153ad7e9f44ba80e99316482-Abstract.html).
11. Dong et al. **FlexAttention Part II: FlexAttention for Inference**. PyTorch, 2025. [Trainable-bias documentation](https://pytorch.org/blog/flexattention-for-inference/).
12. NVIDIA. **GEMM Speedups Across Precisions**. Transformer Engine documentation, accessed September 10, 2026. [Guidance](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/speedups.html).

The accompanying [evidence summary](/Users/tarasbobrovytsky/Dev/EdgeConnect/training/docs/training-performance-evidence-20260910.json) records the measurements and their scope. No proposed treatment has yet demonstrated an EdgeConnect Elo/hour multiplier.
