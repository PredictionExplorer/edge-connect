# Elo per wall-clock hour: variant-network audit

Observed September 5, 2026, 04:00–04:20 UTC on the eight-H100 training host.
The user's objective is equal importance for six modes (classic/double, each
standard/pie/handicap) and four board sizes (rings 4/6/8/10).

## Delivered changes and deployment status

One-stone handicap support is implemented locally in self-play and evaluation.
The prepared Stage B profile targets one sixth of actor batches per mode category
and one quarter of learner/actor allocation per board size. Replay quotas still
operate on four aggregate segments, so within-segment mode equality is an expected
sampling target rather than a strict position-count guarantee.

Older zero-default profiles retain their behavior and autonomous provenance.
Changing arena variant allocation now requires a terminal boundary, including
resumable historical crossplay, so old and new allocation evidence cannot mix.
Validation included 148 initial targeted tests (one CUDA-only skip), 123 focused
checks after compatibility fixes, and 18 boundary/resume checks after the final
crossplay guard; these suites overlap. Type checking and lint passed.

The running server remains on Stage A with its existing model and configuration;
the local changes have not been deployed or activated. Final read-only confirmation
showed step 36,959, active service and zero restarts. The only H100 experiments in
this request were bounded standalone projection microbenchmarks; their mixed
results did not justify changing the production model.

## Findings from the running workload

The live immutable release is `variant-f36dc87-384x8`; the active run is
`variant-network`, Stage A. Its 384-wide, eight-group model has 17,402,775
parameters. There is no controlled evidence that this size maximizes Elo/hour.

The following measurements use 1,476 existing five-second monitor observations,
from 02:10:51 through 04:13:46 UTC. Reading these records did not interrupt training.

| Measurement | Observed value |
| --- | ---: |
| Learner steps | 34,400 → 36,607 |
| Ready replay positions | 17,617,323 → 18,742,927 |
| Learner phase: waiting for update-to-data allowance | 90.8% of observations |
| Learner phase: training | 9.1% of observations |
| Mean learner GPU utilization | 6.4% |
| Mean actor GPU 1–6 utilization | 90.7–95.1% |
| Mean GPU 7 utilization, one actor plus intermittent arena | 75.5% |
| Latest one-hour actor production | 538,774 positions; 5,248 games |
| Latest one-hour neural evaluations | 108,950,895 |
| Update-to-data target | 1.0 |

These are sampled utilization/phase measurements, not a GPU profiler trace.
They identify self-play generation as the current throughput constraint. Faster
learner steps alone will mostly increase the learner's waiting time.

All stored replay is currently standard two-stone play, as expected for Stage A.
All workers and GPU hardware checks were healthy. The latest evaluated candidate
(step 29,298) lost 39–81 to the step-19,532 champion. That result does not measure
strength against the previous lineage or performance in the other five modes.

### Teacher transfer has already left the recent window

The transfer loaded 4,806,026 teacher-labelled positions. The learner selects the
most recent one million eligible positions per ring. A read-only replay-ledger
inspection found **zero lineage-transfer positions in those four recent windows**.
The last recorded nonzero `teacher_samples` metric was at step 27,420; subsequent
training metrics through step 36,600 contain no teacher losses. Some freshly
generated history games still have model step zero, so step zero alone is not a
reliable teacher-data identifier.

The documented Stage B gate waits until step 120,000, then requires an evaluation
against the old champion. That step limit is a maximum age cutoff, not the time
when the teacher actually leaves the sampled window. At approximately 1,077
steps/hour, the remaining wait is about 77 hours if throughput remains unchanged.

**Recommendation:** deliberately revise transfer completion to use verified
teacher-window exclusion plus a passing legacy-champion evaluation. Run that
evaluation sooner. If it fails, inspect teacher retention, held-out transfer
quality and raw-versus-EMA strength; waiting for step 120,000 does not itself
restore teacher supervision. This audit does not change the live transition gate.

## Measurement must match the new objective

Use a fixed, balanced evaluation distribution over the **24 mode/ring cells**.
Within handicap cells, fix and publish the severity distribution; reverse seats
with the same openings. Give each cell weight 1/24 regardless of its game length.
Report both the balanced aggregate and individual-cell confidence intervals, with
regression floors so easy modes cannot hide deterioration elsewhere.

Use connected, fixed reference opponents at a consistent search budget. Compare
the champion's gain from a common starting point divided by total provisioned
wall hours, including evaluation, compilation, failures and pauses. At eight
GPUs, one wall hour costs eight GPU-hours. A secondary fixed-time-per-move
evaluation measures the deployed strength/speed tradeoff of different models.

The current report is missing on disk. A read-only regeneration parsed existing
evidence successfully, but its headline fell back to aggregate standard-mode Elo
and selected the latest evaluated candidate by step, including the rejected
candidate. Its ladder starts from the new random bootstrap and includes saturated
one-sided results. It is **not** a reliable balanced champion Elo/hour measurement.
The existing promotion system also promotes on standard play and uses variant
segments as regression vetoes. Adding handicap coverage does not change that
decision rule. A balanced measurement/promotion objective is separate work.

## Ranked improvements

| Priority | Treatment | Why it is promising | Evidence needed before adoption |
| --- | --- | --- | --- |
| 1 | Earlier evidence-based Stage B eligibility | Teacher data already left the active windows; a fixed 120k wait delays five requested modes | Verified selected-window provenance and passing cross-lineage evaluation |
| 2 | Raise learner reuse from UTD 1.0 to 1.5, then consider 2.0 | Uses idle learner capacity without removing self-play actors | Isolated equal-wall-time Elo comparison; held-out losses and per-cell strength; preserve cadence per fresh sample |
| 3 | Remove duplicate local-block projection work | Each node is currently projected once per neighbor, in 16 local blocks | Forward/gradient parity; compiled full-search throughput and memory on H100 |
| 4 | Broadcast/cache relational bias in homogeneous-ring inference | Repeatedly builds the same large geometry bias for every batch row | Weight-refresh invalidation, padding/mixed-ring/export parity, end-to-end actor benchmark |
| 5 | Cheaper promotion screening with a fixed deeper measurement ladder | Current terminal evaluations take about 2.2 hours and pause GPU 7's actor | Paired precision/calibration at 256 versus 1024 searches; preserve uncertainty and balanced-mode measurement |
| 6 | Reduce full-search cost or adjust its frequency | Full searches account for about 87% of nominal leaf work in the 65% fast / 35% full recipe | More useful positions/hour and better Elo/hour, not merely more games |
| 7 | Tune current-schema model depth | Larger nets improve capacity but slow every self-play search | Common teacher/data/start budget and equal provisioned time for 5/6/8 groups |

UTD 1.5 improved old-lineage training throughput from 2,601 to 4,181 steps/hour.
A later +33.1 Elo promotion was confounded by continued learning-rate annealing;
it does not establish a causal Elo gain for the new architecture. The supported
migration is prospective and scales publication intervals proportionally: at
UTD 1.5, candidate 5M→7.5M examples, regular self-play snapshot 3M→4.5M, and
warmup snapshot 1M→1.5M. This prevents a reuse experiment from simultaneously
changing publication frequency per new position.

The existing fast/full search randomization and auxiliary targets are already
valuable ideas from efficient self-play research. The next step is to tune them
for this workload, not simply add features because they helped another game.
[KataGo's self-play efficiency study](https://arxiv.org/abs/1902.10565).

## Architecture and size

The local graph blocks, global attention, symmetry handling and rule conditioning
are a reasonable design for this game family. The capacity decision remains an
empirical question. Compare **the same current rule-capable model** at these sizes:

| Width | Groups | Parameters |
| ---: | ---: | ---: |
| 384 | 5 | 10,929,399 |
| 384 | 6 | 13,087,191 |
| 384 | 8 | 17,402,775 |

The older five-group model had 10,476,983 parameters and different features and
conditioning. Comparing it directly with the current model confounds size with
those changes. The current eight-group deployment demonstrated feasibility
(ring-10 batch-512 learner peak 62.6 GiB; compiled step 0.395 seconds), not optimality.

Keep the current trained model while screening alternatives from a common
teacher/replay split and equal transfer budget. Benchmark inference first; train
only promising candidates. A hypothetical 1.6× slowdown needs enough sample
efficiency improvement to compensate; neither ratio nor benefit has been
established by a controlled comparison here. AlphaZero scaling research finds
that larger networks can improve sample efficiency and that optimal size depends
on compute budget; it does not specify the answer for this game.
[AlphaZero scaling study](https://arxiv.org/abs/2210.00849).

After depth, a parameter-matched attention/FF reallocation is a reasonable
experiment: 12 KV heads with FF multiplier 2.0 versus 3 KV heads with multiplier
2.5. Source-gated/local-heavy variants are lower-priority experiments. None has
demonstrated an Elo/hour gain in this lineage.

Training's relation-bias gradient workaround recomputes explicit FP32 attention
alongside fused attention. FlexAttention's Triton path with a trainable score
modifier could avoid this; the current Flash backend does not support backward
for captured trainable buffers. This is secondary while the learner is waiting.
[Trainable bias support](https://pytorch.org/blog/flexattention-for-inference/),
[backend limitations](https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/).

## Other issues to watch

- EMA decay 0.9999 has a half-life of about 6,931 updates, or about 6.4 hours at
  the observed pace. Compare raw and EMA weights from the same recovery checkpoint
  before changing decay. A faster EMA is not automatically stronger.
- Self-play snapshot cadence changes from 1M to 3M consumed examples after 20M
  examples (about step 39,063), increasing the age of actor weights. Track that
  change; candidate evaluations currently arrive every 5M examples, about nine hours.
- History sampling does not apply the learner's model-age filter. Later in the
  run it could spend search on samples excluded from training while those samples
  still credit the committed-data allowance. Add eligible-fresh-data telemetry
  and filter such history before it becomes material.
- Clipping 100% of recent updates is diagnostic, not proof of a bad clipping
  threshold. Prior clip-2/clip-5 and EMA screens did not demonstrate a passing
  strength improvement. Inspect update/weight norms and strength before tuning.
- GPU memory reservation is not compute utilization. Co-locating another process
  on the learner GPU requires a measured peak-memory budget; it currently reserves
  about 78 GiB, so spare compute does not imply safe spare memory.

## Projection experiment: correctness passed, performance inconclusive

The projection-before-gather prototype passed 38 local checks, including
forward/input/parameter-gradient parity, conditioning, padding, compiled BF16
training/checkpoint restoration and ONNX parity. Bounded H100 batch-128
microbenchmarks then compared both implementations while the existing GPU-7
actor continued running. Timings are subject to that contention.

| Ring | Eager old/new latency ratio | Compiled old/new latency ratio |
| ---: | ---: | ---: |
| 4 | 1.20 | 1.10 |
| 6 | 0.79 | 1.19 |
| 8 | 1.42 | 1.01 |
| 10 | 1.25 | 0.71 |

A ratio above one favors the prototype. Eager outputs matched exactly for the
tested inputs; compiled maximum absolute output difference was below 0.0006.
These short local-block timings do not establish full-model or self-play speed.
In particular, the compiled ring-10 result regressed, so the default model change
was withdrawn. The reusable benchmark and raw measurements are retained in
`scripts/benchmark_local_projection.py` and `docs/local-projection-h100-20260905.json`.
No production model, profile or running process was replaced by the experiment.

## Execution order

1. Finish and test six-mode coverage and the balanced Stage B sampling profile.
2. Complete parity checks and bounded performance measurements for the local-block
   projection optimization; retain the previous release for rollback.
3. Establish the balanced strength measurement, and evaluate transfer completion.
4. Screen UTD 1.5 against a fixed control, then actor inference/search efficiency.
5. Run a common-start model-size pilot only after the measurement is trustworthy.

Use one-seed screens to reject weak treatments cheaply and confirm only the
selected treatment against control on additional seeds. Charge all experiment
overhead to the wall-time objective. An inconclusive small pilot is not proof
that treatments are equivalent.

## CPU follow-up: measured spare capacity

At 05:01 UTC, a 15-second sample averaged **9.44% CPU busy / 90.56% idle**.
The host exposes two Xeon Platinum 8480+ sockets: 104 physical cores and 208
hardware threads. Individual actors consumed about 1.2–1.8 logical CPU equivalents
despite eight configured native/numerical worker threads. About 1.6 TiB of host RAM
was available, mostly reclaimable filesystem cache. CPU affinity already matches
the GPU's NUMA node; permitting both memory nodes does not prove remote-memory
traffic is a bottleneck.

Native game logic, feature generation and tree search already release the Python
interpreter and use Rayon. Each actor nevertheless executes a synchronous cycle:
CPU select → GPU neural evaluation → CPU backup. There is at most one outstanding
neural request per tree. Two independent lanes per main actor GPU provide some
overlap. More CPU threads alone will not remove the neural dependency.

A bounded CPU-only test loaded the actual champion checkpoint (step 19,532) and
ran eager BF16 forward passes. The best tested batch for each case gave:

| CPU threads | Ring 4 neural rows/s | Ring 10 neural rows/s |
| ---: | ---: | ---: |
| 8 | 140 (batch 32) | 27 (batch 8) |
| 16 | 266 (batch 32) | 48 (batch 8) |

These use synthetic input tensors, exclude search/encoding, and are not compiled
or quantized. The live GPU fleet produces roughly 30,000 neural rows/s across its
changing ring mixture, so this is a scale comparison, not an equal-work benchmark.
CPU actors for small boards could provide supplementary data; their contribution
must be measured end to end and must preserve the balanced objective. Moving
individual layers between CPU and GPU is a less attractive first experiment
because it adds transfers and synchronization to each request.
Raw measurements: `docs/cpu-inference-20260905.json`.

Best CPU-oriented candidates, in order:

1. **Measure cross-move/game neural-request duplication, then build a bounded RAM
   cache if hits justify it.** Per-tree transpositions already exist, but every move
   builds a fresh search and there is no shared neural-result cache. Key by immutable
   model identity and the complete network input, including rule mode, history,
   handicap/pie and PDA; store raw outputs. D5 canonicalization can follow only with
   correct policy remapping and parity checks. This spends CPU/RAM to avoid GPU work.
2. **Benchmark shared inference batching with independent CPU game cohorts.** Group
   compatible model/ring requests, reuse pinned transfer buffers and overlap CPU
   preparation with GPU execution. One observed batch averaged 121 rows out of128,
   so filling its tail alone offers limited room; larger efficient batches and lower
   synchronization overhead must supply most of the gain. Measure128/256 first.
3. **Benchmark CPU-only small-board actors with explicit BF16.** The default CPU
   auto-precision path is FP32. Reserve cores and measure marginal retained
   positions/hour while verifying GPU production is not slowed.
4. **Separate native-search and BLAS/PyTorch thread budgets.** They currently all
   inherit the same value8. A small Rayon4/8/12 × OMP/MKL1/2 sweep may reduce overhead;
   more thread activity is not itself a win. Keep NUMA-local CPU and memory placement.
5. **Consider bounded exact endgame solving later.** CPU proofs may avoid neural
   leaves or improve labels, but transpositions and clinch finalization already
   remove easy cases. This is an algorithm experiment requiring strength validation.

GPU utilization near95% measures time with a kernel active, not95% of theoretical
compute throughput, so efficient batching may still help a busy GPU.
[NVIDIA definition](https://docs.nvidia.com/deploy/nvidia-smi/index.html).
Shared dynamic batching and NUMA-local CPU execution are established optimization
mechanisms; their benefit here remains to be measured.
[NVIDIA batching guidance](https://docs.nvidia.com/deeplearning/triton-inference-server/archives/triton-inference-server-2670/user-guide/docs/tutorials/Conceptual_Guide/Part_2-improving_resource_utilization/README.html),
[PyTorch CPU tuning](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html).

Local evidence: `variant-capable-network-plan.md`, `model-improvement-roadmap.md`
(UTD, cadence, gate and clipping experiment records), `startrain/model.py`,
`startrain/actor.py`, `startrain/learner.py`, `startrain/replay_store.py`, and
`scripts/strength_efficiency_report.py`. Live evidence: active profile, replay
ledger opened in read-only mode, learner metrics and existing monitor telemetry.
