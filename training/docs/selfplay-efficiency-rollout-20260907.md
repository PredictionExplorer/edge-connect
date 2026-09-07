# Self-play efficiency rollout — September 7, 2026

The user requested measured efficiency improvements while preserving the
17,402,775-parameter model, 85% largest-board training allocation, equal weighting
of all six modes, and largest-board-only promotion. The learner was primarily
waiting for new self-play data; its update-to-data target remains 1.5.

## Changes and controls

Native feature encoding previously scanned the entire board to find its maximum
degree once per node. It now computes that invariant once per feature row, with
identical feature bytes and search results across both feature schemas, all board
sizes, and all six modes.

The opt-in `preserve_broadcast_topology` inference setting owns one copy of shared
topology instead of expanding it across every row. Compatible merges and cache
miss selection preserve that layout. Exact cache keys remain byte-identical;
mutation guards and noncanonical-input fallbacks remain in place. Fully cached
requests avoid constructing an unnecessary merged input batch.

Local paired measurements showed approximately threefold faster native
largest-board feature export, 19% lower CPU preparation time, and 60% less prepared
tensor storage. These are component measurements, not measured Elo/hour gains.
The server's separate native feature-export comparison measured **2.495×**
throughput with exact feature-byte equality. That result supports the native
invariant-caching change; it does not establish a complete self-play speedup.

## Target-host benchmark

`benchmark_actor_throughput.py` compares fixed, identical production search
prefixes. Logical game tasks remain 128 games, with the same immutable checkpoint,
task identities, seeds, rules, simulation budgets, and requested neural rows.
Only task concurrency, native thread count, and inference row capacity vary.
Incomplete prefixes and changed action/visit/replay traces cannot enter a speed
ranking. Prefix decisions are never described as completed games or persisted
training positions.

Every physical CUDA batch bucket is warmed with distinct inputs and an empty
prediction cache. Independent arm processes publish unique ready tokens. The
operator checkpoints training, releases the common barrier, and measures arms on
separate GPUs; sole ownership is verified before and after each measurement.
Timeout and interruption terminate the entire owned arm process group, including
compiler descendants. A deployment recovery timer protects against leaving the
training service stopped if the operator is interrupted.

The baseline runtime is `8425b5b`, and the initial pinned model is candidate
112,309 (`sha256-a8ea68f6b30c716492b6a2904311fd8f9c4855f35107daa98f81414172fd7541`).
Source pins, profiles, benchmark results, and rollback evidence are retained in
`/home/ubuntu/edgeconnect-rollouts/selfplay-efficiency-20260907`.

### Completed seven-arm matrix

Each arm completed 4,096 searched decisions: four fixed 128-game tasks, eight
plies each, on the largest board in standard double-stone mode. Every arm used
1,013,386 search simulations and requested 1,017,482 neural rows. These are
unfinished game prefixes: **zero completed games and zero persisted training
positions**. GPU ownership checks passed for every measured interval. Startup,
warmup, and the release barrier are excluded from the times below.

| Arm | GPU | Concurrent cohorts | Native threads | Inference row cap | Seconds | Search decisions/s | Exact trace gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Original baseline | 1 | 2 | 4 | 256 | 261.62 | 15.66 | Reference only |
| Optimized encoding and broadcast storage | 2 | 2 | 4 | 256 | 252.80 | 16.20 | Rejected |
| Four cohorts | 3 | 4 | 4 | 256 | 241.72 | 16.95 | Rejected |
| Eight native threads | 4 | 4 | 8 | 256 | 246.98 | 16.58 | Rejected |
| Larger inference batches | 5 | 4 | 8 | 512 | 250.43 | 16.36 | Rejected |
| Larger batches and 16 GiB / 400,000-entry total cache | 6 | 4 | 8 | 512 | 258.41 | 15.85 | Rejected |
| Original baseline replicate | 7 | 2 | 4 | 256 | 267.86 | 15.29 | Rejected |

The exact gate was not relaxed after seeing these results. **No tuning was
adopted from this matrix.** Only the reference's comparison with itself passes;
that self-comparison is not evidence of reproducibility or a performance gain.

Even the unchanged baseline replicate differs from the reference: its first
task's action/visit fingerprint differs, its other three action/visit
fingerprints match, and all four replay-prefix fingerprints differ. The
optimized two-cohort arm has the same pattern of matching and differing
fingerprints, but the available hashes cannot establish identical discrepancies
or their cause. Every alternative fails the exact task-trace comparison despite
equal checkpoint identity, search budgets, requested rows, and verified isolated
GPU ownership. The apparent timing differences remain exploratory observations.

The timed inference counters are retained separately from warmup. The original
baseline recorded 149,282 cache hits, 1,004,799 physical neural rows, and 143,454
padding rows. The larger-cache arm recorded 179,286 hits, 1,021,025 physical rows,
and 188,175 padding rows. Its extra cache hits did not reduce physical neural
work or establish a throughput improvement. The four-cohort, four-thread arm
was faster in this sample than the eight-thread arm; more CPU threads therefore
cannot be assumed to improve this workload.

The independent, unmodified comparison output is
`/home/ubuntu/edgeconnect-rollouts/selfplay-efficiency-20260907/matrix-comparison-independent.json`.
Individual reports include actual precision, TF32 settings, CPU affinity,
thread counts, effective per-model cache limits, timed inference deltas, and
before/after GPU ownership evidence.

These prefixes cover early largest-board standard-mode play. They do not measure
complete-game production, later-game search, the six-mode mixture, learner
throughput, or Elo/hour. The CPU component gains do not imply that spare CPU
capacity can eliminate the learner's wait for new self-play data. Gradient
clipping, search effort, sampling proportions, and the update-to-data target
remain unchanged.

## Shutdown incident and recovery

The checkpointed deployment stop exposed a shutdown ordering failure: data-loader
cleanup ran before the final recovery checkpoint could be secured. The last
learner heartbeat reported step **114,179**, while the latest durable recovery
checkpoint is **113,309**. Restarting from that checkpoint discards **870 learner
updates**. Generated replay remains intact; this is lost optimizer progress,
not lost self-play data.

The corrective change saves the recovery checkpoint before loader cleanup and
retries an interrupted checkpoint once after cleanup while propagating errors. The planned restart is from
step **113,309**, preserving its model and optimizer state and the existing
replay. This incident must not be reported as a lossless graceful stop.

Final service, resume-step, worker-health, and backup verification will be added
after the restart.
