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

The controlled deployment stop exposed a shutdown ordering failure: data-loader
cleanup ran before the final recovery checkpoint could be secured. The last
learner heartbeat reported step **114,179**, while the latest durable recovery
checkpoint is **113,309**. Restarting from that checkpoint discards **870 learner
updates**. Generated replay remains intact; this is lost optimizer progress,
not lost self-play data.

The corrective change saves the recovery checkpoint before loader cleanup and
retries an interrupted checkpoint once after cleanup while propagating errors.
Failed or partially completed training cannot replace the valid recovery state.
Tests restore exact model, optimizer, scheduler, step, and consumed-example state
after injected teardown failures. This incident must not be reported as a
lossless graceful stop.

## Deployment and validation

The immutable runtime is
`/home/ubuntu/edgeconnect-releases/variant-selfplay-efficiency-20260907`, pinned to
source commit `585a54a821ff7cbeba553af154b4218585a5b7a5`. All 427 source files
were verified against their recorded checksums. The native artifact SHA-256 is
`8ac6163f61a8299d4fdbc1112746f2509dd48f341bb89450ba07e7bd9a19901a`.

The only profile change enables `preserve_broadcast_topology`; the applied
migration retains the existing UTD segment and evaluation contract. All 75
captured learner, run, status, and arena control files remained byte-identical
through migration. The new profile SHA-256 is
`65be57c66dedd2648879f95202a006cd69efb7b3543e411253d7b04228abd292`.

Validation passed: **1,489 tests, five hardware-dependent skips**, Ruff formatting
and lint, and Pyright. Target-host checks included 246 tests with one skip,
63 learner pipeline/durability tests, and the CUDA pinned-transfer test. A direct
CUDA parity check covered all 24 board/mode combinations and 6,144 positions
using candidate 112,309's real EMA model in BF16. Both flag values produced exact
cache keys, CUDA input bytes/shapes/strides/dtypes, and detailed predictions.
This used the eager model to isolate layout semantics from asynchronous batching.

The service became active at **21:26:28 UTC**, resumed verified recovery step
**113,309**, and reached **114,233 at 21:34:30 UTC**, beyond the old stopped step.
This repeats training work; it does not restore the exact discarded optimizer
trajectory. Runtime metrics confirm 17,402,775 parameters, 85/5/5/5 board weights,
the six-mode quotas, and UTD 1.5. Monitoring and all report/backup timers are
active. Fresh local and disaster-recovery backups completed successfully; the
temporary deployment recovery timer was removed from the active schedule.

One existing startup cost became visible during verification: each actor cohort
serially verifies all ready replay shard checksums under the reconciliation
lock. Stack inspection confirmed workers were progressing through these checks,
not blocked in GPU compilation. These integrity checks were left intact.
By 21:39:58 UTC, the CPU actor completed 16 games and persisted 619 new positions
after restart; a full largest-board production throughput comparison is still pending.
The existing high gradient-clipping and incomplete balanced-strength warnings
remain; no Elo/hour gain is claimed from this rollout.

At **21:39:45 UTC**, the learner was at **114,234**, all workers retained their
initial launch PIDs with zero restarts, and all twelve cohorts on GPUs 1–6 were
searching without failed inference requests. GPU 7 completed initialization and
was safely parked for its scheduled evaluation: both cohorts acknowledged
quiescence, inference was idle, and CUDA synchronization was complete. The same
candidate 97,660 versus champion 19,532 evaluation continued under the existing
largest-board contract. GPU 7's post-restart self-play throughput has not yet
been observed; its normal evaluation handoff remains automatic.
