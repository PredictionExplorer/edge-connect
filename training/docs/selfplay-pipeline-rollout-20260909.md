# Self-play pipeline rollout — September 9, 2026

The user authorized implementation and graceful deployment of compatible work
scheduling, more CPU producers, early durable game publication, rolling game
slots, CUDA Graph inference, and physical inference telemetry. The model remains
17,402,775 parameters. Training retains 85% largest-board allocation, 5% for each
smaller board, equal six-mode importance, and largest-board-only promotion.
Search budgets, inference precision, optimizer state, learning rates and UTD 1.5
remain unchanged.

## Implementation

Compatible work shares a pinned model and board across independent producers;
each producer receives its own mode. Deterministic weighted quotas cover the
joint model-role, board and mode allocation without a producer barrier. A task
has a finite game quota. Model changes or a maximum pin age stop refilling new
games, drain already started games, and retain unstarted work as a continuation.
Champion polling caches verified metadata until the manifest pointer changes.

Streaming publishes each completed game before releasing its trajectory. The
legacy seed contract retains exact trajectories. Rolling slots use an explicit
per-game seed contract whose choices do not depend on slot packing. Completed
games are durable before slots can be reused. Stopping preserves completed games;
unfinished games may still be discarded. Publication telemetry uses cumulative
counters so final task summaries do not count games twice.

CUDA Graph caches are scoped to immutable model identity and exact input layout.
Each capture must pass bitwise output parity before admission. Captured storage
has explicit ownership, parameter mutation invalidates entries, and retained
memory is charged from allocator ownership. Known unsupported captures fall back
to ordinary inference. Arbitrary model failures propagate. A device lock covers
inference, model loading and eviction. Entry limits apply per adapter; byte
budgets are divided across the bounded model registry.

Monitoring distinguishes actual neural calls and rows from broker dispatches,
and reports cache hits, padding, graph replay, validation failures and retained
memory. These are current-process cumulative counters, not interval rates or a
measurement of Elo/hour.

## Validation and deployment protocol

The full local suite passed 1,771 tests with ten hardware-dependent skips. Ruff
and Pyright passed. Target GPU correctness tests passed for graph buffer reuse
and a real small GraphResTNet across all boards and modes. Every production-sized
capture also validates its own outputs before use. Seed/trajectory tests cover
all modes, handicap severities, pie decisions, concurrent tasks, durable
publication, early draining and model lifetime.

The initial profile enables streaming globally while retaining baseline
scheduling on GPUs 1–6. GPU 7 tests four producers, 64 active slots per producer,
128-game task quotas, compatible work, rolling slots, per-game seeds and CUDA
Graphs. The aggregate graph allowance is 16 GiB per actor process, divided over
its six-entry model registry; each adapter can retain up to eight graph shapes.
Normal cooperative evaluation on GPU 7 remains enabled.

Rollout evidence is retained under
`/home/ubuntu/edgeconnect-rollouts/selfplay-pipeline-20260909`. A controlled stop
must save the latest learner heartbeat step, and the profile migrator must
report zero discarded learner steps and preserve pending evaluation state.
Operational canary gates require current worker identities, durable games,
observed slot refill, graph replays, no validation failures or worker restarts,
and unchanged model/training contracts. Operational health does not establish a
throughput or Elo gain; unmatched production tasks cannot provide that claim.

The same pipeline is eligible for the other actor GPUs only after the canary
passes. The streaming-only profile is the operational fallback. Source and
profile hashes, checkpoint boundaries and deployment results are recorded below
when complete.

## Canary boundary

The immutable runtime is commit
`cb3d7ef616dcb9f785bc42628a692319ea778f68`, installed at
`/home/ubuntu/edgeconnect-releases/variant-selfplay-pipeline-20260909`.
The native artifact remains
`8ac6163f61a8299d4fdbc1112746f2509dd48f341bb89450ba07e7bd9a19901a`.
The final target-host runtime checks passed 126 tests in 27.41 seconds.

The controlled stop saved step 143,172, exactly matching the stopped learner
heartbeat: zero uncheckpointed learner updates were lost. The migration retained
all learner and arena control-file hashes. Its profile is
`profile-selfplay-pipeline-canary-20260909.yaml`, SHA-256
`699310d48694fad6055d55ba8d4e997e36e4d221891c0aeabd7b597faf0632b8`.
The service became active at 04:24:45 UTC and resumed that checkpoint.

The read-only canary reporter was committed separately as
`5ee6f714ad55250f0b93cf2d5d9abbf91714c3f0` and installed in the rollout evidence
directory. Its script SHA-256 is
`97ddc0cdd2e676c6d0372ca19f227c8ca4665d9349dfed15420c292755f6aff3`.
Its fourteen additional tests cover current-process identity, cumulative
publication accounting, incomplete tasks, stale evidence, graph failures and
protected output paths. It runs against the frozen runtime; no active release
files were edited to add this operational tool.

## First canary rejected

The initial canary did not pass. In a clean 120-second active interval on the
largest board, it served 2,716 useful neural rows/second with 31.8% padding;
fully active but unmatched baseline actors served 3,626–4,832 useful rows/second
with 0.8–8.6% padding. The canary repeatedly evicted graphs because of its byte
budget, despite having only three or four resident entries. These observations
justify testing more active slots and a larger graph allowance; they do not
establish an end-to-end throughput or Elo gain.

At 04:49:59 UTC, GPU 7 reported an Xid 31 virtual-memory write fault in actor
PID 1459019. The error surfaced asynchronously at a pinned-transfer event query,
which does not identify the originating kernel. All four canary cohorts failed;
the service automatically restarted at 04:51:19 UTC. The learner recovered at
step 143,177, but unfinished self-play work was lost. No GPU 7 games had been
durably published and no slot refill had been observed, so the wider rollout
was withheld and the streaming-only fallback was selected.

The failure after 33 graph captures is consistent with the stream-pool reuse
and cuBLAS workspace-lifetime defect described in
[PyTorch issue 193402](https://github.com/pytorch/pytorch/issues/193402). Independent
live graphs must hold distinct capture streams until reset finishes; allocating
a fresh pooled stream for each capture does not guarantee that lifetime.
The correction and stress-validation results are recorded below when complete.

## Corrective implementation

Capture streams are now explicitly created through NVIDIA's CUDA runtime and
wrapped as external PyTorch streams. They do not belong to PyTorch's reusable
stream pool. Every live graph holds an exclusive lease across all adapters in
the process. Reset and graph-wrapper destruction precede stream reuse or
destruction. Failed cleanup retains ownership. Idle handles are bounded to eight
per device; live handles remain bounded by graph-cache limits. Missing runtime
bindings produce an explicit capture fallback.

Graph-enabled inference adds 96- and 192-row plans between the existing powers
of two. This reduces padding for independently finishing search requests while
retaining a bounded set of shapes. CPU inference and graph-disabled CUDA plans
are unchanged. The next candidate uses four 128-slot producers and 256-game
quotas; the final graph byte allowance is selected from the production-model
memory probe rather than an unmeasured estimate.

Replay startup can now be cancelled while waiting for its reconciliation lock
or between hash chunks. Cancelled construction closes the store, and any
already-discovered quarantine repairs remain consistent. A successful open
still performs all integrity checks. Cohort startup receives a pure shutdown
predicate, separate from its pause-aware self-play callback, so an evaluation
pause cannot hold the global reconciliation lock.

The streaming fallback became active at 04:58:30 UTC under profile
`profile-selfplay-streaming-20260909.yaml`, SHA-256
`f600f903c3072c7961fb73c8b62a3a3e32e28a8dcacea70d45f558bf6324755c`.
Its migration preserved step 143,177 exactly and all learner/arena control-file
hashes. It retains completed-game streaming and baseline scheduling/inference.
