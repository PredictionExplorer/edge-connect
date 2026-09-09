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
