# MCTS throughput release

The September 10 source-only rollout installs commit
`dced953118ac6e6e4a5c5cc88551a41e2979c00e` from
`codex/mcts-search-throughput`. It follows the validated MCTS corrections release
`f41482963e373c1f2644881158171c90ce00a6ac`.

The release activates native bookkeeping improvements: scheduler statistics are
read at phase boundaries, selection scratch storage is allocated only when needed,
and intermediate root-statistic copies are avoided. The existing search algorithm
identity and default deterministic search results are preserved.

It also installs first-visit prediction batching, bounded subtree reuse, and
adaptive self-play budget experiments. These controls remain disabled in the live
frozen profile: width 1, no retained subtree visits, and fixed full-search budgets.
The runtime retains BF16, the current search budgets, four 64-slot actor cohorts,
model architecture, optimizer, update-to-data ratio, and evaluation contract.
See [search execution controls](search-execution.md) for opt-in configuration and
the distinction between correctness checks and playing-strength evidence.

## Validation and immutable staging

The new release was built independently while the previous runtime continued
training. All 478 archived source files passed SHA-256 verification. Its rebuilt
native extension reports execution API version 1 and loads from the new release's
own environment. The extension SHA-256 is
`bf8fa5b0a329771c34c9988d1980229d90adcc9fdaa00c0fc96bc3c802aa5bfa`.
PyTorch remains `2.13.0+cu130`. The release was made root-owned and read-only before
activation.

Target-host checks passed: 188 Python tests covering native sessions, self-play,
arena reuse, serving, budget comparisons, profile migration and resume, plus the
Rust search suite. The H100 probe passed all 12 board/variant cases with eager BF16
inference, widths 1 and 4, request-row limits, fresh/inherited visit accounting and
context invalidation. It observed BF16 linear outputs, allocated at most 52,972,032
bytes, and reserved at most 58,720,256 bytes. This tiny untrained-model probe does
not establish production throughput or playing-strength gains.

Local validation included Python regressions with targeted reruns after resolving
test expectations, Rust tests and strict Clippy, WASM bridge checks, all 234 frontend
tests, TypeScript, ESLint, and the production web build. This deployment targets the
training server; it does not publish a new browser model or website.

## Checkpoint boundary and recovery protection

All ten workers stopped normally. The learner heartbeat and recovery checkpoint
both recorded step 164,369 and 84,156,928 consumed examples. The checkpoint SHA-256
is `756424294ea9b1ddb9275eda35882eb64d0799db8a05d41755156e6f77a13d1e`.
No learner updates were discarded at shutdown.

The cutover uses a separate disaster-recovery namespace:
`/lambda/nfs/texas-north-fs/edgeconnect-dr/manual/mcts-throughput-cutover-20260910`.
It was seeded with 12,764 immutable objects from one committed snapshot. Pinning
that document directly avoided scanning the full historical snapshot archive;
the stopped-boundary snapshot and its independent verifier still validate every
referenced object. Normal backup timers retain their original namespace.

The final snapshot independently verified 20,645,230,691 bytes across 12,224
catalog entries. Its SHA-256 is
`7c4d03d2f3332c33eadce3780905f627cb8c922e0cc6ee3728b5070faf4ac9a9`.

The migration recorded `kind: source-only`, an empty configuration diff, zero
discarded steps, and no update-to-data segment change. All 117 captured learner,
arena and run control files retained identical hashes through migration. The new
frozen profile is
`/home/ubuntu/edgeconnect-runs/variant-network/profile-mcts-throughput-20260910.yaml`;
its bytes match the prior frozen profile. The recorded configuration hash remains
`cef8e21f826199dfb00f06c292c10fb09d2e4f874743c73d2de4b8407310c78b`.

## Activation and readiness

The immutable runtime is
`/home/ubuntu/edgeconnect-releases/variant-mcts-throughput-dced953`.
It launched at 07:34:13 UTC and passed the sustained readiness gate at 07:36:43 UTC.
The final independent health check reached learner step 164,371 and confirmed that
the learner resumed from the exact stopped-boundary recovery checkpoint.

All ten workers had fresh, matching process identities and zero restarts. All
active GPU actors completed neural requests with zero failed requests and zero
inference-worker failures. GPU 7's actor was normally paused for arena evaluation.
All eight H100s passed hardware health checks. Training, monitoring and all three
backup/report timers are active; fresh replay-backup and strength-report jobs
completed successfully. Both temporary deployment recovery units are stopped.

An independent read-only audit confirmed the loaded service definitions, current
source authority, identical profile bytes, native import path and hash, read-only
ownership, and restoration of the original regular disaster-backup namespace.

Operational evidence, saved units, rollout scripts and rollback material are under
`/home/ubuntu/edgeconnect-rollouts/mcts-throughput-20260910`. The prior immutable
release is retained. Final migration and sustained readiness evidence are recorded
in that directory's `cutover` subdirectory.
