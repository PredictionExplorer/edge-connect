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

Concurrency/thread changes require isolated evidence and a subsequent production
check. The native and storage optimizations preserve model math and evaluation
contracts. Gradient clipping, search effort, sampling proportions, and the
learner's update-to-data target are not changed to manufacture higher utilization.
