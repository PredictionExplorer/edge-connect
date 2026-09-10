# Further MCTS optimization

This pass removes redundant search bookkeeping while retaining the current
Gumbel algorithm, budgets, arithmetic and tie handling. It is isolated on
`codex/mcts-search-throughput`; it has not been deployed to the training server.

## Implemented changes

- Root candidates within a sequential-halving phase are already scheduled.
  Native runners now request fresh root Q/visit arrays only when elimination
  needs them. A 640-simulation, 53-candidate search requests intermediate root
  statistics six times instead of 640 times. Final statistics remain unchanged.
- Interior selection uses a reusable FP64 scratch array instead of four
  temporary vectors. It preserves the original FP32 rounding before FP64
  exponentiation and normalization. Scratch is allocated only on first use;
  shallow forced-root searches allocate none.
- Final Gumbel selection computes its maximum-visit scale once, preserving the
  existing treatment of all supplied visits.
- The browser uses the same scheduling shortcut when available and reads its
  invariant root action array once. Correct older WASM builds retain the checked
  fallback. A separate implementation URL revision refreshes the binary without
  changing the search behavior identity.

## Measurements

The release-build native benchmark on an Apple M4 Max measured **1.114× throughput**,
or **10.2% less CPU time**, using the ratio of summed per-case medians. Eight
alternating baseline/candidate runs produced twelve samples per arm for each of
90 cases. All four block comparisons favored the candidate; 89 of 90 individual
case medians improved. The remaining single-simulation case differed by less
than 1%, at microsecond scale.

The fixture covers all four boards and six variants, early/middle/four-empty
positions, balanced/concentrated synthetic policies, production-scaled fast/full
budgets, online/arena budgets, and irregular small budgets. Each case has six
roots. This is a CPU search benchmark with a deliberately cheap evaluator. It
does not establish an H100 self-play speedup, stronger play, or improved Elo/hour.
The fixture mixture is not production-weighted. The benchmark uses the serial
native batch runner; production Python actors also have parallel root traversal.
On the H100 pipeline, faster request preparation can also change which requests
the asynchronous broker combines. Physical batch shapes can affect floating-point
rounding, so the production comparison still needs its own throughput and trace
checks with the real model.

Full comparison traces cover **540 searches**. Requested semantic states and
legal ordering, selected actions, visits, root-statistic float bits and target
float bits were byte-identical: 223,338,033 bytes, SHA-256
`8a35c2a8f4924058e48092103d8fdbeb0edc3e24b536b9731607508c4f48dd76`.
Opaque evaluation tokens are omitted. Evaluator call/row counts also match in
every timing sample. Independent legacy tests cover 4,608 interior-selection
cases and 432 evolving scheduler cases, alongside native and WASM protocol tests.
Final validation passed the Rust workspace tests and strict Clippy, 74 Python
native/integration tests, 71 browser AI tests, TypeScript and changed-file lint.
A rebuilt WASM check also reproduced the six-boundary schedule against the
existing checked API with evolving Q values and identical final selection.

The baseline is `283f6ca3cfd8836f8e408ff8bdc2517c424e9705`. Per-case medians,
all timing samples, machine/compiler details and candidate source hashes are in
`mcts-native-search-20260910.json`.

To reproduce, copy the same `crates/star-search/examples/benchmark_search.rs`
into isolated baseline and candidate checkouts, build it with each checkout's
release profile, and run each binary with `--trace <new-file>` before comparing
the files. Then alternate binary runs using `--repeats 3` across four blocks.
Each case performs an untimed warmup. Do not run builds or other CPU-heavy tests
concurrently with the timing measurements.

## Larger experiments to prioritize next

1. Batch independent first visits to root children, initially only while every
   child is unexpanded and all backups can retain their original order. This
   could reduce interactive neural-call latency; later traversals need explicit
   protection against shared DAG paths and stale statistics.
2. Tune full-search allocation on frozen positions and then in equal-time
   training comparisons. Under the current non-PDA 65% fast / 35% full mixture,
   full searches consume about 87% of base simulations. Reducing them also changes
   the quality and weighting of training targets, so speed alone is insufficient.
3. Measure retained visits in the selected subtree before implementing subtree
   reuse. Reusing network predictions is already supported; inherited visits
   additionally change root scheduling, scaling and effective search budgets.

These experiments should measure useful completed-game throughput and playing
strength at equal time, with model identity and logical work held fixed. Existing
cross-game batching, prediction caching and CUDA graphs remain enabled as before.
