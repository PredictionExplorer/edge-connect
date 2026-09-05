# Actor workload efficiency

All options below preserve the network's parameters and start disabled or at
their old defaults. Measure retained samples per provisioned wall hour and the
fixed-budget champion frontier before adopting throughput treatments.

## Independent CPU cohorts with one GPU inference owner

Set `orchestration.model_refresh.inference.shared_batching: true`, one
`actor_lanes` process per GPU, and `actor_cohorts: 2` on that GPU. Each cohort owns
its native game state, random stream, replay connection, generation, and metrics
file. CPU feature preparation overlaps the single inference worker. The bounded
broker combines requests only for the same immutable model and compatible ring
and inference semantics. A model remains pinned until its cohort's requests finish;
weights are never refreshed beneath a game. Variant score utility stays local to
the cohort, and PDA remains part of each request's features.

The registry holds at most `actor_cohorts + 2` cached models. Configured cache entry
and byte limits are budgets for the whole worker: the registry divides each by
its maximum model count. Every model's cache includes full input keys in its
charged bytes. Zero per-model capacity disables caching. Cohort failures stop
their siblings, all submitted inference finishes before adapter cleanup, and the
existing coordinator grace/kill limits remain the process-level recovery boundary.

Per-cohort JSONL metrics include model identity, game/replay accounting, inference
metrics, solver work, and samples still inside replay eligibility when committed.
The parent worker heartbeat reports individual cohort progress and broker work.
GPU memory peaks for shared cohorts describe the process, not independent lanes.

## CPU reservations and thread budgets

GPU worker `native_threads` controls the process's shared Rayon pool;
`blas_threads` controls PyTorch/OpenMP/MKL/OpenBLAS. Omitted values retain
`cpu_threads`. Their shared pool is not multiplied by the number of cohorts.
NUMA placement continues to use `cpu_affinity` on worker processes.

`orchestration.cpu_actors` adds optional CPU-only actors with explicit BF16,
unique actor IDs, rings 4/6, and CPU affinities disjoint from every GPU worker,
promotion worker, and other CPU actor. These supplemental actors have no assigned
GPU and mask CUDA. Their replay still passes the normal rule, variant, and lineage
contracts. Every CPU batch contains exactly one cohort of `actor_batch_size` games,
so it flushes metrics/replay and can refresh its model after that cohort instead
of retaining the GPU fleet's larger `actor_games_per_batch` workload. GPU actors
retain the configured global game count. Equal learner ring quotas continue to define the training objective;
extra ring-4 generation does not change the learner's four-ring allocation.

Use the Linux target-host diagnostic sweep before reserving production cores:

```bash
python scripts/benchmark_cpu_actor.py \
  --config /absolute/frozen-profile.yaml \
  --checkpoint /absolute/run/learner/champion.json --cpu-affinity 192-207 \
  --native-threads 4 8 --blas-threads 2 4 --batch-sizes 8 16 --rings 4
```

The command prints a matrix by default. Add `--execute --output <new-json-path>`
to measure it, with a separate process and timeout per case. Execution requires a
model manifest/pointer supplied through `--checkpoint`, unless random weights are
explicitly requested with `--random-initialization`. The parent resolves the pointer
once to an immutable manifest and pins its digest across all cases; each child
verifies EMA weights, architecture, game, checkpoint digest/size, lineage and step.
Results record that identity and the played mode. Diagnostic replay lives in a
parent-owned temporary directory that is removed even after a timed-out child.
The measurements establish throughput, not playing strength. No production service
or replay is changed. Compare an untouched fleet control and a reserved-core
treatment over the same wall interval so CPU gains cannot hide GPU actor losses.

## Freshness and exact endgames

Historical checkpoints are filtered before sampling to exclude future steps and
models at or beyond the learner's replay-lag boundary. Historical selection also
keeps the existing run identity and resume-cutover constraints. Commit-time
metrics distinguish `eligible_samples_at_commit` and
`ineligible_samples_at_commit`; later eviction can still remove eligible samples,
so these counts are an eligibility observation, not proof of training consumption.

`selfplay.exact_endgame_max_empty` enables exhaustive CPU solving for small tails
(0 disables, maximum 8). `exact_endgame_max_nodes` bounds every attempted tree
(default 100,000, maximum 1,000,000). A fully explored tree returns a real terminal
board from optimal legal play; exhaustion returns no result and leaves the source
state untouched. Outcome is optimized first, then score margin and the quark
tie-break. Classic turns, consecutive double/handicap placements, swap state, and
history all use the authoritative rules engine. PDA on recorded earlier decisions
is preserved. No policy target is invented for solved moves.

Unlike the existing loser-fill clinch proof, exact tails use the actual principal
variation's terminal score/ownership/alive targets. Replay explicitly records
`final=exact-endgame`, and metrics report attempts, solved/exhausted trees, nodes,
and seconds. Start any benchmark at 3–4 empty nodes; larger exhaustive trees can
cost more CPU than the neural evaluations they avoid.
