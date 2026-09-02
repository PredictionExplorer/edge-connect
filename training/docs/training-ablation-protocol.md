# Training ablation protocol

Training changes are promoted by playing strength per unit of compute, not by
training loss alone. Keep the shipped configuration as the control until the
target H100 host produces reproducible evidence.

## Control

The committed 4-H100 and 8-H100 profiles use:

- champion-only self-play;
- policy and soft-policy targets only on full-search plies;
- a fixed 16-action Gumbel candidate cap;
- ring-scaled simulation counts;
- Muon plus AdamW; and
- the calibrated pair-level e-process for deployment promotion.

## Supported treatments

The following switches are deliberately first-class and recorded in metrics:

- `orchestration.model_refresh.selfplay_source`: `champion`, `candidate`,
  `candidate_champion_mix`, or the self-generated
  `candidate_champion_history_mix`;
- `orchestration.model_refresh.candidate_probability`: candidate share for the
  seeded mixture;
- `orchestration.model_refresh.history_probability` and `history_pool_size`:
  share and bounded log-spaced pool of immutable checkpoints from the same run;
- `selfplay.record_fast_policy_targets`: retain completed-Q policy targets from
  reduced searches;
- `selfplay.fast_policy_weight`: confidence weight applied only to policy and
  soft-policy losses for retained fast-search targets;
- `selfplay.policy_surprise_weight` and `policy_surprise_max_weight`: bounded
  replay weighting from the KL divergence between the network root prior and
  completed-Q target;
- `selfplay.max_considered_ring_exponent`: scale the candidate set with board
  radius; and
- `selfplay.max_considered_cap`: bound the scaled candidate set;
- `learner.use_ring_mixture_curriculum`: make stratified learner replay follow the
  actor small-to-large unlock schedule; and
- actor `actor_batch_size` plus `orchestration.actor_games_per_batch`, changed
  together so the requested cohort can actually fill the larger GPU batch.
- actor `actor_lanes`, evaluated at fixed total leaf work before enabling more
  than one process on a GPU;
- `orchestration.allow_colocated_workers` with one explicitly configured actor
  on the learner GPU, evaluated only from a fresh root with per-process CUDA
  memory, restart, learner-throughput, and fleet-throughput evidence;
- `learner.target_updates_per_new_sample` and
  `learner.candidate_interval_examples`, which make replay ratio and candidate
  cadence explicit instead of accidental consequences of throughput.
- `learner.selfplay_snapshot_interval_examples` plus its warmup horizon and
  interval, which refresh actor models frequently without enqueueing every
  snapshot for promotion.
- `data.shards_per_batch`, which mixes positions from several same-ring shards
  while retaining homogeneous tensor shapes.
- `arena.continuation_pairs_per_ring`, which increases GPU occupancy only after
  the unchanged minimum anytime-valid promotion look.
- `arena.simulations` and `arena.max_pairs_per_ring`, the promotion-gate search
  budget and evidence cap. A cheaper gate must be paired with
  `orchestration.historical_evaluation.measure_direct_predecessor` at the
  historical budget so the champion-frontier ladder stays on one Elo scale.
- `arena.promotion_pair_ratios` plus weighted initial, continuation, and maximum
  block counts, which opt into a pre-registered macro-block objective. The
  1/1/1/7 setting weights rings 4/6/8/10 as 10/10/10/70.
- `arena.required_regression_rings`, which is `null` for the legacy all-ring
  guard contract. Setting it to `[]` removes every blocking per-ring floor while
  retaining per-ring confidence sequences as diagnostics.
- `orchestration.training_objective`, which remains `generalist` unless an
  explicitly frozen profile selects `ring10_only`. The latter requires actor
  and learner weights `[0, 0, 0, 1]`, a single-ring arena on ring 10, and no
  smaller-ring regression guards.
- `orchestration.plateau.action: reduce_lr_keep_weights`, which clears stale
  optimizer moments and lowers rates without discarding the learner branch.
- `orchestration.plateau.minimum_learning_rate_scale` and
  `restore_scale_on_promotion`, which bound plateau recovery to one absolute,
  floored multiplier of the profile's reference rates and return to the
  reference after the next promotion. Reductions are triggered only by
  conclusive rejections; budget exhaustion (`reject_max_pairs`) releases the
  replay-lag cap without touching rates.

Candidate/champion/history mixing keeps pointer roles and run identities strict.
Models are refreshed only between complete game batches, so no game contains
weights from two checkpoints.

## Frozen ring-10 optimizer calibration

Optimizer/clipping calibration is deliberately separate from live Elo training.
Generate only the complete `ring10-optimizer-calibration` suite. Its explicit
arms are:

- `ring10-optimizer-runtime-effective-control` (clip norm 1);
- `ring10-optimizer-clip-norm-2`;
- `ring10-optimizer-clip-norm-5`; and
- the follow-on `ring10-optimizer-0.5x-effective-lr`.

All four retain runtime Muon+AdamW routing; AdamW-only profiles are rejected.
The follow-on halves both configured runtime learning-rate groups. The older
`lr-quarter` treatment remains accepted for historical plans and still has its
historical behavior (the same 0.5 multiplier), but new calibration evidence
uses the unambiguous `0.5x effective-LR` label.

The base profile must be the stopped runtime's exact frozen profile. The
terminal-boundary controller additionally reads the hash-pinned recovery
checkpoint and requires each optimizer group's `initial_lr` to equal the
scheduler `base_lrs`. Those recovered Muon and AdamW rates replace the YAML
defaults before profiles are generated, so runtime plateau reductions cannot be
silently lost.

Run each arm with `scripts/run_frozen_replay_optimizer_calibration.py` against
one content-addressed champion publication and an explicit ready-shard cutoff.
The runner opens the replay manifest in SQLite read-only/query-only mode,
hashes the logical cutoff, verifies selected shard hashes, and makes stable,
disjoint train/holdout partitions. It initializes the raw model and a fresh EMA
from champion EMA weights, creates empty optimizer/scheduler state, and writes
only under the requested output directory. An arm declares at most two
H100-hours; `--dry-run` validates and prints the frozen contract without
creating output. A paused invocation can resume only when its source,
partition, config, and run-contract hashes still match.

Compare the complete suite with
`scripts/compare_frozen_replay_optimizer_calibration.py`. A treatment must have
finite training/evaluation, strict optimizer and reference parity, at least
90% of control learner throughput, and a strictly positive one-sided paired
bootstrap lower bound on held-out policy/value-composite loss reduction.
Bonferroni allocation preserves the configured familywise confidence across
all non-control arms.
Gradient-clipping reduction is reported only as a diagnostic and never passes
an arm by itself. No passing arm, invalid evidence, or a top-score tie retains
the runtime-effective control. These artifacts are diagnostic calibration
evidence and do not authorize production promotion.

`scripts/run_frozen_replay_optimizer_calibration_queue.py` owns resumable arm
state and runs deterministic sequential waves until shared-replay concurrency
has separately passed a throughput test. A unique gate-passing treatment
produces a derived Elo screen plan containing only runtime control and that
treatment. No winner or a tie produces no screen plan and returns to the
protected runtime control.

The autonomous profile adds a stronger provenance contract: every treatment
starts with random weights, empty replay, a new run identity, and no external
positions. Its fixed Elo ladder may evaluate historical checkpoints, but those
games never enter replay.

## Experiment design

1. Change one treatment at a time unless the experiment is explicitly
   factorial.
2. Use at least three run seeds and unique run roots. Never reuse replay across
   incompatible treatments.
3. Match treatments by retained positions and realistic leaf evaluations, then
   also report wall-clock GPU-hours.
4. Preserve the same arena openings, roles, search budget, and model architecture
   for each comparison.
5. Evaluate every ring independently for a generalist objective. A
   `ring10_only` experiment evaluates only ring 10 and must not imply preserved
   strength on smaller boards.
6. Keep negative and inconclusive results. Do not repeatedly tune on one arena
   seed.
7. Use successive halving for expensive scratch treatments: equal-leaf pilots
   first, then at least three seeds for any treatment promoted to a long run.

Runtime arena pair-count overrides are permitted only in an isolated
fixed-manifest throughput benchmark. They are not deployable profile settings,
must retain the frozen statistical/search contract in their evidence, and must
not claim outcome equivalence when batch-derived search seeds differ.

The weighted-generalist 55/65/70 matrix is an explicit exception to per-ring
non-inferiority gating. Its arms share one weighted promotion objective and
must not be compared in the same selector as legacy guarded arms. Selecting a
weighted winner accepts that rings 4, 6, or 8 may regress without blocking
promotion; diagnostic reporting is not a guarantee.

The `ring10_only` objective is a separate, stronger exception. It generates
training replay only on ring 10 and spends all promotion games on ring 10.
Rings 4, 6, and 8 remain valid inference inputs because the shared game, feature,
and model schemas are unchanged, but their playing strength is intentionally
outside the acceptance contract. Ring-10-only, weighted-generalist, and guarded
generalist evidence must never be mixed in one selector.

## Required metrics

Each report must retain:

- leaf evaluations/second/H100 and games/hour;
- learner examples/second and replay wait time;
- policy-supervision rate and full/fast search mix;
- candidate/champion role share and model lag;
- game length, search entropy, and ring distribution;
- paired aggregate and per-ring Elo intervals;
- weighted macro-block progress, weighted aggregate Elo, and its anytime-valid
  interval whenever a weighted promotion objective is configured;
- autonomous checkpoint-ladder Elo slope per billion leaf evaluations and
  provisioned GPU-hour;
- peak memory, replay I/O, restarts, and failed/quarantined shards; and
- final strength divided by GPU-hours and leaf evaluations.

Fleet throughput must come from completed counter deltas over explicit wall
intervals, merged by physical GPU when lanes overlap. Never sum stale
latest-batch gauges. The primary Elo-efficiency denominator includes every
provisioned GPU-hour, including arena pauses, cooldowns, restarts, and idle
capacity; active-device time is diagnostic only.

Generate `scripts/strength_efficiency_report.py` for every treatment and control.
Count every provisioned GPU-hour, including learner stalls and arena pause intervals,
rather than normalizing away idle hardware.

An optimization is accepted only when it preserves correctness gates and either
improves the lower confidence bound on strength per compute or materially
improves throughput without a detectable strength regression.

## Later research

Regret-guided restarts, calibrated resignation, FP8, and multi-leaf search
remain experimental. Policy-surprise weighting is implemented, but self-play
forks and regret buffers still require a versioned contract and deterministic
CPU parity tests before an H100 ablation.

