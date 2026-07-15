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
- `learner.target_updates_per_new_sample` and
  `learner.candidate_interval_examples`, which make replay ratio and candidate
  cadence explicit instead of accidental consequences of throughput.
- `learner.selfplay_snapshot_interval_examples` plus its warmup horizon and
  interval, which refresh actor models frequently without enqueueing every
  snapshot for promotion.
- `data.shards_per_batch`, which mixes positions from several same-ring shards
  while retaining homogeneous tensor shapes.
- `orchestration.plateau.action: reduce_lr_keep_weights`, which clears stale
  optimizer moments and lowers rates without discarding the learner branch.

Candidate/champion/history mixing keeps pointer roles and run identities strict.
Models are refreshed only between complete game batches, so no game contains
weights from two checkpoints.

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
5. Evaluate every ring independently in addition to the aggregate result.
6. Keep negative and inconclusive results. Do not repeatedly tune on one arena
   seed.
7. Use successive halving for expensive scratch treatments: equal-leaf pilots
   first, then at least three seeds for any treatment promoted to a long run.

## Required metrics

Each report must retain:

- leaf evaluations/second/H100 and games/hour;
- learner examples/second and replay wait time;
- policy-supervision rate and full/fast search mix;
- candidate/champion role share and model lag;
- game length, search entropy, and ring distribution;
- paired aggregate and per-ring Elo intervals;
- autonomous checkpoint-ladder Elo slope per billion leaf evaluations and
  provisioned GPU-hour;
- peak memory, replay I/O, restarts, and failed/quarantined shards; and
- final strength divided by GPU-hours and leaf evaluations.

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

