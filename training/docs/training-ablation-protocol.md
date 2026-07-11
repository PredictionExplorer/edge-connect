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

- `orchestration.model_refresh.selfplay_source`: `champion`, `candidate`, or
  `candidate_champion_mix`;
- `orchestration.model_refresh.candidate_probability`: candidate share for the
  seeded mixture;
- `selfplay.record_fast_policy_targets`: retain completed-Q policy targets from
  reduced searches;
- `selfplay.max_considered_ring_exponent`: scale the candidate set with board
  radius; and
- `selfplay.max_considered_cap`: bound the scaled candidate set;
- `learner.use_ring_mixture_curriculum`: make stratified learner replay follow the
  actor small-to-large unlock schedule; and
- actor `actor_batch_size` plus `orchestration.actor_games_per_batch`, changed
  together so the requested cohort can actually fill the larger GPU batch.

Candidate/champion mixing keeps pointer roles and run identities strict. Models
are refreshed only between complete game batches, so no game contains weights
from two checkpoints.

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

## Required metrics

Each report must retain:

- leaf evaluations/second/H100 and games/hour;
- learner examples/second and replay wait time;
- policy-supervision rate and full/fast search mix;
- candidate/champion role share and model lag;
- pass rate, game length, search entropy, and ring distribution;
- paired aggregate and per-ring Elo intervals;
- peak memory, replay I/O, restarts, and failed/quarantined shards; and
- final strength divided by GPU-hours and leaf evaluations.

An optimization is accepted only when it preserves correctness gates and either
improves the lower confidence bound on strength per compute or materially
improves throughput without a detectable strength regression.

## Later research

Regret-guided restarts, replay prioritization, calibrated resignation, FP8, and
multi-leaf search remain experimental. Introduce each behind a versioned config
field, add deterministic CPU parity tests first, and do not enable it in the
shipped profiles before an H100 ablation passes this protocol.

