# StarTrain model-improvement roadmap

This document is the source of truth for improving ring-10 strength per
provisioned wall-clock hour. Update it only from immutable experiment evidence;
do not infer progress from loss, throughput, or promotion counts alone.

## Scientific contract

- Primary metric: pair-valid chronological champion-frontier ring-10 Elo lower
  bound per total provisioned wall hour.
- Count all eight H100s from measurement start through resource release.
- Screen one-factor treatments on seed 17.
- Confirm only control plus one selected treatment on seeds 17, 18, and 19.
- Adoption requires a strictly positive candidate-LCB minus control-UCB in
  every seed and at least 20% median point Elo/hour improvement.
- Use complete role-reversed pairs, frozen openings, chronological error
  spending, fixed budgets, immutable releases, and common anchors.
- Stop only for hardware faults, corruption, non-finite training, repeated
  fatal restarts, or preregistered anytime-valid futility.

## Current baseline

- Model: `GraphResTNet`, schema v2, width 384, five RRT groups, 12 query heads,
  3 KV heads, bottleneck ratio 0.5, FF multiplier 2.5.
- Parameters: 10,476,983.
- Recovery anchor: step 465,582,
  `sha256-e3f78b8b62e8feccc0f0eed718aafcae863ffdbb3d09439b34ebffe401ffeda4`.
- Seed-17 control: completed fixed eight-hour budget; five terminal candidates,
  zero promotions.
- Seed-17 three-lane arm: active at roadmap creation; first candidate rejected,
  champion unchanged.
- Seed-18 boundary barrier: armed. Seed 17 will finalize before seed 18 may
  acquire its queue lock.
- Durability: active-arm NFS snapshots, campaign control-plane snapshots,
  independent Mac mirrors, and verified continuity fallback are active.

## Status legend

- `[ ]` not started
- `[-]` in progress
- `[x]` completed and evidence linked
- `[!]` blocked; reason must be recorded

## Phase 0: preserve seed 17 and add clean campaign holds

- [-] Finish seed-17 three-lane arm without interruption.
- [x] Arm reversible seed-18 queue-lock barrier.
- [x] Back up and Mac-ack barrier/control-plane evidence.
- [ ] Finalize and inspect seed-17 pair-valid comparison.
- [ ] If no promotion/frontier gain, keep seeds 18/19 paused.
- [ ] If seed 17 is positive, release the barrier and resume the pinned campaign.
- [-] Implement first-class `seed_boundary_hold_path`.
- [x] Persist `paused`, completed seed, next seed, and
  `operator_resume_required`.
- [x] Trigger continuity after pause/success.
- [ ] Verify deployed pause/resume from immutable
  campaign state.

## Phase 1: training dynamics

### Instrumentation

- [-] Optimizer routing hash and parameter counts.
- [-] Per-group update/weight norms and effective learning rates.
- [-] Gradient clipping frequency and non-finite counts.
- [-] Scheduler age and segment-relative position.
- [-] Raw-versus-EMA distance and effective EMA turnover.
- [-] Replay source-role share, branch cutoff, and lag quantiles.
- [-] Clinch-conditioned policy/outcome/score/ownership/alive losses.

### One-factor treatments

- [-] Reference-validated optimizer alternative versus Muon+AdamW.
- [-] Example-normalized EMA / longer candidate cadence.
- [-] Champion-only versus 50/50 candidate/champion self-play.
- [-] Synthetic clinch auxiliaries versus exact-outcome-only clinch targets.

### Gates

- [ ] Frozen-replay calibration, no more than 2 H100-hours per variant.
- [ ] Finite training and optimizer reference parity.
- [ ] Complete replay cutoff/source-share evidence.
- [ ] Held-out policy/value calibration improvement.
- [ ] H100 throughput remains within the preregistered limit.
- [ ] Seed-17 pair-valid Elo/hour is positive before confirmation.

## Phase 2: parameter-matched attention reallocation

- [-] Add suite `ring10-attention-reallocation`.
- [ ] Control: `kv_heads=3`, `ff_multiplier=2.5`.
- [ ] Treatment: `kv_heads=12`, `ff_multiplier=2.0`.
- [x] Assert exact equality at 10,476,983 parameters.
- [-] Add strict checkpoint-shape incompatibility gate.
- [-] Add fresh scratch ring-10 roots with empty replay.
- [-] Add heterogeneous-model arena support with identical game/rules/features.
- [x] Add a locked fixed-budget scratch queue with a baseline pin and direct
  control/treatment cross-play.
- [ ] Require D5/output parity and no more than 10% evaluator-throughput loss.
- [ ] Require peak allocation no greater than 72 GiB.
- [ ] Run two-arm eight-hour seed-17 pilot.

## Phase 3: relational/local-heavy trunk

- [-] Add configurable `local_operator`.
- [-] Add configurable `local_blocks_per_group`.
- [-] Implement source-conditioned edge gating.
- [x] Parameter-match local-heavy variants within 0.1% of the Phase-2 winner.
  Current local-heavy profile is 10,476,953 parameters, exactly 30 below the
  10,476,983 reference; it is near-matched, not exactly matched.
- [-] Run `ring10-relational` control/local-heavy/gated-local screen.
- [ ] Only if local gating fails, implement D5-invariant graph-relative bias.
- [ ] Verify all-ring D5 equivariance, native parity, padding, compile, ONNX,
  distillation, HBM, and actor/arena throughput.

## Phase 4: capacity scaling

- [ ] Carry forward the winning training dynamics and relational design.
- [x] Baseline 384x5: 10,476,983 parameters.
- [x] Depth treatment 384x7: 14,614,199 parameters.
- [x] Width treatment 512x5: 18,556,727 parameters.
- [x] Preserve 32-wide heads for width 512.
- [ ] Benchmark learner batch, actor leaves/s, arena service time, and HBM.
- [ ] Match samples, search leaves, seeds, and provisioned H100-hours.
- [ ] Prefer the smallest statistically equivalent model.
- [ ] Advance only through the standard three-seed Elo/hour adoption gate.

## Experiment registry

For each experiment, add one block:

```text
ID:
Phase:
Status:
Hypothesis:
Commit:
Release:
Control:
Treatment:
Anchor/replay cutoff:
Seeds:
Budget:
System gates:
Statistical gate:
NFS snapshot:
Mac acknowledgement:
Result:
Decision:
```

## Compute ledger

Record provisioned rather than utilized GPU-hours.

```text
Experiment:
Arms:
Wall hours per arm:
Provisioned H100-hours:
Cumulative roadmap H100-hours:
```

## Decision log

### 2026-08-17 — Preserve seed 17

Decision: do not stop the active three-lane arm. Interrupting it would make
seed-17 comparison evidence incomplete. Arm a fail-closed seed-18 lock barrier
instead, then inspect the finalized seed-17 frontier before spending another
256 H100-hours.

### 2026-08-17 — Capacity is not the first intervention

Decision: diagnose training dynamics and target/replay freshness before scaling.
Repeated rejected candidates are evidence of harmful or inconclusive updates,
not evidence of a demonstrated model-capacity ceiling.
