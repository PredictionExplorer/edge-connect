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

- [x] Reference-validated optimizer alternative versus Muon+AdamW: seed-17
  screen completed with no promoted frontier gain.
- [x] Example-normalized EMA: seed-17 screen completed with no promoted
  frontier gain.
- [x] Champion-only versus 50/50 candidate/champion self-play: seed-17 screen
  completed with no promoted frontier gain.
- [x] Synthetic clinch auxiliaries versus exact-outcome-only clinch targets:
  seed-17 screen completed with no promoted frontier gain.
- [ ] Frozen-live candidate publication cadence: exact live control versus a
  five-million-learner-example cadence, with no other profile change.

### Gates

- [ ] Frozen-replay calibration, no more than 2 H100-hours per variant.
- [ ] Finite training and optimizer reference parity.
- [ ] Complete replay cutoff/source-share evidence.
- [ ] Held-out policy/value calibration improvement.
- [ ] H100 throughput remains within the preregistered limit.
- [ ] Seed-17 pair-valid Elo/hour is positive before confirmation.

### Preregistered live cadence trial

- [ ] Suite: `ring10-live-cadence`.
- [ ] Control: `ring10-live-cadence-control`, preserving every learner cadence
  field from the frozen live ring-10-only profile.
- [ ] Treatment: `ring10-live-cadence-5m`, changing only
  `learner.candidate_interval_examples` to `5000000`.
- [ ] Invariants: UTD `1.0`; identical model, optimizer, replay policy,
  topology, actor self-play, model-refresh policy, arena, champion anchor, and
  replay cutoff.
- [ ] Exclusions: do not queue the null optimizer, EMA, freshness, or clinch
  arms and do not use the canonicalizing `ring10-optimization` control.
- [ ] Screen: seed 17, fixed eight-hour and two-billion-leaf budget per arm,
  charging all eight H100s through resource release.
- [ ] Operational gate: candidate arrival/service ratio `<= 1.20`, or at least
  a 25% relative reduction from control.
- [ ] Statistical gate: eligible positive pair-valid chronological
  champion-frontier ring-10 Elo/hour evidence at screening; confirmation still
  requires positive candidate-LCB minus control-UCB in every seed and at least
  20% median point Elo/hour improvement across seeds 17, 18, and 19.

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

```text
ID: R10-LIVE-CADENCE-01
Phase: 1 — training dynamics / arena backlog
Status: preregistered; not started
Hypothesis: reducing candidate publication frequency clears arena backlog
  without reducing pair-valid champion-frontier Elo gained per wall hour
Commit: to be frozen before execution
Release: to be frozen before execution
Control: ring10-live-cadence-control
Treatment: ring10-live-cadence-5m
Anchor/replay cutoff: one common stopped live ring-10-only boundary; pending
Seeds: 17 screen; 17, 18, 19 only after both screening gates pass
Budget: 8 hours / 2B leaves per screen arm; 12 hours per confirmation arm
System gates: arrival/service <=1.20 or >=25% relative reduction versus control
Statistical gate: standard pair-valid Elo/hour screening and three-seed gate
NFS snapshot: pending
Mac acknowledgement: pending
Result: pending
Decision: pending
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

### 2026-08-22 — Isolate the live cadence transition

Decision: preregister only an exact frozen-live control and a five-million
example candidate-publication treatment. The completed optimizer, EMA,
freshness, and clinch screen had no promoted frontier gain, so those arms are
excluded. Backlog relief is necessary but insufficient: the treatment must also
pass the standard pair-valid Elo/hour gates.
