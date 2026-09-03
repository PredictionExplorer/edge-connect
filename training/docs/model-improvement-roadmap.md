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
- Durability: Lambda-attached active-arm disaster snapshots, campaign
  control-plane snapshots, and verified continuity fallback are active.
- Champion frontier as of 2026-09-02: step 742,979 at +832 connected ring-10
  Elo over the anchor; unchanged since 2026-08-30 08:32 UTC. Frontier gain per
  training hour by learning-rate regime: 7.0 (Muon 3.0e-4), 3.6 (1.3e-4), 1.7
  (6.1e-5), 1.3 (2.9e-5), 0.0 (1.4e-5). The regimes were produced by plateau
  resets compounding through the champion lineage, not by a schedule.

## Status legend

- `[ ]` not started
- `[-]` in progress
- `[x]` completed and evidence linked
- `[!]` blocked; reason must be recorded

## Phase 0: preserve seed 17 and add clean campaign holds

- [-] Finish seed-17 three-lane arm without interruption.
- [x] Arm reversible seed-18 queue-lock barrier.
- [x] Back up barrier/control-plane evidence to the attached Lambda filesystem.
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

- [x] Optimizer routing hash and parameter counts.
- [x] Per-group update/weight norms and effective learning rates.
- [x] Gradient clipping frequency, pre/post-clip norms, severity, and non-finite counts.
- [x] Scheduler age and segment-relative position.
- [x] Raw-versus-EMA distance and effective EMA turnover.
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
- [x] Frozen-live candidate publication cadence: exact live control versus a
  five-million-learner-example cadence, with no other profile change.

### Gates

- [-] Frozen-replay calibration, no more than 2 H100-hours per variant:
  runner, comparator, and terminal-boundary staging are implemented; H100
  execution is pending.
- [-] Finite training and optimizer reference parity: enforced by the staged
  comparator; execution is pending.
- [-] Complete replay cutoff/source-share evidence: hash-pinned by the staged
  runner; execution is pending.
- [-] Held-out policy/value calibration improvement: one-sided paired bootstrap
  gate implemented; execution is pending.
- [-] H100 throughput remains within the preregistered 90% control floor;
  execution is pending.
- [ ] Seed-17 pair-valid Elo/hour is positive before confirmation.

### Preregistered live cadence trial

- [x] Suite: `ring10-live-cadence`.
- [x] Control: `ring10-live-cadence-control`, preserving every learner cadence
  field from the frozen live ring-10-only profile.
- [x] Treatment: `ring10-live-cadence-5m`, changing only
  `learner.candidate_interval_examples` to `5000000`.
- [x] Invariants: UTD `1.0`; identical model, optimizer, replay policy,
  topology, actor self-play, model-refresh policy, arena, champion anchor, and
  replay cutoff.
- [x] Exclusions: do not queue the null optimizer, EMA, freshness, or clinch
  arms and do not use the canonicalizing `ring10-optimization` control.
- [x] Screen: seed 17, fixed eight-hour and two-billion-leaf budget per arm,
  charging all eight H100s through resource release.
- [x] Operational gate: treatment ratio `1.00` versus control `1.25`, clearing
  the absolute `<= 1.20` gate.
- [!] Statistical gate: both arms retained the common champion and gained zero
  pair-valid frontier Elo/hour, so confirmation was not authorized. The
  standard gate still requires positive pair-valid chronological
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
Lambda snapshot verification:
Result:
Decision:
```

```text
ID: R10-LIVE-CADENCE-01
Phase: 1 — training dynamics / arena backlog
Status: seed-17 screen complete; confirmation rejected
Hypothesis: reducing candidate publication frequency clears arena backlog
  without reducing pair-valid champion-frontier Elo gained per wall hour
Commit: 874f5b9366e5e097b8f4cfc9af8e21053dc2abca
Release: main-874f5b9-elo-opt
Control: ring10-live-cadence-control
Treatment: ring10-live-cadence-5m
Anchor/replay cutoff: common champion step 586699,
  sha256-6ea82195c4c33b298903635db19b1d61112e9ab2ca6011aefadacb66455d07de
Seeds: 17 only; seeds 18/19 remain held because the statistical gate failed
Budget: 8 hours / 2B leaves per screen arm; 12 hours per confirmation arm
System gates: arrival/service <=1.20 or >=25% relative reduction versus control
Statistical gate: standard pair-valid Elo/hour screening and three-seed gate
Lambda snapshot verification: verified distinct treatment snapshots under
  edgeconnect-dr/elo-optimization/cadence-seed17-v2
Result: treatment arrival/service 1.00 versus control 1.25, approximately 4.6%
  higher actor and learner throughput, but zero promotions and zero pair-valid
  champion-frontier Elo/hour in both arms
Decision: retain the runtime workload; do not confirm or adopt the 5M cadence
```

```text
ID: R10-OPTIMIZER-CAL-01
Phase: 1 — frozen-replay optimizer/clipping calibration
Status: v6 deployed and armed; waiting for a strictly newer quiescent terminal
  while the verified fallback continues training
Hypothesis: a less restrictive clip norm or a further 0.5x effective-LR
  reduction improves held-out policy/value calibration without sacrificing
  learner throughput, then converts to positive pair-valid Elo/hour
Commit: 44fe41dbacc30619f964c175dfd4117ccfce69c4
Release: main-44fe41d-elo-v6
Policy: optimizer-auto-lambda-v6, SHA-256
  06cd535197f80018bd527cd0acd440ae7bf513d0a89c5d628ef85fa7891f6aa3
Control: exact runtime-effective frozen source profile
Treatments: clip norm 2; clip norm 5; follow-on 0.5x effective LR
Anchor/replay cutoff: atomically selected from the next quiescent terminal
  boundary; current live roots are never edited
Seeds: frozen-replay diagnostic, then seed-17 control versus unique winner;
  seeds 17/18/19 only after positive screening
Budget: no more than 2 H100-hours per frozen arm; 8 hours / 2B leaves per
  seed-17 Elo arm
System gates: finite/reference parity, read-only replay, >=90% control
  throughput, complete backups and source release
Statistical gate: strictly positive one-sided held-out composite lower bound;
  pair-valid Elo/hour remains the advancement and adoption authority
Lambda snapshot verification: `lambda_attached`; a complete snapshot newer than
  verified source release is required before source cutover completes
Result: v4 accepted a quiescent step-719537 plateau boundary, then PyTorch
  Inductor/Triton attempted to write `/root/.triton` inside a read-only home;
  continuity released the hold and resumed the verified fallback. V5 completed
  only policy pinning/waiting and was retired before any boundary or source
  mutation when its release was superseded. V6 passed local and Linux suites,
  sandbox cache-write verification, policy verify/probe, and a side-effect-free
  waiting run; monitoring and 15-minute strength reports are active
Decision: preserve v4/v5 evidence, keep their activators retired, and retain the
  fresh v6 policy with per-arm allowlisted compile caches. Automatic fallback
  to runtime control remains mandatory on no winner, tie, invalid evidence, or
  any cutover failure. Superseded 2026-09-02: v6 completed on 2026-09-01
  00:09-01:35 UTC with no passing treatment; every arm tested an equal or lower
  effective rate, which is consistent with the rate itself being the problem
```

```text
ID: R10-LR-RECOVERY-01
Phase: 1 — training dynamics / learning-rate governance
Status: cut over 2026-09-02 04:52 UTC (36.8 minutes of downtime from the
  04:15:42 stop); 12-hour verification in progress
Hypothesis: the collapse of frontier Elo per hour is caused by plateau resets
  compounding a 0.5x learning-rate cut through every champion checkpoint
  (Muon 6.1e-4 -> 3.06e-4 -> 1.33e-4 -> 6.1e-5 -> 2.9e-5 -> 1.38e-5, five
  champion generations, 44x lower and ~70x below the profile schedule), and
  restoring the Muon 3.0e-4 regime with non-compounding recovery and
  candidate-mixed self-play restores positive Elo per hour
Commit: 73d6632c201d7c2232f9afbedc53c2bc6d19955d
Release: main-73d6632-lr-recovery (release-manifest sha256
  4c6402c7bdd4456e127cef331d58a7e34cf7951a98ae668e9c1b6245340773bc)
Workload: continuity primary `lr-recovery-742979`, run root
  /home/ubuntu/edgeconnect-recovery/lr-recovery-ring10-lr-recovery-3e-4-seed17,
  profile sha256 ab77e3b36cd04cff74357a48c107b6a938409d14c04ad79a89ff3fdacc4ff016;
  `fallback-lkg` (the stopped source) remains the verified last-known-good
  fallback and the rollback path
Control: the stopped fallback-lkg runtime (champion 742,979, Muon 1.35e-5)
Treatment: ring10-lr-recovery-3e-4 forked from the same root and warm-started
  from champion 742,979 with a fresh optimizer/scheduler (Muon 3.0e-4, AdamW
  4.5e-6, warmup 2000, min_lr_ratio 0.33), reduce_lr_keep_weights plateau
  recovery floored at 0.25 with restore-on-promotion, and
  candidate_champion_mix self-play at 0.8 with 1M-example snapshots
Anchor/replay cutoff: champion 742,979
  sha256-70a4e7ad5a8e1a41a60fd597f1603826df6c44f64e79c5649c202bfc03a8939b;
  replay hard-linked from the fallback root; stale candidates excluded by the
  warm-start resume cutover
Seeds: 17 (production continuation, not a preregistered screen)
Budget: continuous; success is judged on the live frontier
System gates: finite training, unchanged actor throughput (~365 samples/s),
  learner UTD 1.0, arena unchanged (1024 simulations, 50/50/200 pairs)
Statistical gate: at least one promotion within 12 hours and candidate point
  Elo versus champion trending above +50; abort on two consecutive conclusive
  rejections below -50 Elo or any non-finite event
Lambda snapshot verification: fallback snapshot
  1788322266613655941-ba1ed78f68e9da46cef7a0f0f24eec26ef5a6df0760d7df0d15609235909533b
  completed 04:15:28 UTC (14 seconds before the stop); a final post-stop
  snapshot was started at 04:16:33; the fork's own 14-minute snapshot timer
  publishes to edgeconnect-dr/continuity/lr-recovery-742979
Result: twelve-hour verification (16:55 UTC) — system healthy: Muon 3.0e-4 at
  multiplier 1.0 throughout, UTD 1.00, zero non-finite events, zero plateau
  events, eight candidates on the 1.55-hour cadence, actors ~370 samples/s,
  hourly losses flat (total 3.44-3.47, policy ~1.37, gradient norm ~8.3 versus
  3.37/1.33/9.4 under the collapsed rate). Frontier evidence at the unchanged
  1024-simulation gate: 746,886 +1.7 (400 games, inconclusive; warmup-era
  candidate), 754,700 +8.7 (400 games, inconclusive), 762,514 +11.6 after 300
  games (continuing). Trend positive and monotone but all evaluations remain
  inconclusive at the 400-game cap, so the twelve-hour promotion criterion was
  not testable at this gate; the abort criteria were not approached
Decision: keep the recovery rates; proceed with R10-ARENA-GATE-02 so the gate
  can conclude on effects of this size, then judge Elo per hour on the
  1024-simulation measurement ladder
Notes: the fork refused sixteen root-owned SQLite temp sidecars left in the
  source's recovery/replay-manifest by the root-run terminal-boundary
  pipelines; their ownership was corrected to ubuntu before forking. The
  completed seed-17 handoff artifact was removed from the continuity manifest
  so the fresh state does not treat it as a new failure; seeds 18/19 remain
  registered. The stale `primary-recovery` workload entry was replaced. The
  post-stop disaster snapshot of the stopped source entered an SQLite backup
  restart loop (13 TB read, 4 MB written in 68 minutes) and was stopped; the
  earlier verified snapshot remains `latest`. The same loop left temp sidecars
  after the Aug 22 stopped-boundary snapshots, so the snapshot tool's
  replay-ledger step misbehaves on stopped roots and needs a fix.
```

```text
ID: R10-ARENA-GATE-02
Phase: 1 — promotion evidence efficiency
Status: live since 2026-09-02 17:32 UTC (in-place migration at learner step
  775,955; four minutes of GPU idle time, stopped at the arena boundary right
  after candidate 762,514's terminal decision so no evaluation mixes budgets)
Hypothesis: gating at 256 simulations with a 600-pair cap concludes true +35
  to +45 Elo candidates in about 500 games (roughly 1.2 hours) instead of
  exhausting a 400-game cap, while a 100-pair measurement crossplay at 1024
  simulations against the direct predecessor keeps the champion-frontier ladder
  on its historical scale
Commit: 2ccd895 (migration chain 22df338 -> 2ccd895; the root's source
  authority had stayed at the parent's commit through the R1 release cutover,
  which the migration reason records)
Release: main-2ccd895-arena-gate (profile-arena-gate.yaml, sha256 d1be1905...)
Control: the R10-LR-RECOVERY-01 gate (1024 simulations, 50/50/200 pairs)
Treatment: arena.simulations 256, arena.max_pairs_per_ring 600,
  historical_evaluation {measure_direct_predecessor, simulations 1024,
  max_considered 32, pairs 50/100, every_promotions 2}
Anchor/replay cutoff: none; applied in place with migrate_continuous_profile.py
Seeds: 17 (production continuation)
Budget: continuous
System gates: gate evaluations conclude (inconclusive fraction well below the
  62% observed under the 400-game cap); measurement links exist for every
  promotion; arena occupancy including measurement stays below one GPU
Statistical gate: the 1024-simulation ladder (report
  autonomous_elo.search_budget.ladder) continues to extend the anchor chain;
  256-simulation results are excluded from it by construction
Lambda snapshot verification: fork snapshot timer continues unchanged
Result: first gate evaluation (candidate 774,235) concluded `reject` after
  400 games in 56 minutes of arena time (17:36-18:32 UTC; 13 minutes per
  100-game wave versus 57 at 1024 simulations), conclusive with anytime upper
  bound +35.3 Elo against the +35 alternative; point estimate -19.1 Elo
  (score 0.4725). Candidate 778,142 at -20.9 after 200 games, continuing. The
  gate now resolves in about an hour what the 400-game 1024-simulation gate
  left open in 3.8 hours. Auditing what the fast verdicts feed exposed the
  keep-weights lag bug recorded under R10-PLATEAU-LAG-03
Decision: keep the gate; the anneal it now drives is corrected in
  R10-PLATEAU-LAG-03
```

```text
ID: R10-PLATEAU-LAG-03
Phase: 1 — training dynamics / plateau policy
Status: live since 2026-09-02 19:37 UTC (runtime-only cutover at learner step
  780,872, two minutes of GPU idle, stopped right after candidate 778,142's
  conclusive rejection while the arena was idle; lag was 37,900 steps)
Hypothesis: the reduce_lr_keep_weights plateau policy inherited
  reset_from_champion's lag gating, and with weights never rewound the lag
  grows without bound between promotions, so past the 60,000-step soft cap the
  learner would (a) pause during every arena evaluation (about 40% learner
  idle at the 256-simulation gate), (b) fire a "hard replay lag" recovery on
  every verdict, resetting the conclusive streak so no further rate stage
  could ever trigger, and (c) keep 20% of actor throughput on champion games
  the learner's replay window already excludes. The parent run shows the
  original design's cost from the other side: three plateau_reset events
  (Aug 31 03:39, Sep 1 00:09, Sep 1 22:43) each rewound the learner from
  ~783,000 to 742,979 the moment lag reached the 40,000-step cap, discarding
  40,000 steps of training (including +33 and +38 Elo candidates) every ~16 h
Commit: 636f79ff509670d236e3b62be1aaa45de6ebcc59
Release: main-636f79f-plateau-lag (release-manifest sha256
  418e1ff8c8fe71ba1683ef8dfab8e69d7dff84df2cf6d6bbb1d613f02e2d900e; profile
  profile-arena-gate.yaml unchanged, sha256 d1be1905...)
Treatment (code only, profile unchanged):
  - keep-weights policy ignores champion lag entirely: never pauses, recovers
    only on a conclusive streak, and treats a multiplier already at the floor
    as nothing to do
  - recovery stages descend from the active multiplier
    (max(floor, current * scale): 3.0e-4 -> 1.5e-4 -> 7.5e-5 with the
    production 0.5/0.25), still restored on promotion, so the cross-champion
    non-compounding guarantee is unchanged while a plateau gets a real anneal
    instead of a single cut
  - actors play the candidate instead of a champion that is
    learner.max_replay_lag_steps or more behind the learner, recording
    champion_selfplay_stale
  - reset_from_champion keeps its lag-gated pause/reset semantics
Control: R10-ARENA-GATE-02 runtime (main-2ccd895-arena-gate)
Seeds: 17 (production continuation)
Budget: continuous
System gates: learner steps per hour unchanged (~2,600) after lag passes
  60,000; no plateau_recovery events with reason hard_replay_lag; actor
  heartbeats show champion_selfplay_stale once lag passes 60,000 and the
  learner's eligible-sample fraction stays at 1.0
Statistical gate: the first stage (Muon 1.5e-4) fires on the next conclusive
  streak after cutover; judge the anneal by whether candidates evaluated under
  1.5e-4 and 7.5e-5 close the gap to champion 742,979 at 256 simulations
  (-19.1 and -22.6 so far under 3.0e-4) and by the first promotion
Result: first stage fired 14 seconds after startup (19:37:57 UTC,
  plateau_recovery from the pre-existing streak of two: 774,235 -19.1 and
  778,142 -22.6, both conclusive at 400 games), Muon 3.0e-4 -> 1.5e-4,
  optimizer moments cleared, streak reset, learner kept training. 21:26 UTC:
  learner 2,590 steps/hour with no pauses (lag 42,345); mean training loss
  3.367 over the last hour versus 3.44-3.47 under 3.0e-4; the first candidate
  with any annealed training (782,049, about 1,200 steps at 1.5e-4 before
  publication) stood at 0.0 Elo after 500 games against -19.1 and -22.6 for
  the two candidates before it, still under evaluation. Side effect found: the
  recovery's new resume cutover broke the disaster snapshot (see the decision
  log entry "Keep disaster snapshots through plateau recovery")
Decision: pending
```

```text
ID: R10-UTD-04
Phase: 1 — learner utilization
Status: live since 2026-09-03 02:03 UTC (in-place migration at learner step
  796,464, 2.5 minutes of GPU idle, applied right after candidate 785,956's
  terminal verdict while the arena had no pairs persisted for 793,770)
Hypothesis: the learner GPU is idle 86% of the time because update-to-data 1.0
  allows one 512-example update per 512 newly generated samples (2,560 steps
  per hour against a 0.19-second step; 19,000 steps per hour unthrottled).
  Raising the target to 1.5 adds 50% more optimizer steps per generated sample
  at zero hardware cost and raises frontier Elo per hour, because data
  generation, not optimization, bounds progress
Commit: 496ad19 (migration chain 2ccd895 -> 496ad19; the R3/R4 runtime
  releases did not change the root's source authority)
Release: main-496ad19-utd-1p5 (release-manifest sha256 de5301c9...; profile
  profile-utd-1.5.yaml sha256 32e2958a...); prospective segment baselined at
  examples_consumed 407,789,568 and committed replay samples 592,021,716
Control: R10-PLATEAU-LAG-03 runtime at UTD 1.0 (candidates 782,049 and
  785,956 under the 1.5e-4 anneal: +13.3 and +18.5 at 1,200 games, both
  reject_max_pairs)
Treatment: learner.target_updates_per_new_sample 1.0 -> 1.5 with the
  publication cadence held constant per new replay sample
  (candidate_interval_examples 2M -> 3M, selfplay_snapshot_interval_examples
  1M -> 1.5M); prospective UTD segment baselined at the migration boundary
Anchor/replay cutoff: none; in place with migrate_continuous_profile.py
Seeds: 17 (production continuation)
Budget: continuous
System gates: learner steps per hour rises to about 3,800 within two hours;
  segment_updates_per_new_sample settles at 1.50; candidate cadence stays
  about 1.5 hours; actor samples per second unchanged (about 340); the
  first restart after migration passes preflight with the checkpoint carrying
  the new target and segment; disaster snapshots keep verifying
Statistical gate: judged over the next four to six candidates at 256
  simulations against champion 742,979 (or its successor) compared with the
  1.5e-4/UTD-1.0 candidates; abort back to UTD 1.0 by migration if training
  loss diverges from the 1.5e-4 trend (mean total loss above 3.55 for two
  consecutive hours) or two consecutive candidates conclude below -30 Elo
Result: first minutes: learner heartbeat target 1.5, segment ratio 1.42 and
  rising toward 1.5 after a twelve-minute wait for samples past the new
  baseline, then 7,900 steps per hour spending the accumulated allowance;
  first disaster snapshot after cutover (02:12 UTC) verified with the new
  segment; steady-state rate and candidate evidence pending
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

```text
Experiment: R10-LR-RECOVERY-01 cutover
Arms: 1 (production continuation)
Wall hours per arm: 0.61 of downtime (04:15:42-04:52:28 UTC, 2026-09-02)
Provisioned H100-hours: 4.9 idle during the cutover
Cumulative roadmap H100-hours: not tracked before this entry
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

### 2026-09-02 — Use the idle learner: update-to-data 1.5

Decision: with the rate, gate, plateau, and snapshot fixes live, the largest
remaining idle capacity is the learner GPU (0% utilization 86% of the time under
UTD 1.0; 0.19-second steps throttled to about 2,560 per hour). Raise the target
to 1.5 as pre-registered, holding publications constant per newly generated
sample by scaling the example-based intervals, through the continuous migrator's
new prospective UTD segment support (ported from the autonomous migrator). Apply
it at the arena boundary after the first fully annealed candidate has reported,
so the anneal has one attributable verdict at UTD 1.0; waiting for more would
idle the learner for hours to sharpen an attribution that a single live run
cannot make clean anyway. The third actor lane follows as a separate change once
UTD 1.5 has its own candidates.

### 2026-09-02 — Keep disaster snapshots through plateau recovery

Decision: the first plateau recovery (19:37 UTC) wrote a new resume cutover and
the 14-minute disaster snapshot failed closed on every run afterwards with
`active warm-start marker disagrees with resume cutover` (seven failures by
21:16, last verified snapshot 19:36). The equality rule is only meaningful until
the learner's own first cutover; afterwards the warm-start marker is provenance.
Treat a cutover created after the marker's `cutover_created_ns` as superseding
it, keep the marker in the catalog, and ship as a runtime-only release with a
snapshot run and verify immediately after cutover. Unstamped markers keep the
strict rule. Shipped as main-c5559a7-dr-warm-start (commit c5559a7,
release-manifest sha256 52b3a864...): cutover 21:37:37-21:39:02 UTC (85
seconds of GPU idle, learner step 785,7xx, resumed at multiplier 0.5 from the
checkpoint's governor state), first snapshot 21:40:30 verified ok with 8,484
catalog files; disaster coverage gap 19:36-21:40. Follow-up: the snapshot
tool's SQLite restart loop on stopped roots (noted under R10-LR-RECOVERY-01)
remains open.

### 2026-09-02 — Decouple keep-weights plateau recovery from champion lag

Decision: the first conclusive verdict from the cheap gate (774,235 rejected in
56 minutes) made the plateau counter fast, which exposed that the
keep-weights policy still carried the rewinding policy's lag gates. Weights are
never rewound under `reduce_lr_keep_weights`, so champion lag is not a reason
to idle the learner and grows without bound between promotions; past the
60,000-step cap the policy would have paused training during every evaluation
and reset the conclusive streak on every verdict, silencing the anneal for good.
Make the keep-weights policy streak-only at any lag, let stages descend from the
active multiplier to the floor (a real anneal, still restored on promotion), and
stop actors from playing a champion whose games the replay window discards. Ship
as a runtime-only release before the live run reaches the cap. Leave
`reset_from_champion` semantics untouched.

### 2026-09-02 — Separate the promotion gate from the measurement scale

Decision: 62% of completed evaluations ended `reject_max_pairs` because the
mixture e-process needs about 250-300 pairs to promote a true +40 and about 380
pairs to reject a true 0, while the cap was 200 pairs at 1024 simulations (3.8
hours per candidate). Make gate games cheap (256 simulations, 600-pair cap) so
evidence concludes, and add a mandatory measurement crossplay at 1024
simulations between each new champion and its predecessor so the historical
Elo ladder remains comparable. Fit the ladder per search budget; never mix
budgets. Apply as an in-place migration after the learning-rate recovery has
twelve hours of attributable evidence.

### 2026-09-02 — Stop compounding plateau learning-rate cuts

Decision: the plateau policy, not model capacity or the gate alone, caused the
zero-Elo plateau. `reset_from_champion` restored the champion checkpoint and
scaled the rates stored inside it by 0.5, so each new champion inherited the
cut and the next reset halved it again; frontier gain fell 7.0 → 3.6 → 1.7 →
1.3 → 0.0 Elo per training hour in lockstep with five halvings. Ship a
learning-rate governor that keeps the profile rates as the reference and applies
one floored, absolute multiplier that restores on promotion; count only
conclusive rejections toward recovery; require that contract in the continuous
validator; and warm-start champion 742,979 at the proven Muon 3.0e-4 regime with
candidate-mixed self-play. Do not return to the YAML 5e-3 schedule: that regime
collapsed the model on 2026-08-18 (candidates scored 0-2%). Keep the arena
unchanged in this release so the effect of the rate is attributable; the gate
budget redesign follows as a separate in-place migration.

### 2026-08-31 — Preserve the sandbox and isolate compiled calibration

Decision: do not weaken `ProtectHome`, permit `/root/.triton`, disable
production-faithful compilation, or reuse failed v4 state. Configure distinct
arm-owned Inductor/Triton caches under the already allowlisted calibration
output, require cache/runtime provenance in comparison evidence, retire prior
activators, and arm a new v6 policy against the current stale promotion digest.
Keep cadence, 19-lane, learner-sharing, and UTD changes out of this experiment;
none has positive pair-valid frontier Elo/hour authority.
