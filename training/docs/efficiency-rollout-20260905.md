# Training efficiency rollout — 2026-09-05

User authorization: implement prior recommendations, preserve the 17,402,775
parameter model size, commit all changes, and deploy gracefully to the training
server. Equal importance applies to six game modes and four board sizes.

## Work in progress

- Existing uncommitted one-stone handicap, balanced Stage B sampling, migration
  compatibility and audit/benchmark work is part of this release.
- Inference work (architecture_audit): bounded exact cache and deduplication,
  shared asynchronous inference, reusable pinned transfers, homogeneous geometry
  bias optimization, parity/compile benchmarks. Preserve checkpoint tensor shapes.
- CPU work (status_guidance): shared actor cohorts, CPU-only small-board actors,
  separate native/BLAS budgets, stale-history filtering, bounded exact endgames,
  recovery/drain integration and telemetry.
- Strength work (variant_contract): balanced 24-cell arena and promotion, fixed
  severity/paired schedules, distinct256screen/1024measurement contracts,
  champion-frontier strength/hour reporting and legacy compatibility.
- Root: typed config and migration integration, evidence-based transfer gate,
  raw/EMA diagnostic, profile selection/benchmarks, report service, all checks,
  immutable release build, checkpointed cutover, rollback, Git commit.

## Deployment facts

- SSH: ubuntu@192.222.52.230, existing pinned known-host file
  `~/.ssh/edgeconnect-training-known_hosts`.
- Active root: `/home/ubuntu/edgeconnect-runs/variant-network`.
- Active release: `/home/ubuntu/edgeconnect-releases/variant-f36dc87-384x8`.
- Unit: `edgeconnect-startrain-variant-network.service`.
- Active profile: `<run-root>/profile.yaml`, Stage A, UTD1.0.
- Unit KillMode=mixed, SIGTERM, TimeoutStopSec1000; old immutable release retained.
- Last initial check: step37,799, all workers active, zero service restarts,
  terminal arena rejection at candidate29,298 vs champion19,532.
- Legacy teacher: `/home/ubuntu/edgeconnect-recovery/lr-recovery-ring10-lr-recovery-3e-4-seed17/learner/checkpoints/sha256-37c38c80d9ff1c9de2422cc3aca14d5f15bc22c5d49931e0941de545c91c11fe.pt`, step864,090.
- DR root: `/lambda/nfs/texas-north-fs/edgeconnect-dr/variant-network`.
- Host tools: `/home/ubuntu/.local/bin/uv`, `/home/ubuntu/.cargo/bin/cargo`;
  Python3.11, PyTorch2.13.0+cu130. Passwordless sudo available.
- CPU logical sibling pairs are adjacent:0/1 core0,2/3 core1,etc.; node0=0–103,
  node1=104–207. CPU-only workers must reserve whole physical cores and be excluded
  from GPU-worker affinity masks.

## Rollout rules

Build and test a separate release while the old service continues. Do not edit the
root-owned active release. Preserve exact old unit/profile and recovery provenance.
Before cutover: verified replay and disaster backup, terminal arena allocation
boundary, complete candidate/recovery checkpoint, ledger integrity and CPU/GPU
preflight. Migration is prospective for UTD; publication intervals scale with UTD.
Start the new immutable runtime only after the old coordinator exits. Verify
advancing learner steps, all worker heartbeats, valid checkpoints/backups, cache/
batch telemetry and actual per-mode replay after activation. If runtime validation
fails, gracefully stop and restore the previous unit/profile from the saved bundle.

Experimental controls are implemented and deployed as capabilities; only validated
settings are enabled. No numerical speed or Elo/hour gain is claimed from GPU
utilization or microbenchmarks alone. Model depth/width/parameter tensor shapes
remain unchanged. The earlier projection-order experiment was inconclusive and is
not a production default without new evidence.

## Validation before cutover

- Full local Python suite: 1,067 passed, 5 hardware-specific skips.
- H100 host targeted inference/cache/broker/actor/endgame suite: 59 passed.
- Strict Python type checking and lint passed. Native Rust checks and all-six-mode
  search-seed partition/reorder parity passed.
- CPU-only real-game canary: same17.4M champion,16BLAS/4native threads on CPUs0–31,
  8games,336samples in223.99seconds (1.50positions/s), no dropped games; exactsolver
  solved1tail in65nodes. Raw evidence: `cpu-actor-canary-20260905.json`.
- Shared-GPU inference preflight passed output parity and exposed cache-miss CPU
  overhead. Key construction was moved to producers and vectorized; isolated host
  comparison is required before enabling the final configuration.
- Readiness now binds the active profile checksum and fullrunidentity, verifies
  actualrecentwindowteacherexclusion and independentpaired search streams. Legacy
  strength certification retains the documented1024search/within15Elo standard;
  the running256search comparison is a screen and cannot certify that gate.

## Curriculum activation decision

For the September5 user-authorized rollout, start the equal-six-mode curriculum
once the selected teacher windows are empty. This is an explicit curriculum
decision, not a declaration that legacy strength has been recovered. The
`check_transfer_readiness.py --purpose curriculum` result exposes both
`curriculum_ready` and `legacy_strength_certified`; the latter still requires the
1024-search paired within15Elo criterion. The legacy reference remains preserved.

A stopped in-flight legacy arena is retained byte-for-byte in the rollback
bundle and the run. Switching to balanced evaluation uses a distinct versioned
result namespace and records `evaluation_contract_transition` in the migration
journal. Other in-place arena-contract changes still require a terminal boundary.

## Deployment outcome

- Training release: `variant-efficiency-20260905`, source commit
  `17f69736b1bf155a5fcf6e696448d4ca0da93588`, root-owned and read-only with406
  source hashes and the compiled native artifact hash recorded.
- Graceful stop: step39,249, coordinator exit0, no forced kill. Restart:
  September5 07:17UTC from the same verified recovery checkpoint, with zero
  uncheckpointed updates discarded. Original release and unit files remain in
  `/home/ubuntu/edgeconnect-rollbacks/variant-20260905`.
- Active profile: `/home/ubuntu/edgeconnect-runs/variant-network/profile-efficiency-20260905.yaml`.
  Equal-six-mode StageB, equal ring allocation, prospective UTD1.5, shared GPU
  inference/cohorts, bounded caches and pinned buffers, one reserved CPU actor,
  and bounded exact tails are enabled.
- At07:48UTC: learner step39,687 (+438);149,517 newly committed positions
  include all six mode categories. No service/worker restarts. GPU7's actor is
  intentionally paused while its balanced measurement arena uses the GPU.
- Replay backup and disaster snapshot both succeeded; the latter was verified
  with a source cutoff approximately three minutes old at07:50UTC. Strength
  reports run every15minutes; the first balanced ladder measurement is pending.
- Final local full suite:1,084 passed,5 hardware-specific skips. Host targeted
  suite:59 passed. Additional counter-isolation, native seed parity, padding,
  and monitor tests passed. No Elo/hour gain is claimed from these checks.
- The legacy256-search screen completed120games: pre-rollout champion19,532
  won2 and lost118 against legacychampion864,090. This establishes a weak
  starting baseline, not the strength of the latest training weights, and does
  not certify legacy-strength recovery.
- The monitor-only follow-up counts configured CPU workers separately and
  checks shared cohorts against parent liveness and their own ring evidence.
  It is deployed as a separately pinned script so the training runtime does not
  need another restart. No proactive user-notification channel is configured;
  server health checks and configured recoveries are automatic.
