# Ring-10 Elo-per-hour ablation runbook

This runbook optimizes ring-10 Elo gained per wall-clock hour. The default
contract treats rings 4, 6, and 8 as non-inferiority guardrails; the explicitly
separate `ring10-only` contract trains and evaluates only ring 10. It
complements the [training ablation protocol](training-ablation-protocol.md) and
does not weaken the paired arena or its anytime-valid promotion test.

## Acceptance contract

- Primary metric: one-sided 95% lower confidence bound for ring-10
  Bradley-Terry Elo gained per wall hour.
- Count all eight provisioned GPUs, including replay waits, arena leases, and
  idle periods.
- Default guard margin: `-35 Elo` for each of rings 4, 6, and 8.
- Eliminate a treatment after `reject_ring_regression`, replay corruption,
  hardware failure, or an incomplete fixed-budget run.
- Promote a treatment only after three seeds, a positive lower confidence
  bound versus control, and at least 20% median ring-10 Elo/hour improvement.

For a pre-registered `ring10-only` plan, omit the smaller-ring guard margin and
use only ring-10 games for both training and promotion. Keep all eight
provisioned GPUs in the wall-clock denominator. Smaller boards remain playable,
but their strength is outside the experiment contract and may regress.

## Safety model

Never edit or fork a live run root. Wait for the current arena result, run the
durable snapshot service, stop the coordinator gracefully, and verify that
`coordinator.lock` is absent. Treatment roots retain the parent's run identity
so copied replay and checkpoints remain valid, but they are isolated branches:
never merge their replay or model pointers into the parent.

`fork_elo_ablation.py` copies mutable files and hard-links only immutable replay
shards, checkpoints, manifests, and recovery checkpoints. It rotates prior
runtime metrics into `ablation-parent/` and writes `ablation.json`.

## Prepare one-seed pilots

Run from `training/` after copying the frozen active profile to a stable path:

```bash
python scripts/prepare_elo_ablation.py \
  --base-config /absolute/path/to/frozen-control.yaml \
  --source-run-root /absolute/path/to/stopped-control \
  --output-dir /absolute/path/to/pilot-profiles-seed17 \
  --run-root-parent /absolute/path/to/pilot-runs \
  --run-id <run-id-from-source-run.json> \
  --prefix ring10-pilot \
  --seed 17 \
  --wall-budget-hours 8 \
  --leaf-budget 2000000000 \
  --guard-floor-elo -35
```

The default matrix is:

- `control`
- `utd-1`
- `plateau-keep`
- `freshness-mix`
- `ring10-70`
- `search-quality`

The command refuses to overwrite its output and records profile digests in
`ablation-plan.json`.

### Weighted-generalist matrix

Run the weighted objective as a separate 55/65/70 matrix:

- `weighted-control` keeps the frozen control training mix (currently
  15/15/15/55 after its large-board transition);
- `ring10-65-weighted` trains 10/10/15/65; and
- `ring10-70-weighted` trains 10/10/10/70.

All three promote on complete 1/1/1/7 arena blocks for rings 4/6/8/10, with 15
initial blocks, 10-block continuations, and a 50-block cap. Prepare them with
repeated `--treatment` options. Do not mix weighted and legacy treatments in
one plan because their promotion objectives and guard contracts differ.

This matrix intentionally sets `required_regression_rings: []` and clears
blocking per-ring floor overrides. Per-ring Elo and anytime intervals remain
in reports as diagnostics, but a weighted winner can regress on rings 4, 6, or
8 and still promote. There is no per-ring non-inferiority guarantee; use the
legacy matrix when one is required.

### Ring-10-only treatment

Prepare `ring10-only` as a separate plan. Do not mix it with the guarded legacy
or weighted-generalist matrices. Its frozen contract is:

```yaml
orchestration:
  training_objective: ring10_only
  ring_mixture:
    step_weights:
      - from_step: 0
        weights: [0.0, 0.0, 0.0, 1.0]
arena:
  rings: [10]
  promotion_pair_ratios: {}
  required_regression_rings: []
  per_ring_regression_floor_elo: {}
```

The actor scheduler therefore creates only ring-10 games, learner readiness and
batch quotas count only ring-10 replay, and every arena pair is ring 10. Keep
`game.rings: [4, 6, 8, 10]`; changing the game contract would break checkpoint
and serving compatibility rather than merely narrowing the training objective.

Before warm-starting a long ring-10-only run from an archived checkpoint,
freeze the candidate manifest allowlist and evaluation profile, then use
`evaluate_archived_manifests.py` to run fresh ring-10-only matches against one
common champion baseline. Select only an independently confirmed improvement;
if none passes, warm-start from the existing champion. Write selection results
outside the parent run root and preserve the parent unchanged for rollback.

### Ring-10 efficiency suite

Prepare the utilization treatments as a separate ring-10-only plan:

```bash
python scripts/prepare_elo_ablation.py \
  --base-config /absolute/path/to/frozen-ring10-control.yaml \
  --source-run-root /absolute/path/to/stopped-ring10-control \
  --output-dir /absolute/path/to/ring10-efficiency-profiles-seed17 \
  --run-root-parent /absolute/path/to/ring10-efficiency-runs \
  --run-id <run-id-from-source-run.json> \
  --prefix ring10-efficiency \
  --seed 17 \
  --wall-budget-hours 8 \
  --leaf-budget 2000000000 \
  --suite ring10-efficiency
```

The frozen queue order is:

1. `ring10-only`: unchanged 13-lane control;
2. `ring10-learner-slack-64`: one batch-64 actor shares GPU 0 with the
   replay-limited learner while promotion remains pause-shared on GPU 7; and
3. `ring10-actor-lanes-3`: three actor lanes on GPUs 1–6 while GPU 7 remains
   single-lane for the arena pause protocol.

The learner-slack arm is a fresh-root topology treatment, not a profile
migration. Abort it on CUDA OOM, learner or actor restart loops, a material
learner-throughput loss without fleet throughput gain, or weaker ring-10
champion-frontier Elo per provisioned GPU-hour. One-seed equal-leaf pilots are
screening evidence only. Repeat surviving treatments with seeds 17, 18, and 19
before adoption.

Fork this suite with its own labels rather than the default matrix:

```bash
for treatment in \
  ring10-only ring10-learner-slack-64 ring10-actor-lanes-3
do
  python scripts/fork_elo_ablation.py \
    --source-run-root /absolute/path/to/stopped-ring10-control \
    --plan /absolute/path/to/ring10-efficiency-profiles-seed17/ablation-plan.json \
    --treatment "$treatment"
done
```

The queue comparator reads the empty guard-ring contract from the plan. If a
comparison must be regenerated directly, pass `--no-guard-rings`; never add
the generalist ring 4/6/8 floors to these runs.

### Benchmark arena occupancy without changing promotion

The production ring-10 profile already uses its largest validator-approved
continuation wave: 25 pairs. A 50-pair continuation cannot be expressed as a
deployable profile without weakening the continuous-service contract. Test the
occupancy hypothesis against fixed manifests instead:

```bash
python scripts/prepare_arena_occupancy_benchmark.py \
  --source-run-root /absolute/path/to/stopped-ring10-control \
  --profile /absolute/path/to/frozen-ring10-control.yaml \
  --candidate-manifest /absolute/path/to/manifest-candidate.json \
  --output-dir /absolute/path/to/arena-occupancy-plan \
  --device cuda:7 \
  --physical-gpu-index 7 \
  --execution-lock /var/lib/edgeconnect/elo-ablation-execution.lock \
  --repeats 4

python scripts/benchmark_arena_occupancy.py \
  --plan /absolute/path/to/arena-occupancy-plan/benchmark-plan.json \
  --output-dir /absolute/path/to/arena-occupancy-results
```

The benchmark warms both strategies and counterbalances four paired repeats.
Each repeat evaluates the same 50 pair indices as either two production-sized
25-pair chunks or one benchmark-only 50-pair chunk. It records GPU utilization,
power, memory, evaluator rows/second, serialized inference time, queue wait,
peak CUDA allocation, exact pair evidence, and runtime/device provenance. The
50-pair arm is explicitly non-deployable. Different batch sizes use different
batch-derived search seeds, so the report must not claim game-outcome
equivalence or update a production profile.

Preparation requires a clean Git revision and no source `coordinator.lock`.
Execution acquires the coordinator-compatible source lock for the benchmark
lifetime plus the same host execution lock used by ablation queues and
continuity, and releases both only after CUDA cleanup and terminal evidence
publication. It also refuses a physical GPU with another compute process and
checks that the logical PyTorch device UUID matches the physical GPU sampled
through `nvidia-smi`. No replay, checkpoint, pointer, metric, or profile in the
source root is modified.

## Fork treatments

Fork every arm before running any arm so all treatments have the same source
state:

```bash
for treatment in \
  control utd-1 plateau-keep freshness-mix ring10-70 search-quality
do
  python scripts/fork_elo_ablation.py \
    --source-run-root /absolute/path/to/stopped-control \
    --plan /absolute/path/to/pilot-profiles-seed17/ablation-plan.json \
    --treatment "$treatment"
done
```

Verify each root has:

- `ablation.json`
- `profile-elo-ablation.yaml`
- no `coordinator.lock`
- empty live `status/`, `logs/`, and `metrics/` directories
- the expected champion identity in `ablation.json`

## Prepare a clean champion warm start

For post-plateau research, use
`configs/h100-8gpu-champion-warmstart.yaml` and the clean treatments
`control`, `lr-quarter`, `fresh-source`, `hard-replay`, and `fresh-hard`.
Every arm still reuses the same validated replay, but it must discard the weak
learner optimizer trajectory:

```bash
python scripts/preflight_run_state.py \
  --run-root "$TREATMENT_ROOT" \
  --profile "$TREATMENT_ROOT/profile-elo-ablation.yaml" \
  --apply

python scripts/prepare_champion_warm_start.py \
  --run-root "$TREATMENT_ROOT" \
  --profile "$TREATMENT_ROOT/profile-elo-ablation.yaml" \
  --apply

python scripts/preflight_run_state.py \
  --run-root "$TREATMENT_ROOT" \
  --profile "$TREATMENT_ROOT/profile-elo-ablation.yaml"
```

The warm start loads champion EMA weights into both the raw model and a fresh
EMA, creates empty optimizer and scheduler state at segment step zero, preserves
the absolute champion model step, resets cadence, and records an explicit
segment-relative UTD baseline. By default it grants at most one configured
recent replay window as initial credit. Historical self-play manifests older
than the resulting resume cutover are excluded from the opponent pool.

Do not generate the deployment manifest until every arm reports an active
`learner/champion-warm-start.json` marker and a clean preflight.

## Freeze the deployment

Run queues only from a clean, committed checkout. Render these templates to
their final systemd paths:

- `deploy/edgeconnect-startrain-ablation-queue.service.example`
- `deploy/edgeconnect-startrain-ablation-finalize.service.example`
- `deploy/edgeconnect-startrain-ablation-replay-backup.service.example`
- `deploy/edgeconnect-startrain-ablation-replay-backup.timer.example`

Both rendered units must reference the same deployment manifest and environment
file. The environment file can contain host-specific settings, but it must be
root-owned and read-only to the service user. Keep queue state and reports
outside the Git checkout. Pre-create the state, report, and execution-lock
parent directories with write access for the service user.

After all arms have been forked, generate the deployment manifest:

```bash
python scripts/run_elo_ablation_queue.py manifest \
  --plan /absolute/path/to/pilot-profiles-seed17/ablation-plan.json \
  --output /etc/edgeconnect/elo-ablation-seed17.json \
  --training-dir "$PWD" \
  --queue-unit /etc/systemd/system/edgeconnect-startrain-ablation-queue.service \
  --finalize-unit /etc/systemd/system/edgeconnect-startrain-ablation-finalize.service \
  --replay-backup-service-unit /etc/systemd/system/edgeconnect-startrain-ablation-seed17-replay-backup.service \
  --replay-backup-timer-unit /etc/systemd/system/edgeconnect-startrain-ablation-seed17-replay-backup.timer \
  --replay-backup-interval-seconds 3600 \
  --replay-backup-retain 3 \
  --environment-file /etc/edgeconnect/elo-ablation-seed17.env \
  --state /var/lib/edgeconnect/elo-ablation-seed17/queue.json \
  --comparison-output /var/lib/edgeconnect/elo-ablation-seed17/comparison.json \
  --execution-lock /var/lib/edgeconnect/elo-ablation-execution.lock \
  --max-transient-retries 2 \
  --retry-delay-seconds 30
```

The manifest pins the Git commit and clean-tree requirement, plan and installed
profiles, queue/runner/comparator/backup scripts, rendered systemd units,
environment file, and seed snapshot identity, model/recovery pointers, and
replay ledger. Backup flags are optional for old or short diagnostic manifests;
use them for every unattended queue. Generation refuses a dirty checkout when
it infers `HEAD`. Every launch verifies the commit and all digests again; a
mixed-revision launch is refused before an arm starts.

Verify the exact installed deployment before enabling it:

```bash
python scripts/run_elo_ablation_queue.py verify \
  --manifest /etc/edgeconnect/elo-ablation-seed17.json
```

Do not edit a profile, unit, script, environment file, or seed snapshot after
manifest generation. Generate a new manifest from one coherent revision
instead.

## Run and recover the queue

Enable the queue and its rendered replay-backup timer on the 8-H100 host:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now edgeconnect-startrain-ablation-seed17-replay-backup.timer
sudo systemctl enable --now edgeconnect-startrain-ablation-queue.service
```

`queue.json` is an atomically replaced state file. It records every arm as
`pending`, `running`, `completed`, or `failed`, plus attempt counts, transient
failures, the frozen policy, and finalization status. A non-blocking file lock
protects the state, while the shared execution lock allows only one ablation
deployment to use the host. Use the same execution-lock path for every
deployment on that host. On restart, a stale `running` arm is reconciled from
`ablation.json` and resumed rather than skipped.

The timer reads `queue.json` but never writes it. It backs up only the single
arm whose status is `running`, uses a non-blocking per-run file lock to avoid
overlap with manual or arm-boundary backups, and no-ops during transitions or
after queue completion. Each arm runner performs a fail-closed replay integrity
check and restores only a verified rotating backup before state preflight. The
queue attempts one final verified backup after every arm attempt and records
its status in queue and comparison evidence. A final backup failure is visible
but cannot rewrite a completed measurement; corrupt replay with no valid backup
blocks the arm before the orchestrator starts.

The per-arm runner records one of these durable outcomes:

- `budget_completion`: the wall or evaluator-row budget was reached. The
  evidence cutoff is recorded before shutdown; teardown health and the final
  resource-release time are recorded separately.
- `transient_crash`: the orchestrator died from a signal or the queue was
  interrupted. The queue retries only up to its frozen transient retry limit.
- `fatal_orchestrator_exit`: the orchestrator exited normally before its budget,
  including explicit hardware-health or exhausted-worker exit codes.

The original measurement start is retained across retries. Wall-clock downtime
therefore remains charged to the arm, and a restart after the wall deadline
records budget completion without starting a fresh measurement window.

`measurement_cutoff_ns` limits which arena evidence is eligible.
`resource_released_ns` includes graceful drain, forced cleanup, and idle
recovery time in the provisioned-hour denominator. A non-clean exit after the
cutoff is accepted only as `complete_with_warning` when state preflight passes
and every terminal failure is proven to have occurred after the cutoff. A
pre-cutoff or untimestamped fatal remains ineligible. Missing resource-release
evidence blocks both queue advancement and automatic continuity handoff.

An isolated fatal arm is quarantined by metadata; its files are preserved. The
queue always finalizes the partial report and writes a continuity handoff
request. The host continuity controller decides whether another arm is safe or
whether to resume the immutable last-known-good workload. The queue itself
never invokes `systemctl`.

For a local diagnostic only, one arm can still be invoked directly:

```bash
python scripts/run_elo_ablation.py \
  --config /absolute/path/to/treatment/profile-elo-ablation.yaml
```

The direct runner uses the same durable attempt metadata and resumes only a
`running` or `transient_crash` measurement. It performs the same fail-closed
replay restore before state preflight and refuses completed and fatal arms.

## Always-run finalization

The queue rebuilds the comparison in a `finally` path on success, interruption,
or arm failure. The queue unit also names the finalizer in both `OnSuccess=` and
`OnFailure=`, so systemd retries finalization even if the queue process cannot
run its own cleanup. The finalizer is idempotent:

```bash
python scripts/run_elo_ablation_queue.py finalize \
  --manifest /etc/edgeconnect/elo-ablation-seed17.json
```

The report always includes every configured arm. Pending and failed arms receive
the `queue_arm_incomplete` ineligibility reason and are never ranked. An
`incomplete` comparison is expected after a failed arm; it is evidence, not a
successful experiment.

Install
`deploy/edgeconnect-startrain-continuity-trigger.conf.example` as a systemd
drop-in for immediate handoff on queue success or failure. The periodic
continuity timer is still required as an independent backstop.

## Throughput screening

Before the strength pilots, run the existing bounded inference sweep:

```bash
python scripts/h100_system_benchmark.py \
  --config /absolute/path/to/control/profile-elo-ablation.yaml \
  --output-dir /absolute/path/to/system-benchmark \
  --rings 10 \
  --batch-sizes 128 160 192 \
  --repeats 3
```

Keep a systems treatment only if ring-10 evaluator throughput improves by at
least 15%, correctness remains exact, and the treatment does not reduce fresh
samples per provisioned hour.

## Compare and advance

The queue writes the final comparison path frozen in its deployment manifest.
Use `compare_elo_ablation.py` directly only to regenerate or inspect a report.
It ranks only eligible treatments. One-seed pilots are successive-halving
evidence, not deployment evidence. Advance the best two plus control to three
12-hour seeds. Test a combined profile only after both one-factor treatments
independently pass.

```bash
python scripts/compare_elo_ablation.py \
  --run control=/absolute/path/to/control \
  --run plateau-keep=/absolute/path/to/plateau-keep \
  --run ring10-70=/absolute/path/to/ring10-70 \
  --provisioned-gpus 8 \
  --guard-ring 4 --guard-ring 6 --guard-ring 8 \
  --guard-floor-elo -35 \
  --output /absolute/path/to/elo-comparison.json
```

The comparator uses `ablation.json` for the exact evidence and resource
intervals, requires one common champion anchor, honors the runner's durable
outcome, and marks incomplete measurements, failed queue arms, parse failures,
missing guard evidence, or ring-regression decisions ineligible. Its deployment
metric is guarded champion-frontier ring-10 Elo lower bound per total
provisioned wall hour. The latest terminal candidate remains a diagnostic and
is never cherry-picked as the deployed winner.

For a weighted-only plan, pass `--no-guard-rings`. When every arm exposes the
same weighted objective, the comparator instead ranks chronological promoted
champion frontiers by weighted Elo lower bound per total provisioned wall hour.
Ring-10 and all per-ring summaries remain secondary diagnostics. Mixed weighted
and legacy objectives are ineligible rather than being ranked on whichever
metric looks best.

For dependent experiment stages, use `run_staged_elo_pipeline.py`. It accepts
only a hash-verified upstream winner snapshot and refuses a downstream fork
whose champion anchor or canonical stage specification is stale. Every
downstream fork receives a new weights-only champion warm-start, recovery
pointer, resume cutover, optimizer, scheduler, and cadence boundary at the
selected champion; `resume_latest` must never select an inherited rejected
checkpoint. Futility policies are pre-registered, stop-only, and require
anytime-valid upper bounds; they never grant promotion authority. Weighted
stages use `guard_rings: []`, `promotion_objective: weighted_aggregate`, and
weighted-aggregate confidence-sequence evidence. Empty guards remove per-ring
futility vetoes, not lifecycle, integrity, fixed-budget, or common-anchor
requirements.

## Rollback

The parent run remains stopped and unchanged during pilots. If no treatment
passes, discard the treatment roots and resume the exact frozen parent profile.
For a winning treatment, create a new continuous run root and complete a
24-hour canary; do not repoint the historical parent's champion or replay.
