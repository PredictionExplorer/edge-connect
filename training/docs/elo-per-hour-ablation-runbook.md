# Ring-10 Elo-per-hour ablation runbook

This runbook optimizes ring-10 Elo gained per wall-clock hour while treating
rings 4, 6, and 8 as non-inferiority guardrails. It complements the
[training ablation protocol](training-ablation-protocol.md); it does not weaken
the paired arena or its anytime-valid promotion test.

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
  --environment-file /etc/edgeconnect/elo-ablation-seed17.env \
  --state /var/lib/edgeconnect/elo-ablation-seed17/queue.json \
  --comparison-output /var/lib/edgeconnect/elo-ablation-seed17/comparison.json \
  --execution-lock /var/lib/edgeconnect/elo-ablation-execution.lock \
  --max-transient-retries 2 \
  --retry-delay-seconds 30
```

The manifest pins the Git commit and clean-tree requirement, plan and installed
profiles, queue/runner/comparator scripts, rendered systemd units, environment
file, and seed snapshot identity, model/recovery pointers, and replay ledger.
Generation refuses a dirty checkout when it infers `HEAD`. Every launch verifies
the commit and all digests again; a mixed-revision launch is refused before an
arm starts.

Verify the exact installed deployment before enabling it:

```bash
python scripts/run_elo_ablation_queue.py verify \
  --manifest /etc/edgeconnect/elo-ablation-seed17.json
```

Do not edit a profile, unit, script, environment file, or seed snapshot after
manifest generation. Generate a new manifest from one coherent revision
instead.

## Run and recover the queue

Start only the queue unit on the 8-H100 host:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now edgeconnect-startrain-ablation-queue.service
```

`queue.json` is an atomically replaced state file. It records every arm as
`pending`, `running`, `completed`, or `failed`, plus attempt counts, transient
failures, the frozen policy, and finalization status. A non-blocking file lock
protects the state, while the shared execution lock allows only one ablation
deployment to use the host. Use the same execution-lock path for every
deployment on that host. On restart, a stale `running` arm is reconciled from
`ablation.json` and resumed rather than skipped.

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
`running` or `transient_crash` measurement. It refuses completed and fatal arms.

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
