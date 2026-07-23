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

## Run one bounded arm

The runner marks the exact measurement interval, forwards graceful signals to
the orchestrator, and stops at the first wall or leaf budget:

```bash
python scripts/run_elo_ablation.py \
  --config /absolute/path/to/treatment/profile-elo-ablation.yaml
```

For unattended operation, instantiate
`deploy/edgeconnect-startrain-ablation.service.example`. Use only one treatment
unit at a time on a single 8-H100 host.

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

Generate a strength-efficiency report for each arm, then use
`compare_elo_ablation.py` to rank only eligible treatments. One-seed pilots are
successive-halving evidence, not deployment evidence. Advance the best two plus
control to three 12-hour seeds. Test a combined profile only after both
one-factor treatments independently pass.

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

The comparator uses `ablation.json` for the exact fixed-budget wall interval,
requires one common champion anchor, and marks incomplete measurements, parse
failures, missing guard evidence, or ring-regression decisions ineligible.

## Rollback

The parent run remains stopped and unchanged during pilots. If no treatment
passes, discard the treatment roots and resume the exact frozen parent profile.
For a winning treatment, create a new continuous run root and complete a
24-hour canary; do not repoint the historical parent's champion or replay.
