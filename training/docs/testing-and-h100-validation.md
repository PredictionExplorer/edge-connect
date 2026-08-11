# Testing and H100 validation

The repository uses four explicit validation tiers. A lower tier passing never
implies that a higher tier passed.

For the end-to-end host preparation, frozen profile, process supervision,
monitoring, graceful stop and recovery sequence, use the
[production H100 training runbook](production-h100-training-runbook.md).

## Tier 1: deterministic CPU checks

Run on every pull request:

```bash
cd training
uv sync --extra test --extra serve --extra onnx --locked
uv run maturin develop --release --locked --manifest-path crates/star-py/Cargo.toml
uv run ruff check startrain starserve tests scripts
uv run ruff format --check startrain starserve tests scripts
uv run pyright
uv run pytest --require-native -m "not cuda and not multi_gpu and not soak" \
  --cov --cov-report=json:coverage.json
uv run python scripts/check_coverage.py coverage.json
uv run python scripts/benchmark_native_features.py --batch-size 256

cargo +1.93.0 fmt --all --check
cargo +1.93.0 clippy --workspace --all-targets --locked -- -D warnings
cargo +1.93.0 test --workspace --locked
```

From the repository root:

```bash
npm ci
npm audit --audit-level=moderate
npm run typecheck
npm run lint
npm run test:coverage
npm run build
npm run test:e2e
```

Native tests must not silently skip in CI. `--require-native` turns a missing
or rules-v1 PyO3 extension into a collection error.

## Tier 2: mutation and contract checks

The scheduled mutation workflow targets the rules, scoring, protocol, replay,
loss, self-play, arena, and search code:

```bash
npm run test:mutation

cd training
uv run mutmut run
cargo mutants --package star-engine --package star-search
```

The Python mirror must match the canonical v2 bytes and fingerprint in
`src/lib/star/rules.ts`:

```bash
uv run pytest tests/test_conformance_fixture.py
# expected fingerprint: fnv1a64:2da3783519381453
```

## Tier 3: one-GPU CUDA validation

Tests marked `cuda` exercise BF16 compilation and repeated inference. The
target-host benchmark includes the complete native-state decoding, feature
encoding, host transfer, model execution, and legal-logit return boundary:

```bash
cd training
uv run python scripts/hardware_health_preflight.py \
  --config configs/h100-8gpu.yaml
uv run pytest --require-native -m "cuda and not multi_gpu and not soak"
uv run python scripts/hardware_preflight.py \
  --config configs/h100-8gpu.yaml \
  --rings 6
uv run python scripts/hardware_preflight.py \
  --config configs/h100-8gpu.yaml \
  --rings 10
```

The hardware-health gate is fail-closed. It must report every configured GPU
healthy before CUDA correctness or throughput results are accepted.

Both representative board sizes must sustain at least 5,000 realistic leaf
evaluations per second per H100. Keep the emitted JSON with the run artifacts;
it records latency, throughput, model size, CUDA/PyTorch versions, and peak
allocated memory.

## Tier 4: multi-GPU and soak validation

First prove real NCCL gradient synchronization:

```bash
cd training
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
uv run torchrun --standalone --nproc-per-node 2 \
  scripts/nccl_smoke.py \
  --config configs/h100-8gpu.yaml
```

Then run the long tests and one complete orchestration lifecycle:

```bash
uv run pytest --require-native --run-soak -m soak
startrain-orchestrate --config configs/h100-8gpu.yaml
```

The orchestration soak is complete only after it demonstrates:

- sustained actor and learner progress without unexplained stalls;
- one process-scoped learner loader-pool start, stable worker PIDs across at
  least 200 replay-window refreshes, and no `SemLock` or `resource_tracker`
  diagnostics;
- replay writes, quarantine, restart, and checkpoint resume on the target NVMe;
- at least one candidate-to-arena terminal decision;
- bounded GPU memory and stable thermals;
- graceful drain after SIGTERM; and
- metrics sufficient to reproduce games/hour, leaf evaluations/second,
  learner examples/second, and promotion latency.

Before certifying unattended training, run an isolated continuity canary with a
separate primary and verified fallback root. The canary must exercise:

1. wall-budget expiry while a GPU health probe is in flight;
2. one actor exit and one learner exit, each followed by bounded recovery;
3. coordinator termination followed by systemd/continuity reconciliation;
4. a malformed canary checkpoint and replay shard, proving verified fallback or
   quarantine without modifying the production root;
5. an unavailable `nvidia-smi` response, proving bounded transient handling;
6. a synthetic unsafe GPU report, proving that active work is stopped and
   fallback is blocked; and
7. queue completion/failure handoff to the verified last-known-good workload.

The queue fault matrix must also run a replay backup between manifest
verification and the next arm. The create-once replay initialization marker
must remain byte-identical, its semantic identity pin must still verify, and
tampering with `run_id` or `generation_family` must fail closed. A completed
single-seed queue must request fallback and must never authorize adoption.
Cross-seed adoption tests require exactly seeds 17/18/19, externally pinned
comparison and policy digests, a positive conservative advantage in every
seed, at least 20% median point improvement, and a fresh-root 24-hour canary
plan.

Do not inject real ECC errors, reset a production GPU, or fill the host
filesystem. Use isolated artifacts and deterministic fault hooks. Certification
requires:

- no false `hardware_health_failure` at a planned shutdown;
- no orphan compute process or second coordinator;
- complete role-reversed arena pairs only;
- resource release within 180 seconds for the full 8-H100 role-partitioned
  canary profile;
- productive fallback learner and actor progress within 180 seconds after an
  immediate handoff, or 300 seconds through the timer backstop; and
- all teardown, retry, fallback, and idle time included in provisioned wall
  hours.

## Certification rule

CPU, browser, or mocked distributed tests cannot certify H100 readiness.
Until Tier 3 and Tier 4 evidence is attached to a run, documentation and
release notes must say that CUDA/NCCL production validation is pending.

