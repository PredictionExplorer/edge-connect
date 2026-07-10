# Testing and H100 validation

The repository uses four explicit validation tiers. A lower tier passing never
implies that a higher tier passed.

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
PyO3 extension into a collection error.

## Tier 2: mutation and contract checks

The scheduled mutation workflow targets the rules, scoring, protocol, replay,
loss, self-play, arena, and search code:

```bash
npm run test:mutation

cd training
uv run mutmut run
cargo mutants --package star-engine --package star-search
```

The deterministic conformance fixture must regenerate byte-for-byte:

```bash
node scripts/export-star-conformance.mjs /tmp/conformance-v1.json
cmp testdata/star/conformance-v1.json /tmp/conformance-v1.json
```

## Tier 3: one-GPU CUDA validation

Tests marked `cuda` exercise BF16 compilation and repeated inference. The
target-host benchmark includes the complete native-state decoding, feature
encoding, host transfer, model execution, and legal-logit return boundary:

```bash
cd training
uv run pytest --require-native -m "cuda and not multi_gpu and not soak"
uv run python scripts/hardware_preflight.py \
  --config configs/h100-8gpu.yaml \
  --rings 6
uv run python scripts/hardware_preflight.py \
  --config configs/h100-8gpu.yaml \
  --rings 12
```

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
- replay writes, quarantine, restart, and checkpoint resume on the target NVMe;
- at least one candidate-to-arena terminal decision;
- bounded GPU memory and stable thermals;
- graceful drain after SIGTERM; and
- metrics sufficient to reproduce games/hour, leaf evaluations/second,
  learner examples/second, and promotion latency.

## Certification rule

CPU, browser, or mocked distributed tests cannot certify H100 readiness.
Until Tier 3 and Tier 4 evidence is attached to a run, documentation and
release notes must say that CUDA/NCCL production validation is pending.

