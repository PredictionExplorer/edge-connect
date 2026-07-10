# Double *Star AI operator guide

This directory contains the implemented training, arena, serving and browser-export
pipeline for no-pie Double *Star on 3–12 rings.

**No trained model is checked into this repository.** The code and tests establish the
pipeline contracts; they do not establish strong or superhuman play. `starserve` needs a
valid `champion.json`, and local browser AI needs a separately distilled and published
model.

## Architecture

- `crates/star-engine`: authoritative Rust board, rules, scoring and D5 symmetry.
- `crates/star-search`: batched exact-state Gumbel AlphaZero search with Sequential
  Halving and completed-Q statistics.
- `crates/star-py`: the `star_native` PyO3 boundary used by Python actors and serving.
- `startrain`: graph feature encoding, `GraphResTNet`, Gumbel self-play, replay,
  learning, EMA checkpoints, arenas, promotion and orchestration.
- `starserve`: one-process FastAPI service that combines an EMA champion with native
  search.
- `crates/star-wasm`: the same Rust rules/search contract compiled for local browser
  inference.

`GraphResTNet` alternates local edge-aware residual blocks with global grouped-query
attention. It predicts policy, WDL, score margin, ownership, alive stones and a
KataGo-style soft-policy auxiliary. Self-play searches atomic placements and pass
actions, which preserves Double *Star's same-player first/second-placement semantics.

Research inspirations:

- [Gumbel AlphaZero](https://openreview.net/forum?id=bERaNdoegnO) for root policy
  improvement and Sequential Halving.
- [KataGo methods](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md)
  for auxiliary targets and sample-efficiency techniques.
- [ResTNet](https://www.ijcai.org/proceedings/2025/828) for repeated local/global
  residual-transformer groups.
- [Regret-Guided Search Control](https://arxiv.org/abs/2602.20809) as an experimental
  future ablation only. RGSC is not implemented or enabled in this pipeline.

Use the [training ablation protocol](docs/training-ablation-protocol.md) before
changing shipped self-play sources, target retention, candidate scaling, precision or
search settings.

## Prerequisites

The documented baseline is:

- Linux for CUDA training and serving;
- Rust 1.93 (`training/Cargo.toml` requires it);
- Python 3.11;
- `uv` or `pip`, plus maturin 1.14.1 for `star_native`;
- a CUDA-enabled PyTorch build compatible with the host NVIDIA driver;
- 4 or 8 H100s for the supplied continuous profiles;
- Node.js/npm for the web application, and `wasm-pack` plus the
  `wasm32-unknown-unknown` target for browser publication.

Use fast durable local storage for `runs/`. Replay uses immutable compressed shards and
a SQLite WAL manifest, while checkpoints and manifests are immutable and
content-addressed.

The supplied layouts are single-host profiles:

- `configs/h100-4gpu.yaml`: GPU 0 learner, GPUs 1–2 actors, GPU 3 arena.
- `configs/h100-8gpu.yaml`: GPU 0 learner, GPUs 1–6 actors, GPU 7 arena.
- `configs/h100.yaml`: one-board-size standalone smoke/tuning profile, not a continuous
  all-ring run.

Both continuous profiles set `distributed.enabled: false`; they do not use NCCL.

## Install and build `star_native`

Run from `training/`. The CUDA wheel index below matches the CUDA 12.6 service image;
select a different official PyTorch index when the installed driver requires it.

```bash
cd training
rustup toolchain install 1.93.0 --profile minimal

uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install --index-url https://download.pytorch.org/whl/cu126 "torch>=2.4"
uv pip install "maturin==1.14.1" -e ".[test,serve,onnx]"
maturin develop --release --locked --manifest-path crates/star-py/Cargo.toml

rustc +1.93.0 --version
python --version
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "import star_native; print(star_native.native_rules_hash_tag())"
```

For a CPU-only workstation, omit the CUDA-index command and let the project dependency
resolve an appropriate CPU PyTorch build. With `pip`, create the Python 3.11 virtual
environment normally and replace each `uv pip` invocation with `python -m pip`.

Re-run `maturin develop` after changing Rust code. `star_native` is required for real
self-play and native end-to-end tests; without it, those pytest cases skip.

## Full validation

With the environment active:

```bash
cd training
cargo +1.93.0 fmt --all --check
cargo +1.93.0 test --workspace --locked
python -m ruff check startrain starserve tests
python -m pytest

cd ..
npm test
npm run lint
npm run build
```

These tests cover CPU behavior, conformance, replay, orchestration command construction,
promotion, serving and browser export. They do **not** replace a real H100, CUDA or NCCL
soak. See [testing and H100 validation](docs/testing-and-h100-validation.md) for the
enforced coverage gates, mutation suites, hardware benchmark commands and certification
rule.

## Local CPU smoke

Direct self-play and learner CLIs require `--run-identity` to name a durable `run.json`
file. It is a path, not a run-name string. Create a fresh identity, generate four tiny
games with an untrained CPU-smoke model, then consume replay for one learner step:

```bash
cd training
source .venv/bin/activate
export RUN=runs/cpu-smoke
mkdir -p "$RUN"

python - <<'PY'
import os
from startrain.runtime import load_or_create_run_identity

identity = load_or_create_run_identity(
    f"{os.environ['RUN']}/run.json",
    requested_run_id="cpu-smoke-v1",
)
print(identity)
PY

startrain-selfplay \
  --config configs/small.yaml \
  --replay-store "$RUN/replay" \
  --run-identity "$RUN/run.json" \
  --actor-id cpu-smoke \
  --device cpu \
  --cpu-smoke \
  --rings 3 \
  --games 4

startrain-train \
  --config configs/small.yaml \
  --replay-store "$RUN/replay" \
  --output "$RUN/learner" \
  --run-identity "$RUN/run.json" \
  --device cpu \
  --steps 1
```

Choose a new directory and run ID for a clean repeat. Outside `--cpu-smoke`,
`startrain-selfplay --checkpoint` expects an immutable champion model manifest, not a
raw `.pt` file.

## Start a 4- or 8-H100 run

First verify all GPUs and CUDA PyTorch:

```bash
nvidia-smi -L
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.device_count())"
```

Copy the selected YAML before a production run. Give it a unique
`orchestration.directories.root`; optionally set `orchestration.run_id` to a stable
operator-chosen identifier. Relative run roots are resolved from the launch working
directory, so launch from `training/`.

Start exactly one profile:

```bash
startrain-orchestrate --config configs/h100-4gpu.yaml
```

or:

```bash
startrain-orchestrate --config configs/h100-8gpu.yaml
```

The orchestrator has no `--run-id` or `--run-identity` option. It atomically creates
`<run-root>/run.json`, then passes that required identity path to every learner, actor
and promotion child process. An existing `run.json` is reused; a configured `run_id`
that disagrees with it is rejected.

For custom DDP, configure at least two learner GPU entries with equal CPU budgets and
set `orchestration.distributed.enabled: true` with backend `nccl`. The coordinator then
constructs the `torch.distributed.run` command and required run-identity arguments.

## Run files and model lifecycle

For the 8-GPU profile, `runs/h100-8gpu/` contains:

- `run.json`: immutable run ID, generation family and creation time.
- `replay/manifest.sqlite3`: WAL replay index and actor generation leases.
- `replay/shards/` and `replay/quarantine/`: immutable replay and rejected corrupt
  shards.
- `learner/checkpoints/` and `learner/manifests/`: immutable EMA artifacts.
- `learner/candidate.json`: atomic pointer to the latest learner candidate.
- `learner/champion.json`: atomic deployment pointer and the actor source used by the
  shipped profiles. `starserve` accepts only this role.
- `learner/metrics.jsonl` and `learner/learner-complete.json`: training metrics and
  final completion identity.
- `arena/promotion-status.json` and arena result JSON: persisted paired evaluation.
- `status/`: coordinator and worker heartbeats.
- `metrics/`: coordinator and actor JSONL metrics.
- `logs/`: one combined stdout/stderr log per child.

At startup the learner publishes an initial candidate. The shipped H100 profiles
explicitly allow the promotion supervisor to bootstrap that first candidate as
champion, which releases actors waiting for `champion.json`. Later candidates are
immutable EMA checkpoints emitted every 5,000 learner steps.

The arena compares each candidate with the current champion using reversed-role pairs,
all rings, forced and unforced openings, a pair-level mixture-betting e-process and
anytime-valid per-ring regression checks. The shipped gate takes 25 new pairs per ring
per look, requires at least 50, allows at most 200, tests a +35 Elo alternative against
0 Elo, and rejects a material ring regression. Only a `promote` result atomically
advances `champion.json`; inconclusive results accumulate more non-overlapping pairs.
The shipped `model_refresh.selfplay_source: champion` means rejected candidates never
feed self-play. Research ablations may select `candidate` or
`candidate_champion_mix`; the selected role and policy-supervision rate are written to
actor metrics and model swaps still occur only at complete batch boundaries.

If candidate/champion lag reaches the configured plateau, the learner pauses for a
terminal arena result. After three terminal rejections, the shipped profile resets
learner/optimizer/EMA state from the champion. At the target step, actors drain at a
complete cohort boundary and the coordinator waits for the final candidate's terminal
arena decision.

## Monitoring and recovery

Set the run root, then inspect coordinator state, promotion and logs:

```bash
RUN=runs/h100-8gpu
jq . "$RUN/status/coordinator.json"
jq . "$RUN/arena/promotion-status.json"
tail -F "$RUN"/logs/*.log
tail -F "$RUN"/learner/metrics.jsonl "$RUN"/metrics/*.jsonl
watch -n 2 nvidia-smi
```

Useful replay checks:

```bash
sqlite3 "$RUN/replay/manifest.sqlite3" \
  "select state, count(*) from shards group by state;"
```

The coordinator treats a heartbeat older than 180 seconds or unchanged progress for
1,800 seconds as a failure. It restarts workers with bounded exponential backoff, up to
eight restarts. If it exits nonzero:

1. inspect `status/coordinator.json` and the named worker log;
2. correct the external cause (CUDA OOM, disk full, driver failure or invalid artifact);
3. rerun the same orchestration command against the same run root.

`resume_latest` reloads the last candidate, replay reopening reconciles orphaned files,
and committed corrupt/missing shards are quarantined. Do not delete or regenerate
`run.json`, manually repoint candidate/champion files, or mix schema-v3 replay from
different run identities. Use a new run root for an incompatible config or schema.

Stop with SIGINT/SIGTERM and allow the configured grace period so complete cohorts and
SQLite/checkpoint writes can finish. A live `coordinator.lock` prevents two
coordinators from owning one root; an abandoned lock is removed only when its PID is no
longer live. Retention is enabled in dry-run mode, so review GC metrics before enabling
deletion.

## Benchmark and recalibration gates

The native CPU smoke benchmark is:

```bash
RAYON_NUM_THREADS=32 \
  cargo +1.93.0 run -p star-py --example actor_throughput --release --locked
```

It measures native scoring and uniform-evaluator search only. It is not an H100
end-to-end benchmark.

Before committing to a long run, use these planning gates:

1. Sustain at least 5,000 realistic leaf evaluations/s/H100 with production batching.
2. After 10,000 games, measure actual game length, pass rate, search mix, CPU saturation
   and ring distribution; rescale the forecast from those measurements.
3. By 100,000 games, beat random/greedy nearly perfectly and fixed shallow search
   convincingly on every ring.
4. By 500,000 games, run at least 200 paired arena games per ring and fit strength gain
   against log training volume.
5. Treat "superhuman" as an external evaluation, not a pipeline output: use one
   checkpoint across all sizes, balanced roles and serious blind matches against the
   strongest available humans.

The original planning budget was about 2 million games, 120 million retained positions
and 22 billion leaf evaluations. After a stable pipeline, the rough wall-clock bands
were:

- 8 H100s: 9–16 days base, 3–6 days optimistic, 2.5–3.5 months pessimistic.
- 4 H100s: 18–28 days base, 5–8 days optimistic, 5–7 months pessimistic.

These are capacity estimates, not strength guarantees. They carry roughly **3–5×
uncertainty**, primarily because Double *Star has no modern rating pool or established
neural baseline.

**A real H100 soak is still required.** The shipped profiles and automated tests have
not demonstrated sustained CUDA throughput, thermals, recovery or full
candidate-to-arena cycles on the target host. **A real NCCL soak is also still
required before enabling custom multi-learner DDP.** The shipped 4/8-H100 profiles do
not exercise NCCL, and unit tests validate only DDP command construction and CPU-side
coordination.

## Serve the private champion

Update `configs/starserve.yaml` to point at the intended experiment config and
`champion.json`. The sample path does not exist until training has produced a champion.

```bash
export STARSERVE_BEARER_TOKEN="replace-with-a-secret"
starserve --config configs/starserve.yaml --check-config
starserve --config configs/starserve.yaml
```

The YAML names `STARSERVE_BEARER_TOKEN`; the secret itself must exist only in the
environment. `starserve` rejects candidate pointers, verifies the immutable manifest
and EMA checkpoint, and reloads a new champion between requests. An invalid replacement
keeps the prior champion live and degrades health.

For the browser, keep the GPU service private and let Next.js proxy the fixed
same-origin routes:

```bash
export STAR_AI_SERVER_URL="http://127.0.0.1:8080"
export STAR_AI_BEARER_TOKEN="$STARSERVE_BEARER_TOKEN"
cd ..
npm run dev
```

- `STAR_AI_SERVER_URL` and `STAR_AI_BEARER_TOKEN` are server-only Next.js variables.
- Leave `NEXT_PUBLIC_STAR_AI_URL` unset to use `/v1/move` and `/v1/health`.
- Never put the token in a `NEXT_PUBLIC_*` variable or a URL.
- Optional `NEXT_PUBLIC_STAR_AI_SIMULATIONS` and
  `NEXT_PUBLIC_STAR_AI_MAX_CONSIDERED` must not exceed the limits in
  `configs/starserve.yaml`.
- `starserve` health is unauthenticated; move/analyze requests use the bearer token when
  configured. Direct-browser CORS origins must be explicit; `*` is rejected.

See [serving and distillation](docs/serving-and-distillation.md) for the API contract and
container notes.

## Distill and publish browser AI

Edit `configs/distill-browser.yaml` to point at validated replay and a champion. Choose
a new `export.model_version` for every release; output artifacts are immutable.

```bash
cd training
startrain-distill --config configs/distill-browser.yaml

cd ..
rustup target add wasm32-unknown-unknown --toolchain 1.93.0
cargo +1.93.0 install wasm-pack --locked
RUSTUP_TOOLCHAIN=1.93.0 npm run build:star-wasm

cd training
startrain-publish-browser \
  --manifest runs/browser/star-browser-v1.browser.json \
  --target ../public/models/star \
  --wasm-source ../public/models/star/wasm
```

The sample config emits `star-browser-v1.pt`, `star-browser-v1.fp16.onnx` and
`star-browser-v1.browser.json`. Publication verifies checkpoint, ONNX and WASM
integrity, copies the immutable ONNX artifact, and replaces
`public/models/star/manifest.json` last. The web app reports local AI unavailable until
that canonical manifest and all referenced artifacts exist.
