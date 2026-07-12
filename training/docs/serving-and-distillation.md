# Model serving and browser distillation

Start with the [operator guide](../README.md) to install Rust 1.93, Python 3.11,
CUDA PyTorch and the native extension. Production trainers should complete the
[H100 training runbook](production-h100-training-runbook.md) before using the
publication steps below. From `training/`, the equivalent pip setup is:

```bash
python -m pip install "maturin==1.14.1" -e ".[serve,onnx]"
maturin develop --release --locked --manifest-path crates/star-py/Cargo.toml
```

No model is included in the repository. `starserve` can start only after
`configs/starserve.yaml` points to an existing champion:

```bash
export STARSERVE_BEARER_TOKEN="replace-with-a-secret"
starserve --config configs/starserve.yaml
```

`starserve` loads only EMA weights named by an atomic `champion.json` pointer.
The pointer references an immutable, content-addressed manifest and checkpoint;
SHA-256, byte length, run identity, generation family, experiment
configuration, finalized rules hash, feature hash, and checkpoint step are
verified before the model becomes ready. Candidate pointers are rejected. The
champion pointer is checked at request boundaries. A valid replacement is
installed only when no request is active; an invalid replacement leaves the
previous champion live and marks health as degraded.

## Mac-local champion service

The full champion can run on Apple silicon through the existing Python service.
The browser and Next.js development server must run on the same Mac because a
remote website cannot connect to that Mac's `127.0.0.1`.

Install the service and native search extension from a repository checkout on
the Mac:

```bash
cd /path/to/EdgeConnect/training
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "maturin==1.14.1" -e ".[serve]"
rustup toolchain install 1.93.0
RUSTUP_TOOLCHAIN=1.93.0 maturin develop --release --locked \
  --manifest-path crates/star-py/Cargo.toml
python -c 'import star_native, torch; print(star_native.__file__); print("mps_available=", torch.backends.mps.is_available())'
```

Create each snapshot on the training host, where the atomic pointer and its
immutable artifacts are on one filesystem. The output directory must not
exist. The exporter reads one complete `champion.json`, rejects candidate
pointers and paths that escape the publication root, verifies hashes and byte
lengths, copies only that pointer's manifest/checkpoint, and validates the
copied EMA checkpoint against a derived FP32/non-compiled experiment profile.

Run the export remotely and transfer the new, versioned directory with SSH and
`rsync`:

```bash
export TRAIN_HOST=trainer.example
export TRAINING_ROOT=/srv/EdgeConnect/training
export REMOTE_CHAMPION=runs/h100-8gpu-optimized/learner/champion.json
export REMOTE_PROFILE=configs/h100-8gpu-optimized.yaml
export SNAPSHOT_ID="$(date -u +%Y%m%dT%H%M%SZ)"
export REMOTE_SNAPSHOT="/tmp/edgeconnect-champion-$SNAPSHOT_ID"

ssh "$TRAIN_HOST" \
  "cd '$TRAINING_ROOT' && .venv/bin/python scripts/export_champion_snapshot.py \
  --champion '$REMOTE_CHAMPION' \
  --profile '$REMOTE_PROFILE' \
  --output '$REMOTE_SNAPSHOT'"

export LOCAL_SNAPSHOT="$HOME/.local/share/edgeconnect/$SNAPSHOT_ID"
mkdir -p "$LOCAL_SNAPSHOT"
rsync -a "$TRAIN_HOST:$REMOTE_SNAPSHOT/" "$LOCAL_SNAPSHOT/"
```

The bundle is relocatable. `champion.json` retains its manifest path, the
manifest retains its checkpoint path, and `starserve-mac.yaml` uses only
bundle-relative paths. The generated server config binds `127.0.0.1`, selects
`mps`, has no CORS origins or bearer token, and limits analysis to one
concurrent request. Its derived experiment profile uses FP32 and disables
`torch.compile`.

Start and verify the Mac service:

```bash
cd /path/to/EdgeConnect/training
source .venv/bin/activate
starserve --config "$LOCAL_SNAPSHOT/starserve-mac.yaml" --check-config
starserve --config "$LOCAL_SNAPSHOT/starserve-mac.yaml"
```

In another terminal, verify readiness and launch Next.js with the private
loopback endpoint:

```bash
curl --fail --silent http://127.0.0.1:8080/v2/health | python -m json.tool

cd /path/to/EdgeConnect
export STAR_AI_SERVER_URL=http://127.0.0.1:8080
export NEXT_PUBLIC_STAR_AI_DEVTOOLS=1
npm run dev
```

The equivalent `.env.local` entries are:

```dotenv
STAR_AI_SERVER_URL=http://127.0.0.1:8080
NEXT_PUBLIC_STAR_AI_DEVTOOLS=1
```

If startup reports that MPS is unavailable, or batch-one measurements show
CPU is faster, use the same verified bundle with the explicit CPU override:

```bash
starserve --config "$LOCAL_SNAPSHOT/starserve-mac.yaml" --device cpu
```

Refresh by exporting and syncing a new `SNAPSHOT_ID`, stopping the old process,
and launching from the new directory. Do not rsync over an active snapshot;
both exporter and versioned-directory workflow intentionally refuse unsafe
in-place replacement.

## Training publication lifecycle

Launch the single-host pipeline with an explicit topology:

```bash
startrain-orchestrate --config configs/h100-8gpu.yaml
```

The coordinator durably creates `<run-root>/run.json` and passes its path to each child.
The orchestrator itself has no run-identity flag. Direct `selfplay`, `train`, `actor`
and `promote` CLIs require `--run-identity <path-to-run.json>`; this argument is a file
path, not a run-name string. Every actor leases monotonically
increasing generations from the WAL replay manifest. Game IDs and all search
seeds derive from the run, generation family, actor, generation, game, and ply.
Completed game IDs are unique in SQLite, so a restarted actor cannot commit the
same game twice.

The learner writes immutable `sha256-<digest>.pt` checkpoints, immutable
content-hashed manifests, and an atomic `candidate.json` pointer. `starserve`
never reads that pointer. Shipped profiles keep actors on `champion.json`;
research runs may explicitly select the latest candidate or a seeded
candidate/champion mixture through `model_refresh.selfplay_source`. Pointer
roles and run identities remain mandatory, and actors switch only between
complete game batches. The coordinator-managed arena supervisor
bootstraps the first `champion.json` only when
`promotion.bootstrap_initial_champion` is explicitly enabled. Every later
candidate is compared with the immutable champion using role-paired games,
binary game outcomes, pair win-count summaries, diagnostic paired bootstrap intervals,
anytime-valid
ring floors, and both forced and unforced openings. Promotion uses a
pair-level mixture-betting e-process with Ville thresholds. Its observation
unit is the complete role-reversed pair, so arbitrary correlation inside each
pair is retained; validity assumes the sequence satisfies the documented
conditional-mean/martingale condition. A `continue` decision is nonterminal:
pair IDs and outcomes are persisted, and new non-overlapping pairs are added
until a terminal boundary or `arena.max_pairs_per_ring`.

Only an arena `promote` result atomically replaces the champion pointer;
under the shipped champion-only policy, rejected candidates never generate
replay. Candidate or mixture self-play is an explicit ablation and is recorded
in actor metrics. Newer candidates supersede older unfinished candidates
explicitly. The 4-GPU and 8-GPU presets reserve GPU 3
and GPU 7 respectively for arena work, so learner and arena CUDA allocations
do not overlap.
The optimized 8-GPU profile instead keeps the learner continuous on GPU 0 and
pause-shares GPU 7 between `actor-gpu-7` and arena/promotion. Promotion
atomically publishes a tokenized request; the coordinator acknowledges that
exact token only after the actor has exited and been reaped. CUDA evaluators
are created only after that acknowledgement. The coordinator restores the
actor once the result, promotion status, and any champion pointer update are
durable. Lease heartbeat expiry or arena exit first stops/reaps the arena
owner, then restores the actor; shutdown and final drain suppress restoration.

Custom layouts may still pause-share a learner GPU. The coordinator waits for
a fresh learner `arena_gpu_pause` progress acknowledgement before granting the
same tokenized lease. Pause sharing must overlap exactly one configured learner
or actor GPU; all uncoordinated overlap is rejected.

Model pointers use pointer-relative manifest paths, and immutable manifests use
manifest-relative checkpoint paths. The whole learner artifact tree can be
relocated or mounted at a different container path without invalidating digest
verification.

Actors observe shutdown only between complete 64-game cohorts and continue
champion self-play during a learner plateau. At the configured candidate versus
champion lag, inconclusive candidates keep receiving arena pairs while learner
updates pause. Terminal rejections may permit another candidate or, after the
configured rejection streak, reset learner/optimizer/EMA state from the
champion. The hard replay-lag boundary is never crossed while paused, so
champion replay remains eligible.

When the learner reaches its configured target it writes
`learner-complete.json`. The coordinator stops actors at their next exact
cohort boundary, keeps the arena supervisor alive, and exits successfully only
after that final candidate has a terminal promote/reject/ring-regression/max
result.

`train.per_rank_batch_size` is the number of unique samples consumed by each
DDP rank per optimizer step. Global batch size is
`per_rank_batch_size * world_size`. Replay is selected lazily from recent
compressed shards with a bounded per-worker shard cache. The learner waits if
there are not enough unique samples for one global step; it never inflates a
small window by replacement. Rank 0 fixes and broadcasts the exact replay spans
and maximum shard ID to all DDP ranks. Batches are shard-local, so a 512-sample
batch decompresses one selected shard rather than hundreds.

Replay schema v4 and WAL manifest schema v4 are intentionally incompatible with
all older replay. Policies are node-only and outcomes are binary loss/win labels.
Start a new run directory; there is no converter or compatibility loader.
Committed missing or corrupt shards are quarantined and reported; they are not
silently deleted. Replay and model GC honor active replay watermarks and
candidate/champion references. Shipped retention is dry-run until operators
review metrics and choose deletion policy.

`configs/h100.yaml` is explicitly a non-stratified standalone smoke/tuning
profile and cannot wait for absent rings. Use `h100-4gpu.yaml` or
`h100-8gpu.yaml` for continuous all-ring training.

## API

`GET /healthz` and `GET /v2/health` report service, configured device, champion
role/identity, search defaults/maximums/named presets, model, rules, feature,
action, and binary-outcome schemas. `POST /v2/analyze` is the documented
endpoint; `POST /v2/move` exposes the same v2 schema. Health is intentionally
unauthenticated for container orchestration.

The v2 request is strict: unknown fields, coercions, terminal states, incompatible
hashes, malformed semantic states, and over-budget searches are rejected.

```json
{
  "schema_version": 2,
  "rules_hash": "fnv1a64:2da3783519381453",
  "rings": 4,
  "stones": [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
             -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
             -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
             -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
             -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  "to_move": 0,
  "moves_left": 1,
  "opening": true,
  "terminal": false,
  "search": {
    "simulations": 4096,
    "max_considered": 32,
    "seed": 17
  }
}
```

The state is imported through `star_native.StateBatch.from_semantic`; Python
does not replay or reinterpret moves. The response contains one atomic
placement, completed-Q root policy, root Q and visits, `{loss, win}` probabilities,
`P(win) - P(loss)`, the 303-bin `[-151, 151]` score belief, EMA model identity,
and timing.

If `security.bearer_token_env` is configured, send
`Authorization: Bearer <token>`. The token is read only from that environment
variable, not YAML. CORS is an explicit-origin allowlist; wildcard origins are
rejected. Body size, search budget, queue duration, execution duration, and
concurrency are bounded. Timed-out or disconnected work receives a cooperative
cancellation signal and keeps its concurrency slot until the worker stops.

## Private same-origin proxy

The web app defaults to its own `/v2/move` and `/v2/health` route handlers. Keep
`starserve` private and configure the Next.js server with:

```bash
STAR_AI_SERVER_URL=http://starserve.internal:8080
STAR_AI_BEARER_TOKEN="$STARSERVE_BEARER_TOKEN"
```

`STAR_AI_SERVER_URL` must be an absolute HTTP(S) URL without embedded credentials.
`STAR_AI_BEARER_TOKEN` must match the environment variable named by
`security.bearer_token_env` in the server YAML. Both variables are server-only; never
use a `NEXT_PUBLIC_*` token. Leave `NEXT_PUBLIC_STAR_AI_URL` unset for the same-origin
proxy. Optional public search-budget variables must stay within the server YAML limits.

Build the CUDA service image from this directory:

```bash
docker build -t edgeconnect-starserve .
docker run --gpus all --read-only \
  -p 8080:8080 \
  -e STARSERVE_CONFIG=/config/starserve.yaml \
  -e STARSERVE_BEARER_TOKEN \
  -v "$PWD/configs:/config:ro" \
  -v "$PWD/runs:/runs:ro" \
  edgeconnect-starserve
```

Adjust manifest paths in the mounted server YAML for the container layout.

## Browser model

`startrain-distill` trains a newly initialized, configurable smaller
`GraphResTNet` from validated replay search/outcome/spatial targets. When a
teacher EMA manifest is configured, per-head logit KL can be enabled for policy,
binary outcome, score margin, ownership, and alive beliefs.

```bash
startrain-distill --config configs/distill-browser.yaml
```

The command emits:

- a regular atomic startrain checkpoint containing raw and EMA state;
- an EMA-weight FP16 ONNX model with dynamic batch, node, degree, and action axes;
- a browser manifest with SHA-256 and byte length for both artifacts, exact
  tensor names/dtypes/shapes, finalized rules/feature/action identifiers,
  architecture, training provenance, and recommended local search settings.

Artifact versions are immutable: the command refuses to overwrite an existing
checkpoint, ONNX model, or manifest. Choose a new `model_version` for each run.

Publish a verified browser release only after distillation succeeds:

```bash
cd ..
RUSTUP_TOOLCHAIN=1.93.0 npm run build:star-wasm
cd training
startrain-publish-browser \
  --manifest runs/browser/star-browser-v2.browser.json \
  --target ../public/models/star \
  --wasm-source ../public/models/star/wasm-2da3783519381453
```

The release command verifies SHA-256 and byte length for both the Python
checkpoint and ONNX artifact, copies the immutable ONNX file, and atomically
replaces `public/models/star/manifest.json` last. An optional WASM build can be
run as the same verified release step:

```bash
startrain-publish-browser \
  --manifest runs/browser/star-browser-v2.browser.json \
  --target ../public/models/star \
  --wasm-cwd .. \
  --wasm-source ../public/models/star/wasm-2da3783519381453 \
  --wasm-build npm run build:star-wasm
```

The verified WASM artifacts are
`wasm/star_wasm.js` and `wasm/star_wasm_bg.wasm`. The build runs first; the
release command stages and verifies both WASM files and the ONNX model, then
replaces the canonical `manifest.json` strictly last.

No placeholder model is checked into the repository. A browser model exists only
after a successful distillation run and ONNX export.
