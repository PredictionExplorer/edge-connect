# Model serving and browser distillation

Start with the [operator guide](../README.md) to install Rust 1.93, Python 3.11,
CUDA PyTorch and the native extension. From `training/`, the equivalent pip setup is:

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
pentanomial summaries, diagnostic paired bootstrap intervals, anytime-valid
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
Custom layouts that intentionally share a learner GPU must set
`promotion.pause_sharing_mode`; the arena then publishes a process-owned pause
lease and all DDP ranks stop launching training steps until that lease clears.
Arena/actor overlap is always rejected.

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

Replay schema v3 and WAL manifest schema v3 are intentionally not compatible
with pre-audit replay. Start a new run directory rather than mixing legacy
shards that lack run, generation, game, ply, and immutable-model provenance.
Committed missing or corrupt shards are quarantined and reported; they are not
silently deleted. Replay and model GC honor active replay watermarks and
candidate/champion references. Shipped retention is dry-run until operators
review metrics and choose deletion policy.

`configs/h100.yaml` is explicitly a non-stratified standalone smoke/tuning
profile and cannot wait for absent rings. Use `h100-4gpu.yaml` or
`h100-8gpu.yaml` for continuous all-ring training.

## API

`GET /healthz` and `GET /v1/health` report service, model, rules, feature, and
action schema versions. `POST /v1/analyze` is the documented endpoint;
`POST /v1/move` is an equivalent compatibility route. Health is intentionally
unauthenticated for container orchestration.

The v1 request is strict: unknown fields, coercions, terminal states, incompatible
hashes, malformed semantic states, and over-budget searches are rejected.

```json
{
  "schema_version": 1,
  "rules_hash": "fnv1a64:cdb34fb02be82843",
  "rings": 3,
  "stones": [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
             -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
             -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  "to_move": 0,
  "moves_left": 1,
  "opening": true,
  "pass_streak": 0,
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
placement/pass, completed-Q root policy, root Q and visits, WDL and value,
the 363-bin score belief, EMA model identity, and timing.

If `security.bearer_token_env` is configured, send
`Authorization: Bearer <token>`. The token is read only from that environment
variable, not YAML. CORS is an explicit-origin allowlist; wildcard origins are
rejected. Body size, search budget, queue duration, execution duration, and
concurrency are bounded. Timed-out or disconnected work receives a cooperative
cancellation signal and keeps its concurrency slot until the worker stops.

## Private same-origin proxy

The web app defaults to its own `/v1/move` and `/v1/health` route handlers. Keep
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
WDL, score margin, ownership, and alive beliefs.

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
  --manifest runs/browser/star-browser-v1.browser.json \
  --target ../public/models/star \
  --wasm-source ../public/models/star/wasm
```

The release command verifies SHA-256 and byte length for both the Python
checkpoint and ONNX artifact, copies the immutable ONNX file, and atomically
replaces `public/models/star/manifest.json` last. An optional WASM build can be
run as the same verified release step:

```bash
startrain-publish-browser \
  --manifest runs/browser/star-browser-v1.browser.json \
  --target ../public/models/star \
  --wasm-cwd .. \
  --wasm-source ../public/models/star/wasm \
  --wasm-build npm run build:star-wasm
```

The verified WASM artifacts are
`wasm/star_wasm.js` and `wasm/star_wasm_bg.wasm`. The build runs first; the
release command stages and verifies both WASM files and the ONNX model, then
replaces the canonical `manifest.json` strictly last.

No placeholder model is checked into the repository. A browser model exists only
after a successful distillation run and ONNX export.
