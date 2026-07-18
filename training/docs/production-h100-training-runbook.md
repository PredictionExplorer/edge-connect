# Production H100 training runbook

This is the canonical operator procedure for starting, supervising, stopping,
and resuming a production Double *Star training run on one Linux host with
either four or eight NVIDIA H100 GPUs.

The short command is:

```bash
startrain-orchestrate --config /absolute/path/to/frozen-run-profile.yaml
```

Do not run that command until the installation, storage, profile, CUDA, native
extension, throughput, and recovery checks below have passed.

Every value written as `<like-this>` is an operator-supplied placeholder and
must be replaced before copying the command.

## Scope and current certification status

The supplied layouts are single-host profiles:

- `configs/h100-4gpu.yaml` assigns GPU 0 to the learner, GPUs 1–2 to
  self-play actors, and GPU 3 to arena/promotion.
- `configs/h100-8gpu.yaml` assigns GPU 0 to the learner, GPUs 1–6 to
  self-play actors, and GPU 7 to arena/promotion.
- `configs/h100-8gpu-optimized.yaml` assigns GPU 0 to the learner, GPUs
  1–7 to self-play actors, and pause-shares GPU 7 with arena/promotion.
  The coordinator stops and reaps `actor-gpu-7` before acknowledging the
  arena lease, so learner GPU 0 remains continuous.
- `configs/h100-8gpu-throughput.yaml` keeps that physical role layout, runs
  two actor lanes on GPUs 1–6, keeps GPU 7 single-lane for pause sharing, and
  applies the measured target host's NUMA affinity.
- `configs/h100-8gpu-autonomous.yaml` runs two actor lanes on GPUs 1–7 and
  pause-shares learner GPU 0 with arena/promotion. The learner releases unused
  CUDA cache before each bounded one-wave arena lease and catches up for 30
  minutes between unresolved waves. The profile rejects imported training
  artifacts, records immutable provenance, caps update-to-data, and disables
  measurement-only historical cross-play.

All supplied continuous profiles deliberately use one learner GPU and set
`distributed.enabled: false`. The real training command is therefore
`startrain-orchestrate`, not `torchrun`. NCCL is used only by the preflight
smoke unless an operator creates and validates a separate multi-learner
profile.

There is no trained model in the repository. Passing CPU tests, browser tests,
or mocked distributed tests does not certify H100 readiness. A target host is
production-ready only after the CUDA, throughput, NCCL, recovery, and
candidate-to-arena checks in this runbook pass.

The Dockerfile in `training/` is a `starserve` serving image. It is not the
production training launcher.

## 1. Host requirements

Required:

- Linux with a current NVIDIA data-center driver;
- at least 4 full H100 GPUs; the supplied profiles consume either GPU IDs
  0–3 or GPU IDs 0–7;
- MIG disabled for the GPUs assigned to this run;
- Python 3.11;
- Rust 1.93;
- a CUDA-enabled PyTorch build compatible with the installed driver;
- fast durable local NVMe for replay, checkpoints, SQLite WAL, logs, and
  metrics;
- stable system time and enough cooling/power for a multi-day sustained load.

The current profiles allocate these CPU-thread budgets:

- historical 8-GPU profile: 16 learner + 48 actor + 8 arena = 72 threads.
- optimized 8-GPU profile: 16 learner + 56 actor + 8 arena = 80 threads.
- throughput 8-GPU profile: 16 learner + 104 actor-lane + 8 arena threads =
  128 configured threads; the arena and GPU-7 actor do not run concurrently.
- autonomous 8-GPU profile: 16 learner + 112 actor-lane + 8 arena threads =
  136 configured threads; learner and arena do not compute concurrently.
- 4-GPU profile: 24 learner + 24 actor + 8 arena = 56 threads.

Have at least that many logical CPUs or lower the per-worker budgets in a new
profile and re-run preflight. For an 8-H100 host, 256 GiB RAM and at least
2 TiB free local NVMe are prudent operational starting points, not hard
application checks. More disk is safer because replay retention initially runs
in dry-run mode and therefore does not delete old shards.

Do not place the run root on NFS, SMB, an object-store mount, or another
filesystem with unreliable SQLite locking/fsync semantics.

Inspect the host before installation:

```bash
nvidia-smi -L
nvidia-smi topo -m
nvidia-smi --query-gpu=index,name,memory.total,driver_version,pci.bus_id \
  --format=csv
nvidia-smi --query-gpu=index,mig.mode.current --format=csv
lscpu
free -h
lsblk -o NAME,SIZE,FSTYPE,TYPE,MOUNTPOINTS
```

Before every production start, run the fail-closed health gate against the
frozen profile:

```bash
python scripts/hardware_health_preflight.py \
  --config "$PROFILE" \
  --output "$RUN_ROOT/status/hardware-health-startup.json"
```

The gate rejects missing or unexpected GPUs, non-H100 devices, MIG or ECC
misconfiguration, volatile or aggregate uncorrectable ECC, SRAM threshold
exceeded, pending channel/TPC repair, row-remap failure, and a requested GPU
recovery action. Do not bypass it to resume a run.

If 4–7 H100s are visible, use the 4-GPU profile and leave additional GPUs
unassigned. If at least 8 are visible, use the 8-GPU profile; extra GPUs remain
unassigned. Create and validate a custom topology only when the run should
consume a different mapping or more than eight GPUs.

## 2. Put an immutable code revision on the host

Push the intended commit/tag to a private remote or copy the repository to the
server. Avoid copying local build artifacts.

Example remote workflow:

```bash
git clone <repository-url> "$HOME/EdgeConnect"
cd "$HOME/EdgeConnect"
git checkout <commit-or-release-tag>
git status --short
git rev-parse HEAD
```

`git status --short` should be empty. Record the output of `git rev-parse HEAD`
with the run metadata.

If a Git remote is unavailable, copy from a workstation:

```bash
rsync -az \
  --exclude node_modules \
  --exclude .next \
  --exclude training/.venv \
  --exclude training/target \
  --exclude training/runs \
  /local/path/EdgeConnect/ \
  <server>:"$HOME/EdgeConnect/"
```

Do not edit source code in place after training starts. Use a new commit and a
new run root for incompatible code or schema changes.

## 3. Install base operating-system packages

Ubuntu 24.04 example:

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential \
  ca-certificates \
  curl \
  git \
  jq \
  sqlite3 \
  tmux \
  util-linux
```

The NVIDIA driver and `nvidia-smi` must already work. Driver installation is
host/vendor-specific and intentionally not automated by this repository.

## 4. Install Rust, uv, and Python 3.11

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
  | sh -s -- -y --profile minimal --default-toolchain 1.93.0
source "$HOME/.cargo/env"
rustup toolchain install 1.93.0 --profile minimal

curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv python install 3.11

rustc +1.93.0 --version
uv --version
uv python find 3.11
```

## 5. Choose exactly one PyTorch installation route

First run:

```bash
nvidia-smi
```

The "CUDA Version" shown by `nvidia-smi` is the maximum CUDA runtime supported
by the installed driver, not necessarily a system toolkit installation.

### Route A: locked environment for a CUDA 13-capable driver

The committed lock currently resolves PyTorch 2.13 and its Linux CUDA 13
runtime dependencies. Use this route when the installed driver supports
CUDA 13:

```bash
cd "$HOME/EdgeConnect/training"

uv sync \
  --python 3.11 \
  --extra test \
  --extra serve \
  --extra onnx \
  --locked

source .venv/bin/activate
```

This is the most reproducible route because package hashes and versions come
from `uv.lock`.

### Route B: CUDA 12.6-compatible PyTorch wheel

Use this route when the host driver cannot load the locked CUDA 13 runtime but
does support CUDA 12.6:

```bash
cd "$HOME/EdgeConnect/training"

uv venv --python 3.11 .venv
source .venv/bin/activate

uv pip install \
  --index-url https://download.pytorch.org/whl/cu126 \
  "torch>=2.13"

uv pip install "maturin==1.14.1" -e ".[test,serve,onnx]"
```

Do not run `uv sync --locked` after Route B; it would replace the selected
PyTorch build with the locked one. Record the actual `pip freeze`, PyTorch
version, CUDA runtime, and driver with the run.

## 6. Build the native Rust/Python extension

From `training/`, with `.venv` activated:

```bash
maturin develop \
  --release \
  --locked \
  --manifest-path crates/star-py/Cargo.toml
```

Re-run this command after every Rust change.

Verify the runtime:

```bash
python - <<'PY'
import torch
import star_native

print("PyTorch:", torch.__version__)
print("Bundled CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
print("Rules hash:", star_native.native_rules_hash_tag())

assert torch.cuda.is_available()
assert torch.cuda.is_bf16_supported()

for index in range(torch.cuda.device_count()):
    print(
        index,
        torch.cuda.get_device_name(index),
        torch.cuda.get_device_capability(index),
        torch.cuda.get_device_properties(index).total_memory,
    )
PY
```

An H100 should report compute capability `(9, 0)`. The rules hash must match the
repository contract. A failure here is an installation failure, not a reason
to bypass native tests.

## 7. Select and prepare fast local storage

Choose the actual NVMe mount and verify free space and write permissions:

```bash
export RUNS_BASE="/mnt/nvme/edgeconnect"  # change for this host
mkdir -p "$RUNS_BASE"
test -w "$RUNS_BASE"
df -h "$RUNS_BASE"
```

The run root contains immutable replay shards and checkpoints plus a SQLite WAL
manifest. Monitor both bytes and inode usage during the run:

```bash
df -h "$RUNS_BASE"
df -i "$RUNS_BASE"
```

## 8. Create a frozen per-run profile

Choose exactly one source profile:

```bash
# At least eight H100s; consumes GPU IDs 0-7:
export SOURCE_PROFILE="configs/h100-8gpu.yaml"

# Strict autonomous scratch training on eight H100s:
# export SOURCE_PROFILE="configs/h100-8gpu-autonomous.yaml"

# Four to seven H100s; consumes GPU IDs 0-3:
# export SOURCE_PROFILE="configs/h100-4gpu.yaml"
```

Create a unique UTC run ID, absolute run root, and copied profile:

```bash
cd "$HOME/EdgeConnect/training"
source .venv/bin/activate

export RUN_ID="star-$(date -u +%Y%m%dT%H%M%SZ)"
export RUN_ROOT="$RUNS_BASE/$RUN_ID"
export PROFILE="$RUN_ROOT/profile.yaml"

mkdir -p "$RUN_ROOT"
cp "$SOURCE_PROFILE" "$PROFILE"
```

Pin `orchestration.run_id` and the absolute NVMe root:

```bash
python - "$PROFILE" "$RUN_ROOT" "$RUN_ID" <<'PY'
import sys
from pathlib import Path
import yaml

profile, run_root, run_id = sys.argv[1:]
path = Path(profile)
config = yaml.safe_load(path.read_text(encoding="utf-8"))

config["orchestration"]["run_id"] = run_id
config["orchestration"]["directories"]["root"] = run_root

path.write_text(
    yaml.safe_dump(config, sort_keys=False),
    encoding="utf-8",
)

print("Profile:", path.resolve())
print("Run ID:", run_id)
print("Run root:", run_root)
PY
```

For an explicitly operator-controlled continuous run, add the following before
freezing the profile. Finite profiles remain the default:

```yaml
selfplay:
  max_considered_ring_exponent: 1.0
  max_considered_cap: 32
  record_fast_policy_targets: true
  fast_policy_weight: 0.25
learner:
  steps: 1000000
  unlimited: true
  recovery_interval_steps: 1000
orchestration:
  ring_mixture:
    step_weights:
      - from_step: 360000
        weights: [0.15, 0.15, 0.15, 0.55]
      - from_step: 1000000
        weights: [0.1, 0.1, 0.1, 0.7]
  plateau:
    consecutive_terminal_rejections: 2
    reset_learning_rate_scale: 0.5
  retention:
    enabled: true
    dry_run: false
    recovery_dry_run: false
```

`steps` remains the monitoring milestone. The cosine scheduler still reaches
`min_lr_ratio` at one million and then holds that floor. Recovery checkpoints
are not promotion candidates and do not change the throughput profile's
15,000-step arena cadence. Autonomous profiles publish promotion candidates by
`candidate_interval_examples` while independently refreshing actor models with
the `selfplay_snapshot_*` example cadence.
At the hard replay-lag boundary, a terminal rejection resets to the champion
even if candidate supersession prevented the configured rejection streak.
Each plateau reset scales the restored optimizer and scheduler rates by 0.5.

Validate and print the effective topology:

```bash
python - "$PROFILE" <<'PY'
import sys
from startrain.config import load_config

config = load_config(sys.argv[1])
print("Profile:", config.profile)
print("Run ID:", config.orchestration.run_id)
print("Run root:", config.orchestration.directories.root)
print("GPU roles:", [(g.gpu_id, g.role) for g in config.orchestration.gpus])
print("Arena GPU:", config.orchestration.promotion.gpu_id)
print("Self-play source:", config.orchestration.model_refresh.selfplay_source)
print("Precision:", config.train.precision)
print("Compile:", config.train.compile)
PY
```

Freeze the profile after validation:

```bash
chmod 0444 "$PROFILE"
sha256sum "$PROFILE" | tee "$RUN_ROOT/profile.sha256"
git rev-parse HEAD | tee "$RUN_ROOT/source-commit.txt"
python -m pip freeze | tee "$RUN_ROOT/python-environment.txt"
nvidia-smi -q > "$RUN_ROOT/nvidia-smi-before.txt"
```

If a profile change is required, create a new profile and normally a new run
root. Do not silently change an active run's experiment definition.

For `h100-8gpu-autonomous.yaml`, the selected run root may contain the frozen
profile and operator evidence, but must not contain `run.json`, replay shards,
model pointers, manifests, checkpoints, or arena results at first launch. The
coordinator writes `autonomous-provenance.json` immediately after `run.json`.
Every resume verifies the run identity and complete frozen-profile digest.
Never copy a prior checkpoint or replay ledger into an autonomous root.

## 9. Run deterministic CPU/native validation

```bash
cd "$HOME/EdgeConnect/training"
source .venv/bin/activate

cargo +1.93.0 fmt --all --check
cargo +1.93.0 clippy --workspace --all-targets --locked -- -D warnings
cargo +1.93.0 test --workspace --locked

python -m ruff check startrain starserve tests scripts
python -m ruff format --check startrain starserve tests scripts
python -m pyright
python -m pytest \
  --require-native \
  -m "not cuda and not multi_gpu and not soak"
```

Also measure the native feature path:

```bash
python scripts/benchmark_native_features.py \
  --rings 6 \
  --batch-size 256
```

The benchmark must report exact parity and a native path speedup. A missing
`star_native` module or skipped native suite is a hard failure.

## 10. Run one-GPU CUDA validation

```bash
python -m pytest \
  --require-native \
  -m "cuda and not multi_gpu and not soak"
```

This checks BF16, compilation, forward/backward execution, finite gradients,
and repeated inference memory behavior.

Run the production-boundary benchmark at representative board sizes:

```bash
python scripts/hardware_preflight.py \
  --config "$PROFILE" \
  --device cuda:0 \
  --rings 6 \
  --batch-size 64

python scripts/hardware_preflight.py \
  --config "$PROFILE" \
  --device cuda:0 \
  --rings 10 \
  --batch-size 64
```

The benchmark includes native-state decoding, schema-v3 features, host/device
transfer, compiled BF16 model execution, and legal-policy return. Both commands
must exit zero and meet the configured threshold of at least 5,000 realistic
leaf evaluations per second per H100.

For a replacement run, also benchmark a production-sized shard with
`scripts/benchmark_replay_pipeline.py`. The learner soak must report less than
10% data-wait time and at least 85% GPU-0 duty after compile warm-up.
`examples_per_second` is end-to-end wall throughput;
`device_examples_per_second` isolates GPU execution.

Keep the emitted JSON in the run root:

```bash
python scripts/hardware_preflight.py \
  --config "$PROFILE" \
  --device cuda:0 \
  --rings 6 \
  --batch-size 64 \
  | tee "$RUN_ROOT/hardware-preflight-r6.json"

python scripts/hardware_preflight.py \
  --config "$PROFILE" \
  --device cuda:0 \
  --rings 10 \
  --batch-size 64 \
  | tee "$RUN_ROOT/hardware-preflight-r10.json"
```

For a repeated ring/batch matrix with immutable JSON/JSONL evidence, run:

```bash
python scripts/h100_system_benchmark.py \
  --config "$PROFILE" \
  --output-dir "$RUN_ROOT/system-benchmark" \
  --rings 6 10 \
  --batch-sizes 64 128 256 \
  --repeats 3 \
  --metrics-root "$RUN_ROOT"
```

Do not start the long run if either gate fails. Capture the JSON and profiler
evidence before changing batch size, model size, compile settings, or search
parameters.

## 11. Verify two-GPU NCCL

The shipped run does not use multi-GPU DDP, but proving the host interconnect
and collective stack is still a valuable preflight:

```bash
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
torchrun \
  --standalone \
  --nproc-per-node 2 \
  scripts/nccl_smoke.py \
  --all-rings \
  --config "$PROFILE" \
  | tee "$RUN_ROOT/nccl-smoke.json"
```

This command must report matching post-step parameters across both ranks.
If it fails, fix driver, NCCL, topology, firewall, or container/runtime issues
before enabling any custom multi-learner profile.

## 12. First launch: use tmux

For the first target-host soak, use `tmux` so the operator can inspect the
coordinator directly while surviving SSH disconnects.

From `training/`:

```bash
cd "$HOME/EdgeConnect/training"
source .venv/bin/activate

tmux new-session -d -s "$RUN_ID" \
  "bash -lc 'set -o pipefail; cd \"$PWD\" && source .venv/bin/activate && startrain-orchestrate --config \"$PROFILE\" 2>&1 | tee \"$RUN_ROOT/coordinator-console.log\"'"
```

Confirm that the process started:

```bash
tmux list-sessions
tmux capture-pane -pt "$RUN_ID" -S -100
```

Attach interactively:

```bash
tmux attach -t "$RUN_ID"
```

Detach without stopping training by pressing `Ctrl-B`, then `D`.

The orchestrator itself creates `<run-root>/run.json`. Do not create
`run.json` manually and do not pass a separate run identity on the
orchestrator command line.

## 13. Long unattended launch: systemd

After a successful tmux soak, systemd is preferable for a multi-day run because
it restarts the coordinator after a process failure and starts it after host
reboot.

The repository includes `deploy/edgeconnect-startrain.service.example`.
Generate a host-specific unit:

```bash
cd "$HOME/EdgeConnect/training"

sed \
  -e "s|@USER@|$USER|g" \
  -e "s|@TRAINING_DIR@|$PWD|g" \
  -e "s|@PROFILE@|$PROFILE|g" \
  -e "s|@RUN_ROOT@|$RUN_ROOT|g" \
  deploy/edgeconnect-startrain.service.example \
  > "/tmp/edgeconnect-startrain-$RUN_ID.service"

sudo install -m 0644 \
  "/tmp/edgeconnect-startrain-$RUN_ID.service" \
  "/etc/systemd/system/edgeconnect-startrain-$RUN_ID.service"

sudo systemctl daemon-reload
sudo systemctl enable --now "edgeconnect-startrain-$RUN_ID.service"
```

For an unlimited profile, generate the unit from
`deploy/edgeconnect-startrain-continuous.service.example`. It validates the
continuous settings, fails closed on configured GPU health before replay
recovery, uses `Restart=always`, disables systemd's start-burst
cutoff, and feeds a watchdog from the coordinator loop. A deliberate
`systemctl stop` still suppresses restart. Finite profiles must use the regular
`Restart=on-failure` template. Install the replay-ledger backup service/timer
from `deploy/edgeconnect-startrain-backup.*.example` with the same `@USER@`,
`@TRAINING_DIR@`, `@RUN_ROOT@`, and `@RUN_ID@` substitutions.
For autonomous runs, also install
`deploy/edgeconnect-startrain-report.{service,timer}.example`, replace
`@PROVISIONED_GPUS@`, and enable the timer to refresh
`strength-efficiency.json` every 15 minutes.

The GPU gate is an `ExecCondition`: an unhealthy or unqueryable device leaves
the unit inactive instead of entering a restart loop. Runtime health failure
uses coordinator exit status 78, which both templates explicitly exclude from
restart. An operator must re-run the health gate and start the unit after
remediation.

Keep the template's `KillMode=mixed`: systemd sends the graceful signal only to
the coordinator, allowing it to unwind learner DataLoader children and actor
search waves, while still hard-killing the complete cgroup after
`TimeoutStopSec` if graceful shutdown fails.

Inspect it:

```bash
systemctl status "edgeconnect-startrain-$RUN_ID.service"
journalctl -u "edgeconnect-startrain-$RUN_ID.service" -f
```

Do not run tmux and systemd coordinators against the same run root at the same
time.

Do not `git pull`, switch branches, rebuild the editable environment, or modify
source inside an active unit's `WorkingDirectory`. Actor and coordinator
restarts import from that physical checkout. Fetch updates into a separate
detached worktree and use that worktree only for monitoring or a future run.

## 14. Expected startup lifecycle

Normal startup order:

1. The coordinator creates/reuses the immutable `run.json`.
2. The learner publishes an initial EMA candidate.
3. The promotion supervisor bootstraps that candidate as `champion.json`
   because the shipped profiles explicitly enable bootstrap.
4. Actor processes select the configured champion/candidate/history source only
   between complete game batches. Every history model belongs to the same run.
5. Actors produce ring-homogeneous replay shards and commit them to SQLite.
6. The learner waits until replay count and per-ring uniqueness gates pass.
7. The learner begins BF16 compiled updates and publishes later promotion
   candidates at the frozen profile's cadence: 5,000 steps in the historical
   control, 15,000 in the throughput profile, or
   `candidate_interval_examples` in the autonomous profile. Autonomous actors
   refresh from separate self-play snapshots every one million examples during
   warmup and every three million afterward.
8. The arena accumulates paired, role-reversed games and applies the
   anytime-valid promotion gate.

The learner can legitimately remain in `replay_wait` while actors generate
data. The curriculum initially permits ring 4, then rings 4 and 6 until total
samples reach one million, and only then opens rings 4/6/8/10. Since the learner
requires a minimum unique count on every ring, initial training can wait beyond
the nominal aggregate replay minimum. Actor progress during that period is the
important health signal.

GPU 7 in the 8-GPU profile (GPU 3 in the 4-GPU profile) can be mostly idle
between arena evaluations. That is expected.

## 15. Monitor the run

Re-establish the run path in every shell:

```bash
export RUN_ROOT="/mnt/nvme/edgeconnect/<run-id>"
```

For a detachable, once-per-minute operator summary, use the read-only monitor
from a checkout that is not modified by the active run:

```bash
export UNIT="edgeconnect-startrain-<run-id>.service"
export MONITOR_TRAINING="$HOME/edgeconnect-releases/main-<sha>/training"
export MONITOR_PYTHON="$HOME/edge-connect-local/training/.venv/bin/python"
export MONITOR_SESSION="startrain-monitor-<run-id>"
export MONITOR_LOG="$RUN_ROOT/operator-monitor.log"

screen -DmS "$MONITOR_SESSION" bash -lc '
  set -o pipefail
  cd "$1"
  "$2" -u scripts/monitor_run.py \
    --run-root "$3" \
    --profile "$4" \
    --unit "$5" \
    --interval 60 \
    --format text 2>&1 |
  tee -a "$6"
' monitor "$MONITOR_TRAINING" "$MONITOR_PYTHON" \
  "$RUN_ROOT" "$PROFILE" "$UNIT" "$MONITOR_LOG"
```

Inspect and detach:

```bash
screen -ls
screen -r "$MONITOR_SESSION"
# Detach interactively with Ctrl-A, then D.
tail -F "$MONITOR_LOG"
```

Stop only the monitor with:

```bash
screen -S "$MONITOR_SESSION" -X quit
```

For structured ingestion, use `--format jsonl`. A one-shot status check is:

```bash
"$MONITOR_PYTHON" -u "$MONITOR_TRAINING/scripts/monitor_run.py" \
  --run-root "$RUN_ROOT" --profile "$PROFILE" --unit "$UNIT" --once
```

GPU health:

```bash
watch -n 2 nvidia-smi
```

Coordinator state:

```bash
watch -n 5 "jq . '$RUN_ROOT/status/coordinator.json'"
```

Promotion state:

```bash
watch -n 10 "jq . '$RUN_ROOT/arena/promotion-status.json'"
```

All process logs:

```bash
tail -F "$RUN_ROOT"/logs/*.log
```

Learner and actor metrics:

```bash
tail -F \
  "$RUN_ROOT/learner/metrics.jsonl" \
  "$RUN_ROOT"/metrics/*.jsonl
```

Replay ledger:

```bash
sqlite3 "$RUN_ROOT/replay/manifest.sqlite3" \
  "select ring, state, count(*) as shards, sum(sample_count) as samples
   from shards
   group by ring, state
   order by ring, state;"
```

Recent game uniqueness:

```bash
sqlite3 "$RUN_ROOT/replay/manifest.sqlite3" \
  "select count(*) as completed_games, count(distinct game_id) as unique_games
   from games;"
```

Storage:

```bash
watch -n 30 "df -h '$RUN_ROOT'; du -sh '$RUN_ROOT'"
```

Useful metric expectations:

- actor `feature_path` should be `rust`;
- learner replay batches should report the native/Rust feature path;
- actor metrics include games/s, samples/s, search simulations/s,
  policy-supervision rate, model role, and model identity;
- the monitor headline actor rates use completed counter deltas over merged
  physical-GPU wall intervals; `latest_batch_rate_sum` is legacy diagnostic
  data and must not be used for capacity or Elo/hour decisions;
- learner metrics include examples/s, step time, loss heads, gradient norm,
  replay counts, and model step;
- shipped profiles report `model_role: champion`;
- policy-supervision rate is expected to be near the configured full-search
  fraction because `record_fast_policy_targets` is false by default.

## 16. Early acceptance gates

Do not extrapolate the full run before measuring real data:

1. Before launch: at least 5,000 realistic leaf evaluations/s/H100 at rings 6
   and 10.
2. After 10,000 games: record game length, search mix, CPU
   saturation, ring distribution, replay growth, and GPU utilization.
3. By 100,000 games: beat random/greedy nearly perfectly and beat a fixed
   shallow search convincingly on every active ring.
4. By 500,000 games: run at least 200 paired arena games per ring and fit
   strength gain against log training volume.
5. Before a superhuman claim: use one checkpoint across all board sizes,
   balanced roles, blind matches, and the strongest available human players.

Current rough capacity estimates:

- 8 H100s: 9–16 days base, 3–6 days optimistic.
- 4 H100s: 18–28 days base, 5–8 days optimistic.

These carry approximately 3–5× uncertainty. Treat them as planning ranges, not
deadlines or strength guarantees.

## 17. Graceful stop

Never use `kill -9` during normal operation.

For tmux:

```bash
tmux send-keys -t "$RUN_ID" C-c
```

For systemd:

```bash
sudo systemctl stop "edgeconnect-startrain-$RUN_ID.service"
```

To keep a continuous unit stopped across reboot:

```bash
sudo systemctl disable --now "edgeconnect-startrain-$RUN_ID.service"
sudo systemctl disable --now "edgeconnect-startrain-$RUN_ID-backup.timer"
```

The profile allows up to 900 seconds for worker termination and complete-cohort
drain. Wait for the coordinator to exit before considering the stop complete.

Verify:

```bash
jq . "$RUN_ROOT/status/coordinator.json"
test -f "$RUN_ROOT/learner/candidate.json"
sqlite3 "$RUN_ROOT/replay/manifest.sqlite3" \
  "pragma integrity_check;"
```

## 18. Resume after SSH loss, process failure, or reboot

An SSH disconnect does not affect tmux or systemd. If the coordinator actually
exited, inspect:

```bash
jq . "$RUN_ROOT/status/coordinator.json"
tail -n 200 "$RUN_ROOT"/logs/*.log
```

Correct the external cause, then start the exact same profile against the exact
same run root:

```bash
cd "$HOME/EdgeConnect/training"
source .venv/bin/activate

export RUN_ID="<existing-run-id>"
export RUNS_BASE="/mnt/nvme/edgeconnect"  # same mount used at creation
export RUN_ROOT="$RUNS_BASE/$RUN_ID"
export PROFILE="$RUN_ROOT/profile.yaml"

tmux new-session -d -s "$RUN_ID" \
  "bash -lc 'set -o pipefail; cd \"$PWD\" && source .venv/bin/activate && startrain-orchestrate --config \"$PROFILE\" 2>&1 | tee \"$RUN_ROOT/coordinator-console.log\"'"
```

The coordinator reuses `run.json`, reconciles replay files, and resumes the
highest-step valid state. Resolution checks the recovery head and journal,
candidate pointer and immutable manifest history, then the champion as a last
resort. Every checkpoint is checked for size, SHA-256, run identity, generation
family, model, and rules compatibility. Rejected artifacts are recorded in
learner metrics instead of trapping the service in a crash loop.

The backup timer uses SQLite's online backup API. Before service start,
`replay_manifest_backup.py check-and-restore` verifies the replay ledger; if it
is corrupt, it preserves the damaged DB/WAL and restores only a verified recent
backup. Replay reconciliation then handles shards newer than that backup.

Do not delete `coordinator.lock` while a coordinator PID is alive. The
coordinator removes an abandoned lock only when the recorded PID is no longer
live.

For indefinite operation, review at least one retention dry-run event before
setting `orchestration.retention.dry_run: false`. Active replay watermarks,
current candidate/champion, arena-referenced manifests, and recent recovery
checkpoints remain protected. Whole-volume loss and persistent GPU/driver
failure still require operator intervention.

## 19. Troubleshooting

### Aggregate SRAM threshold or uncorrectable ECC

Treat `SRAM Threshold Exceeded: Yes`, any volatile uncorrectable ECC, pending
repair/remap, or a non-`None` recovery action as a blocking hardware incident.
Capture `nvidia-smi -q -x`, kernel and service journals, current status,
recovery pointers, and a verified replay-ledger backup. Disable the report and
backup timers, then stop and disable the coordinator through systemd and allow
the full graceful timeout. Do not resume on that GPU because volatile counters
returned to zero.

Before a provider node or GPU swap, copy the incident bundle, current recovery
checkpoint, and verified replay-ledger backup off-host and verify their
recorded SHA-256 digests. Confirm whether the run volume survives a full node
replacement; a ledger without its replay shards is not a complete run backup.

Request provider/NVIDIA Field Diagnostic or DCGM validation and replacement
review. Resume only after eight full H100s pass the health gate, ring-6/ring-10
CUDA preflight, topology checks, and NCCL smoke. A seven-GPU degraded resume is
not supported for an initialized autonomous run.

### `torch.cuda.is_available()` is false

Check:

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

Likely causes are a CPU-only wheel, driver/runtime incompatibility, missing GPU
visibility, or a container/runtime configuration problem. Reinstall the
appropriate PyTorch build; do not modify training code.

### `import star_native` fails

Rebuild in the active environment:

```bash
source .venv/bin/activate
maturin develop \
  --release \
  --locked \
  --manifest-path crates/star-py/Cargo.toml
python -c "import star_native; print(star_native.native_rules_hash_tag())"
```

### Hardware preflight is below 5,000 leaf evaluations/s/H100

Do not start the long run. Save the JSON and inspect CPU saturation, native
feature path, batch occupancy, compilation, PCIe/NVLink topology, clocks,
thermals, and host/device copies. Change one factor at a time in a separate
profile and follow `training-ablation-protocol.md`.

### Actors are waiting for `champion.json`

Inspect learner and promotion logs:

```bash
tail -n 200 "$RUN_ROOT/logs/learner"*.log
tail -n 200 "$RUN_ROOT/logs/"*promotion*.log
ls -l "$RUN_ROOT/learner/candidate.json" "$RUN_ROOT/learner/champion.json"
```

The initial candidate must exist and the promotion supervisor must bootstrap
it. Do not manually copy or edit model pointers.

### Learner remains in `replay_wait`

Check actor heartbeats, actor logs, and per-ring replay counts. During initial
curriculum fill, replay wait is expected. It is unhealthy only when actors are
not advancing, shards are quarantined, or required rings remain unavailable
after the curriculum should have opened them.

### CUDA out of memory

Capture the worker, ring, batch, allocated/reserved memory, and profile. Stop
gracefully. Reduce only the responsible per-rank learner batch or actor batch
in a new profile; do not partially edit an active run definition.

### Stale heartbeat or stall timeout

The coordinator automatically restarts workers with bounded exponential
backoff. Inspect the named worker log before restarting the whole coordinator.
Common external causes are CUDA OOM, driver reset, full disk, blocked I/O, or a
failed model artifact.

### Replay corruption or missing shard

On reopen, committed corrupt/missing shards are quarantined and orphan files
are reconciled. Inspect:

```bash
sqlite3 "$RUN_ROOT/replay/manifest.sqlite3" \
  "select id, relative_path, state, quarantine_reason
   from shards
   where state != 'ready';"
```

Do not delete ledger rows or manually rewrite checksums.

### Disk usage grows continuously

The shipped retention policy has `dry_run: true`. Inspect learner replay-GC
metrics before enabling deletion. Keep the dry run during the initial soak and
provision enough disk.

### Arena takes a long time

Arena uses 1,024 simulations, all rings, role-reversed pairs, repeated looks,
and per-ring regression floors. The arena GPU being busy for an extended
period can be normal. Check that pair counts and promotion status continue to
advance.

### NCCL smoke fails

The shipped single-learner production profile can still run without NCCL, but
do not enable custom DDP. Check driver versions, GPU topology, NCCL environment,
shared memory, and firewall/network interfaces first.

## 20. Controlled migration of an initialized autonomous run

Prefer a new run root for treatment changes. If an initialized autonomous run
must continue across a code/profile migration, preserve an explicit treatment
boundary instead of editing provenance by hand:

1. Build and test a new immutable release directory while the old unit remains
   active.
2. Wait for any arena promotion wave to persist, create a verified online replay
   backup, and archive the profile, provenance, cadence, recovery pointer, and
   systemd units.
3. Stop the coordinator through systemd and allow the configured graceful
   timeout. Confirm the coordinator lock owner is no longer live.
4. Run `scripts/migrate_autonomous_profile.py` in dry-run mode, inspect the
   allowed diff and computed hashes, then repeat with `--apply`.
5. Repoint the training, report, and backup units to the new immutable release
   and newly named frozen profile. Never modify the previous profile in place.
6. Start the coordinator and verify recovery step, examples consumed, cadence,
   replay counts, worker heartbeats, and the migration-segment UTD ratio.

An update-to-data change must be prospective. The migration utility writes
`learner/utd-segment.json` with the current committed replay and consumed-example
baselines, preventing a new ratio from retroactively authorizing updates over
the full historical ledger.

For the Elo-per-wall-clock profile, keep candidate and self-play publication
cadence constant per newly generated replay sample when changing UTD. At UTD
1.25, scale the 5M candidate and 3M steady self-play intervals to 6.25M and
3.75M learner examples respectively.

Promotion may use five-pair waves with a fifteen-pair minimum only while the
existing anytime-valid pair e-process, error levels, search budget, deterministic
opening schedule, and 200-pair maximum remain unchanged. Persist every wave so
the arena can resume after a stop.
`arena.continuation_pairs_per_ring` may increase only post-minimum continuation
batches; it must not alter the initial minimum look or any statistical gate.

Benchmark actor compile modes on the target H100 before selecting one. Compare
the current dynamic full-graph path with static `reduce-overhead` and
`max-autotune`; require fixed-position output parity and a material throughput
win. Keep the current path when the gate is inconclusive.

Rollback means stopping the new unit, restoring the archived profile/provenance
bundle and old immutable release paths, then resuming from the still-valid
recovery pointer. Do not delete post-migration artifacts during rollback.

## 21. Actions that are forbidden during an active run

Do not:

- run two coordinators against the same run root;
- recreate or edit `run.json`;
- manually repoint `candidate.json` or `champion.json`;
- feed actors raw `.pt` files;
- mix replay from different run identities or schema generations;
- edit the frozen profile in place;
- enable retention deletion before reviewing dry-run metrics;
- use `kill -9` except as a last resort after graceful timeouts;
- infer H100 readiness from CPU tests;
- infer superhuman strength from training loss or promotion alone.

## 22. Run artifacts to preserve

Archive:

- frozen YAML profile and SHA-256;
- source commit;
- Python environment;
- pre/post `nvidia-smi -q`;
- hardware and NCCL preflight JSON;
- `run.json`;
- candidate/champion pointers and immutable manifests;
- checkpoints;
- arena results and promotion status;
- learner and actor metrics;
- coordinator/worker logs;
- replay manifest and any quarantine records.

Keep `run.json`, manifests, pointers, and checkpoints together so relative
artifact references remain valid.

## 23. After training

The output is not automatically a browser model:

1. Confirm the final candidate has a terminal arena decision.
2. Use `learner/champion.json` for private `starserve`.
3. Run external strength evaluation across all rings.
4. Distill the validated champion for browser inference.
5. Build Rust/WASM artifacts.
6. Publish ONNX, WASM, and the browser manifest atomically.

Continue with:

- [Model serving and browser distillation](serving-and-distillation.md)
- [Training ablation protocol](training-ablation-protocol.md)
- [Testing and H100 validation](testing-and-h100-validation.md)

## Final launch checklist

Before running `startrain-orchestrate`, confirm all boxes:

- [ ] intended source commit recorded;
- [ ] every GPU ID selected by the 4- or 8-GPU profile is a full H100 and
      visible in the expected order;
- [ ] driver compatible with the selected PyTorch CUDA runtime;
- [ ] BF16 supported;
- [ ] local NVMe root writable with sufficient bytes and inodes;
- [ ] Rust 1.93 and Python 3.11 active;
- [ ] `star_native` release extension built and rules hash verified;
- [ ] CPU/native tests pass without skips;
- [ ] CUDA smoke passes;
- [ ] ring-6 and ring-10 hardware preflight exceeds the throughput floor;
- [ ] NCCL smoke passes or custom DDP is explicitly out of scope;
- [ ] copied profile has a unique run ID and absolute run root;
- [ ] profile, source commit, environment, and preflight evidence archived;
- [ ] only one coordinator will own the run root;
- [ ] operator knows the monitoring and graceful-stop commands.

