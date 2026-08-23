#!/usr/bin/env python3
"""Generate frozen, one-factor H100 Elo-ablation profiles and manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import stat
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from startrain.config import ExperimentConfig, load_config
from startrain.model import model_parameter_count

if __package__:
    from .validate_continuous_profile import validate_continuous_config
else:
    from validate_continuous_profile import validate_continuous_config

SCHEMA_VERSION = 1
REPORT_NAME = "startrain-elo-ablation-plan"
DEFAULT_TREATMENTS = (
    "control",
    "utd-1",
    "plateau-keep",
    "freshness-mix",
    "ring10-70",
    "search-quality",
)
SYSTEM_TREATMENTS = (
    "actor-batch-128",
    "actor-batch-160",
    "actor-batch-192",
    "actor-lanes-3",
    "learner-batch-768",
    "learner-batch-1024",
)
CLEAN_TREATMENTS = (
    "lr-quarter",
    "fresh-source",
    "hard-replay",
    "fresh-hard",
)
WEIGHTED_TREATMENTS = (
    "weighted-control",
    "ring10-65-weighted",
    "ring10-70-weighted",
)
RING10_EFFICIENCY_TREATMENTS = (
    "ring10-only",
    "ring10-learner-slack-64",
    "ring10-actor-lanes-3",
)
RING10_OPTIMIZATION_TREATMENTS = (
    "ring10-optimization-control",
    "ring10-cadence-5m",
    "ring10-freshness-50",
)
RING10_LIVE_CADENCE_TARGET_EXAMPLES = 5_000_000
RING10_LIVE_CADENCE_TREATMENTS = (
    "ring10-live-cadence-control",
    "ring10-live-cadence-5m",
)
RING10_TRAINING_DYNAMICS_TREATMENTS = (
    "ring10-dynamics-control",
    "ring10-dynamics-adamw",
    "ring10-dynamics-ema-1m",
    "ring10-dynamics-freshness-50",
    "ring10-dynamics-clinch-outcome-only",
)
RING10_ATTENTION_TREATMENTS = (
    "ring10-attention-control",
    "ring10-attention-full-kv",
)
RING10_RELATIONAL_TREATMENTS = (
    "ring10-relational-control",
    "ring10-relational-local-heavy",
    "ring10-relational-source-gated",
)
RING10_CAPACITY_TREATMENTS = (
    "ring10-capacity-control",
    "ring10-capacity-depth-7",
    "ring10-capacity-width-512",
)
RING10_ONLY_TREATMENTS = ("ring10-only",)
RING10_OBJECTIVE_TREATMENTS = frozenset(
    (
        *RING10_EFFICIENCY_TREATMENTS,
        *RING10_OPTIMIZATION_TREATMENTS,
        *RING10_LIVE_CADENCE_TREATMENTS,
        *RING10_TRAINING_DYNAMICS_TREATMENTS,
        *RING10_ATTENTION_TREATMENTS,
        *RING10_RELATIONAL_TREATMENTS,
        *RING10_CAPACITY_TREATMENTS,
    )
)
RING10_EFFICIENCY_VARIANTS = frozenset(RING10_EFFICIENCY_TREATMENTS[1:])
TREATMENT_SUITES = {
    "ring10-efficiency": RING10_EFFICIENCY_TREATMENTS,
    "ring10-optimization": RING10_OPTIMIZATION_TREATMENTS,
    "ring10-live-cadence": RING10_LIVE_CADENCE_TREATMENTS,
    "ring10-training-dynamics": RING10_TRAINING_DYNAMICS_TREATMENTS,
    "ring10-attention-reallocation": RING10_ATTENTION_TREATMENTS,
    "ring10-relational": RING10_RELATIONAL_TREATMENTS,
    "ring10-capacity": RING10_CAPACITY_TREATMENTS,
}
SCRATCH_INITIALIZATION_TREATMENTS = frozenset(
    (
        *RING10_ATTENTION_TREATMENTS,
        *RING10_RELATIONAL_TREATMENTS,
        *RING10_CAPACITY_TREATMENTS,
    )
)
EXPECTED_ARCHITECTURE_PARAMETERS = {
    "ring10-attention-control": 10_476_983,
    "ring10-attention-full-kv": 10_476_983,
    "ring10-relational-control": 10_476_983,
    "ring10-relational-local-heavy": 10_476_953,
    "ring10-relational-source-gated": 10_476_983,
    "ring10-capacity-control": 10_476_983,
    "ring10-capacity-depth-7": 14_614_199,
    "ring10-capacity-width-512": 18_556_727,
}
GUARD_RINGS = (4, 6, 8)
WEIGHTED_PROMOTION_PAIR_RATIOS = {4: 1, 6: 1, 8: 1, 10: 7}
WEIGHTED_INITIAL_BLOCKS = 15
WEIGHTED_CONTINUATION_BLOCKS = 10
WEIGHTED_MAX_BLOCKS = 50

RawConfig = dict[str, Any]
Treatment = Callable[[RawConfig], None]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-root-parent", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-run-root", type=Path, required=True)
    parser.add_argument(
        "--winner-snapshot",
        type=Path,
        help="verified upstream comparator snapshot required for staged preparation",
    )
    parser.add_argument(
        "--futility-policy",
        type=Path,
        help="pre-registered stop-only anytime-valid futility policy",
    )
    parser.add_argument("--prefix", default="ring10-elo")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--wall-budget-hours", type=float, default=8.0)
    parser.add_argument("--leaf-budget", type=int, default=2_000_000_000)
    parser.add_argument("--guard-floor-elo", type=float, default=-35.0)
    parser.add_argument(
        "--suite",
        choices=tuple(TREATMENT_SUITES),
        help="pre-registered treatment suite; cannot be combined with --treatment",
    )
    parser.add_argument(
        "--treatment",
        action="append",
        choices=(
            *DEFAULT_TREATMENTS,
            *SYSTEM_TREATMENTS,
            *CLEAN_TREATMENTS,
            *WEIGHTED_TREATMENTS,
            *RING10_EFFICIENCY_TREATMENTS,
            *RING10_OPTIMIZATION_TREATMENTS,
            *RING10_LIVE_CADENCE_TREATMENTS,
            *RING10_TRAINING_DYNAMICS_TREATMENTS,
            *RING10_ATTENTION_TREATMENTS,
            *RING10_RELATIONAL_TREATMENTS,
            *RING10_CAPACITY_TREATMENTS,
        ),
        dest="treatments",
    )
    return parser


def _mapping(config: RawConfig, name: str) -> RawConfig:
    value = config.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _control(_config: RawConfig) -> None:
    return


def _utd_one(config: RawConfig) -> None:
    _mapping(config, "learner")["target_updates_per_new_sample"] = 1.0


def _plateau_keep(config: RawConfig) -> None:
    plateau = _mapping(_mapping(config, "orchestration"), "plateau")
    plateau["action"] = "reduce_lr_keep_weights"
    plateau["clear_optimizer_state_on_recovery"] = True


def _freshness_mix(config: RawConfig) -> None:
    learner = _mapping(config, "learner")
    learner.update(
        {
            "selfplay_snapshot_interval_examples": 3_000_000,
            "selfplay_snapshot_warmup_examples": 20_000_000,
            "selfplay_snapshot_warmup_interval_examples": 1_000_000,
        }
    )
    refresh = _mapping(_mapping(config, "orchestration"), "model_refresh")
    refresh.update(
        {
            "selfplay_source": "candidate_champion_history_mix",
            "candidate_probability": 0.35,
            "history_probability": 0.15,
            "history_pool_size": 8,
        }
    )


def _ring_ten_seventy(config: RawConfig) -> None:
    mixture = _mapping(_mapping(config, "orchestration"), "ring_mixture")
    mixture["step_weights"] = [{"from_step": 0, "weights": [0.10, 0.10, 0.10, 0.70]}]


def _weighted_promotion_objective(config: RawConfig) -> None:
    arena = _mapping(config, "arena")
    arena.update(
        {
            "promotion_pair_ratios": dict(WEIGHTED_PROMOTION_PAIR_RATIOS),
            "required_regression_rings": [],
            "weighted_initial_blocks": WEIGHTED_INITIAL_BLOCKS,
            "weighted_continuation_blocks": WEIGHTED_CONTINUATION_BLOCKS,
            "weighted_max_blocks": WEIGHTED_MAX_BLOCKS,
            # The arena still emits every per-ring confidence sequence. An empty
            # map removes the ablation's hard -35 Elo overrides; the explicit
            # required_regression_rings list controls whether diagnostics gate.
            "per_ring_regression_floor_elo": {},
        }
    )


def _weighted_control(config: RawConfig) -> None:
    _weighted_promotion_objective(config)


def _ring_ten_sixty_five_weighted(config: RawConfig) -> None:
    mixture = _mapping(_mapping(config, "orchestration"), "ring_mixture")
    mixture["step_weights"] = [{"from_step": 0, "weights": [0.10, 0.10, 0.15, 0.65]}]
    _weighted_promotion_objective(config)


def _ring_ten_seventy_weighted(config: RawConfig) -> None:
    _ring_ten_seventy(config)
    _weighted_promotion_objective(config)


def _ring_ten_only(config: RawConfig) -> None:
    orchestration = _mapping(config, "orchestration")
    orchestration["training_objective"] = "ring10_only"
    mixture = _mapping(orchestration, "ring_mixture")
    mixture["step_weights"] = [{"from_step": 0, "weights": [0.0, 0.0, 0.0, 1.0]}]
    _mapping(config, "selfplay")["rings"] = 10
    arena = _mapping(config, "arena")
    arena["rings"] = [10]
    arena["required_regression_rings"] = []
    arena["per_ring_regression_floor_elo"] = {}
    for name in (
        "promotion_pair_ratios",
        "weighted_initial_blocks",
        "weighted_continuation_blocks",
        "weighted_max_blocks",
    ):
        arena.pop(name, None)


def _ring_ten_learner_slack(config: RawConfig) -> None:
    _ring_ten_only(config)
    _learner_slack_actor(config)


def _ring_ten_actor_lanes_three(config: RawConfig) -> None:
    _ring_ten_only(config)
    _actor_lanes_three(config)


def _ring_ten_optimization_control(config: RawConfig) -> None:
    _ring_ten_only(config)
    learner = _mapping(config, "learner")
    learner["target_updates_per_new_sample"] = 1.0
    learner["candidate_interval_examples"] = 2_000_000
    refresh = _mapping(_mapping(config, "orchestration"), "model_refresh")
    refresh.update(
        {
            "selfplay_source": "champion",
            "history_probability": 0.0,
        }
    )


def _ring_ten_cadence_five_million(config: RawConfig) -> None:
    _ring_ten_optimization_control(config)
    _mapping(config, "learner")["candidate_interval_examples"] = 5_000_000


def _ring_ten_live_cadence_control(_config: RawConfig) -> None:
    return


def _ring_ten_live_cadence_five_million(config: RawConfig) -> None:
    _mapping(config, "learner")["candidate_interval_examples"] = (
        RING10_LIVE_CADENCE_TARGET_EXAMPLES
    )


def _ring_ten_freshness_fifty(config: RawConfig) -> None:
    _ring_ten_optimization_control(config)
    learner = _mapping(config, "learner")
    learner.update(
        {
            "selfplay_snapshot_interval_examples": 3_000_000,
            "selfplay_snapshot_warmup_examples": 20_000_000,
            "selfplay_snapshot_warmup_interval_examples": 1_000_000,
        }
    )
    refresh = _mapping(_mapping(config, "orchestration"), "model_refresh")
    refresh.update(
        {
            "selfplay_source": "candidate_champion_mix",
            "candidate_probability": 0.5,
            "history_probability": 0.0,
        }
    )


def _ring_ten_dynamics_control(config: RawConfig) -> None:
    _ring_ten_optimization_control(config)
    _mapping(config, "optimizer")["kind"] = "muon_adamw"
    _mapping(config, "train")["ema_half_life_examples"] = None
    _mapping(config, "selfplay")["clinch_auxiliary_targets"] = "synthetic"
    refresh = _mapping(_mapping(config, "orchestration"), "model_refresh")
    refresh.update(
        {
            "selfplay_source": "champion",
            "candidate_probability": 0.0,
            "history_probability": 0.0,
        }
    )


def _ring_ten_dynamics_adamw(config: RawConfig) -> None:
    _ring_ten_dynamics_control(config)
    _mapping(config, "optimizer")["kind"] = "adamw"


def _ring_ten_dynamics_ema_one_million(config: RawConfig) -> None:
    _ring_ten_dynamics_control(config)
    _mapping(config, "train")["ema_half_life_examples"] = 1_000_000


def _ring_ten_dynamics_freshness_fifty(config: RawConfig) -> None:
    _ring_ten_dynamics_control(config)
    _mapping(_mapping(config, "orchestration"), "model_refresh").update(
        {
            "selfplay_source": "candidate_champion_mix",
            "candidate_probability": 0.5,
            "history_probability": 0.0,
        }
    )


def _ring_ten_dynamics_clinch_outcome_only(config: RawConfig) -> None:
    _ring_ten_dynamics_control(config)
    _mapping(config, "selfplay")["clinch_auxiliary_targets"] = "outcome_only"


def _ring_ten_attention_control(config: RawConfig) -> None:
    _ring_ten_optimization_control(config)
    model = _mapping(config, "model")
    model.update(
        {
            "width": 384,
            "rrt_groups": 5,
            "attention_heads": 12,
            "kv_heads": 3,
            "bottleneck_ratio": 0.5,
            "ff_multiplier": 2.5,
            "dropout": 0.0,
            "rms_norm_eps": 1e-6,
            "local_operator": "mean",
            "local_blocks_per_group": 2,
        }
    )


def _ring_ten_attention_full_kv(config: RawConfig) -> None:
    _ring_ten_attention_control(config)
    _mapping(config, "model").update(
        {
            "kv_heads": 12,
            "ff_multiplier": 2.0,
        }
    )


def _ring_ten_relational_control(config: RawConfig) -> None:
    _ring_ten_attention_full_kv(config)


def _ring_ten_relational_local_heavy(config: RawConfig) -> None:
    _ring_ten_relational_control(config)
    _mapping(config, "model").update(
        {
            "bottleneck_ratio": 35 / 64,
            "ff_multiplier": 419 / 384,
            "local_blocks_per_group": 3,
        }
    )


def _ring_ten_relational_source_gated(config: RawConfig) -> None:
    _ring_ten_relational_control(config)
    _mapping(config, "model").update(
        {
            "local_operator": "source_gated",
            "ff_multiplier": 5 / 3,
        }
    )


def _ring_ten_capacity_control(config: RawConfig) -> None:
    _ring_ten_attention_control(config)


def _ring_ten_capacity_depth_seven(config: RawConfig) -> None:
    _ring_ten_capacity_control(config)
    _mapping(config, "model")["rrt_groups"] = 7


def _ring_ten_capacity_width_512(config: RawConfig) -> None:
    _ring_ten_capacity_control(config)
    model = _mapping(config, "model")
    query_heads = int(model["attention_heads"])
    kv_heads = int(model["kv_heads"])
    model["width"] = 512
    model["attention_heads"] = 16
    model["kv_heads"] = 16 if kv_heads == query_heads else 4


def _search_quality(config: RawConfig) -> None:
    selfplay = _mapping(config, "selfplay")
    selfplay.update(
        {
            "fast_probability": 0.65,
            "full_probability": 0.35,
            "fast_simulations": 32,
            "full_simulations": 384,
            "max_considered": 32,
            "max_considered_ring_exponent": 1.0,
            "max_considered_cap": 64,
        }
    )


def _actor_batch(config: RawConfig, size: int) -> None:
    selfplay = _mapping(config, "selfplay")
    selfplay["batch_size"] = size
    selfplay["games"] = size
    orchestration = _mapping(config, "orchestration")
    orchestration["actor_games_per_batch"] = size
    workers = orchestration.get("gpus")
    if not isinstance(workers, list):
        raise ValueError("orchestration.gpus must be a list")
    for worker in workers:
        if isinstance(worker, dict) and worker.get("role") == "actor":
            worker["actor_batch_size"] = size


def _actor_lanes_three(config: RawConfig) -> None:
    orchestration = _mapping(config, "orchestration")
    promotion_gpu = _mapping(orchestration, "promotion").get("gpu_id")
    workers = orchestration.get("gpus")
    if not isinstance(workers, list):
        raise ValueError("orchestration.gpus must be a list")
    for worker in workers:
        if (
            isinstance(worker, dict)
            and worker.get("role") == "actor"
            and worker.get("gpu_id") != promotion_gpu
        ):
            worker["actor_lanes"] = 3


def _learner_slack_actor(config: RawConfig) -> None:
    orchestration = _mapping(config, "orchestration")
    workers = orchestration.get("gpus")
    if not isinstance(workers, list):
        raise ValueError("orchestration.gpus must be a list")
    learner_indexes = [
        index
        for index, worker in enumerate(workers)
        if isinstance(worker, dict) and worker.get("role") == "learner"
    ]
    if len(learner_indexes) != 1:
        raise ValueError("learner-slack treatment requires exactly one learner GPU")
    learner_index = learner_indexes[0]
    learner = workers[learner_index]
    assert isinstance(learner, dict)
    learner_gpu = learner.get("gpu_id")
    promotion_gpu = _mapping(orchestration, "promotion").get("gpu_id")
    if learner_gpu != 0 or promotion_gpu != 7:
        raise ValueError(
            "learner-slack treatment requires learner GPU 0 and promotion GPU 7"
        )
    if any(
        isinstance(worker, dict)
        and worker.get("role") == "actor"
        and worker.get("gpu_id") == learner_gpu
        for worker in workers
    ):
        raise ValueError("learner GPU already has a colocated actor")
    colocated_actor: RawConfig = {
        "gpu_id": learner_gpu,
        "role": "actor",
        "cpu_threads": 8,
        "actor_batch_size": 64,
        "actor_lanes": 1,
    }
    affinity = learner.get("cpu_affinity")
    if affinity is not None:
        colocated_actor["cpu_affinity"] = affinity
    orchestration["allow_colocated_workers"] = True
    workers.insert(learner_index + 1, colocated_actor)


def _learner_batch(config: RawConfig, size: int) -> None:
    _mapping(config, "train")["per_rank_batch_size"] = size
    _mapping(config, "learner")["target_updates_per_new_sample"] = 1.0


def _lr_quarter(config: RawConfig) -> None:
    optimizer = _mapping(config, "optimizer")
    optimizer["adamw_lr"] = float(optimizer["adamw_lr"]) * 0.5
    optimizer["muon_lr"] = float(optimizer["muon_lr"]) * 0.5


def _fresh_source(config: RawConfig) -> None:
    learner = _mapping(config, "learner")
    learner.update(
        {
            "selfplay_snapshot_interval_examples": 3_000_000,
            "selfplay_snapshot_warmup_examples": 20_000_000,
            "selfplay_snapshot_warmup_interval_examples": 1_000_000,
        }
    )
    refresh = _mapping(_mapping(config, "orchestration"), "model_refresh")
    refresh.update(
        {
            "selfplay_source": "candidate_champion_history_mix",
            "candidate_probability": 0.5,
            "history_probability": 0.15,
            "history_pool_size": 8,
        }
    )


def _hard_replay(config: RawConfig) -> None:
    data = _mapping(config, "data")
    data["shard_cache_size"] = max(16, int(data.get("shard_cache_size", 1)))
    data["shards_per_batch"] = 4
    selfplay = _mapping(config, "selfplay")
    selfplay["policy_surprise_weight"] = 0.5
    selfplay["policy_surprise_max_weight"] = 4.0


def _fresh_hard(config: RawConfig) -> None:
    _fresh_source(config)
    _hard_replay(config)


TREATMENTS: dict[str, Treatment] = {
    "control": _control,
    "utd-1": _utd_one,
    "plateau-keep": _plateau_keep,
    "freshness-mix": _freshness_mix,
    "ring10-70": _ring_ten_seventy,
    "search-quality": _search_quality,
    "actor-batch-128": lambda config: _actor_batch(config, 128),
    "actor-batch-160": lambda config: _actor_batch(config, 160),
    "actor-batch-192": lambda config: _actor_batch(config, 192),
    "actor-lanes-3": _actor_lanes_three,
    "learner-batch-768": lambda config: _learner_batch(config, 768),
    "learner-batch-1024": lambda config: _learner_batch(config, 1024),
    "lr-quarter": _lr_quarter,
    "fresh-source": _fresh_source,
    "hard-replay": _hard_replay,
    "fresh-hard": _fresh_hard,
    "weighted-control": _weighted_control,
    "ring10-65-weighted": _ring_ten_sixty_five_weighted,
    "ring10-70-weighted": _ring_ten_seventy_weighted,
    "ring10-only": _ring_ten_only,
    "ring10-learner-slack-64": _ring_ten_learner_slack,
    "ring10-actor-lanes-3": _ring_ten_actor_lanes_three,
    "ring10-optimization-control": _ring_ten_optimization_control,
    "ring10-cadence-5m": _ring_ten_cadence_five_million,
    "ring10-freshness-50": _ring_ten_freshness_fifty,
    "ring10-live-cadence-control": _ring_ten_live_cadence_control,
    "ring10-live-cadence-5m": _ring_ten_live_cadence_five_million,
    "ring10-dynamics-control": _ring_ten_dynamics_control,
    "ring10-dynamics-adamw": _ring_ten_dynamics_adamw,
    "ring10-dynamics-ema-1m": _ring_ten_dynamics_ema_one_million,
    "ring10-dynamics-freshness-50": _ring_ten_dynamics_freshness_fifty,
    "ring10-dynamics-clinch-outcome-only": (_ring_ten_dynamics_clinch_outcome_only),
    "ring10-attention-control": _ring_ten_attention_control,
    "ring10-attention-full-kv": _ring_ten_attention_full_kv,
    "ring10-relational-control": _ring_ten_relational_control,
    "ring10-relational-local-heavy": _ring_ten_relational_local_heavy,
    "ring10-relational-source-gated": _ring_ten_relational_source_gated,
    "ring10-capacity-control": _ring_ten_capacity_control,
    "ring10-capacity-depth-7": _ring_ten_capacity_depth_seven,
    "ring10-capacity-width-512": _ring_ten_capacity_width_512,
}


def _load_raw(path: Path) -> RawConfig:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("base config must contain a mapping")
    return loaded


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_replay_cutoff(source_run_root: Path) -> int:
    manifest = source_run_root / "replay" / "manifest.sqlite3"
    if not manifest.is_file() or manifest.is_symlink():
        raise ValueError("training-dynamics suite requires a source replay manifest")
    try:
        uri = f"{manifest.resolve().as_uri()}?mode=ro"
        with sqlite3.connect(uri, uri=True, timeout=30.0) as connection:
            row = connection.execute(
                "SELECT COALESCE(MAX(id), 0) FROM shards WHERE state = 'ready'"
            ).fetchone()
    except sqlite3.Error as error:
        raise ValueError(f"cannot read source replay cutoff: {error}") from error
    cutoff = int(row[0]) if row is not None else 0
    if cutoff < 0:
        raise ValueError("source replay cutoff cannot be negative")
    return cutoff


def _read_json(path: Path) -> dict[str, object]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return loaded


def _read_verified_json_artifact(
    path: Path,
    expected_sha256: str,
    *,
    name: str,
) -> dict[str, object]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"winner snapshot {name} artifact is not a regular file")
        chunks = []
        while block := os.read(descriptor, 1024 * 1024):
            chunks.append(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ValueError(f"winner snapshot {name} artifact changed while reading")
    payload = b"".join(chunks)
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError(f"winner snapshot {name} artifact is stale")
    try:
        loaded = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"winner snapshot {name} artifact is invalid JSON") from error
    if not isinstance(loaded, dict):
        raise ValueError(f"winner snapshot {name} artifact must contain an object")
    return loaded


def winner_snapshot_from_document(document: Mapping[str, object]) -> dict[str, object]:
    """Extract a direct snapshot or a comparator selector snapshot."""
    if document.get("status") == "verified" and "champion" in document:
        return dict(document)
    selector = document.get("selector")
    if not isinstance(selector, Mapping) or selector.get("status") != "verified":
        raise ValueError("upstream selector has not emitted a verified winner")
    snapshot = selector.get("winner_snapshot")
    if not isinstance(snapshot, Mapping):
        raise ValueError("verified upstream selector has no winner snapshot")
    return dict(snapshot)


def verify_winner_snapshot(
    source_run_root: Path,
    snapshot: Mapping[str, object],
) -> dict[str, object]:
    """Verify a staged source is still the selector's immutable winner."""
    source = source_run_root.expanduser().resolve()
    if snapshot.get("status") != "verified":
        raise ValueError("winner snapshot status is not verified")
    raw_root = snapshot.get("run_root")
    if not isinstance(raw_root, str) or Path(raw_root).expanduser().resolve() != source:
        raise ValueError("winner snapshot run root does not match staged source")
    champion = snapshot.get("champion")
    if not isinstance(champion, Mapping):
        raise ValueError("winner snapshot champion is missing")
    identity = champion.get("model_identity")
    step = champion.get("model_step")
    if (
        not isinstance(identity, str)
        or not identity
        or type(step) is not int
        or step < 0
    ):
        raise ValueError("winner snapshot champion identity is invalid")
    run_identity = snapshot.get("run_identity")
    if not isinstance(run_identity, Mapping):
        raise ValueError("winner snapshot run identity is missing")
    run_path = source / "run.json"
    champion_path = source / "learner" / "champion.json"
    run_artifact = snapshot.get("run_identity_artifact")
    champion_artifact = snapshot.get("champion_pointer_artifact")
    loaded_artifacts: dict[str, dict[str, object]] = {}
    for name, artifact, expected_path in (
        ("run identity", run_artifact, run_path),
        ("champion pointer", champion_artifact, champion_path),
    ):
        if not isinstance(artifact, Mapping):
            raise ValueError(f"winner snapshot {name} artifact is missing")
        artifact_path = artifact.get("path")
        digest = artifact.get("sha256")
        if (
            not isinstance(artifact_path, str)
            or Path(artifact_path).expanduser().resolve() != expected_path
            or not isinstance(digest, str)
        ):
            raise ValueError(f"winner snapshot {name} artifact is stale")
        loaded_artifacts[name] = _read_verified_json_artifact(
            expected_path,
            digest,
            name=name,
        )
    current_run = loaded_artifacts["run identity"]
    current_champion = loaded_artifacts["champion pointer"]
    if any(
        current_run.get(field) != run_identity.get(field)
        for field in ("run_id", "generation_family", "created_ns")
    ):
        raise ValueError("winner snapshot run identity is stale")
    if (
        current_champion.get("model_identity") != identity
        or current_champion.get("model_step") != step
    ):
        raise ValueError("winner snapshot champion identity is stale")
    return dict(snapshot)


def validate_futility_policy(
    policy: Mapping[str, object],
    *,
    guard_rings: Sequence[int],
    guard_floor_elo: float,
) -> dict[str, object]:
    """Validate a stop-only policy without granting any promotion authority."""
    if (
        policy.get("schema_version") != 1
        or policy.get("report") != "startrain-elo-futility-policy"
    ):
        raise ValueError("unsupported Elo futility policy")
    if type(policy.get("enabled")) is not bool:
        raise ValueError("futility policy enabled must be boolean")
    if policy.get("decision_scope") != "stop_only":
        raise ValueError("futility policy must be stop-only")
    if policy.get("evidence") != "anytime_valid_confidence_sequences":
        raise ValueError("futility policy must require anytime-valid evidence")
    minimum_games = policy.get("minimum_decisive_games")
    if type(minimum_games) is not int or minimum_games <= 0:
        raise ValueError("futility minimum decisive games must be positive")
    guard = policy.get("guard_regression")
    if not isinstance(guard, Mapping):
        raise ValueError("futility guard-regression policy is missing")
    if guard.get("rings") != list(guard_rings):
        raise ValueError("futility guard rings differ from the ablation guardrails")
    if guard.get("floor_elo") != guard_floor_elo:
        raise ValueError("futility guard floor differs from the ablation guardrail")
    if guard.get("arm_upper_field") != "anytime_upper_elo":
        raise ValueError("futility guard regression must use an anytime upper bound")
    control = policy.get("control_comparison")
    if not isinstance(control, Mapping):
        raise ValueError("futility control-comparison policy is missing")
    advantage = control.get("minimum_advantage_elo")
    if isinstance(advantage, bool) or not isinstance(advantage, int | float):
        raise ValueError("futility minimum control advantage must be numeric")
    objective = control.get("objective", "ring_10")
    if objective not in {"ring_10", "weighted_aggregate"}:
        raise ValueError("futility control objective is unsupported")
    if objective == "ring_10" and control.get("ring") != 10:
        raise ValueError("futility ring-10 control comparison must use ring 10")
    selected_ring = control.get("ring")
    if (
        objective == "weighted_aggregate"
        and selected_ring is not None
        and (selected_ring != "weighted")
    ):
        raise ValueError("weighted futility comparison cannot select one ring")
    if (
        control.get("arm_upper_field") != "anytime_upper_elo"
        or control.get("control_lower_field") != "anytime_lower_elo"
    ):
        raise ValueError("futility control comparison must use anytime Elo bounds")
    return json.loads(json.dumps(dict(policy), allow_nan=False))


def training_objective_for_treatments(treatments: Sequence[str]) -> str:
    objectives = {
        "ring10_only" if name in RING10_OBJECTIVE_TREATMENTS else "generalist"
        for name in treatments
    }
    if len(objectives) != 1:
        raise ValueError(
            "incompatible training objectives require separate ablation plans"
        )
    return next(iter(objectives))


def promotion_objective_for_treatments(treatments: Sequence[str]) -> str:
    training_objective = training_objective_for_treatments(treatments)
    if training_objective == "ring10_only":
        return "ring_10_only"
    weighted = [name in WEIGHTED_TREATMENTS for name in treatments]
    if any(weighted) and not all(weighted):
        raise ValueError(
            "weighted-objective and legacy treatments require separate ablation plans"
        )
    return "weighted_aggregate" if weighted and all(weighted) else "ring_10_guarded"


def blocking_guard_rings(treatments: Sequence[str]) -> tuple[int, ...]:
    """Return the one pre-registered guard policy shared by an ablation matrix."""
    promotion_objective = promotion_objective_for_treatments(treatments)
    return GUARD_RINGS if promotion_objective == "ring_10_guarded" else ()


def resolve_treatments(
    *,
    suite: str | None,
    treatments: Sequence[str] | None,
) -> tuple[str, ...]:
    if suite is not None and treatments:
        raise ValueError("--suite cannot be combined with --treatment")
    if suite is not None:
        try:
            return TREATMENT_SUITES[suite]
        except KeyError as error:
            raise ValueError(f"unknown treatment suite: {suite}") from error
    return tuple(treatments or DEFAULT_TREATMENTS)


def _validate_ring10_efficiency_base(experiment: ExperimentConfig) -> None:
    validate_continuous_config(experiment)
    orchestration = experiment.orchestration
    if orchestration.training_objective != "ring10_only":
        raise ValueError("ring10 suites require a ring10_only base profile")
    if orchestration.allow_colocated_workers:
        raise ValueError("ring10 control must not colocate workers")
    learners = orchestration.learner_gpus
    if (
        len(learners) != 1
        or learners[0].gpu_id != 0
        or learners[0].cpu_threads != 16
        or learners[0].cpu_affinity != "0-103"
    ):
        raise ValueError("ring10 control requires the H100 learner layout")
    actors = {gpu.gpu_id: gpu for gpu in orchestration.actor_gpus}
    if set(actors) != set(range(1, 8)):
        raise ValueError("ring10 control requires actor GPUs 1 through 7")
    for gpu_id, actor in actors.items():
        expected_affinity = "0-103" if gpu_id <= 3 else "104-207"
        expected_lanes = 1 if gpu_id == 7 else 2
        if (
            actor.cpu_threads != 8
            or actor.actor_batch_size != 128
            or actor.actor_lanes != expected_lanes
            or actor.cpu_affinity != expected_affinity
        ):
            raise ValueError(f"ring10 control actor GPU {gpu_id} topology drifted")
    promotion = orchestration.promotion
    if (
        orchestration.actor_games_per_batch != 128
        or promotion.gpu_id != 7
        or not promotion.pause_sharing_mode
    ):
        raise ValueError("ring10 control promotion topology drifted")


def _validate_ring10_live_cadence_base(
    raw: RawConfig,
    experiment: ExperimentConfig,
) -> None:
    if experiment.orchestration.training_objective != "ring10_only":
        raise ValueError(
            "ring10 live cadence suite requires a ring10_only base profile"
        )
    validate_continuous_config(experiment)
    learner = _mapping(raw, "learner")
    if experiment.learner.target_updates_per_new_sample != 1.0:
        raise ValueError("ring10 live cadence suite requires frozen UTD=1.0")
    if "candidate_interval_examples" not in learner:
        raise ValueError(
            "ring10 live cadence suite requires an explicit "
            "learner.candidate_interval_examples"
        )
    candidate_interval = learner["candidate_interval_examples"]
    if (
        isinstance(candidate_interval, bool)
        or not isinstance(candidate_interval, int)
        or candidate_interval <= 0
        or candidate_interval >= RING10_LIVE_CADENCE_TARGET_EXAMPLES
    ):
        raise ValueError(
            "ring10 live cadence suite requires a positive frozen candidate cadence "
            f"below {RING10_LIVE_CADENCE_TARGET_EXAMPLES}"
        )
    treatment = replace(
        experiment,
        learner=replace(
            experiment.learner,
            candidate_interval_examples=RING10_LIVE_CADENCE_TARGET_EXAMPLES,
        ),
    )
    validate_continuous_config(treatment)


def _validate_ring10_live_cadence_transition(
    before: RawConfig,
    after: RawConfig,
    treatment: str,
) -> None:
    expected = deepcopy(before)
    if treatment == "ring10-live-cadence-5m":
        _mapping(expected, "learner")["candidate_interval_examples"] = (
            RING10_LIVE_CADENCE_TARGET_EXAMPLES
        )
    elif treatment != "ring10-live-cadence-control":
        raise ValueError(f"unsupported ring10 live cadence treatment: {treatment}")
    if after != expected:
        raise ValueError(
            f"{treatment} violated the frozen live one-factor cadence contract"
        )


def _validate_inputs(
    *,
    prefix: str,
    seed: int,
    wall_budget_hours: float,
    leaf_budget: int,
    guard_floor_elo: float,
    treatments: Sequence[str],
) -> None:
    if not prefix or prefix.strip() != prefix or "/" in prefix:
        raise ValueError("prefix must be a non-empty path-safe name")
    if isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be non-negative")
    if wall_budget_hours <= 0:
        raise ValueError("wall budget must be positive")
    if isinstance(leaf_budget, bool) or leaf_budget <= 0:
        raise ValueError("leaf budget must be positive")
    if guard_floor_elo >= 0:
        raise ValueError("guard floor must be a negative non-inferiority margin")
    if not treatments or len(set(treatments)) != len(treatments):
        raise ValueError("treatments must be non-empty and unique")
    unknown = sorted(set(treatments) - TREATMENTS.keys())
    if unknown:
        raise ValueError(f"unknown treatments: {unknown}")


def _configure_common(
    raw: RawConfig,
    *,
    run_root: Path,
    run_id: str,
    seed: int,
    guard_floor_elo: float,
    guard_rings: Sequence[int],
) -> None:
    _mapping(raw, "train")["seed"] = seed
    _mapping(raw, "selfplay")["seed"] = seed
    _mapping(raw, "arena")["seed"] = seed
    orchestration = _mapping(raw, "orchestration")
    orchestration["run_id"] = run_id
    _mapping(orchestration, "directories")["root"] = str(run_root)
    if guard_rings:
        arena = _mapping(raw, "arena")
        floors = arena.get("per_ring_regression_floor_elo")
        if floors is None:
            floors = {}
            arena["per_ring_regression_floor_elo"] = floors
        if not isinstance(floors, dict):
            raise ValueError("arena.per_ring_regression_floor_elo must be a mapping")
        for ring in guard_rings:
            floors[ring] = guard_floor_elo


def _validate_profile(
    path: Path,
    *,
    expected_root: Path,
    expected_run_id: str,
    expected_seed: int,
    guard_floor_elo: float,
    expected_training_objective: str,
    expected_promotion_objective: str,
) -> ExperimentConfig:
    loaded = load_config(path)
    if loaded.profile != "continuous" or not loaded.orchestration.enabled:
        raise ValueError("ablation profiles must be continuous orchestrated profiles")
    if loaded.orchestration.directories.root != str(expected_root):
        raise ValueError("generated run root did not round-trip")
    if loaded.orchestration.run_id != expected_run_id:
        raise ValueError("generated run ID did not round-trip")
    if (
        loaded.train.seed != expected_seed
        or loaded.selfplay.seed != expected_seed
        or loaded.arena.seed != expected_seed
    ):
        raise ValueError("generated treatment seeds did not round-trip")
    if loaded.orchestration.training_objective != expected_training_objective:
        raise ValueError("generated treatment training objective did not round-trip")
    if expected_training_objective == "ring10_only":
        if expected_promotion_objective != "ring_10_only":
            raise ValueError("ring10_only treatment has an incompatible objective")
        validate_continuous_config(loaded)
    elif expected_promotion_objective == "weighted_aggregate":
        if loaded.arena.per_ring_regression_floor_elo:
            raise ValueError("weighted objective retained blocking per-ring floors")
        if loaded.arena.required_regression_rings != ():
            raise ValueError("weighted objective retained required regression rings")
        if loaded.arena.promotion_pair_ratios != WEIGHTED_PROMOTION_PAIR_RATIOS:
            raise ValueError("weighted promotion pair ratios did not round-trip")
        if (
            loaded.arena.weighted_initial_blocks != WEIGHTED_INITIAL_BLOCKS
            or loaded.arena.weighted_continuation_blocks != WEIGHTED_CONTINUATION_BLOCKS
            or loaded.arena.weighted_max_blocks != WEIGHTED_MAX_BLOCKS
        ):
            raise ValueError("weighted promotion block limits did not round-trip")
    elif expected_promotion_objective == "ring_10_guarded":
        for ring in GUARD_RINGS:
            if loaded.arena.per_ring_regression_floor_elo.get(ring) != guard_floor_elo:
                raise ValueError(f"ring {ring} guard floor did not round-trip")
    else:
        raise ValueError(
            f"unsupported generated promotion objective: {expected_promotion_objective}"
        )
    return loaded


def prepare_elo_ablation(
    *,
    base_config: Path,
    output_dir: Path,
    run_root_parent: Path,
    run_id: str,
    source_run_root: Path,
    prefix: str,
    seed: int,
    wall_budget_hours: float,
    leaf_budget: int,
    guard_floor_elo: float,
    treatments: Sequence[str],
    winner_snapshot: Mapping[str, object] | None = None,
    futility_policy: Mapping[str, object] | None = None,
    guard_rings: Sequence[int] | None = None,
    suite: str | None = None,
) -> dict[str, object]:
    if suite is not None:
        expected = TREATMENT_SUITES.get(suite)
        if expected is None:
            raise ValueError(f"unknown treatment suite: {suite}")
        if tuple(treatments) != expected:
            raise ValueError("treatments do not match the selected suite")
    if any(name in RING10_LIVE_CADENCE_TREATMENTS for name in treatments) and (
        suite != "ring10-live-cadence"
        or tuple(treatments) != RING10_LIVE_CADENCE_TREATMENTS
    ):
        raise ValueError(
            "ring10 live cadence treatments require the complete "
            "--suite ring10-live-cadence"
        )
    _validate_inputs(
        prefix=prefix,
        seed=seed,
        wall_budget_hours=wall_budget_hours,
        leaf_budget=leaf_budget,
        guard_floor_elo=guard_floor_elo,
        treatments=treatments,
    )
    base = base_config.expanduser().resolve()
    source = source_run_root.expanduser().resolve()
    destination = output_dir.expanduser().resolve()
    root_parent = run_root_parent.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"output directory already exists: {destination}")
    if not base.is_file():
        raise FileNotFoundError(f"base config does not exist: {base}")
    if not source.is_dir():
        raise FileNotFoundError(f"source run root does not exist: {source}")
    scratch_treatments = [
        treatment in SCRATCH_INITIALIZATION_TREATMENTS for treatment in treatments
    ]
    if any(scratch_treatments) and not all(scratch_treatments):
        raise ValueError(
            "scratch architecture and forked training treatments require "
            "separate ablation plans"
        )
    initialization = "scratch" if all(scratch_treatments) else "fork"
    training_objective = training_objective_for_treatments(treatments)
    promotion_objective = promotion_objective_for_treatments(treatments)
    inferred_guard_rings = blocking_guard_rings(treatments)
    configured_guard_rings = (
        inferred_guard_rings if guard_rings is None else tuple(guard_rings)
    )
    if configured_guard_rings != inferred_guard_rings:
        raise ValueError(
            "guard rings do not match the treatments' pre-registered objective"
        )
    verified_winner = (
        verify_winner_snapshot(source, winner_snapshot)
        if winner_snapshot is not None
        else None
    )
    registered_futility = (
        validate_futility_policy(
            futility_policy,
            guard_rings=configured_guard_rings,
            guard_floor_elo=guard_floor_elo,
        )
        if futility_policy is not None
        else None
    )
    base_sha256 = _sha256(base)
    raw_base = _load_raw(base)
    if suite == "ring10-live-cadence":
        _validate_ring10_live_cadence_base(raw_base, load_config(base))
    elif (suite is not None and suite.startswith("ring10-")) or any(
        treatment in RING10_EFFICIENCY_VARIANTS
        or treatment in RING10_TRAINING_DYNAMICS_TREATMENTS
        or treatment in SCRATCH_INITIALIZATION_TREATMENTS
        for treatment in treatments
    ):
        _validate_ring10_efficiency_base(load_config(base))
    replay_cutoff = (
        _source_replay_cutoff(source)
        if any(
            treatment in RING10_TRAINING_DYNAMICS_TREATMENTS for treatment in treatments
        )
        else None
    )
    destination.mkdir(parents=True)
    generated = []
    for treatment_name in treatments:
        run_root = root_parent / f"{prefix}-{treatment_name}-seed{seed}"
        treatment_run_id = (
            "scratch-"
            + hashlib.sha256(
                f"{run_id}:{suite}:{treatment_name}:{seed}".encode("utf-8")
            ).hexdigest()[:24]
            if initialization == "scratch"
            else run_id
        )
        profile = deepcopy(raw_base)
        _mapping(profile, "orchestration")["training_objective"] = training_objective
        _configure_common(
            profile,
            run_root=run_root,
            run_id=treatment_run_id,
            seed=seed,
            guard_floor_elo=guard_floor_elo,
            guard_rings=configured_guard_rings,
        )
        before_treatment = deepcopy(profile)
        TREATMENTS[treatment_name](profile)
        if treatment_name in RING10_LIVE_CADENCE_TREATMENTS:
            _validate_ring10_live_cadence_transition(
                before_treatment,
                profile,
                treatment_name,
            )
        if replay_cutoff is not None:
            _mapping(profile, "learner")["minimum_replay_shard_id_exclusive"] = (
                replay_cutoff
            )
        profile_path = destination / f"{treatment_name}.yaml"
        profile_path.write_text(
            yaml.safe_dump(profile, sort_keys=False),
            encoding="utf-8",
        )
        loaded_profile = _validate_profile(
            profile_path,
            expected_root=run_root,
            expected_run_id=treatment_run_id,
            expected_seed=seed,
            guard_floor_elo=guard_floor_elo,
            expected_training_objective=training_objective,
            expected_promotion_objective=promotion_objective,
        )
        expected_parameters = EXPECTED_ARCHITECTURE_PARAMETERS.get(treatment_name)
        if (
            expected_parameters is not None
            and model_parameter_count(loaded_profile.model) != expected_parameters
        ):
            raise ValueError(
                f"{treatment_name} parameter contract differs from "
                f"{expected_parameters}"
            )
        generated.append(
            {
                "treatment": treatment_name,
                "training_objective": training_objective,
                "promotion_objective": promotion_objective,
                "run_id": treatment_run_id,
                "profile": str(profile_path),
                "profile_sha256": _sha256(profile_path),
                "run_root": str(run_root),
            }
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "report": REPORT_NAME,
        "suite": suite,
        "initialization": initialization,
        "base_config": str(base),
        "base_config_sha256": base_sha256,
        "source_run_root": str(source),
        "source_replay_cutoff": replay_cutoff,
        "source_winner_snapshot": verified_winner,
        "futility_policy": registered_futility,
        "run_id": run_id,
        "prefix": prefix,
        "seed": seed,
        "wall_budget_seconds": wall_budget_hours * 3600.0,
        "leaf_budget": leaf_budget,
        "guard_rings": list(configured_guard_rings),
        "guard_floor_elo": guard_floor_elo,
        "training_objective": training_objective,
        "promotion_objective": promotion_objective,
        "per_ring_guarantees": bool(configured_guard_rings),
        "treatments": generated,
    }
    manifest_path = destination / "ablation-plan.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        treatments = resolve_treatments(
            suite=arguments.suite,
            treatments=arguments.treatments,
        )
        winner_snapshot = (
            winner_snapshot_from_document(_read_json(arguments.winner_snapshot))
            if arguments.winner_snapshot is not None
            else None
        )
        futility_policy = (
            _read_json(arguments.futility_policy)
            if arguments.futility_policy is not None
            else None
        )
        manifest = prepare_elo_ablation(
            base_config=arguments.base_config,
            output_dir=arguments.output_dir,
            run_root_parent=arguments.run_root_parent,
            run_id=arguments.run_id,
            source_run_root=arguments.source_run_root,
            prefix=arguments.prefix,
            seed=arguments.seed,
            wall_budget_hours=arguments.wall_budget_hours,
            leaf_budget=arguments.leaf_budget,
            guard_floor_elo=arguments.guard_floor_elo,
            treatments=treatments,
            winner_snapshot=winner_snapshot,
            futility_policy=futility_policy,
            suite=arguments.suite,
        )
    except (FileExistsError, FileNotFoundError, OSError, ValueError) as error:
        print(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "report": REPORT_NAME,
                    "status": "error",
                    "error": f"{type(error).__name__}: {error}",
                },
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps({"status": "ok", **manifest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
