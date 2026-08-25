from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from scripts import run_frozen_replay_optimizer_calibration as calibration
from startrain.checkpoint import ExponentialMovingAverage, write_model_pointer
from startrain.config import load_config
from startrain.contracts import FEATURE_SCHEMA_HASH, RULES_HASH, RULES_HASH_WIRE
from startrain.features import DoubleStarPosition
from startrain.learner import ImmutableModelPublisher
from startrain.model import GraphResTNet
from startrain.optim import build_optimizer
from startrain.replay import ReplaySample, write_replay_shard
from startrain.runtime import RunIdentity
from startrain.topology import get_topology
from startrain.training import build_scheduler

TRAINING_ROOT = Path(__file__).parents[1]
RUN_ID = "calibration-run"
FAMILY = "calibration-family"


def _tiny_ring10_config(tmp_path: Path) -> Path:
    raw = yaml.safe_load(
        (TRAINING_ROOT / "configs" / "h100-8gpu-ring10-only.yaml").read_text(
            encoding="utf-8"
        )
    )
    raw["model"].update(
        {
            "width": 8,
            "rrt_groups": 1,
            "attention_heads": 1,
            "kv_heads": 1,
            "ff_multiplier": 1.0,
            "local_blocks_per_group": 1,
        }
    )
    raw["optimizer"].update(
        {
            "adamw_lr": 0.001,
            "muon_lr": 0.01,
            "min_muon_elements": 1,
            "fallback_to_adamw": False,
        }
    )
    raw["train"].update(
        {
            "per_rank_batch_size": 1,
            "precision": "fp32",
            "compile": False,
            "ema_decay": 0.9,
            "gradient_clip_norm": 1.0,
            "scheduler": {
                "warmup_steps": 0,
                "total_steps": 10,
                "min_lr_ratio": 0.1,
            },
        }
    )
    raw["data"].update(
        {
            "d5_augmentation": False,
            "workers": 0,
            "pin_memory": False,
        }
    )
    raw["learner"]["device"] = "cpu"
    path = tmp_path / "control.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def _sample(index: int) -> ReplaySample:
    topology = get_topology(10)
    stones = torch.full((topology.n,), -1, dtype=torch.int8)
    stones[index % 5] = index % 2
    position = DoubleStarPosition(
        rings=10,
        stones=stones,
        to_move=(index + 1) % 2,
        moves_left=1,
        opening=False,
        terminal=False,
    )
    policy = (stones.numpy() == -1).astype(np.float32)
    policy /= policy.sum()
    return ReplaySample.from_position(
        position,
        policy=policy,
        final_score=None,
        search_provenance="frozen-calibration",
        policy_provenance="frozen-calibration",
        run_id=RUN_ID,
        generation_family=FAMILY,
        actor_id="calibration-actor",
        generation=0,
        game_id=f"calibration-game-{index}",
        model_identity="source-model",
    )


def _write_replay(replay_root: Path, *, samples: int = 8) -> tuple[Path, Path]:
    shard = write_replay_shard(
        replay_root / "shards" / "ring10.npz",
        [_sample(index) for index in range(samples)],
    )
    manifest = replay_root / "manifest.sqlite3"
    with sqlite3.connect(manifest) as connection:
        connection.executescript(
            """
            CREATE TABLE store_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE shards (
                id INTEGER PRIMARY KEY,
                relative_path TEXT NOT NULL,
                sample_count INTEGER NOT NULL,
                ring INTEGER NOT NULL,
                model_step INTEGER NOT NULL,
                model_identity TEXT NOT NULL,
                run_id TEXT NOT NULL,
                generation_family TEXT NOT NULL,
                state TEXT NOT NULL,
                rules_hash TEXT NOT NULL,
                feature_schema_hash TEXT NOT NULL,
                checksum_sha256 TEXT NOT NULL
            );
            """
        )
        connection.executemany(
            "INSERT INTO store_metadata(key, value) VALUES (?, ?)",
            (
                ("manifest_schema_version", str(calibration.MANIFEST_SCHEMA_VERSION)),
                ("rules_hash", RULES_HASH_WIRE),
                ("feature_schema_hash", f"{FEATURE_SCHEMA_HASH:016x}"),
            ),
        )
        connection.execute(
            """
            INSERT INTO shards(
                id, relative_path, sample_count, ring, model_step,
                model_identity, run_id, generation_family, state,
                rules_hash, feature_schema_hash, checksum_sha256
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                7,
                str(shard.relative_to(replay_root)),
                samples,
                10,
                1,
                "source-model",
                RUN_ID,
                FAMILY,
                "ready",
                f"{RULES_HASH:016x}",
                f"{FEATURE_SCHEMA_HASH:016x}",
                calibration.sha256_file(shard),
            ),
        )
    return manifest, shard


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    config_path = _tiny_ring10_config(tmp_path)
    config = load_config(config_path)
    model = GraphResTNet(config.model)
    optimizer = build_optimizer(model, config.optimizer)
    scheduler = build_scheduler(optimizer, config.train.scheduler)
    ema = ExponentialMovingAverage(model, decay=config.train.ema_decay)
    publisher = ImmutableModelPublisher(
        tmp_path / "source" / "learner",
        RunIdentity(tmp_path / "source" / "run.json", RUN_ID, FAMILY, 1),
    )
    candidate = publisher.publish(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema,
        step=1,
        epoch=0,
        config=config.as_dict(),
    )
    champion = publisher.root / "champion.json"
    write_model_pointer(
        champion,
        candidate,
        role="champion",
        promotion_result="bootstrap",
    )
    replay_root = tmp_path / "source" / "replay"
    manifest, shard = _write_replay(replay_root)
    return config_path, champion, replay_root, manifest, shard


def _snapshot(paths: tuple[Path, ...]) -> dict[Path, tuple[bytes, int]]:
    return {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths}


def _settings(
    tmp_path: Path,
    config: Path,
    champion: Path,
    replay_root: Path,
    *,
    dry_run: bool,
    stop_after_steps: int | None = None,
) -> calibration.CalibrationSettings:
    return calibration.CalibrationSettings(
        config=config,
        champion=champion,
        replay_root=replay_root,
        replay_cutoff=7,
        output_dir=tmp_path / "calibration-output",
        arm=calibration.CONTROL_ARM,
        steps=2,
        batch_size=1,
        evaluation_batch_size=1,
        max_samples=8,
        holdout_fraction=0.25,
        seed=23,
        device="cpu",
        budget_h100_hours=0.01,
        checkpoint_interval=1,
        stop_after_steps=stop_after_steps,
        dry_run=dry_run,
    )


def test_frozen_replay_dry_run_is_hash_pinned_and_read_only(tmp_path: Path) -> None:
    config, champion, replay_root, manifest, shard = _fixture(tmp_path)
    before = _snapshot((champion, manifest, shard))
    settings = _settings(
        tmp_path,
        config,
        champion,
        replay_root,
        dry_run=True,
    )

    with calibration.open_replay_read_only(replay_root) as connection:
        assert connection.execute("PRAGMA query_only").fetchone()[0] == 1
        with pytest.raises(sqlite3.OperationalError):
            connection.execute("DELETE FROM shards")

    result = calibration.run_calibration(settings)
    repeated = calibration.run_calibration(settings)

    assert result["status"] == "dry_run"
    assert result["replay"]["cutoff"] == 7
    assert len(result["replay"]["cutoff_sha256"]) == 64
    assert result["partition"]["disjoint"] is True
    assert repeated["partition"] == result["partition"]
    assert result["training"]["budget_h100_hours"] == 0.01
    assert not (tmp_path / "calibration-output").exists()
    assert _snapshot((champion, manifest, shard)) == before
    with pytest.raises(ValueError, match=r"\(0, 2\] H100-hours"):
        calibration.run_calibration(replace(settings, budget_h100_hours=2.01))


def test_cpu_calibration_resumes_from_fresh_champion_ema(tmp_path: Path) -> None:
    config, champion, replay_root, manifest, shard = _fixture(tmp_path)
    before = _snapshot((champion, manifest, shard))
    paused = calibration.run_calibration(
        _settings(
            tmp_path,
            config,
            champion,
            replay_root,
            dry_run=False,
            stop_after_steps=1,
        )
    )
    assert paused["status"] == "paused"
    assert paused["progress"]["completed_steps"] == 1

    resumed_settings = replace(
        _settings(
            tmp_path,
            config,
            champion,
            replay_root,
            dry_run=False,
        ),
        stop_after_steps=None,
    )
    result = calibration.run_calibration(resumed_settings)

    assert result["status"] == "complete"
    assert result["training"]["completed_steps"] == 2
    assert result["training"]["finite"] is True
    assert result["optimizer"]["fresh_from_champion_ema"] is True
    assert result["optimizer"]["source_optimizer_loaded"] is False
    assert result["heldout"]["samples"] == 2
    assert result["heldout"]["batches"] == 2
    unsigned = dict(result)
    expected = unsigned.pop("result_sha256")
    assert calibration._digest(unsigned) == expected
    persisted = json.loads(
        (tmp_path / "calibration-output" / "result.json").read_text(encoding="utf-8")
    )
    assert persisted == result
    assert _snapshot((champion, manifest, shard)) == before


def test_resume_charges_wall_clock_downtime_to_h100_budget(
    tmp_path: Path,
) -> None:
    config, champion, replay_root, _manifest, _shard = _fixture(tmp_path)
    settings = _settings(
        tmp_path,
        config,
        champion,
        replay_root,
        dry_run=False,
        stop_after_steps=1,
    )
    paused = calibration.run_calibration(settings)
    assert paused["status"] == "paused"
    state_path = settings.output_dir / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["measurement_started_ns"] = time.time_ns() - int(2.1 * 3600 * 1e9)
    state_path.write_text(json.dumps(state), encoding="utf-8")

    exhausted = calibration.run_calibration(
        replace(settings, stop_after_steps=None),
    )

    assert exhausted["status"] == "budget_exhausted"
    assert exhausted["progress"]["completed_steps"] == 1
