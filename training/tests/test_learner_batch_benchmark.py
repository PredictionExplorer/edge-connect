from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts import benchmark_learner_batches as benchmark
from startrain.checkpoint import ExponentialMovingAverage, save_checkpoint, sha256_file
from startrain.config import load_config
from startrain.contracts import FEATURE_SCHEMA_HASH, RULES_HASH, RULES_HASH_WIRE
from startrain.features import DoubleStarPosition
from startrain.model import GraphResTNet
from startrain.optim import build_optimizer
from startrain.replay import ReplaySample, write_replay_shard
from startrain.topology import get_topology
from startrain.training import build_scheduler

TRAINING_ROOT = Path(__file__).parents[1]
RUN_ID = "benchmark-run"
GENERATION_FAMILY = "benchmark-family"


def _result(
    batch_size: int,
    *,
    throughput: float,
    peak_allocated: int,
) -> dict[str, object]:
    return {
        "batch_size": batch_size,
        "status": "ok",
        "throughput": {
            "end_to_end_samples_per_second": throughput,
            "cuda_device_samples_per_second": None,
        },
        "memory_bytes": {
            "peak_allocated": peak_allocated,
            "peak_reserved": peak_allocated,
        },
    }


def test_recommendation_uses_throughput_hbm_gate_and_smallest_winner() -> None:
    results = [
        _result(512, throughput=100.0, peak_allocated=20),
        _result(768, throughput=115.0, peak_allocated=30),
        _result(1024, throughput=130.0, peak_allocated=40),
    ]

    recommendation = benchmark.recommend_batch(results)

    assert recommendation["status"] == "larger_batch_selected"
    assert recommendation["selected_batch_size"] == 768
    assert recommendation["passing_larger_batch_sizes"] == [768, 1024]

    results[1] = _result(768, throughput=114.99, peak_allocated=30)
    results[2] = _result(
        1024,
        throughput=140.0,
        peak_allocated=benchmark.MAXIMUM_PEAK_ALLOCATED_BYTES + 1,
    )
    recommendation = benchmark.recommend_batch(results)
    assert recommendation["status"] == "keep_baseline"
    assert recommendation["selected_batch_size"] == 512


def test_batch_matrix_isolates_oom_and_other_errors() -> None:
    attempted: list[int] = []

    def runner(batch_size: int) -> dict[str, object]:
        attempted.append(batch_size)
        if batch_size == 768:
            raise torch.cuda.OutOfMemoryError("synthetic CUDA OOM")
        if batch_size == 1024:
            raise RuntimeError("synthetic replay failure")
        return _result(
            batch_size,
            throughput=float(batch_size),
            peak_allocated=0,
        )

    results = benchmark.run_batch_matrix((512, 768, 1024, 1280), runner)

    assert attempted == [512, 768, 1024, 1280]
    assert [result["status"] for result in results] == [
        "ok",
        "oom",
        "error",
        "ok",
    ]
    assert results[1]["error"]["type"] == "CUDAOutOfMemoryError"
    assert results[2]["error"]["type"] == "RuntimeError"


def _tiny_config(tmp_path: Path) -> Path:
    source = (TRAINING_ROOT / "configs" / "small.yaml").read_text(encoding="utf-8")
    source = source.replace("width: 64", "width: 8")
    source = source.replace("rrt_groups: 5", "rrt_groups: 1")
    source = source.replace("attention_heads: 4", "attention_heads: 1")
    path = tmp_path / "tiny.yaml"
    path.write_text(source, encoding="utf-8")
    return path


def _sample() -> ReplaySample:
    topology = get_topology(4)
    stones = torch.full((topology.n,), -1, dtype=torch.int8)
    stones[0] = 0
    position = DoubleStarPosition(
        rings=4,
        stones=stones,
        to_move=1,
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
        search_provenance="benchmark",
        policy_provenance="benchmark",
        run_id=RUN_ID,
        generation_family=GENERATION_FAMILY,
        actor_id="benchmark-actor",
        generation=0,
        game_id="benchmark-game",
        model_identity="manual",
    )


def _write_manifest(replay_root: Path, shard: Path) -> Path:
    manifest = replay_root / "manifest.sqlite3"
    connection = sqlite3.connect(manifest)
    try:
        connection.executescript(
            """
            CREATE TABLE store_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE shards (
                id INTEGER PRIMARY KEY,
                relative_path TEXT NOT NULL,
                created_ns INTEGER NOT NULL,
                sample_count INTEGER NOT NULL,
                ring INTEGER NOT NULL,
                phase_min INTEGER NOT NULL,
                phase_max INTEGER NOT NULL,
                model_version TEXT NOT NULL,
                model_step INTEGER NOT NULL,
                model_identity TEXT NOT NULL,
                run_id TEXT NOT NULL,
                generation_family TEXT NOT NULL,
                actor_id TEXT NOT NULL,
                generation INTEGER NOT NULL,
                game_count INTEGER NOT NULL,
                state TEXT NOT NULL,
                quarantine_reason TEXT,
                rules_hash TEXT NOT NULL,
                feature_schema_hash TEXT NOT NULL,
                checksum_sha256 TEXT NOT NULL
            );
            """
        )
        connection.executemany(
            "INSERT INTO store_metadata(key, value) VALUES (?, ?)",
            (
                (
                    "manifest_schema_version",
                    str(benchmark.MANIFEST_SCHEMA_VERSION),
                ),
                ("rules_hash", RULES_HASH_WIRE),
                ("feature_schema_hash", f"{FEATURE_SCHEMA_HASH:016x}"),
            ),
        )
        connection.execute(
            """
            INSERT INTO shards(
                id, relative_path, created_ns, sample_count, ring,
                phase_min, phase_max, model_version, model_step,
                model_identity, run_id, generation_family, actor_id,
                generation, game_count, state, quarantine_reason,
                rules_hash, feature_schema_hash, checksum_sha256
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                1,
                str(shard.relative_to(replay_root)),
                1,
                1,
                4,
                0,
                0,
                "manual",
                0,
                "manual",
                RUN_ID,
                GENERATION_FAMILY,
                "benchmark-actor",
                0,
                1,
                "ready",
                None,
                f"{RULES_HASH:016x}",
                f"{FEATURE_SCHEMA_HASH:016x}",
                sha256_file(shard),
            ),
        )
        connection.commit()
    finally:
        connection.close()
    return manifest


def _artifacts(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    config_path = _tiny_config(tmp_path)
    config = load_config(config_path)
    model = GraphResTNet(config.model)
    optimizer = build_optimizer(model, config.optimizer)
    scheduler = build_scheduler(optimizer, config.train.scheduler)
    ema = ExponentialMovingAverage(model, decay=config.train.ema_decay)
    checkpoint = save_checkpoint(
        tmp_path / "checkpoint.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema,
        step=1,
        config=config.as_dict(),
        extra={
            "run_id": RUN_ID,
            "generation_family": GENERATION_FAMILY,
        },
    )
    replay_root = tmp_path / "replay"
    shard_directory = replay_root / "shards"
    shard_directory.mkdir(parents=True)
    shard = write_replay_shard(shard_directory / "shard.npz", [_sample()])
    manifest = _write_manifest(replay_root, shard)
    return config_path, checkpoint, replay_root, manifest


def _snapshot(paths: tuple[Path, ...]) -> dict[Path, tuple[bytes, int]]:
    return {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths}


def test_manifest_connection_and_full_cli_are_read_only(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config, checkpoint, replay_root, manifest = _artifacts(tmp_path)
    shard = replay_root / "shards" / "shard.npz"
    before = _snapshot((config, checkpoint, manifest, shard))

    with benchmark.open_replay_manifest_read_only(replay_root) as connection:
        assert connection.execute("PRAGMA query_only").fetchone()[0] == 1
        with pytest.raises(sqlite3.OperationalError):
            connection.execute("CREATE TABLE forbidden(value INTEGER)")

    output = tmp_path / "benchmark.json"
    exit_code = benchmark.main(
        [
            "--config",
            str(config),
            "--checkpoint",
            str(checkpoint),
            "--replay-root",
            str(replay_root),
            "--device",
            "cpu",
            "--batch-sizes",
            "1",
            "--warmups",
            "0",
            "--repeats",
            "1",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["replay"]["manifest_open_mode"] == "ro"
    assert payload["batches"][0]["status"] == "ok"
    assert payload["batches"][0]["checkpoint_state"]["ema_loaded"] is True
    assert json.loads(output.read_text(encoding="utf-8")) == payload
    assert _snapshot((config, checkpoint, manifest, shard)) == before
