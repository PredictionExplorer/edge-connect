from __future__ import annotations

import json

import numpy as np
import torch

from scripts.benchmark_replay_pipeline import (
    BENCHMARK_NAME,
    benchmark_replay_shard,
    main,
)
from startrain.features import DoubleStarPosition
from startrain.replay import ReplaySample, write_replay_shard
from startrain.scoring import PlayerScore, ScoreResult
from startrain.topology import get_topology


def _sample(index: int) -> ReplaySample:
    topology = get_topology(4)
    stones = torch.full((topology.n,), -1, dtype=torch.int8)
    stones[index % topology.n] = index % 2
    position = DoubleStarPosition(
        rings=4,
        stones=stones,
        to_move=(index + 1) % 2,
        moves_left=1,
        opening=False,
        terminal=False,
    )
    legal = stones.numpy() == -1
    policy = legal.astype(np.float32)
    policy /= policy.sum()
    return ReplaySample.from_position(
        position,
        policy=policy,
        final_score=ScoreResult(
            players=(
                PlayerScore(10, 3, 1, 1, 0, 11),
                PlayerScore(5, 2, 1, 0, 0, 5),
            ),
            node_owner=torch.zeros(topology.n, dtype=torch.int8),
            alive_stone=torch.zeros(topology.n, dtype=torch.bool),
            contested_peries=0,
            leader=0,
        ),
        search_provenance=f"benchmark:{index}",
        policy_provenance="benchmark",
        game_id=f"game-benchmark-{index}",
    )


def test_benchmark_reports_decode_and_selected_row_rates(tmp_path) -> None:
    path = write_replay_shard(
        tmp_path / "benchmark.npz",
        [_sample(index) for index in range(8)],
    )

    result = benchmark_replay_shard(path, rows=3, repeats=2)

    assert result["benchmark"] == BENCHMARK_NAME
    assert result["sample_count"] == 8
    assert result["selected_rows"] == 3
    assert result["npz_members_loaded_per_repeat"] == 30
    assert result["decode_seconds"]["count"] == 2
    assert result["selected_row_materialization_seconds"]["minimum"] >= 0
    assert result["selected_rows_per_second"] > 0


def test_benchmark_cli_is_bounded_and_reports_errors(tmp_path, capsys) -> None:
    missing = tmp_path / "missing.npz"

    assert main(["--shard", str(missing), "--repeats", "1"]) == 2
    error = json.loads(capsys.readouterr().out)
    assert error["status"] == "error"

    path = write_replay_shard(tmp_path / "valid.npz", [_sample(0), _sample(1)])
    assert (
        main(
            [
                "--shard",
                str(path),
                "--rows",
                "1",
                "--repeats",
                "1",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["selected_rows"] == 1
