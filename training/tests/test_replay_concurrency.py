from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import threading

import numpy as np
import torch

from startrain.features import DoubleStarPosition
from startrain.replay import ReplaySample
from startrain.replay_store import ReplayStore
from startrain.runtime import RunIdentity
from startrain.scoring import score_position
from startrain.topology import get_topology


MODEL_IDENTITY = "sha256-" + "c" * 64


def test_cold_replay_store_allows_simultaneous_wal_initialization(tmp_path) -> None:
    root = tmp_path / "cold-replay"
    identity = RunIdentity(
        tmp_path / "run.json",
        "run-cold-concurrent",
        "family-cold-concurrent",
        1,
    )
    actors = 8
    barrier = threading.Barrier(actors)

    def open_actor(actor: int) -> tuple[int, str]:
        barrier.wait()
        with ReplayStore(root) as store:
            generation = store.lease_generation(identity, f"actor-{actor}")
            journal = store.connection.execute("PRAGMA journal_mode").fetchone()[0]
            return generation, str(journal)

    with ThreadPoolExecutor(max_workers=actors) as executor:
        results = list(executor.map(open_actor, range(actors)))

    assert results == [(0, "wal")] * actors


def _sample(
    identity: RunIdentity,
    *,
    actor_id: str,
    generation: int,
    game_id: str,
) -> ReplaySample:
    topology = get_topology(3)
    stones = torch.full((topology.n,), -1, dtype=torch.int8)
    position = DoubleStarPosition(
        rings=3,
        stones=stones,
        to_move=0,
        moves_left=1,
        opening=True,
        pass_streak=0,
        terminal=False,
    )
    policy = np.ones(topology.n + 1, dtype=np.float32)
    policy /= policy.sum()
    return ReplaySample.from_position(
        position,
        policy=policy,
        final_score=score_position(topology, stones),
        search_provenance="concurrency-test",
        policy_provenance="completed-q",
        run_id=identity.run_id,
        generation_family=identity.generation_family,
        actor_id=actor_id,
        generation=generation,
        game_id=game_id,
        model_identity=MODEL_IDENTITY,
    )


def test_sqlite_wal_accepts_concurrent_actor_commits_without_lost_games(
    tmp_path,
) -> None:
    root = tmp_path / "replay"
    identity = RunIdentity(
        tmp_path / "run.json",
        "run-concurrent",
        "family-concurrent",
        1,
    )
    with ReplayStore(root) as store:
        store.register_run(identity)

    actors = 4
    shards_per_actor = 5

    def write_actor(actor: int) -> None:
        actor_id = f"actor-{actor}"
        with ReplayStore(root) as store:
            generation = store.lease_generation(identity, actor_id)
            for shard in range(shards_per_actor):
                sample = _sample(
                    identity,
                    actor_id=actor_id,
                    generation=generation,
                    game_id=f"game-{actor}-{shard}",
                )
                store.append(
                    [sample],
                    phase_min=0,
                    phase_max=0,
                    model_version=MODEL_IDENTITY,
                    model_step=0,
                    model_identity=MODEL_IDENTITY,
                    run_id=identity.run_id,
                    generation_family=identity.generation_family,
                    actor_id=actor_id,
                    generation=generation,
                )

    with ThreadPoolExecutor(max_workers=actors) as executor:
        list(executor.map(write_actor, range(actors)))

    expected = actors * shards_per_actor
    with ReplayStore(root) as store:
        shard_count = store.connection.execute(
            "SELECT COUNT(*) FROM shards WHERE state = 'ready'"
        ).fetchone()[0]
        game_count = store.connection.execute("SELECT COUNT(*) FROM games").fetchone()[
            0
        ]
        assert shard_count == expected
        assert game_count == expected
        assert not any(store.reconciliation_metrics.values())
