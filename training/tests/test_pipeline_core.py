from __future__ import annotations

import json
import hashlib
import os
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from startrain.config import DataConfig, LearnerConfig, SchedulerConfig, TrainConfig
from startrain.contracts import (
    TARGET_ALIVE,
    TARGET_OWNERSHIP,
    TARGET_POLICY,
    TARGET_SCORE_MARGIN,
    TARGET_WDL,
)
from startrain.inference import (
    GraphInferenceAdapter,
    InferenceConfig,
    InferenceResponse,
)
from startrain.learner import (
    LazyShardReplayDataset,
    LearnerLoop,
    UniqueReplayBatchSampler,
    plateau_policy_decision,
)
from startrain.losses import LossWeights
from startrain.model import GraphResTNet, ModelConfig, StarModelOutput
from startrain.native import BITBOARD_WORDS
from startrain.optim import OptimizerConfig, build_optimizer
from startrain.replay import ReplaySample, write_replay_shard
from startrain.replay_store import (
    DuplicateGameError,
    ReplayCursor,
    ReplaySelection,
    ReplaySpan,
    ReplayStore,
    ShardRecord,
)
from startrain.runtime import RunIdentity
from startrain.scoring import score_position
from startrain.selfplay import SelfPlayActor, SelfPlayConfig, SelfPlayIdentity
from startrain.topology import get_topology
from startrain.training import build_scheduler
from startrain.checkpoint import ExponentialMovingAverage
from startrain.features import DoubleStarPosition


def pack_mask(mask: torch.Tensor) -> list[int]:
    words = [0] * BITBOARD_WORDS
    for node in torch.nonzero(mask, as_tuple=False).flatten().tolist():
        words[node // 64] |= 1 << (node % 64)
    return words


@dataclass
class FakeStateData:
    rings: int
    node_count: int
    batch_size: int
    zero_bits: list[int]
    one_bits: list[int]
    legal_bits: list[int]
    hashes: list[int]
    stones_placed: list[int]
    to_move: list[int]
    moves_left: list[int]
    opening: list[bool]
    mid_turn: list[bool]
    pass_streak: list[int]
    terminal: list[bool]
    pass_legal: list[bool]


def state_data(position: DoubleStarPosition) -> FakeStateData:
    legal = (position.stones == -1) & (not position.terminal)
    return FakeStateData(
        rings=position.rings,
        node_count=position.stones.numel(),
        batch_size=1,
        zero_bits=pack_mask(position.stones == 0),
        one_bits=pack_mask(position.stones == 1),
        legal_bits=pack_mask(legal),
        hashes=[123],
        stones_placed=[int((position.stones >= 0).sum())],
        to_move=[position.to_move],
        moves_left=[position.moves_left],
        opening=[position.opening],
        mid_turn=[position.moves_left == 1 and not position.opening],
        pass_streak=[position.pass_streak],
        terminal=[position.terminal],
        pass_legal=[not position.terminal],
    )


@dataclass
class FakeEvalBatch:
    tokens: list[int]
    states: FakeStateData
    legal_offsets: list[int]
    legal_actions: list[int]

    def __len__(self) -> int:
        return len(self.tokens)


class FixedNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, *arguments: torch.Tensor) -> StarModelOutput:
        node_features = arguments[0]
        legal = arguments[-1]
        batch, nodes = node_features.shape[:2]
        policy = torch.arange(
            nodes + 1, device=node_features.device, dtype=node_features.dtype
        ).expand(batch, -1)
        policy = policy + self.anchor
        policy = policy.masked_fill(~legal, torch.finfo(policy.dtype).min)
        margin = torch.zeros(batch, 363, device=node_features.device)
        margin[:, 181] = 4
        return StarModelOutput(
            policy_logits=policy,
            wdl_logits=torch.tensor(
                [[0.0, 0.0, 2.0]], device=node_features.device
            ).expand(batch, -1),
            score_margin_logits=margin,
            ownership_logits=torch.zeros(batch, nodes, 3, device=node_features.device),
            alive_logits=torch.zeros(batch, nodes, device=node_features.device),
            soft_policy_logits=policy,
        )


def opening_position() -> DoubleStarPosition:
    topology = get_topology(3)
    return DoubleStarPosition(
        rings=3,
        stones=torch.full((topology.n,), -1, dtype=torch.int8),
        to_move=0,
        moves_left=1,
        opening=True,
        pass_streak=0,
        terminal=False,
    )


def test_inference_maps_dense_logits_to_native_legal_order_and_keeps_pass() -> None:
    position = opening_position()
    actions = list(range(position.stones.numel())) + [-1]
    requests = FakeEvalBatch(
        tokens=[99],
        states=state_data(position),
        legal_offsets=[0, len(actions)],
        legal_actions=actions,
    )
    adapter = GraphInferenceAdapter(
        FixedNetwork(),
        config=InferenceConfig(initial_pass_logit_penalty=2.5),
        model_version="fixed",
    )
    response = adapter.evaluate(requests)
    assert response.tokens == [99]
    assert response.policy_offsets == [0, len(actions)]
    assert response.policy_logits[:-1] == pytest.approx(
        list(range(position.stones.numel()))
    )
    assert response.policy_logits[-1] == pytest.approx(position.stones.numel() - 2.5)
    assert response.values[0] > 0
    assert actions[-1] == -1


class FakeEvaluator:
    model_version = "fake-v1"
    model_step = 7
    model_identity = "sha256-" + "f" * 64

    def evaluate(self, requests: FakeEvalBatch) -> InferenceResponse:
        return InferenceResponse(
            tokens=list(requests.tokens),
            values=[0.0] * len(requests),
            policy_offsets=list(requests.legal_offsets),
            policy_logits=[0.0] * len(requests.legal_actions),
        )


class FakeStateBatch:
    def __init__(self, rings: int, batch_size: int) -> None:
        assert batch_size == 1
        self.position = opening_position()
        self.last_action = -2

    def data(self) -> FakeStateData:
        return state_data(self.position)

    def apply_many(self, indices: list[int], actions: list[int]) -> None:
        assert indices == [0]
        self.last_action = actions[0]
        stones = self.position.stones.clone()
        if actions[0] >= 0:
            stones[actions[0]] = self.position.to_move
        self.position = DoubleStarPosition(
            rings=3,
            stones=stones,
            to_move=0,
            moves_left=2,
            opening=False,
            pass_streak=2,
            terminal=True,
        )

    def reset_many(self, indices: list[int]) -> None:
        assert indices == [0]
        self.position = opening_position()

    def score_data(self) -> object:
        score = score_position(get_topology(3), self.position.stones)
        components: list[int] = []
        for player in score.players:
            components.extend(
                [
                    player.peries,
                    player.quarks,
                    player.stars,
                    player.quark_peri,
                    player.award,
                    player.total,
                ]
            )
        components.extend([score.contested_peries, score.leader])
        value = (
            0.0
            if score.leader == -1
            else (1.0 if score.leader == self.position.to_move else -1.0)
        )
        margin = (
            score.players[self.position.to_move].total
            - score.players[1 - self.position.to_move].total
        )
        return SimpleNamespace(
            batch_size=1,
            node_count=30,
            components=components,
            node_owner=score.node_owner.tolist(),
            alive_bits=pack_mask(score.alive_stone),
            winner=[score.leader],
            terminal_value=[value],
            wdl_class=[int(value) + 1],
            score_margin=[margin],
            terminal_reason=[2],
        )

    def trajectory_data(self) -> object:
        return SimpleNamespace(
            batch_size=1,
            last_move=[self.last_action],
            current_turn_offsets=[0, 0],
            current_turn_moves=[],
            turn_count=[1],
        )


class FakeSearchBatch:
    seeds: list[int] = []

    def __init__(self, states: FakeStateBatch, **options: object) -> None:
        self.states = states
        self.initialized = False
        self.seeds.append(int(options["deterministic_seed"]))

    def root_requests(self) -> FakeEvalBatch:
        data = self.states.data()
        actions = list(range(30)) + [-1]
        return FakeEvalBatch([55], data, [0, len(actions)], actions)

    def initialize_roots(self, *buffers: object) -> None:
        self.initialized = True

    def is_done(self) -> bool:
        return self.initialized

    def next_requests(self) -> FakeEvalBatch:
        raise AssertionError("fake search is complete after root initialization")

    def submit(self, *buffers: object) -> None:
        raise AssertionError("fake search has no leaf requests")

    def results(self) -> object:
        target = [1.0] + [0.0] * 30
        return SimpleNamespace(
            selected_actions=[0],
            terminal=[False],
            terminal_values=[0.0],
            action_offsets=[0, 31],
            actions=list(range(30)) + [-1],
            visits=[1] + [0] * 30,
            q_values=[0.0] * 31,
            policy_target=target,
        )


class FakeNative:
    StateBatch = FakeStateBatch
    SearchBatch = FakeSearchBatch


class CapturingSink:
    def __init__(self) -> None:
        self.samples: list[ReplaySample] = []
        self.calls: list[dict[str, object]] = []

    def append(self, samples: list[ReplaySample], **metadata: object) -> object:
        self.samples.extend(samples)
        self.calls.append(metadata)
        return object()


@pytest.mark.parametrize("full", [False, True])
def test_selfplay_retains_all_targets_but_policy_only_for_full_search(
    full: bool,
) -> None:
    sink = CapturingSink()
    config = SelfPlayConfig(
        rings=3,
        batch_size=1,
        games=1,
        fast_probability=0.0 if full else 1.0,
        full_probability=1.0 if full else 0.0,
        fast_simulations=2,
        full_simulations=4,
        simulation_reference_rings=3,
        max_considered=2,
        shard_size=8,
        seed=41,
    )
    summaries = SelfPlayActor(FakeNative, FakeEvaluator(), sink, config).run()
    assert len(summaries) == 1
    assert summaries[0].model_version == "fake-v1"
    assert len(sink.samples) == 1
    target_mask = sink.samples[0].target_mask
    required = TARGET_WDL | TARGET_SCORE_MARGIN | TARGET_OWNERSHIP | TARGET_ALIVE
    assert target_mask & required == required
    assert bool(target_mask & TARGET_POLICY) is full
    if full:
        assert sink.samples[0].policy[0] == 1
        assert "completed-q" in sink.samples[0].policy_provenance
    else:
        assert not sink.samples[0].policy.any()
    assert sink.calls[0]["model_step"] == 7


def test_selfplay_playout_randomization_is_seeded_and_board_scaled() -> None:
    config = SelfPlayConfig(
        rings=3,
        batch_size=1,
        games=1,
        fast_probability=0.5,
        full_probability=0.5,
        fast_simulations=8,
        full_simulations=32,
        simulation_reference_rings=6,
        simulation_ring_exponent=1.0,
        max_considered=2,
        seed=1234,
    )
    assert config.simulation_budget(full=False) == 4
    assert config.simulation_budget(full=True) == 16
    runs = []
    for _ in range(2):
        FakeSearchBatch.seeds.clear()
        sink = CapturingSink()
        SelfPlayActor(FakeNative, FakeEvaluator(), sink, config).run()
        runs.append(
            (
                list(FakeSearchBatch.seeds),
                sink.samples[0].policy.copy(),
                sink.samples[0].search_provenance,
            )
        )
    assert runs[0][0] == runs[1][0]
    np.testing.assert_array_equal(runs[0][1], runs[1][1])
    assert runs[0][2] == runs[1][2]


def test_selfplay_exact_cohort_never_resets_or_drops_uneven_games(
    monkeypatch,
) -> None:
    class UnevenStates:
        reset_calls = 0

        def __init__(self, rings: int, batch_size: int) -> None:
            assert rings == 3 and batch_size == 2
            self.remaining = [1, 3]
            self.turns = [0, 0]

        def data(self):
            return SimpleNamespace(
                terminal=[remaining == 0 for remaining in self.remaining],
                stones_placed=list(self.turns),
            )

        def apply_many(self, indices, _actions):
            for row in indices:
                self.remaining[row] -= 1
                self.turns[row] += 1

        def reset_many(self, _indices):
            UnevenStates.reset_calls += 1
            raise AssertionError("exact cohorts must never reset rows")

        def score_data(self):
            return SimpleNamespace(
                terminal_value=[0.0, 0.0],
                score_margin=[0, 0],
                terminal_reason=[2, 2],
                winner=[-1, -1],
            )

        def trajectory_data(self):
            return SimpleNamespace()

    class UnevenSearch:
        def __init__(self, states, **_options):
            self.states = states
            self.initialized = False

        def root_requests(self):
            active = sum(value > 0 for value in self.states.remaining)
            return SimpleNamespace(
                tokens=list(range(active)),
                legal_offsets=list(range(active + 1)),
                legal_actions=[0] * active,
                __len__=lambda: active,
            )

        def initialize_roots(self, *_buffers):
            self.initialized = True

        def is_done(self):
            return self.initialized

        def next_requests(self):
            raise AssertionError

        def submit(self, *_buffers):
            raise AssertionError

        def results(self):
            terminal = [value == 0 for value in self.states.remaining]
            actions = [0 for value in self.states.remaining if value > 0]
            offsets = [0]
            for is_terminal in terminal:
                offsets.append(offsets[-1] + (0 if is_terminal else 1))
            return SimpleNamespace(
                selected_actions=[-2 if is_terminal else 0 for is_terminal in terminal],
                terminal=terminal,
                action_offsets=offsets,
                actions=actions,
                policy_target=[1.0] * len(actions),
            )

    class UnevenEvaluator(FakeEvaluator):
        def evaluate(self, requests):
            count = len(requests.tokens)
            return InferenceResponse(
                tokens=list(requests.tokens),
                values=[0.0] * count,
                policy_offsets=list(requests.legal_offsets),
                policy_logits=[0.0] * count,
            )

    topology = get_topology(3)
    position = opening_position()
    final_score = score_position(topology, position.stones)
    monkeypatch.setattr(
        "startrain.selfplay.positions_from_native",
        lambda data: [position for _ in data.terminal],
    )
    monkeypatch.setattr(
        "startrain.selfplay.score_results_from_native",
        lambda _data: [final_score, final_score],
    )
    monkeypatch.setattr(
        "startrain.selfplay.trajectory_rows_from_native",
        lambda _data: [
            SimpleNamespace(turn_count=1, last_move=0),
            SimpleNamespace(turn_count=3, last_move=0),
        ],
    )
    sink = CapturingSink()
    summaries = SelfPlayActor(
        SimpleNamespace(StateBatch=UnevenStates, SearchBatch=UnevenSearch),
        UnevenEvaluator(),
        sink,
        SelfPlayConfig(
            rings=3,
            batch_size=2,
            games=2,
            fast_probability=0.0,
            full_probability=1.0,
            fast_simulations=1,
            full_simulations=1,
            max_considered=1,
            shard_size=64,
        ),
        SelfPlayIdentity("run-test", "family-test", "actor-test", 4),
    ).run()
    assert len(summaries) == 2
    assert sorted(summary.samples for summary in summaries) == [1, 3]
    assert len(sink.samples) == 4
    assert UnevenStates.reset_calls == 0
    assert len({summary.game_id for summary in summaries}) == 2
    for game_id in {sample.game_id for sample in sink.samples}:
        assert sorted(
            sample.ply for sample in sink.samples if sample.game_id == game_id
        ) == list(range(sum(sample.game_id == game_id for sample in sink.samples)))

    stopping = {"requested": False}
    drained_sink = CapturingSink()

    def request_stop(**details):
        if details.get("completed_games") == 2:
            stopping["requested"] = True

    drained = SelfPlayActor(
        SimpleNamespace(StateBatch=UnevenStates, SearchBatch=UnevenSearch),
        UnevenEvaluator(),
        drained_sink,
        SelfPlayConfig(
            rings=3,
            batch_size=2,
            games=4,
            fast_probability=0.0,
            full_probability=1.0,
            fast_simulations=1,
            full_simulations=1,
            max_considered=1,
            shard_size=64,
        ),
        SelfPlayIdentity("run-test", "family-test", "actor-stop", 5),
    ).run(
        stop_requested=lambda: stopping["requested"],
        progress=request_stop,
    )
    assert len(drained) == 2
    assert len(drained_sink.samples) == sum(summary.samples for summary in drained)


def make_replay_sample(
    rings: int = 3,
    *,
    identity: RunIdentity | None = None,
    actor_id: str = "actor-test",
    generation: int = 0,
    game_id: str | None = None,
    ply: int = 0,
    model_identity: str = "sha256-" + "1" * 64,
) -> ReplaySample:
    topology = get_topology(rings)
    stones = torch.full((topology.n,), -1, dtype=torch.int8)
    position = DoubleStarPosition(
        rings=rings,
        stones=stones,
        to_move=0,
        moves_left=2,
        opening=False,
        pass_streak=0,
        terminal=False,
    )
    policy = np.ones(topology.n + 1, dtype=np.float32)
    policy /= policy.sum()
    return ReplaySample.from_position(
        position,
        policy=policy,
        final_score=score_position(topology, stones),
        search_provenance="test",
        policy_provenance="completed-q",
        run_id=identity.run_id if identity is not None else "manual",
        generation_family=(
            identity.generation_family if identity is not None else "manual"
        ),
        actor_id=actor_id,
        generation=generation,
        game_id=game_id,
        ply=ply,
        model_identity=model_identity,
    )


def run_identity(tmp_path) -> RunIdentity:
    return RunIdentity(
        tmp_path / "run.json",
        "run-test",
        "family-test",
        1,
    )


def test_replay_store_manifest_recency_lag_and_cursor(tmp_path) -> None:
    identity = run_identity(tmp_path)
    with ReplayStore(tmp_path / "replay") as store:
        generation = store.lease_generation(identity, "actor-test")
        assert generation == 0
        first = store.append(
            [make_replay_sample(identity=identity, game_id="game-first")],
            phase_min=0,
            phase_max=3,
            model_version="sha256-" + "1" * 64,
            model_step=1,
            model_identity="sha256-" + "1" * 64,
            run_id=identity.run_id,
            generation_family=identity.generation_family,
            actor_id="actor-test",
            generation=0,
        )
        second = store.append(
            [
                make_replay_sample(
                    identity=identity,
                    game_id="game-second",
                    ply=0,
                ),
                make_replay_sample(
                    identity=identity,
                    game_id="game-second",
                    ply=1,
                ),
            ],
            phase_min=4,
            phase_max=8,
            model_version="sha256-" + "1" * 64,
            model_step=10,
            model_identity="sha256-" + "1" * 64,
            run_id=identity.run_id,
            generation_family=identity.generation_family,
            actor_id="actor-test",
            generation=0,
        )
        assert first.path.exists() and second.path.exists()
        assert store.connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        recent = store.load_recent_samples(
            sample_window=2,
            run_id=identity.run_id,
            generation_family=identity.generation_family,
            current_model_step=10,
            max_model_lag_steps=2,
        )
        assert len(recent) == 2
        future = store.append(
            [make_replay_sample(identity=identity, game_id="game-future")],
            phase_min=0,
            phase_max=0,
            model_version="sha256-" + "1" * 64,
            model_step=11,
            model_identity="sha256-" + "1" * 64,
            run_id=identity.run_id,
            generation_family=identity.generation_family,
            actor_id="actor-test",
            generation=0,
        )
        eligible = store.recent_shards(
            sample_window=100,
            run_id=identity.run_id,
            generation_family=identity.generation_family,
            current_model_step=10,
            max_model_lag_steps=10,
        )
        assert future.shard_id not in {record.shard_id for record in eligible}
        before = set(store.shard_directory.glob("*.npz"))
        with pytest.raises(DuplicateGameError):
            store.append(
                [
                    make_replay_sample(
                        identity=identity,
                        game_id="game-second",
                        ply=0,
                    ),
                    make_replay_sample(
                        identity=identity,
                        game_id="game-second",
                        ply=1,
                    ),
                ],
                phase_min=0,
                phase_max=1,
                model_version="sha256-" + "1" * 64,
                model_step=10,
                model_identity="sha256-" + "1" * 64,
                run_id=identity.run_id,
                generation_family=identity.generation_family,
                actor_id="actor-test",
                generation=0,
            )
        assert set(store.shard_directory.glob("*.npz")) == before
        store.set_cursor("learner", ReplayCursor(first.shard_id, 0))
        consumed = list(store.iter_after_cursor("learner"))
        assert [record.shard_id for record, _ in consumed] == [
            first.shard_id,
            second.shard_id,
            future.shard_id,
        ]
        store.set_cursor("learner", ReplayCursor(second.shard_id + 1, 0))
        assert [
            record.shard_id for record, _ in store.iter_after_cursor("learner")
        ] == [future.shard_id]
        assert store.lease_generation(identity, "actor-test") == 1
    orphan = tmp_path / "replay" / "shards" / "orphan.npz"
    orphan.write_bytes(b"orphan")
    os.utime(orphan, (1, 1))
    with ReplayStore(tmp_path / "replay"):
        assert not orphan.exists()


def test_replay_gc_watermark_and_quarantine_preserve_committed_ledger(
    tmp_path,
) -> None:
    identity = run_identity(tmp_path)
    records = []
    with ReplayStore(tmp_path / "gc-replay") as store:
        generation = store.lease_generation(identity, "actor-test")
        for index in range(3):
            records.append(
                store.append(
                    [
                        make_replay_sample(
                            identity=identity,
                            game_id=f"game-gc-{index}",
                        )
                    ],
                    phase_min=0,
                    phase_max=0,
                    model_version="sha256-" + "1" * 64,
                    model_step=0,
                    model_identity="sha256-" + "1" * 64,
                    run_id=identity.run_id,
                    generation_family=identity.generation_family,
                    actor_id="actor-test",
                    generation=generation,
                )
            )
        protected = ReplaySelection(
            (ReplaySpan(records[0], 0, 1),),
            {3: 1},
            records[-1].shard_id,
        )
        store.set_gc_watermark("learner", protected)
        dry = store.collect_garbage(
            run_id=identity.run_id,
            generation_family=identity.generation_family,
            retain_shards_per_ring=1,
            dry_run=True,
        )
        assert dry["candidate_shards"] == 1
        assert all(record.path.exists() for record in records)
        deleted = store.collect_garbage(
            run_id=identity.run_id,
            generation_family=identity.generation_family,
            retain_shards_per_ring=1,
            dry_run=False,
        )
        assert deleted["deleted_shards"] == 1
        assert records[0].path.exists()
        assert not records[1].path.exists()
        assert records[2].path.exists()
        store.clear_gc_watermark("learner")

    records[0].path.unlink()
    with records[2].path.open("ab") as stream:
        stream.write(b"corrupt")
    with ReplayStore(tmp_path / "gc-replay") as store:
        states = {
            int(row["id"]): (row["state"], row["quarantine_reason"])
            for row in store.connection.execute(
                "SELECT id, state, quarantine_reason FROM shards"
            )
        }
        assert states[records[0].shard_id][0] == "quarantined"
        assert "missing" in states[records[0].shard_id][1]
        assert states[records[2].shard_id][0] == "quarantined"
        assert "checksum" in states[records[2].shard_id][1]
        game_count = store.connection.execute("SELECT COUNT(*) FROM games").fetchone()[
            0
        ]
        assert game_count == 3


def tiny_model() -> GraphResTNet:
    return GraphResTNet(
        ModelConfig(
            width=8,
            rrt_groups=1,
            attention_heads=2,
            kv_heads=1,
        )
    )


def test_learner_runs_homogeneous_batch_and_publishes_atomically(tmp_path) -> None:
    identity = run_identity(tmp_path)
    with ReplayStore(tmp_path / "replay") as store:
        generation = store.lease_generation(identity, "actor-test")
        assert generation == 0
        store.append(
            [
                make_replay_sample(identity=identity, game_id="game-learner-a"),
                make_replay_sample(identity=identity, game_id="game-learner-b"),
            ],
            phase_min=0,
            phase_max=0,
            model_version="sha256-" + "1" * 64,
            model_step=0,
            model_identity="sha256-" + "1" * 64,
            run_id=identity.run_id,
            generation_family=identity.generation_family,
            actor_id="actor-test",
            generation=0,
        )
        model = tiny_model()
        optimizer = build_optimizer(model, OptimizerConfig(kind="adamw"))
        scheduler = build_scheduler(
            optimizer, SchedulerConfig(warmup_steps=0, total_steps=4)
        )
        ema = ExponentialMovingAverage(model, decay=0.9)
        learner = LearnerLoop(
            store=store,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            ema=ema,
            output_directory=tmp_path / "learner",
            learner_config=LearnerConfig(
                steps=1,
                recent_samples_per_ring=8,
                max_replay_lag_steps=10,
                steps_per_window=1,
                candidate_interval=1,
                metrics_interval=1,
                device="cpu",
            ),
            train_config=TrainConfig(
                per_rank_batch_size=2,
                precision="fp32",
                compile=False,
                gradient_clip_norm=1.0,
                scheduler=SchedulerConfig(warmup_steps=0, total_steps=4),
            ),
            data_config=DataConfig(workers=0, ring_stratified=False),
            loss_weights=LossWeights(),
            seed=5,
            serialized_config={"schema_version": 2},
            run_identity=identity,
        )
        assert learner.run(steps=1) == 1
        manifest_path = tmp_path / "learner" / "candidate.json"
        manifest = json.loads(manifest_path.read_text())
        assert manifest["model_step"] == 1
        assert manifest["role"] == "candidate"
        published = learner.publisher.publish(
            model=learner.model,
            optimizer=learner.optimizer,
            scheduler=learner.scheduler,
            ema=learner.ema,
            step=learner.step,
            epoch=learner.epoch,
            config=learner.serialized_config,
        )
        checkpoint = published.checkpoint
        assert checkpoint.exists()
        assert not (tmp_path / "learner" / "champion.json").exists()
        assert (tmp_path / "learner" / "metrics.jsonl").read_text().count("\n") == 1

        restored_model = tiny_model()
        restored_optimizer = build_optimizer(
            restored_model, OptimizerConfig(kind="adamw")
        )
        restored_scheduler = build_scheduler(
            restored_optimizer, SchedulerConfig(warmup_steps=0, total_steps=4)
        )
        restored = LearnerLoop(
            store=store,
            model=restored_model,
            optimizer=restored_optimizer,
            scheduler=restored_scheduler,
            ema=ExponentialMovingAverage(restored_model, decay=0.9),
            output_directory=tmp_path / "restored",
            learner_config=learner.learner_config,
            train_config=learner.train_config,
            data_config=learner.data_config,
            loss_weights=LossWeights(),
            seed=5,
            serialized_config={"schema_version": 2},
            run_identity=identity,
        )
        restored.resume(
            checkpoint,
            expected_sha256=published.checkpoint_sha256,
            expected_bytes=published.checkpoint_bytes,
        )
        assert restored.step == 1


def test_learner_publishes_initial_ema_then_waits_for_minimum_replay(
    tmp_path,
) -> None:
    identity = run_identity(tmp_path)
    with ReplayStore(tmp_path / "empty-replay") as store:
        generation = store.lease_generation(identity, "actor-test")
        store.append(
            [make_replay_sample(identity=identity, game_id="game-only-sample")],
            phase_min=0,
            phase_max=0,
            model_version="sha256-" + "1" * 64,
            model_step=0,
            model_identity="sha256-" + "1" * 64,
            run_id=identity.run_id,
            generation_family=identity.generation_family,
            actor_id="actor-test",
            generation=generation,
        )
        model = tiny_model()
        optimizer = build_optimizer(model, OptimizerConfig(kind="adamw"))
        scheduler = build_scheduler(
            optimizer, SchedulerConfig(warmup_steps=0, total_steps=4)
        )
        learner = LearnerLoop(
            store=store,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            ema=ExponentialMovingAverage(model, decay=0.9),
            output_directory=tmp_path / "waiting-learner",
            learner_config=LearnerConfig(
                steps=1,
                minimum_replay_samples=1,
                recent_samples_per_ring=8,
                max_replay_lag_steps=10,
                steps_per_window=1,
                candidate_interval=1,
                metrics_interval=1,
                replay_poll_seconds=0.001,
                replay_wait_timeout_seconds=0.01,
                device="cpu",
            ),
            train_config=TrainConfig(
                per_rank_batch_size=2,
                scheduler=SchedulerConfig(warmup_steps=0, total_steps=4),
            ),
            data_config=DataConfig(workers=0, ring_stratified=False),
            loss_weights=LossWeights(),
            seed=5,
            serialized_config={"schema_version": 2},
            run_identity=identity,
        )
        with pytest.raises(TimeoutError, match="minimum replay"):
            learner.run()
        manifest = json.loads(
            (tmp_path / "waiting-learner" / "candidate.json").read_text()
        )
        assert manifest["model_step"] == 0
        assert manifest["role"] == "candidate"


def test_lazy_replay_sampler_is_unique_deterministic_and_homogeneous(
    tmp_path,
) -> None:
    identity = run_identity(tmp_path)
    with ReplayStore(tmp_path / "lazy-replay") as store:
        generation = store.lease_generation(identity, "actor-test")
        assert generation == 0
        for ring in (3, 4):
            samples = [
                make_replay_sample(
                    ring,
                    identity=identity,
                    game_id=f"game-ring-{ring}-{index}",
                )
                for index in range(4)
            ]
            store.append(
                samples,
                phase_min=0,
                phase_max=0,
                model_version="sha256-" + "1" * 64,
                model_step=0,
                model_identity="sha256-" + "1" * 64,
                run_id=identity.run_id,
                generation_family=identity.generation_family,
                actor_id="actor-test",
                generation=0,
            )
        selection = store.select_recent_spans(
            rings=(3, 4),
            per_ring_quota=4,
            run_id=identity.run_id,
            generation_family=identity.generation_family,
            current_model_step=0,
            max_model_lag_steps=0,
        )
        assert selection.max_shard_id == 2
    dataset = LazyShardReplayDataset(
        selection,
        seed=9,
        epoch=2,
        augmentation_enabled=False,
        shard_cache_size=1,
    )

    def batches() -> list[list[int]]:
        return list(
            UniqueReplayBatchSampler(
                dataset,
                batch_size=2,
                batches=4,
                seed=9,
                epoch=2,
                ring_stratified=True,
            )
        )

    selected = batches()
    assert selected == batches()
    flattened = [index for batch in selected for index in batch]
    assert len(flattened) == len(set(flattened)) == 8
    for batch in selected:
        assert len({dataset[index].rings for index in batch}) == 1


def test_shard_aware_512_batch_decompresses_one_selected_shard(tmp_path) -> None:
    sample = make_replay_sample(game_id="game-amplification")
    spans = []
    for shard_id in range(1, 5):
        path = write_replay_shard(
            tmp_path / f"shard-{shard_id}.npz",
            [sample] * 512,
        )
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        record = ShardRecord(
            shard_id=shard_id,
            path=path,
            created_ns=shard_id,
            sample_count=512,
            ring=3,
            phase_min=0,
            phase_max=0,
            model_version="sha256-" + "1" * 64,
            model_step=0,
            model_identity="sha256-" + "1" * 64,
            run_id="run-test",
            generation_family="family-test",
            actor_id="actor-test",
            generation=0,
            game_count=512,
            checksum_sha256=digest,
            state="ready",
            quarantine_reason=None,
        )
        spans.append(ReplaySpan(record, 0, 512))
    dataset = LazyShardReplayDataset(
        ReplaySelection(tuple(spans), {3: 2048}, 4),
        seed=1,
        epoch=0,
        augmentation_enabled=False,
        shard_cache_size=1,
    )
    batch = next(
        iter(
            UniqueReplayBatchSampler(
                dataset,
                batch_size=512,
                batches=1,
                seed=1,
                epoch=0,
                ring_stratified=True,
            )
        )
    )
    for index in batch:
        dataset[index]
    assert dataset.shard_load_count == 1
    assert dataset.checksum_verification_count == 1


def test_plateau_policy_keeps_champion_replay_live_and_resets_after_rejections() -> (
    None
):
    common = {
        "soft_lag_steps": 20,
        "hard_replay_lag_steps": 50,
        "status_matches_candidate": True,
        "reset_after_rejections": 3,
        "action": "reset_from_champion",
        "reset_already_applied": False,
    }
    assert (
        plateau_policy_decision(
            lag_steps=20,
            terminal_rejection=False,
            rejection_streak=0,
            **common,
        )
        == "pause"
    )
    assert (
        plateau_policy_decision(
            lag_steps=20,
            terminal_rejection=True,
            rejection_streak=2,
            **common,
        )
        == "proceed"
    )
    assert (
        plateau_policy_decision(
            lag_steps=20,
            terminal_rejection=True,
            rejection_streak=3,
            **common,
        )
        == "reset"
    )
    assert (
        plateau_policy_decision(
            lag_steps=50,
            terminal_rejection=True,
            rejection_streak=2,
            **common,
        )
        == "pause"
    )


def test_ddp_replay_selection_metadata_is_broadcast_from_rank_zero(
    monkeypatch,
) -> None:
    expected = ReplaySelection((), {3: 512}, 42)
    learner = object.__new__(LearnerLoop)
    learner.world_size = 2
    learner.model = tiny_model()

    def broadcast(payload, **_options):
        assert payload == [None]
        payload[0] = expected

    monkeypatch.setattr(torch.distributed, "broadcast_object_list", broadcast)
    assert learner._broadcast_object(None) is expected
