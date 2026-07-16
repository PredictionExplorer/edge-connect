from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from startrain.checkpoint import (
    ExponentialMovingAverage,
    load_model_manifest,
    save_checkpoint,
)
from startrain.config import (
    CurriculumStage,
    DataConfig,
    LearnerConfig,
    PlateauConfig,
    RingMixtureConfig,
    RingWeightStage,
    SchedulerConfig,
    TrainConfig,
)
from startrain.contracts import (
    TARGET_ALIVE,
    TARGET_OUTCOME,
    TARGET_OWNERSHIP,
    TARGET_POLICY,
    TARGET_SCORE_MARGIN,
)
from startrain.features import DoubleStarPosition
from startrain.inference import (
    GraphInferenceAdapter,
    InferenceConfig,
    InferenceMetrics,
    InferenceResponse,
)
from startrain.learner import (
    LazyShardReplayDataset,
    LearnerLoop,
    ReplayWindowSession,
    UTDSegmentState,
    UniqueReplayBatchSampler,
    _maximum_cross_shard_groups,
    _weighted_ring_quotas,
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
from startrain.scoring import PlayerScore, ScoreResult, score_position
from startrain.selfplay import (
    SelfPlayActor,
    SelfPlayConfig,
    SelfPlayMetrics,
)
from startrain.topology import SUPPORTED_RINGS, get_topology
from startrain.training import build_scheduler


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
    terminal: list[bool]


def state_data(positions: list[DoubleStarPosition]) -> FakeStateData:
    assert positions and len({position.rings for position in positions}) == 1
    zero_bits: list[int] = []
    one_bits: list[int] = []
    legal_bits: list[int] = []
    for position in positions:
        zero_bits.extend(pack_mask(position.stones == 0))
        one_bits.extend(pack_mask(position.stones == 1))
        legal_bits.extend(pack_mask((position.stones == -1) & (not position.terminal)))
    return FakeStateData(
        rings=positions[0].rings,
        node_count=positions[0].stones.numel(),
        batch_size=len(positions),
        zero_bits=zero_bits,
        one_bits=one_bits,
        legal_bits=legal_bits,
        hashes=list(range(len(positions))),
        stones_placed=[int((position.stones >= 0).sum()) for position in positions],
        to_move=[position.to_move for position in positions],
        moves_left=[position.moves_left for position in positions],
        opening=[position.opening for position in positions],
        mid_turn=[
            position.moves_left == 1 and not position.opening for position in positions
        ],
        terminal=[position.terminal for position in positions],
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
            nodes, device=node_features.device, dtype=node_features.dtype
        ).expand(batch, -1)
        policy = (policy + self.anchor).masked_fill(
            ~legal, torch.finfo(policy.dtype).min
        )
        margin = torch.zeros(batch, 303, device=node_features.device)
        margin[:, 151] = 4
        return StarModelOutput(
            policy_logits=policy,
            outcome_logits=torch.tensor(
                [[0.0, 2.0]], device=node_features.device
            ).expand(batch, -1),
            score_margin_logits=margin,
            ownership_logits=torch.zeros(batch, nodes, 3, device=node_features.device),
            alive_logits=torch.zeros(batch, nodes, device=node_features.device),
            soft_policy_logits=policy,
        )


def opening_position() -> DoubleStarPosition:
    topology = get_topology(4)
    return DoubleStarPosition(
        rings=4,
        stones=torch.full((topology.n,), -1, dtype=torch.int8),
        to_move=0,
        moves_left=1,
        opening=True,
        terminal=False,
    )


def test_inference_maps_node_logits_to_native_legal_order() -> None:
    first = opening_position()
    second_stones = first.stones.clone()
    second_stones[0] = 0
    second = DoubleStarPosition(
        rings=4,
        stones=second_stones,
        to_move=1,
        moves_left=2,
        opening=False,
        terminal=False,
    )
    first_actions = list(range(first.stones.numel()))
    second_actions = list(range(1, second.stones.numel()))
    requests = FakeEvalBatch(
        tokens=[7, 9],
        states=state_data([first, second]),
        legal_offsets=[0, len(first_actions), len(first_actions) + len(second_actions)],
        legal_actions=[*first_actions, *second_actions],
    )
    adapter = GraphInferenceAdapter(FixedNetwork(), model_version="fixed")

    detailed = adapter.evaluate_detailed(requests)

    assert detailed.response.tokens == [7, 9]
    assert detailed.response.policy_offsets == requests.legal_offsets
    assert detailed.response.policy_logits[: len(first_actions)] == pytest.approx(
        first_actions
    )
    assert detailed.response.policy_logits[len(first_actions) :] == pytest.approx(
        second_actions
    )
    assert detailed.outcome_probabilities[0] == pytest.approx([0.11920292, 0.88079708])
    assert detailed.outcome_values[0] == pytest.approx(0.76159416)
    assert all(action >= 0 for action in requests.legal_actions)


def test_inference_preserves_uneven_multirow_node_csr_and_metrics() -> None:
    first = opening_position()
    second_stones = first.stones.clone()
    second_stones[[0, 3]] = torch.tensor([0, 1], dtype=torch.int8)
    second = DoubleStarPosition(
        rings=4,
        stones=second_stones,
        to_move=0,
        moves_left=1,
        opening=False,
        terminal=False,
    )
    first_actions = list(range(first.stones.numel()))
    second_actions = [
        node for node, stone in enumerate(second.stones.tolist()) if stone == -1
    ]
    requests = FakeEvalBatch(
        tokens=[7, 9],
        states=state_data([first, second]),
        legal_offsets=[0, len(first_actions), len(first_actions) + len(second_actions)],
        legal_actions=[*first_actions, *second_actions],
    )
    adapter = GraphInferenceAdapter(FixedNetwork(), model_version="fixed")
    before = adapter.metrics_snapshot()

    response = adapter.evaluate(requests)

    assert response.tokens == [7, 9]
    assert response.policy_offsets == requests.legal_offsets
    assert response.policy_logits[: len(first_actions)] == pytest.approx(first_actions)
    assert response.policy_logits[len(first_actions) :] == pytest.approx(second_actions)
    after = adapter.metrics_snapshot()
    assert after.evaluator_calls == 1
    assert after.evaluator_rows == 2
    assert after.delta(before) == after
    assert adapter.last_feature_path == "python"


def test_inference_validates_configuration_metrics_and_empty_batches() -> None:
    with pytest.raises(ValueError, match="precision"):
        InferenceConfig(precision="fp16")
    with pytest.raises(ValueError, match="score_utility_weight"):
        InferenceConfig(score_utility_weight=1.1)
    with pytest.raises(ValueError, match="monotonic"):
        InferenceMetrics().delta(InferenceMetrics(evaluator_calls=1))

    empty = FakeEvalBatch(
        tokens=[],
        states=state_data([opening_position()]),
        legal_offsets=[0],
        legal_actions=[],
    )
    empty.states.batch_size = 0
    adapter = GraphInferenceAdapter(FixedNetwork())
    assert adapter.evaluate(empty) == InferenceResponse([], [], [0], [])

    malformed = FakeEvalBatch(
        tokens=[1],
        states=state_data([opening_position()]),
        legal_offsets=[1, 1],
        legal_actions=[],
    )
    with pytest.raises(ValueError, match="CSR"):
        adapter.evaluate(malformed)


def decisive_score(position: DoubleStarPosition) -> ScoreResult:
    topology = get_topology(position.rings)
    return ScoreResult(
        players=(
            PlayerScore(10, 3, 1, 1, 0, 11),
            PlayerScore(5, 2, 1, 0, 0, 5),
        ),
        node_owner=torch.zeros(topology.n, dtype=torch.int8),
        alive_stone=torch.zeros(topology.n, dtype=torch.bool),
        contested_peries=0,
        leader=0,
    )


def replay_sample(identity: RunIdentity, game_id: str) -> ReplaySample:
    position = opening_position()
    policy = np.ones(position.stones.numel(), dtype=np.float32)
    policy /= policy.sum()
    return ReplaySample.from_position(
        position,
        policy=policy,
        final_score=decisive_score(position),
        search_provenance="test",
        policy_provenance="completed-q",
        run_id=identity.run_id,
        generation_family=identity.generation_family,
        actor_id="actor-test",
        game_id=game_id,
        model_identity="sha256-" + "1" * 64,
    )


def test_replay_store_accepts_only_current_supported_data(tmp_path) -> None:
    identity = RunIdentity(tmp_path / "run.json", "run-test", "family-test", 1)
    with ReplayStore(tmp_path / "replay") as store:
        generation = store.lease_generation(identity, "actor-test")
        record = store.append(
            [replay_sample(identity, "game-one")],
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
        assert record.ring == 4
        assert store.sample_counts_by_ring(
            run_id=identity.run_id,
            generation_family=identity.generation_family,
        ) == {4: 1, 6: 0, 8: 0, 10: 0}
        with pytest.raises(ValueError, match="one of"):
            store.sample_counts_by_ring(
                (5,),
                run_id=identity.run_id,
                generation_family=identity.generation_family,
            )


def test_curriculum_and_selfplay_reject_noncanonical_rings() -> None:
    mixture = RingMixtureConfig(
        curriculum=(
            CurriculumStage(10, (4,)),
            CurriculumStage(20, (4, 6)),
        )
    )
    assert mixture.active_rings(0) == (4,)
    assert mixture.active_rings(10) == (4, 6)
    assert mixture.active_rings(20) == SUPPORTED_RINGS
    weighted = RingMixtureConfig(
        step_weights=(RingWeightStage(1_000_000, (0.1, 0.1, 0.1, 0.7)),)
    )
    assert weighted.weights_for_step(999_999) is None
    assert weighted.weights_for_step(1_000_000) == (0.1, 0.1, 0.1, 0.7)
    assert weighted.next_weight_step(999_999) == 1_000_000
    assert weighted.next_weight_step(1_000_000) is None

    with pytest.raises(ValueError, match="one of"):
        SelfPlayConfig(rings=5)
    with pytest.raises(ValueError, match="one of"):
        SelfPlayConfig(rings=4.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="selected"):
        CurriculumStage(10, (4, 5))
    with pytest.raises(ValueError, match="unlimited"):
        LearnerConfig(unlimited="yes")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="recovery_interval_steps"):
        LearnerConfig(candidate_interval=10, recovery_interval_steps=11)
    with pytest.raises(ValueError, match="warmup requires"):
        LearnerConfig(selfplay_snapshot_warmup_examples=10)
    with pytest.raises(ValueError, match="cannot be slower"):
        LearnerConfig(
            selfplay_snapshot_interval_examples=10,
            selfplay_snapshot_warmup_examples=100,
            selfplay_snapshot_warmup_interval_examples=20,
        )
    with pytest.raises(ValueError, match="match configured rings"):
        RingMixtureConfig(step_weights=(RingWeightStage(10, (0.3, 0.7)),))


def test_inference_response_submit_contract_is_stable() -> None:
    response = InferenceResponse([1], [0.25], [0, 1], [2.0])
    assert response.submit_args() == ([1], [0.25], [0, 1], [2.0])


def make_replay_sample(
    rings: int = 4,
    *,
    identity: RunIdentity | None = None,
    actor_id: str = "actor-test",
    generation: int = 0,
    game_id: str | None = None,
    ply: int = 0,
    model_identity: str = "sha256-" + "1" * 64,
) -> ReplaySample:
    topology = get_topology(rings)
    position = DoubleStarPosition(
        rings=rings,
        stones=torch.full((topology.n,), -1, dtype=torch.int8),
        to_move=0,
        moves_left=1,
        opening=True,
        terminal=False,
    )
    policy = np.ones(topology.n, dtype=np.float32)
    policy /= policy.sum()
    return ReplaySample.from_position(
        position,
        policy=policy,
        final_score=decisive_score(position),
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


def append_replay(
    store: ReplayStore,
    samples: list[ReplaySample],
    identity: RunIdentity,
    *,
    model_step: int,
    generation: int = 0,
) -> ShardRecord:
    model_identity = "sha256-" + "1" * 64
    return store.append(
        samples,
        phase_min=min(sample.ply for sample in samples),
        phase_max=max(sample.ply for sample in samples),
        model_version=model_identity,
        model_step=model_step,
        model_identity=model_identity,
        run_id=identity.run_id,
        generation_family=identity.generation_family,
        actor_id="actor-test",
        generation=generation,
    )


def test_replay_store_manifest_recency_lag_duplicates_and_cursor(tmp_path) -> None:
    identity = run_identity(tmp_path)
    with ReplayStore(tmp_path / "replay-restored") as store:
        generation = store.lease_generation(identity, "actor-test")
        assert generation == 0
        first = append_replay(
            store,
            [
                make_replay_sample(
                    identity=identity,
                    generation=generation,
                    game_id="game-first",
                )
            ],
            identity,
            model_step=1,
            generation=generation,
        )
        second_samples = [
            make_replay_sample(
                identity=identity,
                generation=generation,
                game_id="game-second",
                ply=ply,
            )
            for ply in range(2)
        ]
        second = append_replay(
            store,
            second_samples,
            identity,
            model_step=10,
            generation=generation,
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
        assert [sample.game_id for sample in recent] == ["game-second"] * 2

        future = append_replay(
            store,
            [
                make_replay_sample(
                    identity=identity,
                    generation=generation,
                    game_id="game-future",
                )
            ],
            identity,
            model_step=11,
            generation=generation,
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
            append_replay(
                store,
                second_samples,
                identity,
                model_step=10,
                generation=generation,
            )
        assert set(store.shard_directory.glob("*.npz")) == before

        assert store.get_cursor("learner") == ReplayCursor()
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

    orphan = tmp_path / "replay-restored" / "shards" / "orphan.npz"
    orphan.write_bytes(b"orphan")
    os.utime(orphan, (1, 1))
    with ReplayStore(tmp_path / "replay-restored") as reconciled:
        assert not orphan.exists()
        assert reconciled.reconciliation_metrics["orphan_files"] == 1


def test_replay_gc_watermark_and_quarantine_preserve_committed_ledger(
    tmp_path,
) -> None:
    identity = run_identity(tmp_path)
    records: list[ShardRecord] = []
    with ReplayStore(tmp_path / "gc-replay") as store:
        generation = store.lease_generation(identity, "actor-test")
        for index in range(3):
            records.append(
                append_replay(
                    store,
                    [
                        make_replay_sample(
                            identity=identity,
                            generation=generation,
                            game_id=f"game-gc-{index}",
                        )
                    ],
                    identity,
                    model_step=0,
                    generation=generation,
                )
            )
        protected = ReplaySelection(
            (ReplaySpan(records[0], 0, 1),),
            {4: 1},
            records[-1].shard_id,
        )
        assert (
            store.total_committed_sample_count(
                run_id=identity.run_id,
                generation_family=identity.generation_family,
            )
            == 3
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
        assert (
            store.total_committed_sample_count(
                run_id=identity.run_id,
                generation_family=identity.generation_family,
            )
            == 3
        )
        store.clear_gc_watermark("learner")
        append_replay(
            store,
            [
                make_replay_sample(
                    identity=identity,
                    generation=generation,
                    game_id="game-gc-after-delete",
                )
            ],
            identity,
            model_step=0,
            generation=generation,
        )
        assert (
            store.total_committed_sample_count(
                run_id=identity.run_id,
                generation_family=identity.generation_family,
            )
            == 4
        )

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
        assert game_count == 4


def test_legacy_replay_counter_migration_fails_closed_after_unknown_gc(
    tmp_path,
) -> None:
    identity = run_identity(tmp_path)
    root = tmp_path / "legacy-counter"
    with ReplayStore(root) as store:
        generation = store.lease_generation(identity, "actor-test")
        for index in range(2):
            append_replay(
                store,
                [
                    make_replay_sample(
                        identity=identity,
                        generation=generation,
                        game_id=f"legacy-counter-{index}",
                    )
                ],
                identity,
                model_step=0,
                generation=generation,
            )
        deleted = store.collect_garbage(
            run_id=identity.run_id,
            generation_family=identity.generation_family,
            retain_shards_per_ring=1,
            dry_run=False,
        )
        assert deleted["deleted_shards"] == 1
        store.connection.execute("DROP TABLE run_counters")

    with ReplayStore(root) as migrated:
        assert (
            migrated.total_committed_sample_count(
                run_id=identity.run_id,
                generation_family=identity.generation_family,
            )
            == 1
        )
        assert not migrated.committed_sample_history_is_complete(
            run_id=identity.run_id,
            generation_family=identity.generation_family,
        )
        learner = object.__new__(LearnerLoop)
        learner.store = migrated
        learner.run_identity = identity
        learner.learner_config = LearnerConfig(target_updates_per_new_sample=1.0)
        learner.train_config = TrainConfig(per_rank_batch_size=1)
        learner.world_size = 1
        learner._latest_total_replay_samples = 0
        learner.examples_consumed = 0
        with pytest.raises(ValueError, match="complete committed-sample history"):
            learner._utd_step_budget()


def tiny_model() -> GraphResTNet:
    return GraphResTNet(
        ModelConfig(
            width=8,
            rrt_groups=1,
            attention_heads=2,
            kv_heads=1,
        )
    )


def make_test_learner(
    store: ReplayStore,
    identity: RunIdentity,
    output_directory,
    *,
    learner_config: LearnerConfig,
    train_config: TrainConfig | None = None,
    data_config: DataConfig | None = None,
    ring_mixture_config: RingMixtureConfig = RingMixtureConfig(),
) -> LearnerLoop:
    train = train_config or TrainConfig(
        per_rank_batch_size=1,
        scheduler=SchedulerConfig(warmup_steps=0, total_steps=100),
    )
    model = tiny_model()
    optimizer = build_optimizer(model, OptimizerConfig(kind="adamw"))
    scheduler = build_scheduler(optimizer, train.scheduler)
    return LearnerLoop(
        store=store,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ExponentialMovingAverage(model, decay=0.9),
        output_directory=output_directory,
        learner_config=learner_config,
        train_config=train,
        data_config=data_config
        or DataConfig(
            workers=0,
            ring_stratified=False,
            d5_augmentation=False,
        ),
        loss_weights=LossWeights(),
        seed=5,
        serialized_config={
            "schema_version": 3,
            "learner": {
                "target_updates_per_new_sample": (
                    learner_config.target_updates_per_new_sample
                )
            },
        },
        run_identity=identity,
        ring_mixture_config=ring_mixture_config,
    )


def curriculum_learner_stub(*, enabled: bool = True) -> LearnerLoop:
    learner = object.__new__(LearnerLoop)
    learner.learner_config = LearnerConfig(
        minimum_replay_samples=4,
        recent_samples_per_ring=16,
        minimum_unique_samples_per_ring=2,
        use_ring_mixture_curriculum=enabled,
    )
    learner.train_config = TrainConfig(per_rank_batch_size=2)
    learner.data_config = DataConfig(ring_stratified=True)
    learner.ring_mixture_config = RingMixtureConfig(
        curriculum=(
            CurriculumStage(until_samples=10, rings=(4,)),
            CurriculumStage(until_samples=20, rings=(4, 6)),
        )
    )
    learner.rank = 0
    learner.world_size = 1
    learner.step = 0
    return learner


def test_learner_curriculum_readiness_expands_only_at_aggregate_boundaries() -> None:
    learner = curriculum_learner_stub()
    early = {ring: 0 for ring in SUPPORTED_RINGS}
    early[4] = 4
    assert learner._active_replay_rings(early) == (4,)
    assert learner._replay_is_ready(early)
    assert not curriculum_learner_stub(enabled=False)._replay_is_ready(early)

    transition = dict(early)
    transition[4] = 10
    assert learner._active_replay_rings(transition) == (4, 6)
    assert not learner._replay_is_ready(transition)

    warmed = dict(transition)
    warmed[6] = 2
    assert learner._replay_is_ready(warmed)

    fully_unlocked = dict(warmed)
    fully_unlocked[4] = 18
    assert learner._active_replay_rings(fully_unlocked) == SUPPORTED_RINGS
    assert not learner._replay_is_ready(fully_unlocked)


def test_learner_curriculum_selects_only_currently_active_rings(tmp_path) -> None:
    class CapturingStore:
        def __init__(self, counts: dict[int, int]) -> None:
            self.counts = counts
            self.selections: list[tuple[int, ...]] = []

        def eligible_sample_counts(
            self, rings: tuple[int, ...], **_metadata: object
        ) -> dict[int, int]:
            assert rings == SUPPORTED_RINGS
            return dict(self.counts)

        def select_recent_spans(
            self, *, rings: tuple[int, ...], **_metadata: object
        ) -> ReplaySelection:
            self.selections.append(rings)
            return ReplaySelection(
                (),
                {ring: self.counts.get(ring, 0) for ring in rings},
                0,
            )

    learner = curriculum_learner_stub()
    early = {ring: 0 for ring in SUPPORTED_RINGS}
    early[4] = 4
    store = CapturingStore(early)
    learner.store = store
    learner.run_identity = run_identity(tmp_path)

    assert tuple(learner._select_replay_spans().samples_by_ring) == (4,)

    store.counts.update({4: 10, 6: 2})
    assert tuple(learner._select_replay_spans().samples_by_ring) == (4, 6)

    store.counts = {ring: 5 for ring in SUPPORTED_RINGS}
    assert tuple(learner._select_replay_spans().samples_by_ring) == SUPPORTED_RINGS
    assert store.selections == [(4,), (4, 6), SUPPORTED_RINGS]


def test_learner_runs_batch_publishes_metrics_and_resumes_example_cadence(
    tmp_path,
) -> None:
    identity = run_identity(tmp_path)
    with ReplayStore(tmp_path / "learner-replay") as store:
        generation = store.lease_generation(identity, "actor-test")
        append_replay(
            store,
            [
                make_replay_sample(
                    identity=identity,
                    generation=generation,
                    game_id=f"game-learner-{suffix}",
                )
                for suffix in ("a", "b")
            ],
            identity,
            model_step=0,
            generation=generation,
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
            data_config=DataConfig(
                workers=0,
                ring_stratified=False,
                d5_augmentation=False,
            ),
            loss_weights=LossWeights(),
            seed=5,
            serialized_config={"schema_version": 3, "learner": {}},
            run_identity=identity,
        )
        assert learner.run(steps=1) == 1
        manifest_path = tmp_path / "learner" / "candidate.json"
        manifest = json.loads(manifest_path.read_text())
        assert manifest["model_step"] == 1
        assert manifest["role"] == "candidate"
        manifest_path.write_text("{broken", encoding="utf-8")
        published = learner.publisher.publish(
            model=learner.model,
            optimizer=learner.optimizer,
            scheduler=learner.scheduler,
            ema=learner.ema,
            step=learner.step,
            epoch=learner.epoch,
            config=learner.serialized_config,
            examples_consumed=learner.examples_consumed,
            global_batch_size=2,
        )
        checkpoint = published.checkpoint
        assert checkpoint.exists()
        history = [
            json.loads(line)
            for line in (tmp_path / "learner" / "model-history.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert history[-1]["model_identity"] == published.model_identity
        assert history[-1]["model_step"] == 1
        assert not (tmp_path / "learner" / "champion.json").exists()
        metric_lines = [
            line
            for line in (tmp_path / "learner" / "metrics.jsonl")
            .read_text()
            .splitlines()
            if "losses" in json.loads(line)
        ]
        assert len(metric_lines) == 1
        metric = json.loads(metric_lines[0])
        assert metric["metrics_interval_steps"] == 1
        assert metric["metrics_interval_wall_seconds"] > 0
        assert metric["step_seconds"] > 0
        assert metric["device_step_seconds"] > 0
        assert metric["examples_per_second"] > 0
        assert metric["device_examples_per_second"] > 0
        assert metric["data_wait_seconds"] >= 0
        assert metric["window_setup_seconds"] >= 0
        assert metric["h2d_seconds"] == 0
        assert metric["examples_consumed"] == 2
        assert metric["total_replay_samples"] == 2
        assert metric["updates_per_new_sample"] == 1

        learner.learner_config = replace(
            learner.learner_config,
            target_updates_per_new_sample=2.0,
            candidate_interval_examples=4,
        )
        (tmp_path / "learner" / "utd-segment.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "run_id": identity.run_id,
                    "generation_family": identity.generation_family,
                    "target_updates_per_new_sample": 2.0,
                    "baseline_examples_consumed": 0,
                    "baseline_committed_replay_samples": 0,
                }
            ),
            encoding="utf-8",
        )
        learner._utd_segment_state = None
        learner._last_candidate_examples = 0
        learner.examples_consumed = 0
        assert learner._utd_step_budget() == 2
        learner.examples_consumed = 2
        assert learner._utd_step_budget() == 1
        assert learner._candidate_due() is False
        learner.examples_consumed = 4
        assert learner._candidate_due() is True

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
            learner_config=replace(
                learner.learner_config,
                use_ring_mixture_curriculum=True,
            ),
            train_config=learner.train_config,
            data_config=learner.data_config,
            loss_weights=LossWeights(),
            seed=5,
            serialized_config={
                "schema_version": 3,
                "learner": {"use_ring_mixture_curriculum": True},
            },
            run_identity=identity,
        )
        restored.resume(
            checkpoint,
            expected_sha256=published.checkpoint_sha256,
            expected_bytes=published.checkpoint_bytes,
        )
        assert restored.step == 1
        assert restored.examples_consumed == 2
        assert restored.learner_config.use_ring_mixture_curriculum is True

        legacy_checkpoint = save_checkpoint(
            tmp_path / "legacy-checkpoint.pt",
            model=restored.model,
            optimizer=restored.optimizer,
            scheduler=restored.scheduler,
            ema=restored.ema,
            step=1,
            epoch=0,
            config=restored.serialized_config,
            extra={
                "run_id": identity.run_id,
                "generation_family": identity.generation_family,
            },
        )
        before_rejected_resume = {
            name: value.detach().clone()
            for name, value in restored.model.state_dict().items()
        }
        before_step = restored.step
        before_examples = restored.examples_consumed
        with pytest.raises(ValueError, match="legacy checkpoint"):
            restored.resume(legacy_checkpoint)
        assert restored.step == before_step
        assert restored.examples_consumed == before_examples
        for name, value in restored.model.state_dict().items():
            torch.testing.assert_close(value, before_rejected_resume[name])

        completion_path = tmp_path / "restored" / "learner-complete.json"
        completion_path.write_text("{}", encoding="utf-8")
        restored.learner_config = replace(
            restored.learner_config,
            steps=1,
            unlimited=True,
            recovery_interval_steps=1,
            target_updates_per_new_sample=None,
            candidate_interval_examples=None,
        )
        assert restored.run(stop_requested=lambda: restored.step >= 2) == 2
        assert not completion_path.exists()
        recovery = json.loads(
            (tmp_path / "restored" / "recovery.json").read_text(encoding="utf-8")
        )
        assert recovery["step"] == 2


def test_persistent_replay_window_reuses_loader_across_utd_waits(
    tmp_path,
    monkeypatch,
) -> None:
    identity = run_identity(tmp_path)
    with ReplayStore(tmp_path / "persistent-replay") as store:
        generation = store.lease_generation(identity, "actor-test")
        append_replay(
            store,
            [
                make_replay_sample(
                    identity=identity,
                    generation=generation,
                    game_id=f"persistent-{index}",
                )
                for index in range(8)
            ],
            identity,
            model_step=0,
            generation=generation,
        )
        learner = make_test_learner(
            store,
            identity,
            tmp_path / "persistent-learner",
            learner_config=LearnerConfig(
                steps=4,
                minimum_replay_samples=1,
                recent_samples_per_ring=16,
                max_replay_lag_steps=10,
                steps_per_window=4,
                candidate_interval=100,
                target_updates_per_new_sample=1.0,
                metrics_interval=1,
                replay_poll_seconds=0.001,
                device="cpu",
            ),
        )
        loader_batches: list[int] = []
        original_loader = learner._loader

        def capture_loader(selection, *, batches):
            loader_batches.append(batches)
            return original_loader(selection, batches=batches)

        budgets = iter((1, 0, 1, 2))

        def utd_budget() -> int:
            learner._latest_total_replay_samples = 8
            return next(budgets)

        selected_indices: list[int] = []
        original_sampler_iter = UniqueReplayBatchSampler.__iter__

        def capture_sampler(sampler):
            for indices in original_sampler_iter(sampler):
                selected_indices.extend(indices)
                yield indices

        watermark_during_wait: list[int] = []

        def progress(**payload) -> None:
            if payload.get("phase") == "update_to_data_wait":
                watermark_during_wait.append(
                    store.connection.execute(
                        "SELECT COUNT(*) FROM gc_watermarks"
                    ).fetchone()[0]
                )

        monkeypatch.setattr(learner, "_loader", capture_loader)
        monkeypatch.setattr(learner, "_utd_step_budget", utd_budget)
        monkeypatch.setattr(UniqueReplayBatchSampler, "__iter__", capture_sampler)
        monkeypatch.setattr("startrain.learner.time.sleep", lambda _seconds: None)

        assert learner.run(steps=4, progress=progress) == 4

        assert loader_batches == [4]
        assert watermark_during_wait == [1]
        assert len(selected_indices) == len(set(selected_indices)) == 4
        assert (
            store.connection.execute(
                "SELECT COUNT(*) FROM gc_watermarks"
            ).fetchone()[0]
            == 0
        )
        events = [
            json.loads(line)
            for line in (tmp_path / "persistent-learner" / "metrics.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        allocations = [
            event
            for event in events
            if event.get("event") == "replay_window_allocated"
        ]
        consumptions = [
            event
            for event in events
            if event.get("event") == "replay_window_consumed"
        ]
        assert len(allocations) == 1
        assert allocations[0]["window_batches_allocated"] == 4
        assert allocations[0]["loader_workers_effective"] == 0
        assert [event["window_reuse"] for event in consumptions] == [
            False,
            True,
            True,
        ]
        assert consumptions[-1]["window_batches_consumed"] == 4


def test_replay_window_refreshes_at_ring_weight_boundary(
    tmp_path,
    monkeypatch,
) -> None:
    identity = run_identity(tmp_path)
    with ReplayStore(tmp_path / "boundary-replay") as store:
        generation = store.lease_generation(identity, "actor-test")
        append_replay(
            store,
            [
                make_replay_sample(
                    identity=identity,
                    generation=generation,
                    game_id=f"boundary-{index}",
                )
                for index in range(8)
            ],
            identity,
            model_step=0,
            generation=generation,
        )
        learner = make_test_learner(
            store,
            identity,
            tmp_path / "boundary-learner",
            learner_config=LearnerConfig(
                steps=4,
                minimum_replay_samples=1,
                recent_samples_per_ring=16,
                max_replay_lag_steps=10,
                steps_per_window=4,
                candidate_interval=100,
                metrics_interval=10,
                device="cpu",
            ),
            ring_mixture_config=RingMixtureConfig(
                step_weights=(RingWeightStage(2, (1.0, 0.0, 0.0, 0.0)),)
            ),
        )
        allocations: list[int] = []
        original_loader = learner._loader

        def capture_loader(selection, *, batches):
            allocations.append(batches)
            return original_loader(selection, batches=batches)

        monkeypatch.setattr(learner, "_loader", capture_loader)

        assert learner.run(steps=4) == 4

        assert allocations == [2, 2]
        events = [
            json.loads(line)
            for line in (tmp_path / "boundary-learner" / "metrics.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if '"event":"replay_window_refreshed"' in line
        ]
        assert [event["window_refresh_reason"] for event in events] == [
            "ring_weight_change",
            "target",
        ]


def test_short_replay_windows_disable_configured_workers() -> None:
    learner = object.__new__(LearnerLoop)
    learner.data_config = DataConfig(
        workers=8,
        min_batches_for_workers=32,
    )

    assert learner._effective_loader_workers(31) == 0
    assert learner._effective_loader_workers(32) == 8


def test_replay_window_shutdown_closes_workers_and_stop_clears_watermark(
    tmp_path,
    monkeypatch,
) -> None:
    class WorkerIterator:
        shutdowns = 0

        def _shutdown_workers(self) -> None:
            self.shutdowns += 1

    worker_iterator = WorkerIterator()
    standalone = ReplayWindowSession(
        selection=ReplaySelection((), {}, 0),
        loader=SimpleNamespace(_iterator=worker_iterator),
        prefetcher=SimpleNamespace(_stream=None),
        batches_allocated=1,
        effective_workers=1,
        setup_seconds=0.0,
        refresh_reason="test",
        opened_step=0,
        opened_epoch=0,
        active_rings=(),
        ring_weights=None,
        recovery_boundary=None,
        ring_weight_boundary=None,
    )
    standalone.shutdown()
    standalone.shutdown()
    assert worker_iterator.shutdowns == 1

    class FailingStream:
        def synchronize(self) -> None:
            raise RuntimeError("stream failed")

    failed_iterator = WorkerIterator()
    failing = ReplayWindowSession(
        selection=ReplaySelection((), {}, 0),
        loader=SimpleNamespace(_iterator=failed_iterator),
        prefetcher=SimpleNamespace(_stream=FailingStream()),
        batches_allocated=1,
        effective_workers=1,
        setup_seconds=0.0,
        refresh_reason="test",
        opened_step=0,
        opened_epoch=0,
        active_rings=(),
        ring_weights=None,
        recovery_boundary=None,
        ring_weight_boundary=None,
    )
    with pytest.raises(RuntimeError, match="stream failed"):
        failing.shutdown()
    assert failed_iterator.shutdowns == 1
    assert failing.closed is True
    assert failing.loader is None
    assert failing.prefetcher is None

    identity = run_identity(tmp_path)
    with ReplayStore(tmp_path / "shutdown-replay") as store:
        generation = store.lease_generation(identity, "actor-test")
        append_replay(
            store,
            [
                make_replay_sample(
                    identity=identity,
                    generation=generation,
                    game_id=f"shutdown-{index}",
                )
                for index in range(4)
            ],
            identity,
            model_step=0,
            generation=generation,
        )
        learner = make_test_learner(
            store,
            identity,
            tmp_path / "shutdown-learner",
            learner_config=LearnerConfig(
                steps=4,
                minimum_replay_samples=1,
                recent_samples_per_ring=8,
                max_replay_lag_steps=10,
                steps_per_window=4,
                candidate_interval=100,
                device="cpu",
            ),
        )
        shutdown_reasons: list[str] = []
        original_close = learner._close_replay_window

        def capture_close(window, *, reason):
            shutdown_reasons.append(reason)
            return original_close(window, reason=reason)

        monkeypatch.setattr(learner, "_close_replay_window", capture_close)

        assert learner.run(
            steps=4,
            stop_requested=lambda: learner.step >= 1,
        ) == 1
        assert shutdown_reasons == ["stop"]
        assert (
            store.connection.execute(
                "SELECT COUNT(*) FROM gc_watermarks"
            ).fetchone()[0]
            == 0
        )


def test_prospective_fractional_utd_state_and_target_mismatch_fail_closed(
    tmp_path,
    monkeypatch,
) -> None:
    identity = run_identity(tmp_path)
    with ReplayStore(tmp_path / "utd-replay") as store:
        monkeypatch.setattr(
            store,
            "total_committed_sample_count",
            lambda **_metadata: 4,
        )
        origin = make_test_learner(
            store,
            identity,
            tmp_path / "utd-origin",
            learner_config=LearnerConfig(
                target_updates_per_new_sample=1.25,
                device="cpu",
            ),
            train_config=TrainConfig(
                per_rank_batch_size=2,
                scheduler=SchedulerConfig(warmup_steps=0, total_steps=100),
            ),
        )

        assert origin._utd_step_budget() == 2
        origin_state = json.loads(
            (tmp_path / "utd-origin" / "utd-segment.json").read_text(
                encoding="utf-8"
            )
        )
        assert origin_state["baseline_examples_consumed"] == 0
        assert origin_state["baseline_committed_replay_samples"] == 0
        published = origin._publish()
        checkpoint_payload = torch.load(
            published.checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        assert checkpoint_payload["extra"]["utd_segment"] == origin_state

        restored_output = tmp_path / "utd-restored"
        restored = make_test_learner(
            store,
            identity,
            restored_output,
            learner_config=LearnerConfig(
                target_updates_per_new_sample=1.25,
                device="cpu",
            ),
            train_config=TrainConfig(
                per_rank_batch_size=2,
                scheduler=SchedulerConfig(warmup_steps=0, total_steps=100),
            ),
        )
        restored.resume(
            published.checkpoint,
            expected_sha256=published.checkpoint_sha256,
            expected_bytes=published.checkpoint_bytes,
        )
        assert not (restored_output / "utd-segment.json").exists()
        assert restored._utd_step_budget() == 2
        assert json.loads(
            (restored_output / "utd-segment.json").read_text(encoding="utf-8")
        ) == origin_state

        migrated_output = tmp_path / "utd-migrated"
        migrated = make_test_learner(
            store,
            identity,
            migrated_output,
            learner_config=LearnerConfig(
                target_updates_per_new_sample=1.25,
                device="cpu",
            ),
            train_config=TrainConfig(
                per_rank_batch_size=2,
                scheduler=SchedulerConfig(warmup_steps=0, total_steps=100),
            ),
        )
        (migrated_output / "utd-segment.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "run_id": identity.run_id,
                    "generation_family": identity.generation_family,
                    "target_updates_per_new_sample": 1.25,
                    "baseline_examples_consumed": 100,
                    "baseline_committed_replay_samples": 80,
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(
            store,
            "total_committed_sample_count",
            lambda **_metadata: 84,
        )
        migrated.step = 50
        migrated.examples_consumed = 100

        assert migrated._utd_step_budget() == 2
        migrated.examples_consumed = 104
        assert migrated._utd_step_budget() == 0
        metrics = migrated._utd_metric_values()
        assert metrics["lifetime_updates_per_new_sample"] == pytest.approx(104 / 84)
        assert metrics["segment_updates_per_new_sample"] == pytest.approx(1.0)

        mismatched = make_test_learner(
            store,
            identity,
            tmp_path / "utd-mismatch",
            learner_config=LearnerConfig(
                target_updates_per_new_sample=1.25,
                device="cpu",
            ),
            train_config=TrainConfig(
                per_rank_batch_size=2,
                scheduler=SchedulerConfig(warmup_steps=0, total_steps=100),
            ),
        )
        mismatched.step = 50
        mismatched.examples_consumed = 100
        mismatched._resume_utd_target = 1.0
        with pytest.raises(ValueError, match="changed its update-to-data target"):
            mismatched._utd_step_budget()

        override_output = tmp_path / "utd-migration-override"
        override = make_test_learner(
            store,
            identity,
            override_output,
            learner_config=LearnerConfig(
                target_updates_per_new_sample=1.25,
                device="cpu",
            ),
            train_config=TrainConfig(
                per_rank_batch_size=2,
                scheduler=SchedulerConfig(warmup_steps=0, total_steps=100),
            ),
        )
        (override_output / "utd-segment.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "run_id": identity.run_id,
                    "generation_family": identity.generation_family,
                    "target_updates_per_new_sample": 1.25,
                    "baseline_examples_consumed": 100,
                    "baseline_committed_replay_samples": 80,
                }
            ),
            encoding="utf-8",
        )
        override.step = 50
        override.examples_consumed = 100
        override._resume_utd_target = 1.0
        override._resume_utd_segment_state = UTDSegmentState(
            run_id=identity.run_id,
            generation_family=identity.generation_family,
            target_updates_per_new_sample=1.0,
            baseline_examples_consumed=0,
            baseline_committed_replay_samples=0,
        )
        assert override._utd_step_budget() == 2


def test_learner_publishes_initial_ema_then_times_out_on_short_batch(
    tmp_path,
) -> None:
    identity = run_identity(tmp_path)
    with ReplayStore(tmp_path / "waiting-replay") as store:
        generation = store.lease_generation(identity, "actor-test")
        append_replay(
            store,
            [
                make_replay_sample(
                    identity=identity,
                    generation=generation,
                    game_id="game-only-sample",
                )
            ],
            identity,
            model_step=0,
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
            serialized_config={"schema_version": 3},
            run_identity=identity,
        )
        with pytest.raises(TimeoutError, match="minimum replay"):
            learner.run()
        manifest = json.loads(
            (tmp_path / "waiting-learner" / "candidate.json").read_text()
        )
        assert manifest["model_step"] == 0
        assert manifest["role"] == "candidate"


def test_decoupled_model_cadence_migrates_existing_run_without_reset(
    tmp_path,
) -> None:
    identity = run_identity(tmp_path)
    with ReplayStore(tmp_path / "cadence-replay") as store:
        model = tiny_model()
        optimizer = build_optimizer(model, OptimizerConfig(kind="adamw"))
        scheduler = build_scheduler(
            optimizer,
            SchedulerConfig(warmup_steps=0, total_steps=100),
        )
        learner = LearnerLoop(
            store=store,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            ema=ExponentialMovingAverage(model, decay=0.9),
            output_directory=tmp_path / "cadence-learner",
            learner_config=LearnerConfig(
                candidate_interval=10,
                candidate_interval_examples=5,
                selfplay_snapshot_interval_examples=3,
                selfplay_snapshot_warmup_examples=20,
                selfplay_snapshot_warmup_interval_examples=1,
                device="cpu",
            ),
            train_config=TrainConfig(
                per_rank_batch_size=1,
                scheduler=SchedulerConfig(warmup_steps=0, total_steps=100),
            ),
            data_config=DataConfig(workers=0),
            loss_weights=LossWeights(),
            seed=5,
            serialized_config={"schema_version": 3},
            run_identity=identity,
        )
        initial = learner._publish()
        assert initial.model_step == 0
        learner.step = 18
        learner.examples_consumed = 18

        learner._load_cadence_state()

        assert learner._last_candidate_examples == 0
        assert learner._last_selfplay_examples == 0
        assert (
            load_model_manifest(
                tmp_path / "cadence-learner" / "selfplay" / "candidate.json"
            ).model_step
            == 0
        )

        candidate, selfplay = learner._publish_due_models()
        assert candidate is not None and candidate.model_step == 18
        assert selfplay is not None and selfplay.model_step == 18

        learner.step = 19
        learner.examples_consumed = 19
        candidate, selfplay = learner._publish_due_models()
        assert candidate is None
        assert selfplay is not None and selfplay.model_step == 19

        learner.step = 22
        learner.examples_consumed = 22
        candidate, selfplay = learner._publish_due_models()
        assert candidate is None
        assert selfplay is not None and selfplay.model_step == 22

        learner.step = 23
        learner.examples_consumed = 23
        candidate, selfplay = learner._publish_due_models()
        assert candidate is not None and candidate.model_step == 23
        assert selfplay is None
        cadence = json.loads(
            (tmp_path / "cadence-learner" / "cadence.json").read_text(encoding="utf-8")
        )
        assert cadence["candidate_examples"] == 23
        assert cadence["selfplay_examples"] == 22
        assert (
            load_model_manifest(
                tmp_path / "cadence-learner" / "candidate.json"
            ).model_step
            == 23
        )
        assert (
            load_model_manifest(
                tmp_path / "cadence-learner" / "selfplay" / "candidate.json"
            ).model_step
            == 22
        )
        learner._last_candidate_examples = None
        learner._last_selfplay_examples = None
        learner._load_cadence_state()
        assert learner._last_candidate_examples == 23
        assert learner._last_selfplay_examples == 22


def test_lazy_replay_sampler_is_unique_deterministic_and_homogeneous(
    tmp_path,
) -> None:
    identity = run_identity(tmp_path)
    with ReplayStore(tmp_path / "lazy-replay") as store:
        generation = store.lease_generation(identity, "actor-test")
        for ring in (4, 6):
            append_replay(
                store,
                [
                    make_replay_sample(
                        ring,
                        identity=identity,
                        generation=generation,
                        game_id=f"game-ring-{ring}-{index}",
                    )
                    for index in range(4)
                ],
                identity,
                model_step=0,
                generation=generation,
            )
        selection = store.select_recent_spans(
            rings=(4, 6),
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


def test_lazy_replay_sampler_mixes_distinct_shards_without_replacement(
    tmp_path,
) -> None:
    spans = []
    for shard_id in range(4):
        path = write_replay_shard(
            tmp_path / f"mixed-{shard_id}.npz",
            [
                make_replay_sample(game_id=f"mixed-{shard_id}-{index}")
                for index in range(8)
            ],
        )
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        spans.append(
            ReplaySpan(
                ShardRecord(
                    shard_id=shard_id + 1,
                    path=path,
                    created_ns=shard_id + 1,
                    sample_count=8,
                    ring=4,
                    phase_min=0,
                    phase_max=0,
                    model_version="sha256-" + "1" * 64,
                    model_step=0,
                    model_identity="sha256-" + "1" * 64,
                    run_id="run-test",
                    generation_family="family-test",
                    actor_id="actor-test",
                    generation=0,
                    game_count=8,
                    checksum_sha256=digest,
                    state="ready",
                    quarantine_reason=None,
                ),
                0,
                8,
            )
        )
    dataset = LazyShardReplayDataset(
        ReplaySelection(tuple(spans), {4: 32}, 4),
        seed=11,
        epoch=0,
        augmentation_enabled=False,
        shard_cache_size=4,
    )
    sampler = UniqueReplayBatchSampler(
        dataset,
        batch_size=4,
        batches=4,
        seed=11,
        epoch=0,
        ring_stratified=True,
        shards_per_batch=2,
    )

    batches = list(sampler)

    assert len({index for batch in batches for index in batch}) == 16
    assert all(len({index // 8 for index in batch}) == 2 for batch in batches)
    assert _maximum_cross_shard_groups([2, 2, 2, 2], shards_per_batch=2) == 4
    assert _maximum_cross_shard_groups([10, 1], shards_per_batch=2) == 1


def test_lazy_replay_sampler_applies_weighted_ring_quotas(tmp_path) -> None:
    spans = []
    samples_by_ring = {}
    for shard_id, ring in enumerate(SUPPORTED_RINGS, start=1):
        samples = [
            make_replay_sample(ring, game_id=f"weighted-{ring}-{index}")
            for index in range(100)
        ]
        path = write_replay_shard(tmp_path / f"ring-{ring}.npz", samples)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        spans.append(
            ReplaySpan(
                ShardRecord(
                    shard_id=shard_id,
                    path=path,
                    created_ns=shard_id,
                    sample_count=100,
                    ring=ring,
                    phase_min=0,
                    phase_max=0,
                    model_version="sha256-" + "1" * 64,
                    model_step=0,
                    model_identity="sha256-" + "1" * 64,
                    run_id="run-test",
                    generation_family="family-test",
                    actor_id="actor-test",
                    generation=0,
                    game_count=100,
                    checksum_sha256=digest,
                    state="ready",
                    quarantine_reason=None,
                ),
                0,
                100,
            )
        )
        samples_by_ring[ring] = 100
    dataset = LazyShardReplayDataset(
        ReplaySelection(tuple(spans), samples_by_ring, 4),
        seed=17,
        epoch=1,
        augmentation_enabled=False,
        shard_cache_size=4,
    )
    sampler = UniqueReplayBatchSampler(
        dataset,
        batch_size=1,
        batches=100,
        seed=17,
        epoch=1,
        ring_stratified=True,
        ring_weights={4: 0.1, 6: 0.1, 8: 0.1, 10: 0.7},
    )
    counts = {ring: 0 for ring in SUPPORTED_RINGS}
    for batch in sampler:
        counts[dataset[batch[0]].rings] += 1
    assert counts == {4: 10, 6: 10, 8: 10, 10: 70}
    with pytest.raises(ValueError, match="cannot satisfy configured proportions"):
        _weighted_ring_quotas(
            100,
            capacities={4: 100, 6: 100, 8: 100, 10: 8},
            weights={4: 0.1, 6: 0.1, 8: 0.1, 10: 0.7},
        )


def test_shard_aware_large_batch_decodes_once_and_materializes_selected_rows(
    tmp_path,
) -> None:
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
            ring=4,
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
        ReplaySelection(tuple(spans), {4: 2048}, 4),
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
    selected = dataset.__getitems__(batch[:17])
    assert len(selected) == 17
    assert dataset.shard_load_count == 1
    assert dataset.checksum_verification_count == 1
    assert dataset.sample_materialization_count == 17


def test_plateau_policy_keeps_replay_live_and_resets_after_rejections() -> None:
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
            lag_steps=19,
            terminal_rejection=False,
            rejection_streak=0,
            **common,
        )
        == "proceed"
    )
    assert (
        plateau_policy_decision(
            lag_steps=20,
            terminal_rejection=False,
            rejection_streak=0,
            **common,
        )
        == "pause"
    )
    autonomous = {
        **common,
        "action": "reduce_lr_keep_weights",
    }
    assert (
        plateau_policy_decision(
            lag_steps=20,
            terminal_rejection=True,
            rejection_streak=3,
            **autonomous,
        )
        == "recover"
    )
    assert (
        plateau_policy_decision(
            lag_steps=50,
            terminal_rejection=True,
            rejection_streak=1,
            **autonomous,
        )
        == "recover"
    )
    assert (
        plateau_policy_decision(
            lag_steps=50,
            terminal_rejection=False,
            rejection_streak=0,
            **{**autonomous, "reset_already_applied": True},
        )
        == "proceed"
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
        == "reset"
    )
    assert (
        plateau_policy_decision(
            lag_steps=50,
            terminal_rejection=False,
            rejection_streak=2,
            **common,
        )
        == "pause"
    )


def test_plateau_learning_rate_scale_updates_optimizer_and_scheduler() -> None:
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    scheduler = build_scheduler(
        optimizer,
        SchedulerConfig(warmup_steps=0, total_steps=10, min_lr_ratio=0.1),
    )
    learner = object.__new__(LearnerLoop)
    learner.optimizer = optimizer
    learner.scheduler = scheduler

    learner._scale_learning_rates(0.5)

    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.5)
    assert optimizer.param_groups[0]["initial_lr"] == pytest.approx(0.5)
    assert scheduler.base_lrs == pytest.approx([0.5])
    assert scheduler.get_last_lr() == pytest.approx([0.5])
    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] < 0.5

    adam = torch.optim.AdamW([parameter], lr=0.1)
    parameter.grad = torch.ones_like(parameter)
    adam.step()
    assert adam.state
    learner.optimizer = adam
    learner._clear_optimizer_state()
    assert not adam.state


def test_plateau_recovery_preserves_weights_and_creates_durable_cutover(
    tmp_path,
    monkeypatch,
) -> None:
    parameter = torch.nn.Parameter(torch.tensor(2.0))
    optimizer = torch.optim.AdamW([parameter], lr=0.1)
    scheduler = build_scheduler(
        optimizer,
        SchedulerConfig(warmup_steps=0, total_steps=100),
    )
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    before = parameter.detach().clone()
    events = []
    cutovers = []
    learner = object.__new__(LearnerLoop)
    learner.rank = 0
    learner.world_size = 1
    learner.step = 100
    learner.optimizer = optimizer
    learner.scheduler = scheduler
    learner._last_recovery_step = 90
    learner._last_plateau_reset = None
    learner.publisher = SimpleNamespace(root=tmp_path / "learner")
    learner.promotion_status_path = tmp_path / "arena" / "promotion-status.json"
    learner.run_identity = SimpleNamespace(
        run_id="run-autonomous",
        generation_family="family-autonomous",
    )
    learner.metrics = SimpleNamespace(append=events.append)
    configured = PlateauConfig(
        enabled=True,
        action="reduce_lr_keep_weights",
        reset_learning_rate_scale=0.5,
        clear_optimizer_state_on_recovery=True,
    )
    action = {
        "kind": "recover",
        "reset_reason": "terminal_rejection_streak",
        "candidate_identity": "candidate",
        "candidate_step": 100,
        "champion_identity": "champion",
        "champion_step": 50,
    }
    learner._plateau_config = lambda: configured
    learner._rank_zero_plateau_action = lambda _configured: action
    learner._broadcast_object = lambda value: value
    learner._distributed_barrier = lambda: None
    learner._maybe_write_recovery_checkpoint = lambda **_kwargs: SimpleNamespace(
        checkpoint_sha256="a" * 64
    )
    monkeypatch.setattr(
        "startrain.learner.write_resume_cutover",
        lambda *args, **kwargs: cutovers.append((args, kwargs)),
    )

    assert learner._plateau_control(
        stop_requested=lambda: False,
        progress=None,
    )

    torch.testing.assert_close(parameter, before)
    assert not optimizer.state
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.05)
    assert learner.step == 100
    assert learner._last_plateau_reset == ("champion", "candidate")
    assert len(cutovers) == 1
    status = json.loads(learner.promotion_status_path.read_text(encoding="utf-8"))
    assert status["decision"] == "plateau_recover"
    assert status["candidate_step"] == 100
    assert events[-1]["event"] == "plateau_recovery"
    assert events[-1]["optimizer_state_cleared"] is True


def test_ddp_replay_selection_metadata_is_broadcast_from_rank_zero(
    monkeypatch,
) -> None:
    expected = ReplaySelection((), {4: 512}, 42)
    learner = object.__new__(LearnerLoop)
    learner.world_size = 2
    learner.model = tiny_model()

    def broadcast(payload, **_options):
        assert payload == [None]
        payload[0] = expected

    monkeypatch.setattr(torch.distributed, "broadcast_object_list", broadcast)
    assert learner._broadcast_object(None) is expected


class OneMoveStateBatch:
    def __init__(self, rings: int, batch_size: int) -> None:
        assert rings == 4 and batch_size == 1
        topology = get_topology(rings)
        stones = torch.zeros(topology.n, dtype=torch.int8)
        stones[-1] = -1
        self.position = DoubleStarPosition(
            rings=4,
            stones=stones,
            to_move=0,
            moves_left=1,
            opening=False,
            terminal=False,
        )
        self.last_action = -1

    def data(self) -> FakeStateData:
        return state_data([self.position])

    def apply_many(self, indices: list[int], actions: list[int]) -> None:
        assert indices == [0]
        assert actions == [self.position.stones.numel() - 1]
        self.last_action = actions[0]
        stones = self.position.stones.clone()
        stones[self.last_action] = 0
        self.position = DoubleStarPosition(
            rings=4,
            stones=stones,
            to_move=0,
            moves_left=0,
            opening=False,
            terminal=True,
        )

    def score_data(self) -> object:
        score = score_position(get_topology(4), self.position.stones)
        assert score.leader == 0
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
        margin = score.players[0].total - score.players[1].total
        return SimpleNamespace(
            batch_size=1,
            node_count=self.position.stones.numel(),
            components=components,
            node_owner=score.node_owner.tolist(),
            alive_bits=pack_mask(score.alive_stone),
            winner=[score.leader],
            terminal_value=[1.0],
            outcome_class=[1],
            score_margin=[margin],
        )

    def trajectory_data(self) -> object:
        return SimpleNamespace(
            batch_size=1,
            last_move=[self.last_action],
            current_turn_offsets=[0, 1],
            current_turn_moves=[self.last_action],
            turn_count=[1],
        )


class OneMoveSearchBatch:
    seeds: list[int] = []

    def __init__(self, states: OneMoveStateBatch, **options: object) -> None:
        self.states = states
        self.initialized = False
        self.seeds.append(int(options["deterministic_seed"]))

    def root_requests(self) -> FakeEvalBatch:
        data = self.states.data()
        action = data.node_count - 1
        return FakeEvalBatch([55], data, [0, 1], [action])

    def initialize_roots(self, *_buffers: object) -> None:
        self.initialized = True

    def is_done(self) -> bool:
        return self.initialized

    def next_requests(self) -> FakeEvalBatch:
        raise AssertionError("one-move search completes at root")

    def submit(self, *_buffers: object) -> None:
        raise AssertionError("one-move search has no leaves")

    def results(self) -> object:
        action = self.states.position.stones.numel() - 1
        return SimpleNamespace(
            selected_actions=[action],
            terminal=[False],
            action_offsets=[0, 1],
            actions=[action],
            visits=[1],
            q_values=[1.0],
            priors=[1.0],
            policy_target=[1.0],
        )


class OneMoveNative:
    StateBatch = OneMoveStateBatch
    SearchBatch = OneMoveSearchBatch


class SelfPlayEvaluator:
    model_version = "fake-v2"
    model_step = 7
    model_identity = "sha256-" + "f" * 64

    def evaluate(self, requests: FakeEvalBatch) -> InferenceResponse:
        return InferenceResponse(
            tokens=list(requests.tokens),
            values=[0.0] * len(requests),
            policy_offsets=list(requests.legal_offsets),
            policy_logits=[0.0] * len(requests.legal_actions),
        )


class CapturingSink:
    def __init__(self) -> None:
        self.samples: list[ReplaySample] = []
        self.calls: list[dict[str, object]] = []

    def append(self, samples: list[ReplaySample], **metadata: object) -> object:
        self.samples.extend(samples)
        self.calls.append(metadata)
        return SimpleNamespace(sample_count=len(samples))


@pytest.mark.parametrize("full", [False, True])
def test_selfplay_retains_binary_targets_and_only_full_search_policy(
    full: bool,
) -> None:
    sink = CapturingSink()
    config = SelfPlayConfig(
        rings=4,
        batch_size=1,
        games=1,
        fast_probability=0.0 if full else 1.0,
        full_probability=1.0 if full else 0.0,
        fast_simulations=2,
        full_simulations=4,
        simulation_reference_rings=4,
        max_considered=2,
        shard_size=8,
        seed=41,
    )
    actor = SelfPlayActor(OneMoveNative, SelfPlayEvaluator(), sink, config)
    before_metrics = actor.metrics_snapshot()
    summaries = actor.run()

    assert len(summaries) == 1
    assert summaries[0].model_version == "fake-v2"
    assert summaries[0].winner == 0
    assert summaries[0].terminal_value == 1.0
    assert len(sink.samples) == 1
    sample = sink.samples[0]
    required = TARGET_OUTCOME | TARGET_SCORE_MARGIN | TARGET_OWNERSHIP | TARGET_ALIVE
    assert sample.target_mask & required == required
    assert bool(sample.target_mask & TARGET_POLICY) is full
    assert sample.policy.shape == (get_topology(4).n,)
    if full:
        assert sample.policy[-1] == 1
        assert "completed-q-full" in sample.policy_provenance
    else:
        assert not sample.policy.any()
        assert sample.policy_provenance == "none"
    assert sink.calls[0]["model_step"] == 7
    metrics = actor.metrics_snapshot()
    assert metrics.completed_decisions == 1
    assert metrics.full_decisions == int(full)
    assert metrics.fast_decisions == int(not full)
    assert metrics.policy_entropy_count == int(full)
    assert metrics.policy_entropy_sum == pytest.approx(0.0)
    assert metrics.replay_append_calls == 1
    assert metrics.replay_append_seconds >= 0
    assert metrics.delta(before_metrics) == metrics


def test_selfplay_randomization_is_seeded_and_board_scaled() -> None:
    config = SelfPlayConfig(
        rings=4,
        batch_size=1,
        games=1,
        fast_probability=0.5,
        full_probability=0.5,
        fast_simulations=8,
        full_simulations=32,
        simulation_reference_rings=8,
        simulation_ring_exponent=1.0,
        max_considered=2,
        seed=1234,
    )
    assert config.simulation_budget(full=False) == 4
    assert config.simulation_budget(full=True) == 16
    runs = []
    for _ in range(2):
        OneMoveSearchBatch.seeds.clear()
        sink = CapturingSink()
        SelfPlayActor(OneMoveNative, SelfPlayEvaluator(), sink, config).run()
        runs.append(
            (
                list(OneMoveSearchBatch.seeds),
                sink.samples[0].policy.copy(),
                sink.samples[0].search_provenance,
            )
        )
    assert runs[0][0] == runs[1][0]
    np.testing.assert_array_equal(runs[0][1], runs[1][1])
    assert runs[0][2] == runs[1][2]


def test_policy_surprise_weighting_preserves_total_mass_and_prioritizes() -> None:
    actor = object.__new__(SelfPlayActor)
    actor.config = SelfPlayConfig(
        policy_surprise_weight=0.5,
        policy_surprise_max_weight=4.0,
    )
    decisions = [
        SimpleNamespace(policy=np.ones(1), policy_surprise=0.1),
        SimpleNamespace(policy=np.ones(1), policy_surprise=0.9),
        SimpleNamespace(policy=None, policy_surprise=0.0),
    ]

    weights = actor._policy_surprise_sample_weights(decisions)

    assert sum(weights) == pytest.approx(3.0)
    assert weights[1] > weights[0]
    assert weights[2] == pytest.approx(1.0)


def test_selfplay_measures_replay_append_bytes_and_metric_monotonicity(
    tmp_path,
) -> None:
    class SizedSink(CapturingSink):
        def append(self, samples: list[ReplaySample], **metadata: object) -> object:
            super().append(samples, **metadata)
            path = tmp_path / "measured-shard.npz"
            path.write_bytes(b"x" * 137)
            return SimpleNamespace(sample_count=len(samples), path=path)

    actor = SelfPlayActor(
        OneMoveNative,
        SelfPlayEvaluator(),
        SizedSink(),
        SelfPlayConfig.cpu_smoke(seed=42),
    )
    actor.run()
    metrics = actor.metrics_snapshot()
    assert metrics.replay_append_calls == 1
    assert metrics.replay_append_bytes == 137
    assert metrics.replay_append_seconds >= 0
    with pytest.raises(ValueError, match="monotonic"):
        SelfPlayMetrics().delta(metrics)
