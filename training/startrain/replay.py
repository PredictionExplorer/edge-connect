"""Validated schema-v3 replay samples and pickle-free heterogeneous shards."""

from __future__ import annotations

import json
import numbers
import os
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .actions import relocate_sample_actions
from .contracts import (
    ACTION_LAYOUT_VERSION,
    ALL_TARGETS,
    FEATURE_SCHEMA_HASH,
)
from .contracts import (
    RULES_HASH,
    RULES_HASH_WIRE,
    RULES_SCHEMA_ID,
    SCORE_MARGIN_MAX,
    SCORE_MARGIN_MIN,
    SOFT_POLICY_TEMPERATURE,
    TARGET_ALIVE,
    TARGET_OWNERSHIP,
    TARGET_POLICY,
    TARGET_SCORE_MARGIN,
    TARGET_SOFT_POLICY,
    TARGET_WDL,
    WDL_DRAW,
    WDL_LOSS,
    WDL_WIN,
)
from .features import (
    DoubleStarPosition,
    EncodedBatch,
    collate_encoded,
    encode_position,
)
from .losses import TrainingTargets
from .runtime import validate_identifier
from .scoring import ScoreResult
from .symmetry import D5Transform
from .topology import get_topology

REPLAY_SCHEMA_VERSION = 3
REPLAY_SHARD_FORMAT = "startrain.replay.npz"
MISSING_LEADER = -2
MISSING_OWNERSHIP = -100
MISSING_ALIVE = 255


class ReplaySchemaError(ValueError):
    pass


def _checked_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ReplaySchemaError(f"{name} must be an integer before conversion")
    return int(value)


def _checked_bool(name: str, value: object) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ReplaySchemaError(f"{name} must be bool")
    return bool(value)


def _integer_array(
    name: str,
    value: object,
    *,
    shape: tuple[int, ...],
    dtype: np.dtype,
) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.integer):
        raise ReplaySchemaError(f"{name} must contain integers before conversion")
    if array.shape != shape:
        raise ReplaySchemaError(f"{name} must have shape {shape}")
    limits = np.iinfo(dtype)
    if array.size and (np.min(array) < limits.min or np.max(array) > limits.max):
        raise ReplaySchemaError(f"{name} cannot be represented as {dtype}")
    return np.ascontiguousarray(array, dtype=dtype)


def _float_array(
    name: str,
    value: object,
    *,
    shape: tuple[int, ...],
) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number):
        raise ReplaySchemaError(f"{name} must be numeric")
    if array.shape != shape:
        raise ReplaySchemaError(f"{name} must have shape {shape}")
    array = np.ascontiguousarray(array, dtype=np.float32)
    if not np.isfinite(array).all() or (array < 0).any():
        raise ReplaySchemaError(f"{name} must be finite and non-negative")
    return array


def katago_soft_policy_target(
    policy: np.ndarray,
    legal_mask: np.ndarray,
    *,
    temperature: float = SOFT_POLICY_TEMPERATURE,
) -> np.ndarray:
    """KataGo auxiliary target: normalize ``policy ** (1 / T)`` at T=4."""

    if temperature != SOFT_POLICY_TEMPERATURE:
        raise ReplaySchemaError("soft-policy temperature must be exactly 4")
    values = np.where(legal_mask, np.asarray(policy, dtype=np.float64), 0.0)
    values = np.power(values, 1.0 / temperature)
    mass = float(values.sum())
    if mass <= 0:
        raise ReplaySchemaError("soft-policy source has no legal mass")
    return (values / mass).astype(np.float32)


@dataclass(slots=True)
class ReplaySample:
    rings: int
    stones: np.ndarray
    to_move: int
    moves_left: int
    opening: bool
    pass_streak: int
    terminal: bool
    policy: np.ndarray
    soft_policy: np.ndarray
    target_mask: int
    final_leader: int
    final_scores: np.ndarray
    final_quarks: np.ndarray
    final_ownership: np.ndarray
    final_alive: np.ndarray
    search_provenance: str
    policy_provenance: str
    run_id: str = "manual"
    generation_family: str = "manual"
    actor_id: str = "manual"
    generation: int = 0
    game_id: str = field(default_factory=lambda: f"manual-{uuid.uuid4().hex}")
    ply: int = 0
    model_identity: str = "manual"
    soft_policy_temperature: float = SOFT_POLICY_TEMPERATURE
    rules_hash: int = RULES_HASH
    feature_schema_hash: int = FEATURE_SCHEMA_HASH
    weight: float = 1.0
    schema_version: int = REPLAY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        schema_version = _checked_int("schema_version", self.schema_version)
        if schema_version != REPLAY_SCHEMA_VERSION:
            raise ReplaySchemaError(f"sample schema {schema_version} is unsupported")
        rings = _checked_int("rings", self.rings)
        topology = get_topology(rings)
        stones = _integer_array(
            "stones", self.stones, shape=(topology.n,), dtype=np.dtype(np.int8)
        )
        if not np.isin(stones, (-1, 0, 1)).all():
            raise ReplaySchemaError("stones must contain only -1, 0, or 1")
        to_move = _checked_int("to_move", self.to_move)
        moves_left = _checked_int("moves_left", self.moves_left)
        pass_streak = _checked_int("pass_streak", self.pass_streak)
        opening = _checked_bool("opening", self.opening)
        terminal = _checked_bool("terminal", self.terminal)

        # Reuse the exact semantic-key validator before storing narrowed dtypes.
        DoubleStarPosition.from_sequence(
            rings=rings,
            stones=stones,
            to_move=to_move,
            moves_left=moves_left,
            opening=opening,
            pass_streak=pass_streak,
            terminal=terminal,
        )

        target_mask = _checked_int("target_mask", self.target_mask)
        if target_mask < 0 or target_mask & ~ALL_TARGETS:
            raise ReplaySchemaError("target_mask contains unknown bits")
        policy = _float_array("policy", self.policy, shape=(topology.n + 1,))
        soft_policy = _float_array(
            "soft_policy", self.soft_policy, shape=(topology.n + 1,)
        )
        final_leader = _checked_int("final_leader", self.final_leader)
        final_scores = _integer_array(
            "final_scores", self.final_scores, shape=(2,), dtype=np.dtype(np.int16)
        )
        final_quarks = _integer_array(
            "final_quarks", self.final_quarks, shape=(2,), dtype=np.dtype(np.int8)
        )
        final_ownership = _integer_array(
            "final_ownership",
            self.final_ownership,
            shape=(topology.n,),
            dtype=np.dtype(np.int8),
        )
        final_alive = _integer_array(
            "final_alive",
            self.final_alive,
            shape=(topology.n,),
            dtype=np.dtype(np.uint8),
        )

        if not isinstance(self.search_provenance, str) or not self.search_provenance:
            raise ReplaySchemaError("search_provenance must be a non-empty string")
        if not isinstance(self.policy_provenance, str) or not self.policy_provenance:
            raise ReplaySchemaError("policy_provenance must be a non-empty string")
        try:
            run_id = validate_identifier("run_id", self.run_id)
            generation_family = validate_identifier(
                "generation_family", self.generation_family
            )
            actor_id = validate_identifier("actor_id", self.actor_id)
            game_id = validate_identifier("game_id", self.game_id)
            model_identity = validate_identifier("model_identity", self.model_identity)
        except ValueError as exc:
            raise ReplaySchemaError(str(exc)) from exc
        generation = _checked_int("generation", self.generation)
        ply = _checked_int("ply", self.ply)
        if generation < 0 or ply < 0:
            raise ReplaySchemaError("generation and ply must be non-negative")
        if float(self.soft_policy_temperature) != SOFT_POLICY_TEMPERATURE:
            raise ReplaySchemaError("soft_policy_temperature must be exactly 4")
        rules_hash = _checked_int("rules_hash", self.rules_hash)
        feature_hash = _checked_int("feature_schema_hash", self.feature_schema_hash)
        if rules_hash != RULES_HASH:
            raise ReplaySchemaError("rules hash does not match the Rust contract")
        if feature_hash != FEATURE_SCHEMA_HASH:
            raise ReplaySchemaError("feature schema hash is incompatible")
        weight = float(self.weight)
        if not np.isfinite(weight) or weight <= 0:
            raise ReplaySchemaError("sample weight must be finite and positive")

        legal = np.zeros(topology.n + 1, dtype=np.bool_)
        if not terminal:
            legal[: topology.n] = stones == -1
            legal[topology.n] = True
        self._validate_policy(
            "policy", policy, legal, bool(target_mask & TARGET_POLICY)
        )
        self._validate_policy(
            "soft_policy",
            soft_policy,
            legal,
            bool(target_mask & TARGET_SOFT_POLICY),
        )
        if target_mask & TARGET_SOFT_POLICY:
            if not target_mask & TARGET_POLICY:
                raise ReplaySchemaError("soft policy requires a policy target")
            expected = katago_soft_policy_target(policy, legal)
            if not np.allclose(soft_policy, expected, atol=2e-6, rtol=2e-6):
                raise ReplaySchemaError(
                    "soft policy is not the T=4 exponent-1/4 target"
                )
        if terminal and target_mask & (TARGET_POLICY | TARGET_SOFT_POLICY):
            raise ReplaySchemaError("terminal samples cannot carry decision policies")

        if target_mask & (TARGET_WDL | TARGET_SCORE_MARGIN):
            if final_leader not in (-1, 0, 1):
                raise ReplaySchemaError("final leader must be -1, 0, or 1")
            if not np.logical_and(final_quarks >= 0, final_quarks <= 5).all():
                raise ReplaySchemaError("final quarks must be in 0..5")
            expected_leader = _leader_from_scores(final_scores, final_quarks)
            if final_leader != expected_leader:
                raise ReplaySchemaError(
                    "final leader disagrees with totals and quark tiebreak"
                )
        elif final_leader != MISSING_LEADER:
            raise ReplaySchemaError("unavailable final result must use missing leader")

        if target_mask & TARGET_SCORE_MARGIN:
            margin = int(final_scores[to_move]) - int(final_scores[1 - to_move])
            if not SCORE_MARGIN_MIN <= margin <= SCORE_MARGIN_MAX:
                raise ReplaySchemaError("score margin is outside [-181, 181]")
        if target_mask & TARGET_OWNERSHIP:
            if not np.isin(final_ownership, (-1, 0, 1)).all():
                raise ReplaySchemaError("final ownership must contain -1, 0, or 1")
        elif not np.all(final_ownership == MISSING_OWNERSHIP):
            raise ReplaySchemaError("missing ownership must use -100")
        if target_mask & TARGET_ALIVE:
            if not np.isin(final_alive, (0, 1)).all():
                raise ReplaySchemaError("final alive target must be binary")
        elif not np.all(final_alive == MISSING_ALIVE):
            raise ReplaySchemaError("missing alive target must use 255")

        self.rings = rings
        self.stones = stones
        self.to_move = to_move
        self.moves_left = moves_left
        self.opening = opening
        self.pass_streak = pass_streak
        self.terminal = terminal
        self.policy = policy
        self.soft_policy = soft_policy
        self.target_mask = target_mask
        self.final_leader = final_leader
        self.final_scores = final_scores
        self.final_quarks = final_quarks
        self.final_ownership = final_ownership
        self.final_alive = final_alive
        self.soft_policy_temperature = SOFT_POLICY_TEMPERATURE
        self.rules_hash = rules_hash
        self.feature_schema_hash = feature_hash
        self.weight = weight
        self.schema_version = schema_version
        self.run_id = run_id
        self.generation_family = generation_family
        self.actor_id = actor_id
        self.generation = generation
        self.game_id = game_id
        self.ply = ply
        self.model_identity = model_identity

    @staticmethod
    def _validate_policy(
        name: str,
        values: np.ndarray,
        legal: np.ndarray,
        available: bool,
    ) -> None:
        if not available:
            if np.any(values != 0):
                raise ReplaySchemaError(f"unavailable {name} must be all zero")
            return
        if np.any(values[~legal] > 1e-8):
            raise ReplaySchemaError(f"{name} has support on an illegal action")
        mass = float(values[legal].sum())
        if not np.isclose(mass, 1.0, atol=1e-5, rtol=1e-5):
            raise ReplaySchemaError(f"{name} legal mass must sum to one")

    @classmethod
    def from_position(
        cls,
        position: DoubleStarPosition,
        *,
        policy: np.ndarray | None,
        final_score: ScoreResult | None,
        search_provenance: str,
        policy_provenance: str,
        include_spatial_targets: bool = True,
        weight: float = 1.0,
        run_id: str = "manual",
        generation_family: str = "manual",
        actor_id: str = "manual",
        generation: int = 0,
        game_id: str | None = None,
        ply: int = 0,
        model_identity: str = "manual",
    ) -> "ReplaySample":
        topology = get_topology(position.rings)
        target_mask = 0
        if policy is None:
            policy_array = np.zeros(topology.n + 1, dtype=np.float32)
            soft_policy = np.zeros_like(policy_array)
        else:
            if position.terminal:
                raise ReplaySchemaError("terminal positions cannot have policy targets")
            policy_array = np.asarray(policy, dtype=np.float32)
            legal = np.concatenate(
                (
                    (position.stones.detach().cpu().numpy() == -1),
                    np.asarray([True]),
                )
            )
            soft_policy = katago_soft_policy_target(policy_array, legal)
            target_mask |= TARGET_POLICY | TARGET_SOFT_POLICY

        if final_score is None:
            final_leader = MISSING_LEADER
            final_scores = np.zeros(2, dtype=np.int16)
            final_quarks = np.zeros(2, dtype=np.int8)
            final_ownership = np.full(topology.n, MISSING_OWNERSHIP, dtype=np.int8)
            final_alive = np.full(topology.n, MISSING_ALIVE, dtype=np.uint8)
        else:
            final_leader = final_score.leader
            final_scores = np.asarray(
                [player.total for player in final_score.players], dtype=np.int16
            )
            final_quarks = np.asarray(
                [player.quarks for player in final_score.players], dtype=np.int8
            )
            target_mask |= TARGET_WDL | TARGET_SCORE_MARGIN
            if include_spatial_targets:
                final_ownership = final_score.node_owner.numpy()
                final_alive = final_score.alive_stone.numpy().astype(np.uint8)
                target_mask |= TARGET_OWNERSHIP | TARGET_ALIVE
            else:
                final_ownership = np.full(topology.n, MISSING_OWNERSHIP, dtype=np.int8)
                final_alive = np.full(topology.n, MISSING_ALIVE, dtype=np.uint8)
        return cls(
            rings=position.rings,
            stones=position.stones.detach().cpu().numpy(),
            to_move=position.to_move,
            moves_left=position.moves_left,
            opening=position.opening,
            pass_streak=position.pass_streak,
            terminal=position.terminal,
            policy=policy_array,
            soft_policy=soft_policy,
            target_mask=target_mask,
            final_leader=final_leader,
            final_scores=final_scores,
            final_quarks=final_quarks,
            final_ownership=final_ownership,
            final_alive=final_alive,
            search_provenance=search_provenance,
            policy_provenance=policy_provenance,
            run_id=run_id,
            generation_family=generation_family,
            actor_id=actor_id,
            generation=generation,
            game_id=game_id or f"manual-{uuid.uuid4().hex}",
            ply=ply,
            model_identity=model_identity,
            weight=weight,
        )

    def to_position(self) -> DoubleStarPosition:
        return DoubleStarPosition.from_sequence(
            rings=self.rings,
            stones=self.stones,
            to_move=self.to_move,
            moves_left=self.moves_left,
            opening=self.opening,
            pass_streak=self.pass_streak,
            terminal=self.terminal,
        )

    def outcome_targets(self) -> tuple[int, int]:
        """Return current-player WDL and margin from absolute final data.

        The WDL leader uses total score first and absolute quark count as the
        authoritative tie-break. Therefore a zero score margin can still be a
        win or loss when the final quark counts differ.
        """

        if not self.target_mask & (TARGET_WDL | TARGET_SCORE_MARGIN):
            raise ReplaySchemaError("final outcome targets are unavailable")
        if self.final_leader == -1:
            wdl = WDL_DRAW
        elif self.final_leader == self.to_move:
            wdl = WDL_WIN
        else:
            wdl = WDL_LOSS
        margin = int(self.final_scores[self.to_move]) - int(
            self.final_scores[1 - self.to_move]
        )
        return wdl, margin


def _leader_from_scores(scores: np.ndarray, quarks: np.ndarray) -> int:
    if scores[0] != scores[1]:
        return 0 if scores[0] > scores[1] else 1
    if quarks[0] != quarks[1]:
        return 0 if quarks[0] > quarks[1] else 1
    return -1


def augment_sample(sample: ReplaySample, transform: D5Transform) -> ReplaySample:
    topology = get_topology(sample.rings)
    permutation = topology.d5_permutation(
        transform.rotation, transform.reflected
    ).numpy()

    def nodes(values: np.ndarray) -> np.ndarray:
        output = np.empty_like(values)
        output[permutation] = values
        return output

    def actions(values: np.ndarray) -> np.ndarray:
        output = np.empty_like(values)
        output[permutation] = values[: topology.n]
        output[topology.n] = values[topology.n]
        return output

    return ReplaySample(
        rings=sample.rings,
        stones=nodes(sample.stones),
        to_move=sample.to_move,
        moves_left=sample.moves_left,
        opening=sample.opening,
        pass_streak=sample.pass_streak,
        terminal=sample.terminal,
        policy=actions(sample.policy),
        soft_policy=actions(sample.soft_policy),
        target_mask=sample.target_mask,
        final_leader=sample.final_leader,
        final_scores=sample.final_scores.copy(),
        final_quarks=sample.final_quarks.copy(),
        final_ownership=nodes(sample.final_ownership),
        final_alive=nodes(sample.final_alive),
        search_provenance=sample.search_provenance,
        policy_provenance=sample.policy_provenance,
        run_id=sample.run_id,
        generation_family=sample.generation_family,
        actor_id=sample.actor_id,
        generation=sample.generation,
        game_id=sample.game_id,
        ply=sample.ply,
        model_identity=sample.model_identity,
        weight=sample.weight,
    )


def _offsets(lengths: Sequence[int]) -> np.ndarray:
    output = np.zeros(len(lengths) + 1, dtype=np.int64)
    output[1:] = np.cumsum(lengths, dtype=np.int64)
    return output


def write_replay_shard(
    destination: str | Path,
    samples: Sequence[ReplaySample],
    *,
    compressed: bool = True,
) -> Path:
    if not samples:
        raise ValueError("cannot write an empty replay shard")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    node_offsets = _offsets([sample.stones.size for sample in samples])
    action_offsets = _offsets([sample.policy.size for sample in samples])
    metadata = {
        "format": REPLAY_SHARD_FORMAT,
        "schema_version": REPLAY_SCHEMA_VERSION,
        "rules_schema": RULES_SCHEMA_ID,
        "rules_hash": RULES_HASH,
        "rules_hash_wire": RULES_HASH_WIRE,
        "feature_schema_hash": FEATURE_SCHEMA_HASH,
        "action_layout_version": ACTION_LAYOUT_VERSION,
        "sample_count": len(samples),
        "soft_policy_temperature": SOFT_POLICY_TEMPERATURE,
    }
    arrays = {
        "metadata": np.asarray(json.dumps(metadata, sort_keys=True)),
        "rings": np.asarray([sample.rings for sample in samples], dtype=np.int8),
        "node_offsets": node_offsets,
        "action_offsets": action_offsets,
        "stones": np.concatenate([sample.stones for sample in samples]),
        "to_move": np.asarray([sample.to_move for sample in samples], dtype=np.int8),
        "moves_left": np.asarray(
            [sample.moves_left for sample in samples], dtype=np.int8
        ),
        "opening": np.asarray([sample.opening for sample in samples], dtype=np.bool_),
        "pass_streak": np.asarray(
            [sample.pass_streak for sample in samples], dtype=np.int8
        ),
        "terminal": np.asarray([sample.terminal for sample in samples], dtype=np.bool_),
        "policy": np.concatenate([sample.policy for sample in samples]),
        "soft_policy": np.concatenate([sample.soft_policy for sample in samples]),
        "target_mask": np.asarray(
            [sample.target_mask for sample in samples], dtype=np.uint16
        ),
        "final_leader": np.asarray(
            [sample.final_leader for sample in samples], dtype=np.int8
        ),
        "final_scores": np.stack([sample.final_scores for sample in samples]),
        "final_quarks": np.stack([sample.final_quarks for sample in samples]),
        "final_ownership": np.concatenate(
            [sample.final_ownership for sample in samples]
        ),
        "final_alive": np.concatenate([sample.final_alive for sample in samples]),
        "search_provenance": np.asarray(
            [sample.search_provenance for sample in samples], dtype=np.str_
        ),
        "policy_provenance": np.asarray(
            [sample.policy_provenance for sample in samples], dtype=np.str_
        ),
        "run_id": np.asarray([sample.run_id for sample in samples], dtype=np.str_),
        "generation_family": np.asarray(
            [sample.generation_family for sample in samples], dtype=np.str_
        ),
        "actor_id": np.asarray([sample.actor_id for sample in samples], dtype=np.str_),
        "generation": np.asarray(
            [sample.generation for sample in samples], dtype=np.int64
        ),
        "game_id": np.asarray([sample.game_id for sample in samples], dtype=np.str_),
        "ply": np.asarray([sample.ply for sample in samples], dtype=np.int32),
        "model_identity": np.asarray(
            [sample.model_identity for sample in samples], dtype=np.str_
        ),
        "soft_policy_temperature": np.asarray(
            [sample.soft_policy_temperature for sample in samples],
            dtype=np.float32,
        ),
        "rules_hash": np.asarray(
            [sample.rules_hash for sample in samples], dtype=np.uint64
        ),
        "feature_schema_hash": np.asarray(
            [sample.feature_schema_hash for sample in samples], dtype=np.uint64
        ),
        "weight": np.asarray([sample.weight for sample in samples], dtype=np.float32),
    }
    writer = np.savez_compressed if compressed else np.savez
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            writer(temporary, **arrays)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, destination)
        descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return destination


def read_replay_shard(source: str | Path) -> list[ReplaySample]:
    with np.load(Path(source), allow_pickle=False) as shard:
        metadata = json.loads(str(shard["metadata"].item()))
        expected_metadata = {
            "format": REPLAY_SHARD_FORMAT,
            "schema_version": REPLAY_SCHEMA_VERSION,
            "rules_schema": RULES_SCHEMA_ID,
            "rules_hash": RULES_HASH,
            "rules_hash_wire": RULES_HASH_WIRE,
            "feature_schema_hash": FEATURE_SCHEMA_HASH,
            "action_layout_version": ACTION_LAYOUT_VERSION,
            "soft_policy_temperature": SOFT_POLICY_TEMPERATURE,
        }
        for key, expected in expected_metadata.items():
            if metadata.get(key) != expected:
                raise ReplaySchemaError(f"incompatible shard metadata: {key}")
        count = _checked_int("sample_count", metadata.get("sample_count"))
        if shard["rings"].shape != (count,):
            raise ReplaySchemaError("shard sample count does not match arrays")
        node_offsets = shard["node_offsets"]
        action_offsets = shard["action_offsets"]
        samples: list[ReplaySample] = []
        for index in range(count):
            node_slice = slice(int(node_offsets[index]), int(node_offsets[index + 1]))
            action_slice = slice(
                int(action_offsets[index]), int(action_offsets[index + 1])
            )
            samples.append(
                ReplaySample(
                    rings=shard["rings"][index],
                    stones=shard["stones"][node_slice].copy(),
                    to_move=shard["to_move"][index],
                    moves_left=shard["moves_left"][index],
                    opening=shard["opening"][index],
                    pass_streak=shard["pass_streak"][index],
                    terminal=shard["terminal"][index],
                    policy=shard["policy"][action_slice].copy(),
                    soft_policy=shard["soft_policy"][action_slice].copy(),
                    target_mask=shard["target_mask"][index],
                    final_leader=shard["final_leader"][index],
                    final_scores=shard["final_scores"][index].copy(),
                    final_quarks=shard["final_quarks"][index].copy(),
                    final_ownership=shard["final_ownership"][node_slice].copy(),
                    final_alive=shard["final_alive"][node_slice].copy(),
                    search_provenance=str(shard["search_provenance"][index]),
                    policy_provenance=str(shard["policy_provenance"][index]),
                    run_id=str(shard["run_id"][index]),
                    generation_family=str(shard["generation_family"][index]),
                    actor_id=str(shard["actor_id"][index]),
                    generation=int(shard["generation"][index]),
                    game_id=str(shard["game_id"][index]),
                    ply=int(shard["ply"][index]),
                    model_identity=str(shard["model_identity"][index]),
                    soft_policy_temperature=float(
                        shard["soft_policy_temperature"][index]
                    ),
                    rules_hash=shard["rules_hash"][index],
                    feature_schema_hash=shard["feature_schema_hash"][index],
                    weight=float(shard["weight"][index]),
                )
            )
    return samples


class ReplayDataset(Dataset[ReplaySample]):
    def __init__(self, samples: Sequence[ReplaySample]) -> None:
        self.samples = list(samples)
        self.rings = [sample.rings for sample in self.samples]

    @classmethod
    def from_shards(cls, paths: Sequence[str | Path]) -> "ReplayDataset":
        samples: list[ReplaySample] = []
        for path in paths:
            samples.extend(read_replay_shard(path))
        return cls(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> ReplaySample:
        return self.samples[index]


@dataclass(frozen=True, slots=True)
class ReplayBatch:
    inputs: EncodedBatch
    targets: TrainingTargets

    def to(
        self,
        device: torch.device | str,
        *,
        feature_dtype: torch.dtype | None = None,
    ) -> "ReplayBatch":
        return ReplayBatch(
            inputs=self.inputs.to(device, feature_dtype=feature_dtype),
            targets=self.targets.to(device),
        )


def collate_replay_samples(samples: Sequence[ReplaySample]) -> ReplayBatch:
    if not samples:
        raise ValueError("cannot collate an empty replay batch")
    inputs = collate_encoded(
        [encode_position(sample.to_position()) for sample in samples]
    )
    batch_size = len(samples)
    max_nodes = inputs.max_nodes
    policy = torch.zeros((batch_size, max_nodes + 1), dtype=torch.float32)
    soft_policy = torch.zeros_like(policy)
    ownership = torch.full((batch_size, max_nodes), -100, dtype=torch.long)
    alive = torch.full((batch_size, max_nodes), -1.0, dtype=torch.float32)
    wdl = torch.zeros(batch_size, dtype=torch.long)
    margin = torch.zeros(batch_size, dtype=torch.long)

    masks = {
        "policy": torch.zeros(batch_size, dtype=torch.bool),
        "wdl": torch.zeros(batch_size, dtype=torch.bool),
        "margin": torch.zeros(batch_size, dtype=torch.bool),
        "ownership": torch.zeros(batch_size, dtype=torch.bool),
        "alive": torch.zeros(batch_size, dtype=torch.bool),
        "soft": torch.zeros(batch_size, dtype=torch.bool),
    }
    for index, sample in enumerate(samples):
        nodes = get_topology(sample.rings).n
        policy[index] = relocate_sample_actions(
            torch.from_numpy(sample.policy),
            sample_nodes=nodes,
            batch_max_nodes=max_nodes,
            fill_value=0.0,
        )
        soft_policy[index] = relocate_sample_actions(
            torch.from_numpy(sample.soft_policy),
            sample_nodes=nodes,
            batch_max_nodes=max_nodes,
            fill_value=0.0,
        )
        masks["policy"][index] = bool(sample.target_mask & TARGET_POLICY)
        masks["soft"][index] = bool(sample.target_mask & TARGET_SOFT_POLICY)
        if sample.target_mask & (TARGET_WDL | TARGET_SCORE_MARGIN):
            sample_wdl, sample_margin = sample.outcome_targets()
            wdl[index] = sample_wdl
            margin[index] = sample_margin
        masks["wdl"][index] = bool(sample.target_mask & TARGET_WDL)
        masks["margin"][index] = bool(sample.target_mask & TARGET_SCORE_MARGIN)
        if sample.target_mask & TARGET_OWNERSHIP:
            absolute_owner = torch.from_numpy(sample.final_ownership)
            ownership[index, :nodes] = torch.where(
                absolute_owner == -1,
                torch.tensor(2),
                torch.where(
                    absolute_owner == sample.to_move,
                    torch.tensor(0),
                    torch.tensor(1),
                ),
            )
            masks["ownership"][index] = True
        if sample.target_mask & TARGET_ALIVE:
            alive[index, :nodes] = torch.from_numpy(sample.final_alive).float()
            masks["alive"][index] = True

    return ReplayBatch(
        inputs=inputs,
        targets=TrainingTargets(
            policy=policy,
            wdl=wdl,
            score_margin=margin,
            ownership=ownership,
            alive=alive,
            soft_policy=soft_policy,
            policy_mask=masks["policy"],
            wdl_mask=masks["wdl"],
            score_margin_mask=masks["margin"],
            ownership_mask=masks["ownership"],
            alive_mask=masks["alive"],
            soft_policy_mask=masks["soft"],
            sample_weight=torch.tensor(
                [sample.weight for sample in samples], dtype=torch.float32
            ),
        ),
    )
