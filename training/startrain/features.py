"""Schema-v2 features that are a pure function of the Rust semantic key."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import torch
from torch import Tensor

from .actions import relocate_sample_actions
from .contracts import FEATURE_SCHEMA_HASH, FEATURE_SCHEMA_VERSION
from .scoring import EMPTY, ScoreResult, score_position
from .topology import MAX_RINGS, StarTopology, get_topology

NODE_FEATURE_NAMES = (
    "empty",
    "current_stone",
    "opponent_stone",
    "owner_current",
    "owner_opponent",
    "owner_unclaimed",
    "alive_current",
    "alive_opponent",
    "is_peri",
    "is_quark",
    "ring_fraction",
    "arm_distance_fraction",
    "degree_fraction",
    "is_bridge",
    "legal",
)
GLOBAL_FEATURE_NAMES = (
    "rings_fraction",
    "occupancy_fraction",
    "current_stone_fraction",
    "opponent_stone_fraction",
    "moves_left_fraction",
    "opening",
    "pass_streak_fraction",
    "terminal",
    "current_total_scaled",
    "opponent_total_scaled",
    "score_margin_scaled",
    "current_peries_fraction",
    "opponent_peries_fraction",
    "current_quarks_fraction",
    "opponent_quarks_fraction",
    "current_stars_fraction",
    "opponent_stars_fraction",
    "contested_peries_fraction",
)
NODE_FEATURE_DIM = len(NODE_FEATURE_NAMES)
GLOBAL_FEATURE_DIM = len(GLOBAL_FEATURE_NAMES)

_INTEGER_DTYPES = {
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
}


def _plain_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


@dataclass(frozen=True, slots=True)
class DoubleStarPosition:
    """Exact Python form of ``star_engine::StateKey``.

    Network inputs depend only on these seven fields: rings, stones, to_move,
    moves_left, opening, pass_streak, and terminal. Terminal states are valid
    data records but have no legal decision policy.
    """

    rings: int
    stones: Tensor
    to_move: int
    moves_left: int
    opening: bool
    pass_streak: int
    terminal: bool

    def __post_init__(self) -> None:
        rings = _plain_int("rings", self.rings)
        topology = get_topology(rings)
        if not isinstance(self.stones, Tensor):
            raise TypeError("stones must be a torch.Tensor")
        if self.stones.dtype not in _INTEGER_DTYPES:
            raise TypeError("stones must use an integer dtype")
        if self.stones.ndim != 1 or self.stones.numel() != topology.n:
            raise ValueError(f"stones must have shape ({topology.n},)")
        values = self.stones.detach().to(device="cpu")
        if not bool(((values == EMPTY) | (values == 0) | (values == 1)).all()):
            raise ValueError("stones must contain only -1, 0, or 1")

        to_move = _plain_int("to_move", self.to_move)
        moves_left = _plain_int("moves_left", self.moves_left)
        pass_streak = _plain_int("pass_streak", self.pass_streak)
        if to_move not in (0, 1):
            raise ValueError("to_move must be 0 or 1")
        if moves_left not in (0, 1, 2):
            raise ValueError("moves_left must be in 0..2")
        if pass_streak not in (0, 1, 2):
            raise ValueError("pass_streak must be in 0..2")
        if type(self.opening) is not bool or type(self.terminal) is not bool:
            raise TypeError("opening and terminal must be bool")

        occupied = int((values != EMPTY).sum())
        board_full = occupied == topology.n
        derived_terminal = board_full or pass_streak == 2
        if self.terminal != derived_terminal:
            raise ValueError("terminal must equal board-full or pass_streak == 2")
        if moves_left == 0 and not board_full:
            raise ValueError("moves_left == 0 is valid only on a full board")
        if board_full and moves_left > 1:
            raise ValueError("a full board may retain at most one placement")
        if self.opening and (
            to_move != 0
            or moves_left != 1
            or pass_streak != 0
            or occupied != 0
            or self.terminal
        ):
            raise ValueError("invalid one-stone opening metadata")

    @classmethod
    def from_sequence(
        cls,
        *,
        rings: int,
        stones: Sequence[int],
        to_move: int,
        moves_left: int,
        opening: bool,
        pass_streak: int,
        terminal: bool,
    ) -> "DoubleStarPosition":
        raw = torch.as_tensor(stones)
        if raw.dtype not in _INTEGER_DTYPES:
            raise TypeError("stones must contain integers before conversion")
        raw_values = raw.detach().to(device="cpu")
        if not bool(
            ((raw_values == EMPTY) | (raw_values == 0) | (raw_values == 1)).all()
        ):
            raise ValueError("stones must contain only -1, 0, or 1 before conversion")
        return cls(
            rings=rings,
            stones=raw.to(dtype=torch.int8).clone(),
            to_move=to_move,
            moves_left=moves_left,
            opening=opening,
            pass_streak=pass_streak,
            terminal=terminal,
        )

    def with_stones(self, stones: Tensor) -> "DoubleStarPosition":
        return replace(self, stones=stones)


@dataclass(frozen=True, slots=True)
class EncodedPosition:
    topology: StarTopology
    node_features: Tensor
    global_features: Tensor
    legal_node_mask: Tensor
    legal_pass: Tensor
    score: ScoreResult


@dataclass(frozen=True, slots=True)
class EncodedBatch:
    node_features: Tensor
    global_features: Tensor
    neighbor_index: Tensor
    neighbor_mask: Tensor
    neighbor_edge_type: Tensor
    node_mask: Tensor
    legal_action_mask: Tensor
    rings: Tensor

    @property
    def batch_size(self) -> int:
        return int(self.node_features.shape[0])

    @property
    def max_nodes(self) -> int:
        return int(self.node_features.shape[1])

    def to(
        self,
        device: torch.device | str,
        *,
        feature_dtype: torch.dtype | None = None,
    ) -> "EncodedBatch":
        dtype = feature_dtype or self.node_features.dtype
        return EncodedBatch(
            node_features=self.node_features.to(device=device, dtype=dtype),
            global_features=self.global_features.to(device=device, dtype=dtype),
            neighbor_index=self.neighbor_index.to(device=device),
            neighbor_mask=self.neighbor_mask.to(device=device),
            neighbor_edge_type=self.neighbor_edge_type.to(device=device),
            node_mask=self.node_mask.to(device=device),
            legal_action_mask=self.legal_action_mask.to(device=device),
            rings=self.rings.to(device=device),
        )

    def model_args(
        self,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        return (
            self.node_features,
            self.global_features,
            self.neighbor_index,
            self.neighbor_mask,
            self.neighbor_edge_type,
            self.node_mask,
            self.legal_action_mask,
        )


def encode_position(
    position: DoubleStarPosition,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> EncodedPosition:
    """Encode schema v2 from the semantic key, with no history leakage."""

    topology = get_topology(position.rings)
    stones = position.stones.detach().to(device="cpu", dtype=torch.int8)
    score = score_position(topology, stones)
    current = position.to_move
    opponent = 1 - current

    empty = stones == EMPTY
    current_stone = stones == current
    opponent_stone = stones == opponent
    owner = score.node_owner
    alive = score.alive_stone
    degrees = topology.neighbor_mask.sum(dim=1)
    ring = topology.ring_of.to(torch.float32)
    position_on_ring = topology.pos_of.to(torch.float32)
    arm_distance = torch.minimum(position_on_ring, ring - position_on_ring) / ring
    legal = empty & (not position.terminal)

    node_features = torch.stack(
        (
            empty,
            current_stone,
            opponent_stone,
            owner == current,
            owner == opponent,
            owner == -1,
            alive & current_stone,
            alive & opponent_stone,
            topology.is_peri,
            topology.is_quark,
            ring / float(position.rings),
            arm_distance,
            degrees.to(torch.float32) / float(topology.max_degree),
            topology.ring_of == 1,
            legal,
        ),
        dim=-1,
    ).to(dtype=dtype)

    occupied = topology.n - int(empty.sum())
    current_count = int(current_stone.sum())
    opponent_count = int(opponent_stone.sum())
    current_score = score.players[current]
    opponent_score = score.players[opponent]
    score_scale = 181.0
    star_scale = max(1.0, topology.peri_count / 2.0)
    global_features = torch.tensor(
        (
            position.rings / MAX_RINGS,
            occupied / topology.n,
            current_count / topology.n,
            opponent_count / topology.n,
            position.moves_left / 2.0,
            float(position.opening),
            position.pass_streak / 2.0,
            float(position.terminal),
            current_score.total / score_scale,
            opponent_score.total / score_scale,
            (current_score.total - opponent_score.total) / score_scale,
            current_score.peries / topology.peri_count,
            opponent_score.peries / topology.peri_count,
            current_score.quarks / 5.0,
            opponent_score.quarks / 5.0,
            current_score.stars / star_scale,
            opponent_score.stars / star_scale,
            score.contested_peries / topology.peri_count,
        ),
        dtype=dtype,
    )

    target_device = torch.device(device) if device is not None else torch.device("cpu")
    return EncodedPosition(
        topology=topology,
        node_features=node_features.to(target_device),
        global_features=global_features.to(target_device),
        legal_node_mask=legal.to(target_device),
        legal_pass=torch.tensor(
            not position.terminal, dtype=torch.bool, device=target_device
        ),
        score=score,
    )


def collate_encoded(positions: Sequence[EncodedPosition]) -> EncodedBatch:
    if not positions:
        raise ValueError("cannot collate an empty batch")
    devices = {position.node_features.device for position in positions}
    dtypes = {position.node_features.dtype for position in positions}
    if len(devices) != 1 or len(dtypes) != 1:
        raise ValueError("all encoded positions must share device and dtype")
    device = positions[0].node_features.device
    dtype = positions[0].node_features.dtype
    batch_size = len(positions)
    max_nodes = max(position.topology.n for position in positions)
    max_degree = max(position.topology.max_degree for position in positions)

    node_features = torch.zeros(
        (batch_size, max_nodes, NODE_FEATURE_DIM), dtype=dtype, device=device
    )
    global_features = torch.stack(
        [position.global_features for position in positions], dim=0
    )
    neighbor_index = torch.zeros(
        (batch_size, max_nodes, max_degree), dtype=torch.long, device=device
    )
    neighbor_mask = torch.zeros(
        (batch_size, max_nodes, max_degree), dtype=torch.bool, device=device
    )
    neighbor_edge_type = torch.zeros(
        (batch_size, max_nodes, max_degree), dtype=torch.long, device=device
    )
    node_mask = torch.zeros((batch_size, max_nodes), dtype=torch.bool, device=device)
    legal_action_mask = torch.zeros(
        (batch_size, max_nodes + 1), dtype=torch.bool, device=device
    )

    for batch_index, position in enumerate(positions):
        nodes = position.topology.n
        degree = position.topology.max_degree
        node_features[batch_index, :nodes] = position.node_features
        node_mask[batch_index, :nodes] = True
        neighbor_index[batch_index, :nodes, :degree] = (
            position.topology.neighbor_index.to(device)
        )
        neighbor_mask[batch_index, :nodes, :degree] = (
            position.topology.neighbor_mask.to(device)
        )
        neighbor_edge_type[batch_index, :nodes, :degree] = (
            position.topology.neighbor_edge_type.to(device)
        )
        sample_legal = torch.cat(
            (position.legal_node_mask, position.legal_pass.reshape(1))
        )
        legal_action_mask[batch_index] = relocate_sample_actions(
            sample_legal,
            sample_nodes=nodes,
            batch_max_nodes=max_nodes,
            fill_value=False,
        )

    return EncodedBatch(
        node_features=node_features,
        global_features=global_features,
        neighbor_index=neighbor_index,
        neighbor_mask=neighbor_mask,
        neighbor_edge_type=neighbor_edge_type,
        node_mask=node_mask,
        legal_action_mask=legal_action_mask,
        rings=torch.tensor(
            [position.topology.rings for position in positions],
            dtype=torch.long,
            device=device,
        ),
    )


def encode_batch(
    positions: Sequence[DoubleStarPosition],
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> EncodedBatch:
    return collate_encoded(
        [
            encode_position(position, dtype=dtype, device=device)
            for position in positions
        ]
    )


assert FEATURE_SCHEMA_VERSION == 2
assert FEATURE_SCHEMA_HASH != 0
