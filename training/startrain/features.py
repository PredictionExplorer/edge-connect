"""Schema-v4 features that are a pure function of the semantic key and context.

The semantic key is the exact Python form of ``star_engine::StateKey``: board,
stones, turn metadata, rule variant, swap state, and the retained placement
history. Two context fields travel with every position but are not game state:
``history_known`` (zero for upgraded legacy samples) and ``pda`` (the playout
doubling advantage of the side to move).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np
import torch
from torch import Tensor

from .contracts import (
    FEATURE_SCHEMA_HASH,
    FEATURE_SCHEMA_VERSION,
    MAX_HANDICAP,
    MAX_PLAYOUT_DOUBLING_ADVANTAGE,
    MODE_TURN_SIZE,
    MODES,
    SCORE_MARGIN_MAX,
    SEGMENT_CLASSIC,
    SEGMENT_HANDICAP,
    SEGMENT_PIE,
    SEGMENT_STANDARD,
)
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
    "placed_this_turn",
    "own_previous_turn",
    "opponent_previous_turn",
    "handicap_stone",
)
GLOBAL_FEATURE_NAMES = (
    "rings_fraction",
    "occupancy_fraction",
    "current_stone_fraction",
    "opponent_stone_fraction",
    "moves_left_fraction",
    "opening",
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
    "turn_size_fraction",
    "handicap_fraction",
    "handicap_phase",
    "handicap_remaining_fraction",
    "pie_pending",
    "swap_available",
    "history_known",
    "playout_doubling_advantage_fraction",
)
NODE_FEATURE_DIM = len(NODE_FEATURE_NAMES)
GLOBAL_FEATURE_DIM = len(GLOBAL_FEATURE_NAMES)

# Bits of the per-node ``history_flags`` byte shared with the native encoder
# and the replay schema.
HISTORY_CURRENT_TURN = 1
HISTORY_OWN_PREVIOUS_TURN = 2
HISTORY_OPPONENT_PREVIOUS_TURN = 4
HISTORY_HANDICAP_STONE = 8

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


def _plain_bool(name: str, value: object) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be bool")
    return value


def _node_set(name: str, value: object, nodes: int) -> Tensor:
    if value is None:
        return torch.zeros(nodes, dtype=torch.bool)
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor or None")
    if value.dtype is not torch.bool:
        raise TypeError(f"{name} must use dtype torch.bool")
    if value.ndim != 1 or value.numel() != nodes:
        raise ValueError(f"{name} must have shape ({nodes},)")
    return value.detach().to(device="cpu")


def variant_label(mode: str, handicap: int, pie: bool) -> str:
    """Compact variant name used in provenance, shard rows, and arena results."""

    if pie:
        return f"pie-{mode}"
    if handicap > 1:
        return f"handicap-{handicap}-{mode}"
    return mode


def variant_segment(mode: str, handicap: int, pie: bool) -> str:
    """Mixture segment: standard, classic, handicap, or pie."""

    if pie:
        return SEGMENT_PIE
    if handicap > 1:
        return SEGMENT_HANDICAP
    if mode == "classic":
        return SEGMENT_CLASSIC
    return SEGMENT_STANDARD


@dataclass(frozen=True, slots=True)
class DoubleStarPosition:
    """Exact Python form of ``star_engine::StateKey`` plus evaluation context.

    Network inputs depend on the board, stones, turn metadata, variant, swap
    state, retained placement history, and the two context fields. Terminal
    states are valid data records but have no policy. History sets default to
    empty (unknown); the opening phase derives its history from the stones.
    """

    rings: int
    stones: Tensor
    to_move: int
    moves_left: int
    opening: bool
    terminal: bool
    mode: str = "double"
    handicap: int = 1
    pie: bool = False
    swap_available: bool = False
    swapped: bool = False
    current_turn: Tensor | None = None
    previous_turn: Tensor | None = None
    own_previous_turn: Tensor | None = None
    handicap_stones: Tensor | None = None
    history_known: bool = True
    pda: int = 0

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
        handicap = _plain_int("handicap", self.handicap)
        pda = _plain_int("pda", self.pda)
        if to_move not in (0, 1):
            raise ValueError("to_move must be 0 or 1")
        if type(self.mode) is not str or self.mode not in MODES:
            raise ValueError("mode must be 'classic' or 'double'")
        if not 1 <= handicap <= MAX_HANDICAP:
            raise ValueError(f"handicap must be in 1..{MAX_HANDICAP}")
        for name in ("opening", "terminal", "pie", "swap_available", "swapped"):
            _plain_bool(name, getattr(self, name))
        _plain_bool("history_known", self.history_known)
        if self.pie and handicap != 1:
            raise ValueError("handicap games cannot use the pie rule")
        if abs(pda) > MAX_PLAYOUT_DOUBLING_ADVANTAGE:
            raise ValueError(
                f"pda must be in -{MAX_PLAYOUT_DOUBLING_ADVANTAGE}.."
                f"{MAX_PLAYOUT_DOUBLING_ADVANTAGE}"
            )

        occupied = int((values != EMPTY).sum())
        board_full = occupied == topology.n
        if self.terminal != board_full:
            raise ValueError("terminal must equal board-full")
        turn_size = MODE_TURN_SIZE[self.mode]
        zero_count = int((values == 0).sum())
        one_count = int((values == 1).sum())
        if self.opening:
            if (
                to_move != 0
                or one_count != 0
                or moves_left < 1
                or moves_left > handicap
                or zero_count != handicap - moves_left
                or self.swap_available
                or self.swapped
                or self.terminal
            ):
                raise ValueError("invalid opening metadata")
        elif board_full:
            if moves_left >= turn_size:
                raise ValueError("a full board retains fewer placements than a turn")
        elif moves_left < 1 or moves_left > turn_size:
            raise ValueError("moves_left must be in 1..turn_size outside the opening")
        if (self.swap_available or self.swapped) and not self.pie:
            raise ValueError("swap flags require the pie rule")
        if self.swap_available and (
            self.swapped
            or self.opening
            or to_move != 1
            or moves_left != turn_size
            or zero_count != 1
            or one_count != 0
        ):
            raise ValueError("swap_available requires the post-opening pie position")

        # History sets: ownership, disjointness, and current-turn size. The
        # opening derives its history from the stones, so supplied sets must
        # agree with them.
        own = values == to_move
        opponent = values == (1 - to_move)
        current_turn = _node_set("current_turn", self.current_turn, topology.n)
        previous_turn = _node_set("previous_turn", self.previous_turn, topology.n)
        own_previous = _node_set(
            "own_previous_turn", self.own_previous_turn, topology.n
        )
        handicap_stones = _node_set("handicap_stones", self.handicap_stones, topology.n)
        if self.opening:
            zero_mask = values == 0
            if (
                (bool(current_turn.any()) and not torch.equal(current_turn, zero_mask))
                or (
                    bool(handicap_stones.any())
                    and not torch.equal(handicap_stones, zero_mask)
                )
                or bool(previous_turn.any())
                or bool(own_previous.any())
            ):
                raise ValueError("opening history must match the placed stones")
            current_turn = zero_mask.clone()
            handicap_stones = zero_mask.clone()
        expected_current = (
            handicap - moves_left if self.opening else turn_size - moves_left
        )
        if (
            bool((current_turn & ~own).any())
            or bool((own_previous & ~own).any())
            or bool((previous_turn & ~opponent).any())
            or bool((handicap_stones & (values == EMPTY)).any())
            or bool((current_turn & own_previous).any())
            or (
                bool(current_turn.any()) and int(current_turn.sum()) != expected_current
            )
        ):
            raise ValueError("invalid placement history")
        object.__setattr__(self, "current_turn", current_turn)
        object.__setattr__(self, "previous_turn", previous_turn)
        object.__setattr__(self, "own_previous_turn", own_previous)
        object.__setattr__(self, "handicap_stones", handicap_stones)

    @classmethod
    def from_sequence(
        cls,
        *,
        rings: int,
        stones: Sequence[int] | Tensor | np.ndarray,
        to_move: int,
        moves_left: int,
        opening: bool,
        terminal: bool,
        mode: str = "double",
        handicap: int = 1,
        pie: bool = False,
        swap_available: bool = False,
        swapped: bool = False,
        history_flags: Sequence[int] | Tensor | np.ndarray | None = None,
        history_known: bool = True,
        pda: int = 0,
    ) -> "DoubleStarPosition":
        raw = torch.as_tensor(stones)
        if raw.dtype not in _INTEGER_DTYPES:
            raise TypeError("stones must contain integers before conversion")
        raw_values = raw.detach().to(device="cpu")
        if not bool(
            ((raw_values == EMPTY) | (raw_values == 0) | (raw_values == 1)).all()
        ):
            raise ValueError("stones must contain only -1, 0, or 1 before conversion")
        sets = history_sets_from_flags(history_flags, raw.numel())
        return cls(
            rings=rings,
            stones=raw.to(dtype=torch.int8).clone(),
            to_move=to_move,
            moves_left=moves_left,
            opening=opening,
            terminal=terminal,
            mode=mode,
            handicap=handicap,
            pie=pie,
            swap_available=swap_available,
            swapped=swapped,
            current_turn=sets[0],
            previous_turn=sets[1],
            own_previous_turn=sets[2],
            handicap_stones=sets[3],
            history_known=history_known,
            pda=pda,
        )

    def with_stones(self, stones: Tensor) -> "DoubleStarPosition":
        return replace(self, stones=stones)

    @property
    def turn_size(self) -> int:
        return MODE_TURN_SIZE[self.mode]

    @property
    def current_turn_total(self) -> int:
        """Placements of the turn in progress: handicap during the opening."""

        return self.handicap if self.opening else self.turn_size

    @property
    def pie_pending(self) -> bool:
        return self.pie and self.opening and self.moves_left == self.handicap

    @property
    def variant_label(self) -> str:
        return variant_label(self.mode, self.handicap, self.pie)

    @property
    def segment(self) -> str:
        return variant_segment(self.mode, self.handicap, self.pie)

    @property
    def is_standard(self) -> bool:
        return self.mode == "double" and self.handicap == 1 and not self.pie

    def history_flags(self) -> Tensor:
        """Per-node ``uint8`` bitfield of the retained placement history."""

        flags = torch.zeros(self.stones.numel(), dtype=torch.uint8)
        assert self.current_turn is not None
        assert self.previous_turn is not None
        assert self.own_previous_turn is not None
        assert self.handicap_stones is not None
        flags[self.current_turn] |= HISTORY_CURRENT_TURN
        flags[self.own_previous_turn] |= HISTORY_OWN_PREVIOUS_TURN
        flags[self.previous_turn] |= HISTORY_OPPONENT_PREVIOUS_TURN
        flags[self.handicap_stones] |= HISTORY_HANDICAP_STONE
        return flags


def history_sets_from_flags(
    history_flags: Sequence[int] | Tensor | np.ndarray | None,
    nodes: int,
) -> tuple[Tensor | None, Tensor | None, Tensor | None, Tensor | None]:
    """Split a per-node bitfield into the four retained placement sets."""

    if history_flags is None:
        return (None, None, None, None)
    flags = torch.as_tensor(history_flags).detach().to(device="cpu")
    if flags.dtype not in _INTEGER_DTYPES or flags.ndim != 1 or flags.numel() != nodes:
        raise ValueError(f"history_flags must be an integer tensor of shape ({nodes},)")
    flags = flags.to(torch.int64)
    if bool((flags < 0).any()) or bool((flags > 15).any()):
        raise ValueError("history_flags must use only the four defined bits")
    return (
        (flags & HISTORY_CURRENT_TURN) != 0,
        (flags & HISTORY_OPPONENT_PREVIOUS_TURN) != 0,
        (flags & HISTORY_OWN_PREVIOUS_TURN) != 0,
        (flags & HISTORY_HANDICAP_STONE) != 0,
    )


@dataclass(frozen=True, slots=True)
class EncodedPosition:
    topology: StarTopology
    node_features: Tensor
    global_features: Tensor
    legal_node_mask: Tensor
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
        non_blocking: bool = False,
    ) -> "EncodedBatch":
        dtype = feature_dtype or self.node_features.dtype
        return EncodedBatch(
            node_features=self.node_features.to(
                device=device, dtype=dtype, non_blocking=non_blocking
            ),
            global_features=self.global_features.to(
                device=device, dtype=dtype, non_blocking=non_blocking
            ),
            neighbor_index=self.neighbor_index.to(
                device=device, non_blocking=non_blocking
            ),
            neighbor_mask=self.neighbor_mask.to(
                device=device, non_blocking=non_blocking
            ),
            neighbor_edge_type=self.neighbor_edge_type.to(
                device=device, non_blocking=non_blocking
            ),
            node_mask=self.node_mask.to(device=device, non_blocking=non_blocking),
            legal_action_mask=self.legal_action_mask.to(
                device=device, non_blocking=non_blocking
            ),
            rings=self.rings.to(device=device, non_blocking=non_blocking),
        )

    def pin_memory(self, *, pin_topology: bool = True) -> "EncodedBatch":
        return EncodedBatch(
            node_features=self.node_features.pin_memory(),
            global_features=self.global_features.pin_memory(),
            neighbor_index=(
                self.neighbor_index.pin_memory()
                if pin_topology
                else self.neighbor_index
            ),
            neighbor_mask=(
                self.neighbor_mask.pin_memory() if pin_topology else self.neighbor_mask
            ),
            neighbor_edge_type=(
                self.neighbor_edge_type.pin_memory()
                if pin_topology
                else self.neighbor_edge_type
            ),
            node_mask=self.node_mask.pin_memory() if pin_topology else self.node_mask,
            legal_action_mask=self.legal_action_mask.pin_memory(),
            rings=self.rings.pin_memory() if pin_topology else self.rings,
        )

    def record_stream(self, stream: torch.Stream) -> None:
        for tensor in (
            self.node_features,
            self.global_features,
            self.neighbor_index,
            self.neighbor_mask,
            self.neighbor_edge_type,
            self.node_mask,
            self.legal_action_mask,
            self.rings,
        ):
            tensor.record_stream(stream)

    def model_args(
        self,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        return (
            self.node_features,
            self.global_features,
            self.neighbor_index,
            self.neighbor_mask,
            self.neighbor_edge_type,
            self.node_mask,
            self.legal_action_mask,
            self.rings,
        )


def encode_position(
    position: DoubleStarPosition,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> EncodedPosition:
    """Encode schema v4 from the semantic key and context."""

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
    assert position.current_turn is not None
    assert position.previous_turn is not None
    assert position.own_previous_turn is not None
    assert position.handicap_stones is not None

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
            position.current_turn,
            position.own_previous_turn,
            position.previous_turn,
            position.handicap_stones,
        ),
        dim=-1,
    ).to(dtype=dtype)

    occupied = topology.n - int(empty.sum())
    current_count = int(current_stone.sum())
    opponent_count = int(opponent_stone.sum())
    current_score = score.players[current]
    opponent_score = score.players[opponent]
    score_scale = float(SCORE_MARGIN_MAX)
    star_scale = max(1.0, topology.peri_count / 2.0)
    turn_total = max(1, position.current_turn_total)
    global_features = torch.tensor(
        (
            position.rings / MAX_RINGS,
            occupied / topology.n,
            current_count / topology.n,
            opponent_count / topology.n,
            position.moves_left / turn_total,
            float(position.opening),
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
            position.turn_size / 2.0,
            position.handicap / MAX_HANDICAP,
            float(position.opening and position.handicap >= 2),
            (position.moves_left / MAX_HANDICAP) if position.opening else 0.0,
            float(position.pie_pending),
            float(position.swap_available),
            float(position.history_known),
            position.pda / MAX_PLAYOUT_DOUBLING_ADVANTAGE,
        ),
        dtype=dtype,
    )

    target_device = torch.device(device) if device is not None else torch.device("cpu")
    return EncodedPosition(
        topology=topology,
        node_features=node_features.to(target_device),
        global_features=global_features.to(target_device),
        legal_node_mask=legal.to(target_device),
        score=score,
    )


def collate_encoded(
    positions: Sequence[EncodedPosition],
    *,
    node_feature_dim: int = NODE_FEATURE_DIM,
) -> EncodedBatch:
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
        (batch_size, max_nodes, node_feature_dim), dtype=dtype, device=device
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
        (batch_size, max_nodes), dtype=torch.bool, device=device
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
        legal_action_mask[batch_index, :nodes] = position.legal_node_mask

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


assert FEATURE_SCHEMA_VERSION == 4
assert FEATURE_SCHEMA_HASH != 0
