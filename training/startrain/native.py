"""Optional adapter for coarse ``star_native.StateData`` buffers."""

from __future__ import annotations

import importlib
import numbers
from dataclasses import dataclass
from collections.abc import Sequence
from types import ModuleType
from typing import Protocol, runtime_checkable

import torch

from .contracts import RULES_HASH, RULES_HASH_WIRE, RULES_SCHEMA_ID
from .features import DoubleStarPosition, EncodedBatch, encode_batch
from .scoring import PlayerScore, ScoreResult
from .topology import get_topology

BITBOARD_WORDS = 7


class NativeCompatibilityError(ValueError):
    pass


@runtime_checkable
class NativeStateDataProtocol(Protocol):
    """Structural subset exposed by ``star_native.StateData``."""

    rings: int
    node_count: int
    batch_size: int
    zero_bits: Sequence[int]
    one_bits: Sequence[int]
    legal_bits: Sequence[int]
    hashes: Sequence[int]
    stones_placed: Sequence[int]
    to_move: Sequence[int]
    moves_left: Sequence[int]
    opening: Sequence[bool]
    mid_turn: Sequence[bool]
    pass_streak: Sequence[int]
    terminal: Sequence[bool]
    pass_legal: Sequence[bool]


@runtime_checkable
class NativeScoreDataProtocol(Protocol):
    batch_size: int
    node_count: int
    components: Sequence[int]
    node_owner: Sequence[int]
    alive_bits: Sequence[int]
    winner: Sequence[int]
    terminal_value: Sequence[float]
    wdl_class: Sequence[int]
    score_margin: Sequence[int]
    terminal_reason: Sequence[int]


@runtime_checkable
class NativeTrajectoryDataProtocol(Protocol):
    batch_size: int
    last_move: Sequence[int]
    current_turn_offsets: Sequence[int]
    current_turn_moves: Sequence[int]
    turn_count: Sequence[int]


@dataclass(frozen=True, slots=True)
class NativeTrajectoryRow:
    last_move: int
    current_turn_moves: tuple[int, ...]
    turn_count: int


def load_star_native(*, required: bool = False) -> ModuleType | None:
    """Import and fingerprint-check the extension only when requested."""

    try:
        module = importlib.import_module("star_native")
    except ModuleNotFoundError as exc:
        if exc.name != "star_native":
            raise
        if required:
            raise NativeCompatibilityError(
                "star_native is not installed; build the optional PyO3 extension"
            ) from None
        return None
    validate_native_module(module)
    return module


def validate_native_module(module: object) -> None:
    hash_function = getattr(module, "native_rules_hash", None)
    if not callable(hash_function):
        raise NativeCompatibilityError("star_native lacks native_rules_hash()")
    native_hash = hash_function()
    if (
        isinstance(native_hash, bool)
        or not isinstance(native_hash, numbers.Integral)
        or int(native_hash) != RULES_HASH
    ):
        raise NativeCompatibilityError(
            "star_native rules hash does not match finalized TypeScript rules"
        )
    hash_tag = getattr(module, "native_rules_hash_tag", None)
    if callable(hash_tag) and hash_tag() != RULES_HASH_WIRE:
        raise NativeCompatibilityError("star_native rules hash tag is incompatible")
    schema = getattr(module, "native_rules_schema", None)
    if callable(schema) and schema() != RULES_SCHEMA_ID:
        raise NativeCompatibilityError("star_native rules schema is incompatible")


def _integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise NativeCompatibilityError(f"{name} must be an integer")
    return int(value)


def _integers(name: str, values: Sequence[int], expected: int) -> list[int]:
    if len(values) != expected:
        raise NativeCompatibilityError(
            f"{name} must contain {expected} values, got {len(values)}"
        )
    output: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
            raise NativeCompatibilityError(f"{name} must contain integers")
        output.append(int(value))
    return output


def _booleans(name: str, values: Sequence[bool], expected: int) -> list[bool]:
    if len(values) != expected or any(type(value) is not bool for value in values):
        raise NativeCompatibilityError(f"{name} must contain {expected} boolean values")
    return list(values)


def _unpack_row(words: Sequence[int], *, row: int, nodes: int) -> torch.Tensor:
    start = row * BITBOARD_WORDS
    row_words = words[start : start + BITBOARD_WORDS]
    output = torch.zeros(nodes, dtype=torch.bool)
    for word_index, word in enumerate(row_words):
        if word < 0 or word >= 1 << 64:
            raise NativeCompatibilityError("bitboard words must be unsigned u64")
        first_node = 64 * word_index
        valid_bits = min(64, max(0, nodes - first_node))
        valid_mask = (1 << valid_bits) - 1 if valid_bits else 0
        if word & ~valid_mask:
            raise NativeCompatibilityError("bitboard contains off-board bits")
        for bit in range(valid_bits):
            if word & (1 << bit):
                output[first_node + bit] = True
    return output


def positions_from_native(
    data: NativeStateDataProtocol,
    *,
    verify_legal_buffers: bool = True,
) -> list[DoubleStarPosition]:
    """Convert one homogeneous native batch into schema-v2 semantic keys."""

    rings = _integer("rings", data.rings)
    topology = get_topology(rings)
    batch_size = _integer("batch_size", data.batch_size)
    if batch_size <= 0:
        raise NativeCompatibilityError("native batch_size must be positive")
    if _integer("node_count", data.node_count) != topology.n:
        raise NativeCompatibilityError("native node_count does not match rings")
    word_count = batch_size * BITBOARD_WORDS
    zero_words = _integers("zero_bits", data.zero_bits, word_count)
    one_words = _integers("one_bits", data.one_bits, word_count)
    legal_words = _integers("legal_bits", data.legal_bits, word_count)
    _integers("hashes", data.hashes, batch_size)
    stones_placed = _integers("stones_placed", data.stones_placed, batch_size)
    to_move = _integers("to_move", data.to_move, batch_size)
    moves_left = _integers("moves_left", data.moves_left, batch_size)
    pass_streak = _integers("pass_streak", data.pass_streak, batch_size)
    opening = _booleans("opening", data.opening, batch_size)
    mid_turn = _booleans("mid_turn", data.mid_turn, batch_size)
    terminal = _booleans("terminal", data.terminal, batch_size)
    pass_legal = _booleans("pass_legal", data.pass_legal, batch_size)

    positions: list[DoubleStarPosition] = []
    for row in range(batch_size):
        zero = _unpack_row(zero_words, row=row, nodes=topology.n)
        one = _unpack_row(one_words, row=row, nodes=topology.n)
        if bool((zero & one).any()):
            raise NativeCompatibilityError("native player bitboards overlap")
        stones = torch.full((topology.n,), -1, dtype=torch.int8)
        stones[zero] = 0
        stones[one] = 1
        if stones_placed[row] != int((stones >= 0).sum()):
            raise NativeCompatibilityError("native stones_placed is inconsistent")
        if mid_turn[row] != (not opening[row] and moves_left[row] == 1):
            raise NativeCompatibilityError("native mid_turn is inconsistent")
        position = DoubleStarPosition(
            rings=rings,
            stones=stones,
            to_move=to_move[row],
            moves_left=moves_left[row],
            opening=opening[row],
            pass_streak=pass_streak[row],
            terminal=terminal[row],
        )
        if verify_legal_buffers:
            native_legal = _unpack_row(legal_words, row=row, nodes=topology.n)
            expected_legal = (stones == -1) & (not terminal[row])
            if not torch.equal(native_legal, expected_legal):
                raise NativeCompatibilityError(
                    "native legal placement buffer disagrees with semantic state"
                )
            if pass_legal[row] != (not terminal[row]):
                raise NativeCompatibilityError(
                    "native pass_legal disagrees with semantic state"
                )
        positions.append(position)
    return positions


def encode_native_state_data(
    data: NativeStateDataProtocol,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
    verify_legal_buffers: bool = True,
) -> EncodedBatch:
    return encode_batch(
        positions_from_native(data, verify_legal_buffers=verify_legal_buffers),
        dtype=dtype,
        device=device,
    )


def score_results_from_native(
    data: NativeScoreDataProtocol,
) -> list[ScoreResult]:
    batch_size = _integer("batch_size", data.batch_size)
    node_count = _integer("node_count", data.node_count)
    components = _integers("components", data.components, batch_size * 14)
    owners = _integers("node_owner", data.node_owner, batch_size * node_count)
    alive_words = _integers("alive_bits", data.alive_bits, batch_size * BITBOARD_WORDS)
    winners = _integers("winner", data.winner, batch_size)
    results: list[ScoreResult] = []
    for row in range(batch_size):
        base = row * 14
        players = []
        for player in range(2):
            offset = base + 6 * player
            players.append(
                PlayerScore(
                    peries=components[offset],
                    quarks=components[offset + 1],
                    stars=components[offset + 2],
                    quark_peri=components[offset + 3],
                    award=components[offset + 4],
                    total=components[offset + 5],
                )
            )
        owner_start = row * node_count
        owner = torch.tensor(
            owners[owner_start : owner_start + node_count], dtype=torch.int8
        )
        if not bool(((owner == -1) | (owner == 0) | (owner == 1)).all()):
            raise NativeCompatibilityError("native ownership has invalid values")
        alive = _unpack_row(alive_words, row=row, nodes=node_count)
        leader = components[base + 13]
        if leader != winners[row] or leader not in (-1, 0, 1):
            raise NativeCompatibilityError("native winner/components disagree")
        results.append(
            ScoreResult(
                players=(players[0], players[1]),
                node_owner=owner,
                alive_stone=alive,
                contested_peries=components[base + 12],
                leader=leader,
            )
        )
    return results


def trajectory_rows_from_native(
    data: NativeTrajectoryDataProtocol,
) -> list[NativeTrajectoryRow]:
    batch_size = _integer("batch_size", data.batch_size)
    last_move = _integers("last_move", data.last_move, batch_size)
    offsets = _integers(
        "current_turn_offsets", data.current_turn_offsets, batch_size + 1
    )
    moves = _integers("current_turn_moves", data.current_turn_moves, offsets[-1])
    turn_count = _integers("turn_count", data.turn_count, batch_size)
    if offsets[0] != 0 or any(
        left > right for left, right in zip(offsets[:-1], offsets[1:], strict=True)
    ):
        raise NativeCompatibilityError("trajectory offsets are invalid")
    return [
        NativeTrajectoryRow(
            last_move=last_move[row],
            current_turn_moves=tuple(moves[offsets[row] : offsets[row + 1]]),
            turn_count=turn_count[row],
        )
        for row in range(batch_size)
    ]
