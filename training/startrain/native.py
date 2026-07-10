"""Optional adapter for coarse ``star_native.StateData`` buffers."""

from __future__ import annotations

import importlib
import numbers
from collections import Counter
from dataclasses import dataclass
from collections.abc import Sequence
from types import ModuleType
from typing import Any, Protocol, cast, runtime_checkable

import numpy as np
import torch

from .contracts import (
    FEATURE_SCHEMA_HASH,
    FEATURE_SCHEMA_VERSION,
    RULES_HASH,
    RULES_HASH_WIRE,
    RULES_SCHEMA_ID,
)
from .features import (
    GLOBAL_FEATURE_DIM,
    NODE_FEATURE_DIM,
    DoubleStarPosition,
    EncodedBatch,
    encode_batch,
)
from .scoring import PlayerScore, ScoreResult
from .topology import get_topology

BITBOARD_WORDS = 7
_NATIVE_FEATURE_PATH_COUNTS: Counter[str] = Counter()


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
class NativeFeatureDataProtocol(Protocol):
    """Contiguous buffers exported by ``star_native.FeatureData``."""

    batch_size: int
    max_nodes: int
    node_feature_dim: int
    global_feature_dim: int
    score_component_dim: int
    feature_schema_version: int
    feature_schema_hash: int
    rings: Any
    node_features: Any
    global_features: Any
    node_mask: Any
    legal_action_mask: Any
    score_components: Any
    node_owner: Any
    alive_stones: Any


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


@dataclass(frozen=True, slots=True)
class NativeFeatureScores:
    components: torch.Tensor
    node_owner: torch.Tensor
    alive_stones: torch.Tensor


def reset_native_feature_path_stats() -> None:
    """Reset process-local fast/fallback-path instrumentation."""

    _NATIVE_FEATURE_PATH_COUNTS.clear()


def native_feature_path_stats() -> dict[str, int]:
    """Return process-local feature path batch and row counters."""

    return dict(_NATIVE_FEATURE_PATH_COUNTS)


def _record_feature_path(source: str, rows: int) -> None:
    _NATIVE_FEATURE_PATH_COUNTS[f"{source}_batches"] += 1
    _NATIVE_FEATURE_PATH_COUNTS[f"{source}_rows"] += rows


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


def _buffer_tensor(
    name: str,
    buffer: Any,
    *,
    dtype: np.dtype[Any] | type[np.generic],
    shape: tuple[int, ...],
) -> torch.Tensor:
    expected = 1
    for dimension in shape:
        expected *= dimension
    try:
        array = np.frombuffer(buffer, dtype=dtype)
    except (TypeError, ValueError) as exc:
        raise NativeCompatibilityError(
            f"{name} is not a compatible contiguous buffer"
        ) from exc
    if array.size != expected:
        raise NativeCompatibilityError(
            f"{name} must contain {expected} values, got {array.size}"
        )
    if not array.flags.writeable:
        raise NativeCompatibilityError(f"{name} must expose writable storage")
    return torch.from_numpy(array.reshape(shape))


def encode_native_feature_data(
    data: NativeFeatureDataProtocol,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
    source: str = "native_feature",
) -> EncodedBatch:
    """Wrap one Rust feature export and add cached graph topology tensors."""

    batch_size = _integer("batch_size", data.batch_size)
    max_nodes = _integer("max_nodes", data.max_nodes)
    if batch_size <= 0 or max_nodes <= 0:
        raise NativeCompatibilityError("native feature dimensions must be positive")
    if _integer("node_feature_dim", data.node_feature_dim) != NODE_FEATURE_DIM:
        raise NativeCompatibilityError("native node feature dimension is incompatible")
    if _integer("global_feature_dim", data.global_feature_dim) != GLOBAL_FEATURE_DIM:
        raise NativeCompatibilityError(
            "native global feature dimension is incompatible"
        )
    if _integer("score_component_dim", data.score_component_dim) != 14:
        raise NativeCompatibilityError(
            "native score component dimension is incompatible"
        )
    if (
        _integer("feature_schema_version", data.feature_schema_version)
        != FEATURE_SCHEMA_VERSION
        or _integer("feature_schema_hash", data.feature_schema_hash)
        != FEATURE_SCHEMA_HASH
    ):
        raise NativeCompatibilityError("native feature schema is incompatible")

    rings_u8 = _buffer_tensor("rings", data.rings, dtype=np.uint8, shape=(batch_size,))
    ring_values = [int(value) for value in rings_u8]
    topologies = {rings: get_topology(rings) for rings in set(ring_values)}
    if max(topology.n for topology in topologies.values()) != max_nodes:
        raise NativeCompatibilityError("native max_nodes disagrees with ring metadata")

    node_features = _buffer_tensor(
        "node_features",
        data.node_features,
        dtype=np.float32,
        shape=(batch_size, max_nodes, NODE_FEATURE_DIM),
    )
    global_features = _buffer_tensor(
        "global_features",
        data.global_features,
        dtype=np.float32,
        shape=(batch_size, GLOBAL_FEATURE_DIM),
    )
    node_mask = _buffer_tensor(
        "node_mask",
        data.node_mask,
        dtype=np.bool_,
        shape=(batch_size, max_nodes),
    )
    legal_action_mask = _buffer_tensor(
        "legal_action_mask",
        data.legal_action_mask,
        dtype=np.bool_,
        shape=(batch_size, max_nodes + 1),
    )
    rings = rings_u8.to(dtype=torch.long)

    max_degree = max(topology.max_degree for topology in topologies.values())
    neighbor_index = torch.zeros((batch_size, max_nodes, max_degree), dtype=torch.long)
    neighbor_mask = torch.zeros((batch_size, max_nodes, max_degree), dtype=torch.bool)
    neighbor_edge_type = torch.zeros(
        (batch_size, max_nodes, max_degree), dtype=torch.long
    )
    for ring_count, topology in topologies.items():
        rows = rings == ring_count
        nodes = topology.n
        degree = topology.max_degree
        neighbor_index[rows, :nodes, :degree] = topology.neighbor_index
        neighbor_mask[rows, :nodes, :degree] = topology.neighbor_mask
        neighbor_edge_type[rows, :nodes, :degree] = topology.neighbor_edge_type

    encoded = EncodedBatch(
        node_features=node_features,
        global_features=global_features,
        neighbor_index=neighbor_index,
        neighbor_mask=neighbor_mask,
        neighbor_edge_type=neighbor_edge_type,
        node_mask=node_mask,
        legal_action_mask=legal_action_mask,
        rings=rings,
    )
    _record_feature_path(source, batch_size)
    if device is not None or dtype != torch.float32:
        target = torch.device(device) if device is not None else torch.device("cpu")
        encoded = encoded.to(target, feature_dtype=dtype)
    return encoded


def score_tensors_from_native_features(
    data: NativeFeatureDataProtocol,
) -> NativeFeatureScores:
    """Wrap the score buffers paired with a native feature export."""

    batch_size = _integer("batch_size", data.batch_size)
    max_nodes = _integer("max_nodes", data.max_nodes)
    component_dim = _integer("score_component_dim", data.score_component_dim)
    if component_dim != 14:
        raise NativeCompatibilityError(
            "native score component dimension is incompatible"
        )
    return NativeFeatureScores(
        components=_buffer_tensor(
            "score_components",
            data.score_components,
            dtype=np.int32,
            shape=(batch_size, component_dim),
        ),
        node_owner=_buffer_tensor(
            "node_owner",
            data.node_owner,
            dtype=np.int8,
            shape=(batch_size, max_nodes),
        ),
        alive_stones=_buffer_tensor(
            "alive_stones",
            data.alive_stones,
            dtype=np.bool_,
            shape=(batch_size, max_nodes),
        ),
    )


def encode_native_state_data(
    data: NativeStateDataProtocol,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
    verify_legal_buffers: bool = True,
) -> EncodedBatch:
    feature_export = getattr(data, "feature_data", None)
    if callable(feature_export):
        return encode_native_feature_data(
            cast(NativeFeatureDataProtocol, feature_export()),
            dtype=dtype,
            device=device,
            source="native_state",
        )
    _record_feature_path("python_state", _integer("batch_size", data.batch_size))
    return encode_batch(
        positions_from_native(data, verify_legal_buffers=verify_legal_buffers),
        dtype=dtype,
        device=device,
    )


def encode_native_semantic_batch(
    *,
    rings: Sequence[int],
    stones: Sequence[np.ndarray],
    to_move: Sequence[int],
    moves_left: Sequence[int],
    opening: Sequence[bool],
    pass_streak: Sequence[int],
    terminal: Sequence[bool],
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> EncodedBatch | None:
    """Encode heterogeneous semantic keys through Rust when the export exists."""

    rows = len(rings)
    if rows == 0:
        raise NativeCompatibilityError("semantic feature batch cannot be empty")
    if not (
        len(stones)
        == len(to_move)
        == len(moves_left)
        == len(opening)
        == len(pass_streak)
        == len(terminal)
        == rows
    ):
        raise NativeCompatibilityError("semantic feature fields disagree on row count")
    module = load_star_native()
    encoder = getattr(module, "encode_semantic_features", None) if module else None
    if not callable(encoder):
        _record_feature_path("python_semantic", rows)
        return None

    ring_values = _integers("rings", rings, rows)
    to_move_values = _integers("to_move", to_move, rows)
    moves_left_values = _integers("moves_left", moves_left, rows)
    pass_streak_values = _integers("pass_streak", pass_streak, rows)
    opening_values = _booleans("opening", opening, rows)
    terminal_values = _booleans("terminal", terminal, rows)
    topologies = [get_topology(ring_count) for ring_count in ring_values]
    for row in range(rows):
        if to_move_values[row] not in (0, 1):
            raise NativeCompatibilityError(f"row {row} has invalid to_move")
        if moves_left_values[row] not in (0, 1, 2):
            raise NativeCompatibilityError(f"row {row} has invalid moves_left")
        if pass_streak_values[row] not in (0, 1, 2):
            raise NativeCompatibilityError(f"row {row} has invalid pass_streak")
    ring_array = np.asarray(ring_values, dtype=np.uint8)
    metadata = np.empty((rows, 5), dtype=np.uint8)
    metadata[:, 0] = to_move_values
    metadata[:, 1] = moves_left_values
    metadata[:, 2] = opening_values
    metadata[:, 3] = pass_streak_values
    metadata[:, 4] = terminal_values
    stone_arrays: list[np.ndarray] = []
    for row, (topology, values) in enumerate(zip(topologies, stones, strict=True)):
        array = np.asarray(values)
        if not np.issubdtype(array.dtype, np.integer):
            raise NativeCompatibilityError(f"stones row {row} must contain integers")
        if array.shape != (topology.n,):
            raise NativeCompatibilityError(
                f"stones row {row} must have shape ({topology.n},)"
            )
        if not np.isin(array, (-1, 0, 1)).all():
            raise NativeCompatibilityError(
                f"stones row {row} must contain only -1, 0, or 1"
            )
        stone_arrays.append(np.ascontiguousarray(array, dtype=np.int8))
    stone_array = np.concatenate(stone_arrays)
    feature_data = cast(
        NativeFeatureDataProtocol,
        encoder(
            ring_array.tobytes(),
            metadata.tobytes(),
            stone_array.tobytes(),
        ),
    )
    return encode_native_feature_data(
        feature_data,
        dtype=dtype,
        device=device,
        source="native_semantic",
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
