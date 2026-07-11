from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from startrain.contracts import RULES_HASH
from startrain.features import DoubleStarPosition, encode_batch
from startrain.native import (
    BITBOARD_WORDS,
    NativeCompatibilityError,
    NativeStateDataProtocol,
    encode_native_state_data,
    positions_from_native,
    validate_native_module,
)
from startrain.topology import get_topology


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


def fake_native_data() -> tuple[FakeStateData, list[DoubleStarPosition]]:
    topology = get_topology(4)
    opening = DoubleStarPosition(
        rings=4,
        stones=torch.full((topology.n,), -1, dtype=torch.int8),
        to_move=0,
        moves_left=1,
        opening=True,
        terminal=False,
    )
    stones = torch.full((topology.n,), -1, dtype=torch.int8)
    stones[0] = 0
    stones[7] = 1
    live = DoubleStarPosition(
        rings=4,
        stones=stones,
        to_move=1,
        moves_left=1,
        opening=False,
        terminal=False,
    )
    positions = [opening, live]
    zero_bits: list[int] = []
    one_bits: list[int] = []
    legal_bits: list[int] = []
    for position in positions:
        zero_bits.extend(pack_mask(position.stones == 0))
        one_bits.extend(pack_mask(position.stones == 1))
        legal_bits.extend(pack_mask(position.stones == -1))
    return (
        FakeStateData(
            rings=4,
            node_count=topology.n,
            batch_size=2,
            zero_bits=zero_bits,
            one_bits=one_bits,
            legal_bits=legal_bits,
            hashes=[11, 12],
            stones_placed=[int((position.stones >= 0).sum()) for position in positions],
            to_move=[position.to_move for position in positions],
            moves_left=[position.moves_left for position in positions],
            opening=[position.opening for position in positions],
            mid_turn=[
                not position.opening and position.moves_left == 1
                for position in positions
            ],
            terminal=[position.terminal for position in positions],
        ),
        positions,
    )


def test_protocol_adapter_matches_direct_feature_encoding() -> None:
    data, expected_positions = fake_native_data()
    assert isinstance(data, NativeStateDataProtocol)
    adapted = positions_from_native(data)
    for expected, actual in zip(expected_positions, adapted, strict=True):
        assert torch.equal(expected.stones, actual.stones)
        assert expected.to_move == actual.to_move
        assert expected.moves_left == actual.moves_left
        assert expected.opening == actual.opening
    direct_batch = encode_batch(expected_positions)
    native_batch = encode_native_state_data(data)
    for direct, native in zip(
        direct_batch.model_args(), native_batch.model_args(), strict=True
    ):
        assert torch.equal(direct, native)


def test_adapter_rejects_bad_legal_buffers_without_native_extension() -> None:
    data, _ = fake_native_data()
    data.legal_bits[0] = 0
    with pytest.raises(NativeCompatibilityError, match="legal placement"):
        positions_from_native(data)


def test_native_module_requires_finalized_rules_hash() -> None:
    validate_native_module(SimpleNamespace(native_rules_hash=lambda: RULES_HASH))
    with pytest.raises(NativeCompatibilityError, match="rules hash"):
        validate_native_module(SimpleNamespace(native_rules_hash=lambda: 1))
    with pytest.raises(NativeCompatibilityError, match="lacks"):
        validate_native_module(SimpleNamespace())
