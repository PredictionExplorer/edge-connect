from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
import torch

from startrain.contracts import FEATURE_SCHEMA_HASH, RULES_HASH
from startrain.features import DoubleStarPosition, encode_batch
from startrain.native import (
    BITBOARD_WORDS,
    NativeCompatibilityError,
    NativeStateDataProtocol,
    NativeVariantStateDataProtocol,
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
class FakeLegacyStateData:
    """A pre-variant export: only the six legacy semantic fields."""

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


@dataclass
class FakeStateData(FakeLegacyStateData):
    current_turn_bits: list[int] = field(default_factory=list)
    previous_turn_bits: list[int] = field(default_factory=list)
    own_previous_turn_bits: list[int] = field(default_factory=list)
    handicap_bits: list[int] = field(default_factory=list)
    mode: list[int] = field(default_factory=list)
    handicap: list[int] = field(default_factory=list)
    pie: list[bool] = field(default_factory=list)
    pie_pending: list[bool] = field(default_factory=list)
    swap_available: list[bool] = field(default_factory=list)
    swapped: list[bool] = field(default_factory=list)
    turn_size: list[int] = field(default_factory=list)
    current_turn_total: list[int] = field(default_factory=list)
    turn_count: list[int] = field(default_factory=list)


def fake_positions() -> list[DoubleStarPosition]:
    topology = get_topology(4)
    opening = DoubleStarPosition(
        rings=4,
        stones=torch.full((topology.n,), -1, dtype=torch.int8),
        to_move=0,
        moves_left=1,
        opening=True,
        terminal=False,
        pie=True,
    )
    stones = torch.full((topology.n,), -1, dtype=torch.int8)
    stones[0] = 0
    stones[7] = 1
    previous = torch.zeros(topology.n, dtype=torch.bool)
    previous[0] = True
    current = torch.zeros(topology.n, dtype=torch.bool)
    current[7] = True
    live = DoubleStarPosition(
        rings=4,
        stones=stones,
        to_move=1,
        moves_left=1,
        opening=False,
        terminal=False,
        current_turn=current,
        previous_turn=previous,
        handicap_stones=previous,
        pda=-2,
    )
    return [opening, live]


def fake_legacy_data(positions: list[DoubleStarPosition]) -> FakeLegacyStateData:
    topology = get_topology(4)
    zero_bits: list[int] = []
    one_bits: list[int] = []
    legal_bits: list[int] = []
    for position in positions:
        zero_bits.extend(pack_mask(position.stones == 0))
        one_bits.extend(pack_mask(position.stones == 1))
        legal_bits.extend(pack_mask(position.stones == -1))
    return FakeLegacyStateData(
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
            not position.opening and position.moves_left == 1 for position in positions
        ],
        terminal=[position.terminal for position in positions],
    )


def fake_native_data() -> tuple[FakeStateData, list[DoubleStarPosition]]:
    positions = fake_positions()
    legacy = fake_legacy_data(positions)
    data = FakeStateData(**legacy.__dict__)
    for position in positions:
        assert position.current_turn is not None
        assert position.previous_turn is not None
        assert position.own_previous_turn is not None
        assert position.handicap_stones is not None
        data.current_turn_bits.extend(pack_mask(position.current_turn))
        data.previous_turn_bits.extend(pack_mask(position.previous_turn))
        data.own_previous_turn_bits.extend(pack_mask(position.own_previous_turn))
        data.handicap_bits.extend(pack_mask(position.handicap_stones))
        data.mode.append(1 if position.mode == "double" else 0)
        data.handicap.append(position.handicap)
        data.pie.append(position.pie)
        data.pie_pending.append(position.pie_pending)
        data.swap_available.append(position.swap_available)
        data.swapped.append(position.swapped)
        data.turn_size.append(position.turn_size)
        data.current_turn_total.append(position.current_turn_total)
        data.turn_count.append(0 if position.opening else 1)
    data.mid_turn = [
        bool(position.current_turn.any()) and position.moves_left > 0
        for position in positions
        if position.current_turn is not None
    ]
    return data, positions


def test_protocol_adapter_matches_direct_feature_encoding() -> None:
    data, expected_positions = fake_native_data()
    assert isinstance(data, NativeStateDataProtocol)
    assert isinstance(data, NativeVariantStateDataProtocol)
    adapted = positions_from_native(data, pda=[0, -2])
    for expected, actual in zip(expected_positions, adapted, strict=True):
        assert torch.equal(expected.stones, actual.stones)
        assert expected.to_move == actual.to_move
        assert expected.moves_left == actual.moves_left
        assert expected.opening == actual.opening
        assert expected.pie == actual.pie
        assert expected.pda == actual.pda
        assert actual.history_known
        assert torch.equal(expected.current_turn, actual.current_turn)
        assert torch.equal(expected.previous_turn, actual.previous_turn)
        assert torch.equal(expected.handicap_stones, actual.handicap_stones)
    direct_batch = encode_batch(expected_positions)
    native_batch = encode_native_state_data(data, pda=[0, -2])
    for direct, native in zip(
        direct_batch.model_args(), native_batch.model_args(), strict=True
    ):
        assert torch.equal(direct, native)


def test_legacy_exports_decode_as_the_standard_game_with_unknown_history() -> None:
    positions = fake_positions()
    data = fake_legacy_data(positions)
    assert isinstance(data, NativeStateDataProtocol)
    assert not isinstance(data, NativeVariantStateDataProtocol)
    adapted = positions_from_native(data)
    for position in adapted:
        assert position.is_standard
        assert not position.history_known
        assert position.previous_turn is not None
        assert not bool(position.previous_turn.any())
    encoded = encode_native_state_data(data)
    assert encoded.global_features[:, 23].tolist() == [0.0, 0.0]


def test_adapter_rejects_bad_legal_buffers_without_native_extension() -> None:
    data, _ = fake_native_data()
    data.legal_bits[0] = 0
    with pytest.raises(NativeCompatibilityError, match="legal placement"):
        positions_from_native(data)


def test_native_module_requires_finalized_rules_hash() -> None:
    class CompatibleStateBatch:
        def complete_clinches(self) -> None:
            pass

    validate_native_module(
        SimpleNamespace(
            native_rules_hash=lambda: RULES_HASH,
            StateBatch=CompatibleStateBatch,
        )
    )
    validate_native_module(
        SimpleNamespace(
            native_rules_hash=lambda: RULES_HASH,
            native_feature_schema_hash=lambda: FEATURE_SCHEMA_HASH,
            StateBatch=CompatibleStateBatch,
        )
    )
    with pytest.raises(NativeCompatibilityError, match="rules hash"):
        validate_native_module(SimpleNamespace(native_rules_hash=lambda: 1))
    with pytest.raises(NativeCompatibilityError, match="feature schema"):
        validate_native_module(
            SimpleNamespace(
                native_rules_hash=lambda: RULES_HASH,
                native_feature_schema_hash=lambda: 0x1234,
                StateBatch=CompatibleStateBatch,
            )
        )
    with pytest.raises(NativeCompatibilityError, match="complete_clinches"):
        validate_native_module(SimpleNamespace(native_rules_hash=lambda: RULES_HASH))
    with pytest.raises(NativeCompatibilityError, match="lacks"):
        validate_native_module(SimpleNamespace())
