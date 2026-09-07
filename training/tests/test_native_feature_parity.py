"""Exact feature contracts across the production variants and legacy schema."""

from dataclasses import fields

import pytest
import torch

from startrain.features import EncodedBatch, encode_batch
from startrain.features_v3 import encode_legacy_batch
from startrain.native import encode_native_state_data, positions_from_native
from startrain.topology import SUPPORTED_RINGS


@pytest.mark.native
@pytest.mark.parametrize("rings", SUPPORTED_RINGS)
@pytest.mark.parametrize("schema_version", (3, 4))
@pytest.mark.parametrize(
    "mode,handicap,pie",
    (
        ("classic", 1, False),
        ("double", 1, False),
        ("classic", 1, True),
        ("double", 1, True),
        ("classic", 9, False),
        ("double", 9, False),
    ),
)
def test_native_features_match_oracle_across_variants_and_game_phases(
    rings: int, schema_version: int, mode: str, handicap: int, pie: bool
) -> None:
    native = pytest.importorskip("star_native")
    states = native.StateBatch(rings, 6, mode=mode, handicap=handicap, pie=pie)
    nodes = states.node_count
    # Opening, first response, retained history, midgame, last move, terminal.
    for row, count in enumerate((0, 1, 3, nodes // 2, nodes - 1, nodes)):
        actions = list(range(count))
        if pie and row == 2:
            actions.insert(1, nodes)
        states.apply_many([row] * len(actions), actions)
    pda = [-3, -2, -1, 1, 2, 3]
    data = states.data()
    positions = positions_from_native(data, pda=pda)
    encoder = encode_batch if schema_version == 4 else encode_legacy_batch
    expected = encoder(positions)
    actual = encode_native_state_data(data, pda=pda, schema_version=schema_version)
    for field in fields(EncodedBatch):
        torch.testing.assert_close(
            getattr(actual, field.name),
            getattr(expected, field.name),
            rtol=0,
            atol=0,
        )
