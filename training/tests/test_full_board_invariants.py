from __future__ import annotations

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from startrain.contracts import SCORE_MARGIN_MAX, SCORE_MARGIN_MIN
from startrain.features import DoubleStarPosition, encode_position
from startrain.scoring import score_position
from startrain.topology import MAX_NODES, SUPPORTED_RINGS, get_topology


@st.composite
def full_boards(draw: st.DrawFn) -> tuple[int, torch.Tensor]:
    rings = draw(st.sampled_from(SUPPORTED_RINGS))
    nodes = get_topology(rings).n
    values = draw(
        st.lists(
            st.integers(min_value=0, max_value=1),
            min_size=nodes,
            max_size=nodes,
        )
    )
    return rings, torch.tensor(values, dtype=torch.int8)


@st.composite
def legal_fill_orders(draw: st.DrawFn) -> tuple[int, list[int]]:
    rings = draw(st.sampled_from(SUPPORTED_RINGS))
    nodes = get_topology(rings).n
    priorities = draw(
        st.lists(
            st.integers(min_value=0, max_value=2**31 - 1),
            min_size=nodes,
            max_size=nodes,
        )
    )
    order = sorted(range(nodes), key=lambda node: (priorities[node], node))
    return rings, order


def assert_decisive_full_score(rings: int, stones: torch.Tensor) -> None:
    topology = get_topology(rings)
    score = score_position(topology, stones)
    zero, one = score.players
    margin = zero.total - one.total

    assert score.contested_peries == 0
    assert zero.peries + one.peries == topology.peri_count
    assert zero.quarks + one.quarks == 5
    assert zero.quark_peri + one.quark_peri == 1
    assert zero.award + one.award == 0
    assert zero.total + one.total == topology.peri_count + 1
    assert margin != 0
    assert abs(margin) % 2 == 1
    assert SCORE_MARGIN_MIN <= margin <= SCORE_MARGIN_MAX
    assert score.leader == (0 if margin > 0 else 1)


@pytest.mark.parametrize("rings", SUPPORTED_RINGS)
@pytest.mark.parametrize("pattern", ("alternating", "ring", "sector"))
def test_deterministic_full_board_patterns_are_terminal_and_bounded(
    rings: int, pattern: str
) -> None:
    topology = get_topology(rings)
    if pattern == "alternating":
        stones = torch.arange(topology.n, dtype=torch.int8) % 2
    elif pattern == "ring":
        stones = topology.ring_of.to(torch.int8) % 2
    else:
        stones = topology.sector_of.to(torch.int8) % 2
    position = DoubleStarPosition(
        rings=rings,
        stones=stones,
        to_move=1,
        moves_left=0,
        opening=False,
        terminal=True,
    )
    encoded = encode_position(position)
    score = score_position(topology, stones)
    margin = score.players[1].total - score.players[0].total

    assert_decisive_full_score(rings, stones)
    assert topology.n <= MAX_NODES
    assert not bool(encoded.legal_node_mask.any())
    assert SCORE_MARGIN_MIN <= margin <= SCORE_MARGIN_MAX
    assert encoded.global_features[9].item() == pytest.approx(margin / SCORE_MARGIN_MAX)


@settings(max_examples=128, deadline=None, derandomize=True, print_blob=True)
@given(board=full_boards())
def test_every_full_board_has_a_decisive_binary_outcome(
    board: tuple[int, torch.Tensor],
) -> None:
    rings, stones = board
    assert_decisive_full_score(rings, stones)
    for to_move in (0, 1):
        position = DoubleStarPosition(
            rings=rings,
            stones=stones,
            to_move=to_move,
            moves_left=1,
            opening=False,
            terminal=True,
        )
        assert not bool(encode_position(position).legal_node_mask.any())


@settings(max_examples=128, deadline=None, derandomize=True, print_blob=True)
@given(game=legal_fill_orders())
def test_every_legal_fill_order_ends_with_a_decisive_winner(
    game: tuple[int, list[int]],
) -> None:
    rings, order = game
    topology = get_topology(rings)
    stones = torch.full((topology.n,), -1, dtype=torch.int8)
    player = 0
    moves_left = 1
    for node in order:
        stones[node] = player
        moves_left -= 1
        if moves_left == 0:
            player = 1 - player
            moves_left = 2

    assert bool((stones >= 0).all())
    assert_decisive_full_score(rings, stones)


@settings(max_examples=100, deadline=None, derandomize=True, print_blob=True)
@given(board=full_boards(), raw_empty=st.integers(min_value=0, max_value=MAX_NODES - 1))
def test_one_empty_position_has_one_legal_fill_and_then_terminates(
    board: tuple[int, torch.Tensor], raw_empty: int
) -> None:
    rings, full_stones = board
    topology = get_topology(rings)
    empty = raw_empty % topology.n
    live_stones = full_stones.clone()
    live_stones[empty] = -1
    live = DoubleStarPosition(
        rings=rings,
        stones=live_stones,
        to_move=0,
        moves_left=1,
        opening=False,
        terminal=False,
    )
    encoded = encode_position(live)
    legal = torch.nonzero(encoded.legal_node_mask, as_tuple=False).flatten()
    assert legal.tolist() == [empty]

    filled_stones = live_stones.clone()
    filled_stones[empty] = live.to_move
    filled = DoubleStarPosition(
        rings=rings,
        stones=filled_stones,
        to_move=live.to_move,
        moves_left=0,
        opening=False,
        terminal=True,
    )
    assert not bool(encode_position(filled).legal_node_mask.any())


@pytest.mark.parametrize("rings", (3, 5, 7, 9, 11, 12, -2, 0))
def test_odd_and_out_of_range_rings_are_rejected(rings: int) -> None:
    with pytest.raises(ValueError, match="one of"):
        get_topology(rings)


@pytest.mark.parametrize("rings", (True, 4.0, "4"))
def test_non_integer_ring_values_are_rejected(rings: object) -> None:
    with pytest.raises(TypeError, match="integer"):
        get_topology(rings)  # type: ignore[arg-type]
