import json
from pathlib import Path

import pytest
import torch

from startrain.actions import extract_sample_actions, relocate_sample_actions
from startrain.contracts import (
    ACTION_LAYOUT_SCHEMA_ID,
    CONFORMANCE_SCHEMA_ID,
    EXTERNAL_FEATURE_SCHEMA_ID,
    RULES_CANONICAL,
    RULES_HASH,
    RULES_HASH_ALGORITHM,
    RULES_HASH_HEX,
    RULES_HASH_WIRE,
    RULES_SCHEMA_ID,
    fnv1a64,
)
from startrain.features import DoubleStarPosition, encode_position
from startrain.scoring import ScoreResult, score_position
from startrain.symmetry import D5Transform
from startrain.topology import get_topology

FIXTURE_PATH = (
    Path(__file__).resolve().parents[2] / "testdata" / "star" / "conformance-v1.json"
)


@pytest.fixture(scope="session")
def conformance() -> dict:
    with FIXTURE_PATH.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def assert_score_matches(expected: dict, actual: ScoreResult) -> None:
    assert [
        {
            "peries": player.peries,
            "quarks": player.quarks,
            "stars": player.stars,
            "quarkPeri": player.quark_peri,
            "award": player.award,
            "total": player.total,
        }
        for player in actual.players
    ] == expected["players"]
    assert actual.node_owner.tolist() == expected["nodeOwner"]
    assert actual.alive_stone.to(torch.uint8).tolist() == expected["aliveStone"]
    assert actual.contested_peries == expected["contestedPeries"]
    assert actual.leader == expected["leader"]


def position_from_fixture(state: dict, rings: int) -> DoubleStarPosition:
    return DoubleStarPosition.from_sequence(
        rings=rings,
        stones=state["stones"],
        to_move=state["toMove"],
        moves_left=state["movesLeft"],
        opening=state["opening"],
        pass_streak=state["passStreak"],
        terminal=state["terminal"] if "terminal" in state else state["over"],
    )


def test_finalized_rules_identifiers_and_canonical_bytes(conformance: dict) -> None:
    assert conformance["schema"] == CONFORMANCE_SCHEMA_ID
    assert conformance["schemas"] == {
        "rules": RULES_SCHEMA_ID,
        "conformance": CONFORMANCE_SCHEMA_ID,
        "modelFeatures": EXTERNAL_FEATURE_SCHEMA_ID,
        "actionLayout": ACTION_LAYOUT_SCHEMA_ID,
    }
    fixture_rules = conformance["rules"]
    assert fixture_rules["contract"]["schema"] == RULES_SCHEMA_ID
    assert fixture_rules["hashAlgorithm"] == RULES_HASH_ALGORITHM
    assert fixture_rules["hash"] == RULES_HASH_WIRE
    assert fixture_rules["canonical"] == RULES_CANONICAL
    assert f"{fnv1a64(RULES_CANONICAL):016x}" == RULES_HASH_HEX
    assert RULES_HASH == int(RULES_HASH_HEX, 16)


def test_fixture_topology_csr_and_all_d5_vectors(conformance: dict) -> None:
    for expected in conformance["boards"]:
        topology = get_topology(expected["rings"])
        assert topology.n == expected["nodeCount"]
        assert topology.peri_count == expected["perimeterCount"]
        assert topology.edge_index.shape[1] // 2 == expected["edgeCount"]
        assert topology.max_degree == expected["maximumDegree"]
        assert topology.sector_of.tolist() == expected["sectorOf"]
        assert topology.ring_of.tolist() == expected["ringOf"]
        assert topology.pos_of.tolist() == expected["positionOf"]
        assert topology.is_peri.to(torch.uint8).tolist() == expected["perimeterMask"]
        assert topology.is_quark.to(torch.uint8).tolist() == expected["quarkMask"]
        assert list(topology.labels) == expected["labels"]
        assert topology.adjacency_offsets.tolist() == expected["adjacencyOffsets"]
        assert topology.adjacency.tolist() == expected["adjacency"]
        assert list(topology.bridge) == expected["bridge"]
        for node in expected["nodes"]:
            start = int(topology.adjacency_offsets[node["id"]])
            end = int(topology.adjacency_offsets[node["id"] + 1])
            assert sorted(topology.adjacency[start:end].tolist()) == node["adjacent"]
        for index, symmetry in enumerate(expected["symmetries"]):
            transform = D5Transform.from_index(index)
            mapping = topology.d5_permutation(transform.rotation, transform.reflected)
            assert transform.index == index
            assert mapping.tolist() == symmetry["map"]
            inverse = torch.empty_like(mapping)
            inverse[mapping] = torch.arange(topology.n)
            assert inverse.tolist() == symmetry["inverseMap"]


def test_fixture_scoring_vectors(conformance: dict) -> None:
    for vector in conformance["scores"]:
        topology = get_topology(vector["rings"])
        actual = score_position(topology, vector["stones"])
        assert_score_matches(vector["expected"], actual)


def test_fixture_terminal_metadata_and_value_perspectives(conformance: dict) -> None:
    for game in conformance["games"]:
        final_state = game["states"][-1]
        position = position_from_fixture(final_state, game["config"]["rings"])
        assert position.terminal
        assert not position.opening
        assert not bool(encode_position(position).legal_node_mask.any())
        assert not bool(encode_position(position).legal_pass)
        if game["terminal"]["reason"] == "double-pass":
            assert position.pass_streak == 2
            assert position.moves_left == 2
        else:
            assert position.pass_streak == 0
            assert position.moves_left in (0, 1)

        score = score_position(get_topology(position.rings), position.stones)
        assert_score_matches(game["terminal"]["score"], score)
        terminal = game["terminal"]
        values = [
            0 if score.leader == -1 else (1 if score.leader == player else -1)
            for player in (0, 1)
        ]
        margins = [
            score.players[0].total - score.players[1].total,
            score.players[1].total - score.players[0].total,
        ]
        assert values == terminal["valuesByPlayer"]
        assert [value + 1 for value in values] == terminal["wdlClassByPlayer"]
        assert margins == terminal["scoreMarginsByPlayer"]
        perspective = terminal["valuePerspective"]
        assert perspective == {
            "kind": "toMove",
            "player": position.to_move,
            "value": values[position.to_move],
            "wdlClass": values[position.to_move] + 1,
            "scoreMargin": margins[position.to_move],
        }


def test_fixture_action_and_pass_layouts(conformance: dict) -> None:
    assert conformance["actionLayouts"]["schema"] == ACTION_LAYOUT_SCHEMA_ID
    for batch in conformance["actionLayouts"]["mixedBatches"]:
        maximum_nodes = batch["maximumNodes"]
        assert batch["batchActionCount"] == maximum_nodes + 1
        for row in batch["rows"]:
            nodes = row["nodeCount"]
            native = torch.arange(nodes + 1)
            padded = relocate_sample_actions(
                native,
                sample_nodes=nodes,
                batch_max_nodes=maximum_nodes,
                fill_value=-999,
            )
            assert row["native"]["passSlot"] == nodes
            assert row["padded"]["passSlot"] == maximum_nodes
            if row["padded"]["paddingSlots"]:
                first, last = row["padded"]["paddingSlots"]
                assert torch.equal(
                    padded[first : last + 1],
                    torch.full((last - first + 1,), -999),
                )
            assert padded[maximum_nodes] == native[nodes]
            assert torch.equal(
                extract_sample_actions(
                    padded,
                    sample_nodes=nodes,
                    batch_max_nodes=maximum_nodes,
                ),
                native,
            )
            for example in row["examples"]:
                if example["action"]["type"] == "pass":
                    assert example["wireCode"] == -1
                    assert example["nativeIndex"] == nodes
                    assert example["paddedIndex"] == maximum_nodes
                else:
                    node = example["action"]["node"]
                    assert example["wireCode"] == node
                    assert example["nativeIndex"] == node
                    assert example["paddedIndex"] == node


def test_fixture_ab_ba_semantic_keys_have_identical_model_features(
    conformance: dict,
) -> None:
    pair = conformance["pairEquivalences"][0]
    rings = pair["config"]["rings"]
    ab = position_from_fixture(pair["ab"]["semanticState"], rings)
    ba = position_from_fixture(pair["ba"]["semanticState"], rings)
    assert pair["ab"]["lastMove"] != pair["ba"]["lastMove"]
    assert torch.equal(ab.stones, ba.stones)
    assert (
        ab.to_move,
        ab.moves_left,
        ab.opening,
        ab.pass_streak,
        ab.terminal,
    ) == (
        ba.to_move,
        ba.moves_left,
        ba.opening,
        ba.pass_streak,
        ba.terminal,
    )
    encoded_ab = encode_position(ab)
    encoded_ba = encode_position(ba)
    assert torch.equal(encoded_ab.node_features, encoded_ba.node_features)
    assert torch.equal(encoded_ab.global_features, encoded_ba.global_features)
    assert torch.equal(encoded_ab.legal_node_mask, encoded_ba.legal_node_mask)
    assert encoded_ab.legal_pass == encoded_ba.legal_pass
