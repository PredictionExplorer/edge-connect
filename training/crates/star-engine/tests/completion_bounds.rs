#![allow(missing_docs)]

use std::sync::Arc;

use proptest::prelude::*;
use serde::Deserialize;
use star_engine::{
    Action, BitBoard, Board, GameState, Player, SUPPORTED_RINGS, ScoringScratch,
    score_completion_bounds,
};

const FIXTURE_JSON: &str = include_str!("../../../../testdata/star/completion-bounds-v1.json");

#[derive(Deserialize)]
struct CompletionFixture {
    schema: String,
    cases: Vec<CompletionCase>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct CompletionCase {
    id: String,
    rings: u8,
    actions: Vec<u16>,
    empty_nodes: u16,
    guaranteed_winner: Option<u8>,
    proof_fill_player: Option<u8>,
    scenarios: Vec<ExpectedScenario>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ExpectedScenario {
    fill_player: u8,
    winner: u8,
    totals: [i16; 2],
    quarks: [i16; 2],
}

fn player(value: u8) -> Player {
    match value {
        0 => Player::Zero,
        1 => Player::One,
        _ => panic!("fixture player must be zero or one"),
    }
}

fn stones_from_cells(cells: &[u8]) -> [BitBoard; 2] {
    let mut stones = [BitBoard::empty(); 2];
    for (node, cell) in cells.iter().copied().enumerate() {
        if cell == 1 || cell == 2 {
            stones[usize::from(cell - 1)].insert(node as u16);
        }
    }
    stones
}

#[test]
fn shared_typescript_vectors_match_native_bounds() {
    let fixture: CompletionFixture = serde_json::from_str(FIXTURE_JSON).unwrap();
    assert_eq!(fixture.schema, "edgeconnect.star.completion-bounds.v1");

    for case in fixture.cases {
        let board = Arc::new(Board::new(case.rings).unwrap());
        let mut state = GameState::new(Arc::clone(&board));
        for action in case.actions {
            state.apply(Action::Place(action)).unwrap();
        }
        let bounds = score_completion_bounds(&board, state.stones());

        assert_eq!(bounds.empty_nodes, case.empty_nodes, "{}", case.id);
        assert_eq!(
            bounds.guaranteed_winner,
            case.guaranteed_winner.map(player),
            "{}",
            case.id
        );
        assert_eq!(case.scenarios.len(), 2, "{}", case.id);
        for expected in case.scenarios {
            let scenario = &bounds.scenarios[usize::from(expected.fill_player)];
            assert_eq!(
                scenario.fill_player,
                player(expected.fill_player),
                "{}",
                case.id
            );
            assert_eq!(
                scenario.score.leader,
                Some(player(expected.winner)),
                "{}",
                case.id
            );
            assert_eq!(
                scenario.score.players.map(|score| score.total),
                expected.totals,
                "{}",
                case.id
            );
            assert_eq!(
                scenario.score.players.map(|score| score.quarks),
                expected.quarks,
                "{}",
                case.id
            );
        }
        assert_eq!(
            bounds
                .loser_filled_scenario()
                .map(|scenario| scenario.fill_player),
            case.proof_fill_player.map(player),
            "{}",
            case.id
        );
    }
}

#[test]
fn empty_board_is_unsettled() {
    for rings in SUPPORTED_RINGS {
        let board = Board::new(rings).unwrap();
        let bounds = score_completion_bounds(&board, [BitBoard::empty(); 2]);

        assert_eq!(bounds.empty_nodes, board.node_count());
        assert_eq!(bounds.scenarios[0].score.leader, Some(Player::Zero));
        assert_eq!(bounds.scenarios[1].score.leader, Some(Player::One));
        assert_eq!(bounds.guaranteed_winner, None);
        assert!(bounds.loser_filled_scenario().is_none());
    }
}

#[test]
fn dominant_positions_clinch_for_both_colors() {
    let board = Board::new(4).unwrap();
    for winner in [Player::Zero, Player::One] {
        let mut stones = [BitBoard::empty(); 2];
        for node in 0..board.node_count() - 1 {
            stones[winner.index()].insert(node);
        }

        let bounds = score_completion_bounds(&board, stones);
        let proof = bounds.loser_filled_scenario().unwrap();

        assert_eq!(bounds.guaranteed_winner, Some(winner));
        assert_eq!(bounds.empty_nodes, 1);
        assert_eq!(proof.fill_player, winner.opponent());
        assert_eq!(proof.score.leader, Some(winner));
        assert_eq!(proof.stones[0].union(proof.stones[1]), board.node_mask());
        assert!(proof.stones[winner.opponent().index()].contains(board.node_count() - 1));
    }
}

proptest! {
    #[test]
    fn guaranteed_winner_wins_every_sampled_completion(
        rings in prop::sample::select(SUPPORTED_RINGS.to_vec()),
        cells in prop::collection::vec(0_u8..3, 275),
        completion in prop::collection::vec(any::<bool>(), 275),
    ) {
        let board = Arc::new(Board::new(rings).unwrap());
        let node_count = usize::from(board.node_count());
        let source = stones_from_cells(&cells[..node_count]);
        let bounds = score_completion_bounds(&board, source);
        let Some(winner) = bounds.guaranteed_winner else {
            return Ok(());
        };

        let mut completed = source;
        for node in 0..board.node_count() {
            if !source[0].contains(node) && !source[1].contains(node) {
                completed[usize::from(completion[usize::from(node)])].insert(node);
            }
        }
        let score = ScoringScratch::default().score(&board, completed);
        prop_assert_eq!(score.leader, Some(winner));
    }

    #[test]
    fn swapping_colors_swaps_the_completion_bound(
        cells in prop::collection::vec(0_u8..3, 50),
    ) {
        let board = Board::new(4).unwrap();
        let stones = stones_from_cells(&cells);
        let original = score_completion_bounds(&board, stones);
        let swapped = score_completion_bounds(&board, [stones[1], stones[0]]);

        prop_assert_eq!(original.empty_nodes, swapped.empty_nodes);
        prop_assert_eq!(
            swapped.guaranteed_winner,
            original.guaranteed_winner.map(Player::opponent),
        );
    }
}
