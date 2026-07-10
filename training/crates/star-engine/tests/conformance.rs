#![allow(missing_docs)]

use std::collections::BTreeSet;
use std::sync::Arc;

use star_engine::{
    Action, BITBOARD_WORDS, BitBoard, Board, D5Maps, GameState, MAX_RINGS, MIN_RINGS, Player,
    PlayerScore, RULES_HASH_VALUE, ScoringScratch, Symmetry, rules_hash,
};

fn edge_count(rings: u8) -> usize {
    let rings = usize::from(rings);
    5 * rings * (rings + 1) / 2 + 5 * (rings * rings - 1) + 5
}

fn position(board: &Board, zero: &[&str], one: &[&str]) -> [BitBoard; 2] {
    let mut stones = [BitBoard::empty(); 2];
    for label in zero {
        stones[0].insert(board.parse_label(label).unwrap());
    }
    for label in one {
        stones[1].insert(board.parse_label(label).unwrap());
    }
    stones
}

#[test]
fn typescript_known_board_counts_and_topology_match() {
    assert_eq!(Board::new(6).unwrap().node_count(), 105);
    assert_eq!(Board::new(8).unwrap().node_count(), 180);
    assert_eq!(Board::new(10).unwrap().node_count(), 275);
    assert_eq!(Board::new(6).unwrap().peri_count(), 30);
    assert_eq!(Board::new(8).unwrap().peri_count(), 40);
    assert_eq!(Board::new(10).unwrap().peri_count(), 50);

    for rings in MIN_RINGS..=MAX_RINGS {
        let board = Board::new(rings).unwrap();
        assert_eq!(
            board.node_count(),
            5 * u16::from(rings) * (u16::from(rings) + 1) / 2
        );
        assert_eq!(board.peri_mask().count(), 5 * u16::from(rings));
        assert_eq!(board.quark_mask().count(), 5);
        assert_eq!(board.edge_count(), edge_count(rings));
        assert_eq!(board.node_mask().words().len(), BITBOARD_WORDS);

        for node in 0..board.node_count() {
            let neighbors: BTreeSet<_> = board.neighbors(node).iter().copied().collect();
            assert_eq!(neighbors.len(), board.neighbors(node).len());
            assert!(!neighbors.contains(&node));
            assert!(neighbors.len() >= 3);
            for neighbor in neighbors {
                assert!(board.neighbors(neighbor).contains(&node));
            }
        }
    }
}

#[test]
fn typescript_known_labels_and_adjacencies_match() {
    let ten = Board::new(10).unwrap();
    assert_eq!(ten.label(ten.index(0, 10, 0).unwrap()), "*00");
    assert_eq!(ten.label(ten.index(4, 10, 0).unwrap()), "R00");
    assert_eq!(ten.label(ten.index(1, 3, 2).unwrap()), "S32");
    for label in ["*00", "S32", "T41", "R98"] {
        assert_eq!(ten.label(ten.parse_label(label).unwrap()), label);
    }

    let board = Board::new(4).unwrap();
    let adjacent = |left: &str, right: &str| {
        let left = board.parse_label(left).unwrap();
        let right = board.parse_label(right).unwrap();
        board.neighbors(left).contains(&right)
    };
    for (left, right) in [
        ("*40", "*41"),
        ("*43", "S40"),
        ("R43", "*40"),
        ("*40", "*30"),
        ("*43", "*32"),
        ("S41", "S30"),
        ("*21", "S10"),
        ("*43", "S30"),
        ("S43", "T30"),
        ("*10", "S10"),
        ("*10", "T10"),
        ("S10", "A10"),
    ] {
        assert!(adjacent(left, right), "{left} must touch {right}");
    }
    for (left, right) in [("*40", "*42"), ("*40", "S40"), ("*10", "*30")] {
        assert!(!adjacent(left, right), "{left} must not touch {right}");
    }
}

#[test]
fn double_star_atomic_protocol_passes_and_undo_match_oracle() {
    let board = Arc::new(Board::new(3).unwrap());
    let mut state = GameState::new(Arc::clone(&board));
    assert_eq!(state.to_move(), Player::Zero);
    assert_eq!(state.moves_left(), 1);

    state.apply(Action::Place(0)).unwrap();
    assert_eq!(state.to_move(), Player::One);
    assert_eq!(state.moves_left(), 2);
    state.apply(Action::Place(1)).unwrap();
    assert_eq!(state.to_move(), Player::One);
    assert_eq!(state.moves_left(), 1);

    let key_before_pass = state.key();
    let (_, undo) = state.apply_reversible(Action::Pass).unwrap();
    assert_eq!(state.to_move(), Player::Zero);
    assert_eq!(state.moves_left(), 2);
    assert_eq!(state.pass_streak(), 1);
    state.undo(undo);
    assert_eq!(state.key(), key_before_pass);

    state.apply(Action::Pass).unwrap();
    state.apply(Action::Pass).unwrap();
    assert!(state.is_terminal());
    assert!(state.legal_actions().is_empty());
    assert!(state.apply(Action::Place(2)).is_err());

    let mut full = GameState::new(board);
    for node in 0..full.board().node_count() {
        full.apply(Action::Place(node)).unwrap();
    }
    assert!(full.is_terminal());
    assert_eq!(full.stones_placed(), full.board().node_count());
    let transformed_full = D5Maps::new(full.board()).state(Symmetry::ALL[6], &full);
    assert!(transformed_full.is_terminal());
    assert_eq!(transformed_full.moves_left(), full.moves_left());
}

#[test]
fn pair_order_has_the_same_semantic_key_and_hash() {
    let board = Arc::new(Board::new(4).unwrap());
    let mut left = GameState::new(Arc::clone(&board));
    left.apply(Action::Place(0)).unwrap();
    let mut right = left.clone();

    left.apply(Action::Place(1)).unwrap();
    left.apply(Action::Place(2)).unwrap();
    right.apply(Action::Place(2)).unwrap();
    right.apply(Action::Place(1)).unwrap();

    assert_eq!(left.key(), right.key());
    assert_eq!(left.hash64(), right.hash64());
    assert_eq!(left.to_move(), Player::Zero);
}

#[test]
fn d5_maps_are_deterministic_bijections_and_graph_automorphisms() {
    for rings in MIN_RINGS..=MAX_RINGS {
        let board = Board::new(rings).unwrap();
        let first = D5Maps::new(&board);
        let second = D5Maps::new(&board);
        for symmetry in Symmetry::ALL {
            assert_eq!(first.map(symmetry), second.map(symmetry));
            let mapped: BTreeSet<_> = first.map(symmetry).iter().copied().collect();
            assert_eq!(mapped.len(), usize::from(board.node_count()));

            for node in 0..board.node_count() {
                let transformed = first.node(symmetry, node);
                assert_eq!(
                    first.node(symmetry.inverse(), transformed),
                    node,
                    "inverse failed for {rings} rings and symmetry {}",
                    symmetry.index()
                );
                for &neighbor in board.neighbors(node) {
                    let mapped_neighbor = first.node(symmetry, neighbor);
                    assert!(board.neighbors(transformed).contains(&mapped_neighbor));
                }
            }
        }
    }
}

#[test]
fn typescript_scoring_fixtures_match_exactly() {
    let board = Board::new(4).unwrap();
    let mut scratch = ScoringScratch::default();

    let fixture_a = position(
        &board,
        &[
            "S10", "S20", "S30", "S40", "A10", "A20", "A30", "A40", "A41",
        ],
        &[
            "*10", "*20", "*30", "*40", "*41", "T10", "T20", "T30", "T40", "R41", "R42", "S42",
        ],
    );
    let score = scratch.score(&board, fixture_a);
    assert_eq!(
        score.players[0],
        PlayerScore {
            peries: 3,
            quarks: 2,
            stars: 1,
            quark_peri: 0,
            award: 2,
            total: 5,
        }
    );
    assert_eq!(
        score.players[1],
        PlayerScore {
            peries: 5,
            quarks: 2,
            stars: 2,
            quark_peri: 0,
            award: -2,
            total: 3,
        }
    );
    assert_eq!(score.contested_peries, 12);
    let dead = board.parse_label("S42").unwrap();
    assert!(!score.alive_stones.contains(dead));
    assert_eq!(score.owner(dead), None);

    let fixture_b = position(
        &board,
        &["*43", "T42", "T43"],
        &["*42", "*32", "S30", "S40"],
    );
    let score = scratch.score(&board, fixture_b);
    assert_eq!(
        score.players[1],
        PlayerScore {
            peries: 3,
            quarks: 1,
            stars: 1,
            quark_peri: 0,
            award: 0,
            total: 3,
        }
    );
    assert_eq!(
        score.owner(board.parse_label("*43").unwrap()),
        Some(Player::One)
    );

    let fixture_c = position(
        &board,
        &["A40", "A41", "A42", "A43", "R40", "R41"],
        &["*40", "*41", "S40", "S41", "T40", "T41"],
    );
    let score = scratch.score(&board, fixture_c);
    assert_eq!(score.players[0].total, 10);
    assert_eq!(score.players[0].stars, 1);
    assert_eq!(score.players[1].total, 3);
    assert_eq!(score.players[1].stars, 3);
    assert_eq!(score.players[1].quark_peri, 1);

    let fixture_d = position(&board, &["*40", "*41"], &["S41", "S42"]);
    let score = scratch.score(&board, fixture_d);
    assert_eq!(score.players[0].total, 2);
    assert_eq!(score.players[1].total, 2);
    assert_eq!(score.leader, Some(Player::Zero));

    let fixture_e = position(
        &board,
        &["*10", "*20", "*30", "*40", "A10", "A20", "A30", "A40"],
        &["S10", "S20", "S30", "S40", "T10", "T20", "T30", "T40"],
    );
    let score = scratch.score(&board, fixture_e);
    assert_eq!(score.players[0].stars, 1);
    assert_eq!(score.players[1].stars, 1);
}

#[test]
fn scoring_is_d5_invariant_and_full_board_identity_holds() {
    let board = Arc::new(Board::new(6).unwrap());
    let maps = D5Maps::new(&board);
    let mut state = GameState::new(Arc::clone(&board));
    for node in [0, 5, 11, 30, 75, 76, 77, 80, 90, 100] {
        state.apply(Action::Place(node)).unwrap();
    }
    let mut scratch = ScoringScratch::default();
    let original = scratch.score_state(&state);
    for symmetry in Symmetry::ALL {
        let transformed = maps.state(symmetry, &state);
        let score = scratch.score_state(&transformed);
        assert_eq!(score.players, original.players);
        assert_eq!(score.contested_peries, original.contested_peries);
    }

    let mut seed = 0x0057_17a5_u64;
    for rings in [3, 4, 5, 6, 8, 10, 12] {
        let board = Board::new(rings).unwrap();
        for _ in 0..40 {
            let mut stones = [BitBoard::empty(); 2];
            for node in 0..board.node_count() {
                seed = seed.wrapping_add(0x9e37_79b9_7f4a_7c15).rotate_left(17)
                    ^ 0xbf58_476d_1ce4_e5b9;
                stones[usize::from((seed & 1) != 0)].insert(node);
            }
            let score = scratch.score(&board, stones);
            let total = score.players[0].total + score.players[1].total;
            assert_eq!(
                total,
                i16::try_from(board.peri_count() - score.contested_peries).unwrap()
                    + score.players[0].quark_peri
                    + score.players[1].quark_peri
            );
            assert!(score.players[0].quark_peri + score.players[1].quark_peri <= 1);
        }
    }

    assert_eq!(rules_hash(), RULES_HASH_VALUE);
    assert_eq!(rules_hash(), 0xcdb3_4fb0_2be8_2843);
}
