//! Bounded exact endgame solving. Budget exhaustion never yields a label.

use std::collections::HashMap;

use crate::{Action, GameState, Player, StateKey, Variant, score_state};

/// A fully searched principal variation and its actual terminal board.
#[derive(Clone, Debug)]
pub struct ExactEndgame {
    /// Terminal board reached by optimal play, with real rule/history state.
    pub terminal: GameState,
    /// Visited nodes, including terminal leaves and exact-cache lookups.
    pub nodes: u64,
}

type Utility = (i8, i32, i32);

const LOWER: Utility = (-2, i32::MIN, i32::MIN);
const UPPER: Utility = (2, i32::MAX, i32::MAX);

#[derive(Clone)]
struct SolvedLine {
    utility: Utility,
    actions: Vec<Action>,
}

struct EndgameSearch {
    nodes: u64,
    limit: u64,
    // Full rule metadata supplements the network's order-free semantic key.
    // Store continuations, not terminal states: equal keys can have different
    // ordered histories, which must come from the caller's actual path.
    exact: HashMap<(StateKey, Variant, bool), SolvedLine>,
}

impl EndgameSearch {
    fn search(
        &mut self,
        state: &GameState,
        mut alpha: Utility,
        mut beta: Utility,
    ) -> Option<SolvedLine> {
        if self.nodes >= self.limit {
            return None;
        }
        self.nodes += 1;
        let key = (state.key(), state.variant(), state.swapped());
        if let Some(solved) = self.exact.get(&key) {
            return Some(solved.clone());
        }
        if state.is_terminal() {
            let score = score_state(state);
            let winner = match score.leader? {
                Player::Zero => 1,
                Player::One => -1,
            };
            let solved = SolvedLine {
                utility: (
                    winner,
                    i32::from(score.players[0].total) - i32::from(score.players[1].total),
                    i32::from(score.players[0].quarks) - i32::from(score.players[1].quarks),
                ),
                actions: Vec::new(),
            };
            self.exact.insert(key, solved.clone());
            return Some(solved);
        }
        let original_alpha = alpha;
        let original_beta = beta;
        let maximize = state.to_move() == Player::Zero;
        let mut actions = state.legal_actions().to_vec();
        if state.swap_available() {
            actions.push(Action::Swap);
        }
        let mut best: Option<SolvedLine> = None;
        for action in actions {
            let mut next = state.clone();
            next.apply(action).ok()?;
            let candidate = self.search(&next, alpha, beta)?;
            if best.as_ref().is_none_or(|current| {
                if maximize {
                    candidate.utility > current.utility
                } else {
                    candidate.utility < current.utility
                }
            }) {
                let mut continuation = Vec::with_capacity(candidate.actions.len() + 1);
                continuation.push(action);
                continuation.extend(candidate.actions);
                best = Some(SolvedLine {
                    utility: candidate.utility,
                    actions: continuation,
                });
            }
            let utility = best.as_ref()?.utility;
            if maximize {
                alpha = alpha.max(utility);
            } else {
                beta = beta.min(utility);
            }
            if alpha >= beta {
                break;
            }
        }
        let best = best?;
        // Standard fail-soft bounds outside the original open window are not
        // exact. Never memoize those cutoffs (or a budget-exhausted traversal).
        if best.utility > original_alpha && best.utility < original_beta {
            self.exact.insert(key, best.clone());
        }
        Some(best)
    }
}

/// Solve at most eight empty nodes with a strict node budget. Both players
/// optimize the true outcome, then their score margin and quark tie-break;
/// atomic double/handicap turns and pie swaps use the authoritative rules.
/// Exact transpositions and alpha-beta bounds avoid redundant work. An
/// incomplete proof returns `None` and never mutates the caller's state.
#[must_use]
pub fn solve_exact_endgame(
    state: &GameState,
    max_empty: u16,
    max_nodes: u64,
) -> Option<ExactEndgame> {
    if max_empty == 0
        || max_empty > 8
        || max_nodes == 0
        || state.is_terminal()
        || state.legal_actions().len() > usize::from(max_empty)
    {
        return None;
    }
    let mut search = EndgameSearch {
        nodes: 0,
        limit: max_nodes,
        exact: HashMap::new(),
    };
    let solved = search.search(state, LOWER, UPPER)?;
    let mut terminal = state.clone();
    for action in solved.actions {
        terminal.apply(action).ok()?;
    }
    if !terminal.is_terminal() {
        return None;
    }
    Some(ExactEndgame {
        terminal,
        nodes: search.nodes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Board, Mode, Variant};
    use proptest::prelude::*;
    use std::sync::Arc;

    fn exhaustive(state: &GameState, nodes: &mut u64) -> (Utility, GameState) {
        *nodes += 1;
        if state.is_terminal() {
            let score = score_state(state);
            return (
                (
                    if score.leader.unwrap() == Player::Zero {
                        1
                    } else {
                        -1
                    },
                    i32::from(score.players[0].total) - i32::from(score.players[1].total),
                    i32::from(score.players[0].quarks) - i32::from(score.players[1].quarks),
                ),
                state.clone(),
            );
        }
        let mut actions = state.legal_actions().to_vec();
        if state.swap_available() {
            actions.push(Action::Swap);
        }
        let mut best: Option<(Utility, GameState)> = None;
        for action in actions {
            let mut next = state.clone();
            next.apply(action).unwrap();
            let candidate = exhaustive(&next, nodes);
            if best.as_ref().is_none_or(|current| {
                if state.to_move() == Player::Zero {
                    candidate.0 > current.0
                } else {
                    candidate.0 < current.0
                }
            }) {
                best = Some(candidate);
            }
        }
        best.unwrap()
    }

    fn tail(rings: u8, variant: Variant, empty: usize, seed: u64) -> GameState {
        let mut state = GameState::with_variant(Arc::new(Board::new(rings).unwrap()), variant);
        let mut random = seed;
        while state.legal_actions().len() > empty {
            random = random
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let actions = state.legal_actions().to_vec();
            state
                .apply(actions[(random as usize) % actions.len()])
                .unwrap();
            if state.swap_available() && seed & 1 == 1 {
                state.apply(Action::Swap).unwrap();
            }
        }
        state
    }

    fn assert_same_terminal(actual: &GameState, expected: &GameState) {
        assert!(actual.is_terminal() && expected.is_terminal());
        assert_eq!(actual.key(), expected.key());
        assert_eq!(actual.variant(), expected.variant());
        assert_eq!(actual.swapped(), expected.swapped());
        assert_eq!(actual.last_move(), expected.last_move());
        assert_eq!(actual.turn_count(), expected.turn_count());
        assert_eq!(actual.current_turn_moves(), expected.current_turn_moves());
        assert_eq!(actual.previous_turn_moves(), expected.previous_turn_moves());
        assert_eq!(
            actual.own_previous_turn_moves(),
            expected.own_previous_turn_moves()
        );
    }

    #[test]
    fn memoized_pruning_matches_exhaustive_values_and_first_tie_terminal_histories() {
        let mut baseline_nodes = 0;
        let mut optimized_nodes = 0;
        for rings in [4, 6, 8, 10] {
            for (mode, handicap, pie) in [
                (Mode::Classic, 1, false),
                (Mode::Double, 1, false),
                (Mode::Classic, 3, false),
                (Mode::Double, 9, false),
                (Mode::Classic, 1, true),
                (Mode::Double, 1, true),
            ] {
                for seed in [17, 28] {
                    for empty in [3, 4, 5] {
                        let state = tail(
                            rings,
                            Variant::new(mode, handicap, pie).unwrap(),
                            empty,
                            seed,
                        );
                        let before = state.clone();
                        let (_, expected) = exhaustive(&state, &mut baseline_nodes);
                        let solved = solve_exact_endgame(&state, 5, 100_000).unwrap();
                        optimized_nodes += solved.nodes;
                        assert_same_terminal(&solved.terminal, &expected);
                        assert_eq!(state.key(), before.key());
                        assert_eq!(state.last_move(), before.last_move());
                        assert_eq!(state.current_turn_moves(), before.current_turn_moves());
                    }
                }
            }
        }
        assert!(optimized_nodes < baseline_nodes);
        eprintln!(
            "144 exact endgames: exhaustive={baseline_nodes}, optimized={optimized_nodes} visited nodes"
        );
    }

    #[test]
    fn exact_node_limit_is_enforced_and_pruning_reduces_required_budget() {
        let state = tail(10, Variant::STANDARD, 6, 17);
        let mut old_nodes = 0;
        let (_, expected) = exhaustive(&state, &mut old_nodes);
        let solved = solve_exact_endgame(&state, 6, 100_000).unwrap();
        assert_same_terminal(&solved.terminal, &expected);
        assert!(solved.nodes < old_nodes);
        assert!(solve_exact_endgame(&state, 6, solved.nodes - 1).is_none());
        let at_limit = solve_exact_endgame(&state, 6, solved.nodes).unwrap();
        assert_eq!(at_limit.nodes, solved.nodes);
        assert_same_terminal(&at_limit.terminal, &expected);
        assert!(solve_exact_endgame(&state, 5, 100_000).is_none());
        assert!(solve_exact_endgame(&state, 0, 100_000).is_none());
        assert!(solve_exact_endgame(&state, 9, 100_000).is_none());
        assert!(solve_exact_endgame(&state, 6, 0).is_none());
        assert!(solve_exact_endgame(&expected, 6, 100_000).is_none());
        eprintln!(
            "ring-10 six-empty endgame: exhaustive={old_nodes}, optimized={}",
            solved.nodes
        );
    }

    #[test]
    fn cutoff_bounds_and_exhausted_searches_are_not_cached_as_exact() {
        let state = tail(4, Variant::STANDARD, 5, 17);
        let (expected, _) = exhaustive(&state, &mut 0);
        let key = (state.key(), state.variant(), state.swapped());
        for (alpha, beta) in [(LOWER, expected), (expected, UPPER)] {
            let mut search = EndgameSearch {
                nodes: 0,
                limit: 100_000,
                exact: HashMap::new(),
            };
            search.search(&state, alpha, beta).unwrap();
            assert!(!search.exact.contains_key(&key));
            let solved = search.search(&state, LOWER, UPPER).unwrap();
            assert_eq!(solved.utility, expected);
            assert_eq!(search.exact.get(&key).unwrap().utility, expected);
        }
        let mut exhausted = EndgameSearch {
            nodes: 0,
            limit: 1,
            exact: HashMap::new(),
        };
        assert!(exhausted.search(&state, LOWER, UPPER).is_none());
        assert_eq!(exhausted.nodes, 1);
        assert!(!exhausted.exact.contains_key(&key));
    }

    #[test]
    fn semantic_cache_hits_replay_continuations_onto_the_current_ordered_history() {
        let start = tail(4, Variant::STANDARD, 5, 17);
        assert_eq!(start.moves_left(), 2);
        let actions = start.legal_actions().to_vec();
        let mut left = start.clone();
        left.apply(actions[0]).unwrap();
        left.apply(actions[1]).unwrap();
        let mut right = start;
        right.apply(actions[1]).unwrap();
        right.apply(actions[0]).unwrap();
        assert_eq!(left.key(), right.key());
        assert_ne!(left.previous_turn_moves(), right.previous_turn_moves());
        let mut search = EndgameSearch {
            nodes: 0,
            limit: 100_000,
            exact: HashMap::new(),
        };
        search.search(&left, LOWER, UPPER).unwrap();
        let before = search.nodes;
        let reused = search.search(&right, LOWER, UPPER).unwrap();
        assert_eq!(search.nodes, before + 1);
        let (_, expected) = exhaustive(&right, &mut 0);
        let mut terminal = right;
        for action in reused.actions {
            terminal.apply(action).unwrap();
        }
        assert_same_terminal(&terminal, &expected);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]
        #[test]
        fn generated_legal_tails_match_exhaustive_proofs(
            ring_index in 0_usize..4,
            variant_index in 0_usize..6,
            empty in 1_usize..7,
            seed in any::<u64>(),
        ) {
            let (mode, handicap, pie) = [
                (Mode::Classic, 1, false), (Mode::Double, 1, false),
                (Mode::Classic, 9, false), (Mode::Double, 9, false),
                (Mode::Classic, 1, true), (Mode::Double, 1, true),
            ][variant_index];
            let state = tail(
                [4, 6, 8, 10][ring_index], Variant::new(mode, handicap, pie).unwrap(), empty, seed,
            );
            let (_, expected) = exhaustive(&state, &mut 0);
            let solved = solve_exact_endgame(&state, 6, 100_000).unwrap();
            assert_same_terminal(&solved.terminal, &expected);
            prop_assert!(solved.nodes <= 100_000);
        }
    }

    #[test]
    fn every_variant_solves_only_with_complete_evidence_and_preserves_input() {
        let board = Arc::new(Board::new(4).unwrap());
        for (mode, handicap, pie) in [
            (Mode::Classic, 1, false),
            (Mode::Double, 1, false),
            (Mode::Classic, 3, false),
            (Mode::Double, 3, false),
            (Mode::Classic, 1, true),
            (Mode::Double, 1, true),
        ] {
            let mut state = GameState::with_variant(
                Arc::clone(&board),
                Variant::new(mode, handicap, pie).unwrap(),
            );
            while state.legal_actions().len() > 4 {
                let action = state.legal_actions().to_vec()[0];
                state.apply(action).unwrap();
                if state.swap_available() {
                    state.apply(Action::Swap).unwrap();
                }
            }
            let before = state.clone();
            assert!(solve_exact_endgame(&state, 4, 1).is_none());
            assert_eq!(state.key(), before.key());
            let solved = solve_exact_endgame(&state, 4, 1000).unwrap();
            assert!(solved.terminal.is_terminal());
            assert_eq!(solved.terminal.variant(), state.variant());
            assert_eq!(solved.terminal.swapped(), state.swapped());
            assert!(score_state(&solved.terminal).leader.is_some());
            assert!(solved.nodes <= 1000);
        }
    }
}
