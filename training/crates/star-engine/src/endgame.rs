//! Bounded exhaustive endgame solving. Budget exhaustion never yields a label.

use crate::{Action, GameState, Player, score_state};

/// A fully searched principal variation and its actual terminal board.
#[derive(Clone, Debug)]
pub struct ExactEndgame {
    /// Terminal board reached by optimal play, with real rule/history state.
    pub terminal: GameState,
    /// Number of searched nodes including terminal leaves.
    pub nodes: u64,
}

/// Solve at most eight empty nodes with a strict node budget. Both players
/// optimize the true outcome, then their score margin and quark tie-break;
/// atomic double/handicap turns and pie swaps use the authoritative rules.
/// An incomplete tree returns `None` and never mutates the caller's state.
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
    fn search(
        state: &GameState,
        nodes: &mut u64,
        limit: u64,
    ) -> Option<((i8, i32, i32), GameState)> {
        if *nodes >= limit {
            return None;
        }
        *nodes += 1;
        if state.is_terminal() {
            let score = score_state(state);
            let winner = match score.leader? {
                Player::Zero => 1,
                Player::One => -1,
            };
            return Some((
                (
                    winner,
                    i32::from(score.players[0].total) - i32::from(score.players[1].total),
                    i32::from(score.players[0].quarks) - i32::from(score.players[1].quarks),
                ),
                state.clone(),
            ));
        }
        let mut actions = state.legal_actions().to_vec();
        if state.swap_available() {
            actions.push(Action::Swap);
        }
        let maximize = state.to_move() == Player::Zero;
        let mut best: Option<((i8, i32, i32), GameState)> = None;
        for action in actions {
            let mut next = state.clone();
            next.apply(action).ok()?;
            let candidate = search(&next, nodes, limit)?;
            if best.as_ref().is_none_or(|current| {
                if maximize {
                    candidate.0 > current.0
                } else {
                    candidate.0 < current.0
                }
            }) {
                best = Some(candidate);
            }
        }
        best
    }
    let mut nodes = 0;
    let (_, terminal) = search(state, &mut nodes, max_nodes)?;
    Some(ExactEndgame { terminal, nodes })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Board, Mode, Variant};
    use std::sync::Arc;

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
