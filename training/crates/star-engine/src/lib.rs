//! Authoritative native rules for Double *Star.
//!
//! The crate intentionally supports exactly one protocol: two placements per
//! turn after the opening player's single stone, atomic place/pass actions,
//! no pie rule, and terminal play on a full board or two consecutive passes.

mod bitboard;
mod board;
mod game;
mod scoring;
mod symmetry;

pub use bitboard::{BITBOARD_WORDS, BitBoard, BitIter};
pub use board::{Board, BoardError, SECTOR_CHARS};
pub use game::{Action, GameError, GameState, LegalActions, Player, StateKey, Transition, Undo};
pub use scoring::{PlayerScore, ScoreResult, ScoringScratch, score_state, terminal_value};
pub use symmetry::{D5_ORDER, D5Maps, Symmetry};

/// Dense node id.
pub type NodeId = u16;

/// Smallest supported board.
pub const MIN_RINGS: u8 = 3;
/// Largest supported board.
pub const MAX_RINGS: u8 = 12;
/// Maximum playable nodes: `5 * 12 * 13 / 2`.
pub const MAX_NODES: usize = 390;
/// Semantic contract version embedded into generated training data.
pub const RULES_VERSION: u32 = 1;
/// Schema of the finalized cross-language rules contract.
pub const RULES_SCHEMA: &str = "edgeconnect.star.rules.v1";
/// Tagged FNV-1a hash of the finalized canonical rules contract.
pub const RULES_HASH: &str = "fnv1a64:cdb34fb02be82843";
/// Raw finalized FNV-1a rules hash.
pub const RULES_HASH_VALUE: u64 = 0xcdb3_4fb0_2be8_2843;
/// Schema of the generated conformance vectors.
pub const CONFORMANCE_SCHEMA: &str = "edgeconnect.star.conformance.v1";
/// Schema of the native nodes-then-pass action layout.
pub const ACTION_LAYOUT_SCHEMA: &str = "edgeconnect.star.action-layout.nodes-then-pass.v1";

/// Stable hash of the complete rules contract.
#[must_use]
pub const fn rules_hash() -> u64 {
    RULES_HASH_VALUE
}
