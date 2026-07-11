//! Authoritative native rules for Double *Star.
//!
//! The crate intentionally supports exactly one protocol: two placements per
//! turn after the opening player's single stone, atomic placement actions,
//! no pie rule, and terminal play exactly when the board is full.

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
pub const MIN_RINGS: u8 = 4;
/// Largest supported board.
pub const MAX_RINGS: u8 = 10;
/// Complete set of supported board sizes.
pub const SUPPORTED_RINGS: [u8; 4] = [4, 6, 8, 10];
/// Maximum playable nodes: `5 * 10 * 11 / 2`.
pub const MAX_NODES: usize = 275;
/// Semantic contract version embedded into generated training data.
pub const RULES_VERSION: u32 = 2;
/// Schema of the finalized cross-language rules contract.
pub const RULES_SCHEMA: &str = "edgeconnect.star.rules.v2";
/// Tagged FNV-1a hash of the finalized canonical rules contract.
pub const RULES_HASH: &str = "fnv1a64:2da3783519381453";
/// Raw finalized FNV-1a rules hash.
pub const RULES_HASH_VALUE: u64 = 0x2da3_7835_1938_1453;
/// Schema of the generated conformance vectors.
pub const CONFORMANCE_SCHEMA: &str = "edgeconnect.star.conformance.v2";
/// Schema of the external model feature contract.
pub const FEATURE_SCHEMA: &str = "edgeconnect.star.model-features.external.v2";
/// Schema of the native nodes-only action layout.
pub const ACTION_LAYOUT_SCHEMA: &str = "edgeconnect.star.action-layout.nodes-only.v1";

/// Stable hash of the complete rules contract.
#[must_use]
pub const fn rules_hash() -> u64 {
    RULES_HASH_VALUE
}
