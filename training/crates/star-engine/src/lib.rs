//! Authoritative native rules for the *Star family.
//!
//! One rules contract covers four variants of one game: standard Double *Star
//! (one opening stone, then two placements per turn), classic *Star (one
//! placement every turn), handicap openings (player 0 places `k` stones
//! consecutively, `k` in `1..=9`), and the pie rule (player 1 may swap once
//! after the opening turn). Placements are atomic actions; the swap is a
//! separate action that is never a policy output. Play ends exactly when the
//! board is full.

mod bitboard;
mod board;
mod game;
mod scoring;
mod symmetry;

pub use bitboard::{BITBOARD_WORDS, BitBoard, BitIter};
pub use board::{Board, BoardError, SECTOR_CHARS};
pub use game::{
    Action, GameError, GameState, LegalActions, MAX_HANDICAP, MAX_TURN_PLACEMENTS, Mode, Player,
    StateKey, StateParts, Transition, Undo, Variant,
};
pub use scoring::{
    CompletionBounds, CompletionScenario, PlayerScore, ScoreResult, ScoringScratch,
    score_completion_bounds, score_state, terminal_value,
};
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
pub const RULES_VERSION: u32 = 3;
/// Schema of the finalized cross-language rules contract.
pub const RULES_SCHEMA: &str = "edgeconnect.star.rules.v3";
/// Tagged FNV-1a hash of the finalized canonical rules contract.
pub const RULES_HASH: &str = "fnv1a64:a5d932b0ef8354e8";
/// Raw finalized FNV-1a rules hash.
pub const RULES_HASH_VALUE: u64 = 0xa5d9_32b0_ef83_54e8;
/// Schema of the generated conformance vectors.
pub const CONFORMANCE_SCHEMA: &str = "edgeconnect.star.conformance.v3";
/// Schema of the external model feature contract.
pub const FEATURE_SCHEMA: &str = "edgeconnect.star.model-features.external.v3";
/// Schema of the native nodes-only action layout.
pub const ACTION_LAYOUT_SCHEMA: &str = "edgeconnect.star.action-layout.nodes-only.v1";

/// Exact canonical bytes of the rules contract. The web client
/// (`src/lib/star/rules.ts`) and the Python mirror (`startrain/contracts.py`)
/// carry the same bytes so every runtime derives the same fingerprint.
pub const RULES_CANONICAL: &str = concat!(
    "double-star/rules-v3;",
    "rings=even:{4,6,8,10};",
    "node-count=5*r*(r+1)/2;",
    "node-order=x:1..r,s:0..4,y:0..x-1;",
    "node-id=5*x*(x-1)/2+s*x+y;",
    "sector-order=*:0,S:1,T:2,A:3,R:4:clockwise;",
    "sector-arithmetic=mod5;",
    "label=sector+ring-char(10->0)+decimal-y;",
    "peri=x==r;",
    "quark=x==r&&y==0;",
    "edges=node-order:cycle,radial,diagonal,corner-cross;then-ring1-k5-lexicographic;",
    "edge-dedupe=first-undirected-insertion;",
    "csr-neighbor-order=edge-insertion-order;",
    "cycle=(s,x,y)-(y<x-1?(s,x,y+1):(s+1,x,0));",
    "radial=x>=2&&y<=x-2?(s,x,y)-(s,x-1,y);",
    "diagonal=x>=2&&y>=1?(s,x,y)-(s,x-1,y-1);",
    "corner-cross=x>=2&&y==x-1?(s,x,y)-(s+1,x-1,0);",
    "bridge=K5((s,1,0),s=0..4);",
    "modes={classic:turn-size-1,double:opening-1-then-2};",
    "handicap=k-consecutive-opening-placements-by-player0,k-in-1..9,k=1-is-standard;",
    "pie=optional:after-first-turn-player1-may-swap,recolor-opening-stones-to-player1,",
    "player0-moves-next-with-full-turn,swap-unavailable-after-any-placement;",
    "handicap-excludes-pie;",
    "variant-in-semantic-key=mode,handicap,pie;",
    "history-in-semantic-key=currentTurn,previousTurn,ownPreviousTurn,handicapStones;",
    "actions=atomic-place|swap;",
    "action-wire=place(node)->node,swap->node-count;",
    "legal-order=empty-node-id-ascending;",
    "native-action-layout=node-u-at-u;",
    "terminal=full;",
    "full-terminal=decrement-movesLeft,retain-actor-and-turnCount,no-endTurn,",
    "movesLeft-below-turn-size,midTurn=(movesLeft>0),lastMove=final-node,",
    "currentTurnMoves=final-partial-turn;",
    "pair-semantic=AB==BA-excluding-lastMove;",
    "stones=empty:-1,players:0,1;",
    "star=same-color-connected-group-with-at-least-two-directly-occupied-peries;",
    "territory=after-dead-removal,maximal-nonalive-component-owned-iff-adjacent-",
    "alive-color-set-is-exactly-one-player;",
    "score=peries+quark-peri+2*(opponent-stars-own-stars);",
    "tiebreak=quarks;",
    "terminal-value=toMove-perspective:win=1,loss=-1,tie=invalid;",
    "outcome-class=loss:0,win:1;",
    "score-margin=toMove-total-opponent-total;",
    "terminal-legal-actions=empty;",
    "d5-order=r0,r1,r2,r3,r4,f0,f1,f2,f3,f4;",
    "d5-coordinate=t=s*x+y(mod5*x);",
    "d5-rk=t+k*x(mod5*x);",
    "d5-fk=k*x-t(mod5*x);",
    "d5-action=map-place-node,swap-fixed",
);

/// Stable hash of the complete rules contract.
#[must_use]
pub const fn rules_hash() -> u64 {
    RULES_HASH_VALUE
}

/// FNV-1a 64-bit hash of a byte string, matching the web and Python mirrors.
#[must_use]
pub const fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    let mut index = 0;
    while index < bytes.len() {
        hash ^= bytes[index] as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        index += 1;
    }
    hash
}

const _: () = assert!(
    fnv1a64(RULES_CANONICAL.as_bytes()) == RULES_HASH_VALUE,
    "RULES_HASH_VALUE must equal the FNV-1a hash of RULES_CANONICAL"
);
