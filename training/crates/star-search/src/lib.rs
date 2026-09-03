//! Evaluator-agnostic MCTS foundations for the *Star variant family.
//!
//! Search edges are atomic placements. Values are stored in
//! each node's current-player perspective, so a backup changes sign exactly
//! when `to_move` changes—not after every atomic stone. Consecutive handicap
//! placements and one-stone classic turns therefore back up without any
//! special case. Exact semantic keys turn the tree into a DAG and reuse
//! `{a,b}` / `{b,a}` completed-turn states. The pie swap is never a search
//! edge: the empty-board root of a pie game reports each opening's value as
//! `-|q|`, and the game driver takes the swap from the responder's root value.

mod batch;
mod evaluation;
mod gumbel;
mod tree;

pub use batch::{
    RootSearchConfig, SearchNonce, SearchResult, SearchRunError, gumbel_search_batch,
    gumbel_search_batch_with_budgets, resolve_root_budget,
};
pub use evaluation::{BatchEvaluator, Evaluation, EvaluationRequest};
pub use gumbel::{GumbelError, GumbelParameters, GumbelSequentialHalving};
pub use tree::{RootActionStats, SearchError, SearchTree, SimulationStart};
