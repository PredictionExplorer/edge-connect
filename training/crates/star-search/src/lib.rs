//! Evaluator-agnostic MCTS foundations for Double *Star.
//!
//! Search edges are atomic placements or pass actions. Values are stored in
//! each node's current-player perspective, so a backup changes sign exactly
//! when `to_move` changes—not after every atomic stone. Exact semantic keys
//! turn the tree into a DAG and reuse `{a,b}` / `{b,a}` completed-turn states.

mod batch;
mod evaluation;
mod gumbel;
mod tree;

pub use batch::{RootSearchConfig, SearchNonce, SearchResult, SearchRunError, gumbel_search_batch};
pub use evaluation::{BatchEvaluator, Evaluation, EvaluationRequest};
pub use gumbel::{GumbelError, GumbelParameters, GumbelSequentialHalving};
pub use tree::{RootActionStats, SearchError, SearchTree, SimulationStart};
