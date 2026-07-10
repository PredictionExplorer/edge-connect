use star_engine::{Action, GameState};

/// One evaluator input. Values and logits are always in `state.to_move()` perspective.
#[derive(Clone, Debug)]
pub struct EvaluationRequest {
    /// Opaque token used to match asynchronous responses.
    pub token: u64,
    /// Exact leaf state.
    pub state: GameState,
    /// Stable action order expected by `policy_logits`.
    pub legal_actions: Vec<Action>,
}

/// Evaluator output for one request.
#[derive(Clone, Debug)]
pub struct Evaluation {
    /// Opaque token copied from the matching request.
    pub token: u64,
    /// Zero-sum value in the leaf state's current-player perspective.
    pub value: f32,
    /// One unnormalized logit per legal action.
    pub policy_logits: Vec<f32>,
}

/// Synchronous batched inference abstraction used by native actors.
pub trait BatchEvaluator {
    /// Evaluator-specific error.
    type Error;

    /// Evaluates one packed batch.
    ///
    /// Responses may be reordered, but each must carry exactly one request
    /// token. The search driver rejects duplicates, omissions, and unknowns.
    fn evaluate_batch(
        &mut self,
        requests: &[EvaluationRequest],
    ) -> Result<Vec<Evaluation>, Self::Error>;
}
