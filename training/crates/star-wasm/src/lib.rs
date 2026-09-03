//! Target-gated `wasm-bindgen` interfaces for browser actors.
//!
//! Native workspace tests do not require a WASM toolchain. The actual binding
//! dependency and exported classes are compiled only for `wasm32`.

/// Indicates whether this build contains JavaScript bindings.
pub const WASM_BINDINGS_ENABLED: bool = cfg!(target_arch = "wasm32");

#[cfg(target_arch = "wasm32")]
mod bindings {
    use std::sync::Arc;

    use star_engine::{
        Action, Board, D5Maps, GameState, MAX_HANDICAP, Mode, RULES_HASH, RULES_SCHEMA,
        ScoringScratch, Symmetry, Variant, rules_hash,
    };
    use star_search::{
        Evaluation, EvaluationRequest, GumbelParameters, GumbelSequentialHalving, SearchTree,
        SimulationStart,
    };
    use wasm_bindgen::prelude::*;

    /// Browser-owned *Star state for any rule variant.
    #[wasm_bindgen]
    pub struct WasmState {
        inner: GameState,
    }

    #[wasm_bindgen]
    impl WasmState {
        /// Creates an empty state. `mode` is `"classic"` or `"double"`,
        /// `handicap` is `1..=9`, and `pie` enables the swap after the opening.
        #[wasm_bindgen(constructor)]
        pub fn new(rings: u8, mode: &str, handicap: u8, pie: bool) -> Result<WasmState, JsValue> {
            let board = Arc::new(Board::new(rings).map_err(js_error)?);
            let mode = Mode::parse(mode)
                .ok_or_else(|| JsValue::from_str("mode must be 'classic' or 'double'"))?;
            let variant = Variant::new(mode, handicap, pie).map_err(js_error)?;
            Ok(Self {
                inner: GameState::with_variant(board, variant),
            })
        }

        /// Creates an empty standard Double *Star state.
        pub fn standard(rings: u8) -> Result<WasmState, JsValue> {
            Self::new(rings, "double", 1, false)
        }

        /// Applies one dense node id.
        pub fn apply(&mut self, node: u16) -> Result<(), JsValue> {
            self.inner.apply(Action::Place(node)).map_err(js_error)?;
            Ok(())
        }

        /// Applies the pie swap; legal only right after the opening turn.
        pub fn swap(&mut self) -> Result<(), JsValue> {
            self.inner.apply(Action::Swap).map_err(js_error)?;
            Ok(())
        }

        /// D5-transformed copy.
        pub fn transformed(&self, symmetry: u8) -> Result<WasmState, JsValue> {
            let symmetry = Symmetry::from_index(symmetry)
                .ok_or_else(|| JsValue::from_str("symmetry must be in 0..10"))?;
            let maps = D5Maps::new(self.inner.board());
            Ok(Self {
                inner: maps.state(symmetry, &self.inner),
            })
        }

        /// Current player's numeric index.
        #[wasm_bindgen(getter)]
        pub fn to_move(&self) -> u8 {
            self.inner.to_move() as u8
        }

        /// Placements remaining in this turn.
        #[wasm_bindgen(getter)]
        pub fn moves_left(&self) -> u8 {
            self.inner.moves_left()
        }

        /// Placements of the turn in progress: handicap during the opening.
        #[wasm_bindgen(getter)]
        pub fn current_turn_total(&self) -> u8 {
            self.inner.current_turn_total()
        }

        /// Whether the opening turn is active.
        #[wasm_bindgen(getter)]
        pub fn opening(&self) -> bool {
            self.inner.is_opening()
        }

        /// Terminal marker.
        #[wasm_bindgen(getter)]
        pub fn terminal(&self) -> bool {
            self.inner.is_terminal()
        }

        /// Turn protocol name: `"classic"` or `"double"`.
        #[wasm_bindgen(getter)]
        pub fn mode(&self) -> String {
            self.inner.mode().name().to_owned()
        }

        /// Consecutive opening placements by player 0.
        #[wasm_bindgen(getter)]
        pub fn handicap(&self) -> u8 {
            self.inner.handicap()
        }

        /// Whether the pie rule is in effect.
        #[wasm_bindgen(getter)]
        pub fn pie(&self) -> bool {
            self.inner.pie()
        }

        /// Empty board of a pie game.
        #[wasm_bindgen(getter)]
        pub fn pie_pending(&self) -> bool {
            self.inner.is_pie_pending()
        }

        /// Whether player 1 may swap right now.
        #[wasm_bindgen(getter)]
        pub fn swap_available(&self) -> bool {
            self.inner.swap_available()
        }

        /// Whether the swap was taken earlier in this game.
        #[wasm_bindgen(getter)]
        pub fn swapped(&self) -> bool {
            self.inner.swapped()
        }

        /// Player-zero fixed bitboard words.
        pub fn zero_bits(&self) -> Vec<u64> {
            self.inner.stones()[0].words().to_vec()
        }

        /// Player-one fixed bitboard words.
        pub fn one_bits(&self) -> Vec<u64> {
            self.inner.stones()[1].words().to_vec()
        }

        /// Legal placement bitboard words.
        pub fn legal_bits(&self) -> Vec<u64> {
            self.inner.legal_actions().placements.words().to_vec()
        }

        /// Placements of the unfinished current turn.
        pub fn current_turn_bits(&self) -> Vec<u64> {
            self.inner.current_turn_set().words().to_vec()
        }

        /// The most recently completed turn (the opponent's).
        pub fn previous_turn_bits(&self) -> Vec<u64> {
            self.inner.previous_turn_set().words().to_vec()
        }

        /// The completed turn before that (the mover's).
        pub fn own_previous_turn_bits(&self) -> Vec<u64> {
            self.inner.own_previous_turn_set().words().to_vec()
        }

        /// Stones placed during the opening phase.
        pub fn handicap_bits(&self) -> Vec<u64> {
            self.inner.handicap_stones().words().to_vec()
        }

        /// Legal node ids in ascending order.
        pub fn legal_actions(&self) -> Vec<u16> {
            self.inner.legal_actions().placements.iter().collect()
        }

        /// Stable deterministic state hash.
        pub fn hash64(&self) -> u64 {
            self.inner.hash64()
        }

        /// Fourteen score integers: six per player, contested, leader.
        pub fn score_components(&self) -> Vec<i32> {
            let score = ScoringScratch::default().score_state(&self.inner);
            let mut components = Vec::with_capacity(14);
            for player in score.players {
                components.extend([
                    i32::from(player.peries),
                    i32::from(player.quarks),
                    i32::from(player.stars),
                    i32::from(player.quark_peri),
                    i32::from(player.award),
                    i32::from(player.total),
                ]);
            }
            components.push(i32::from(score.contested_peries));
            components.push(if self.inner.is_terminal() {
                score
                    .leader
                    .expect("a full *Star board must have a decisive winner") as i32
            } else {
                score.leader.map_or(-1, |player| player as i32)
            });
            components
        }

        /// Rules-contract hash expected by replay data and services.
        pub fn rules_hash() -> u64 {
            rules_hash()
        }

        /// Tagged finalized rules hash.
        pub fn rules_hash_tag() -> String {
            RULES_HASH.to_owned()
        }

        /// Finalized rules schema.
        pub fn rules_schema() -> String {
            RULES_SCHEMA.to_owned()
        }

        /// Largest supported handicap.
        pub fn max_handicap() -> u8 {
            MAX_HANDICAP
        }
    }

    fn placement_nodes(actions: impl IntoIterator<Item = Action>) -> Vec<u16> {
        actions
            .into_iter()
            .map(|action| action.node().expect("search expands placements only"))
            .collect()
    }

    /// Ask/tell atomic-action search tree for browser inference.
    #[wasm_bindgen]
    pub struct WasmSearchTree {
        inner: SearchTree,
        parameters: GumbelParameters,
        pending: Option<EvaluationRequest>,
    }

    #[wasm_bindgen]
    impl WasmSearchTree {
        /// Creates a search tree from a state snapshot.
        #[wasm_bindgen(constructor)]
        pub fn new(
            state: &WasmState,
            c_visit: f32,
            c_scale: f32,
        ) -> Result<WasmSearchTree, JsValue> {
            if state.inner.is_terminal() {
                return Err(JsValue::from_str("cannot search a terminal state"));
            }
            let parameters = GumbelParameters { c_visit, c_scale };
            parameters.validate().map_err(js_error)?;
            Ok(Self {
                inner: SearchTree::new(state.inner.clone()),
                parameters,
                pending: None,
            })
        }

        /// Stable root legal node ids required by `initialize_root`.
        pub fn root_actions(&self) -> Result<Vec<u16>, JsValue> {
            Ok(placement_nodes(
                self.inner.root_request().map_err(js_error)?.legal_actions,
            ))
        }

        /// Token that must accompany the root evaluation response.
        pub fn root_token(&self) -> Result<u64, JsValue> {
            Ok(self.inner.root_request().map_err(js_error)?.token)
        }

        /// Whether root values are reported as the opener's optimal-swap payoff.
        pub fn pie_root_transform(&self) -> bool {
            self.inner.uses_pie_root_transform()
        }

        /// Supplies initial root inference.
        pub fn initialize_root(
            &mut self,
            token: u64,
            value: f32,
            policy_logits: Vec<f32>,
        ) -> Result<(), JsValue> {
            self.inner
                .initialize_root(Evaluation {
                    token,
                    value,
                    policy_logits,
                })
                .map_err(js_error)
        }

        /// Starts one simulation, forcing the supplied root action.
        ///
        /// Returns `true` when leaf inference is required and `false` when an
        /// exact terminal value was backed up immediately.
        pub fn start(&mut self, root_action: u16) -> Result<bool, JsValue> {
            if self.pending.is_some() {
                return Err(JsValue::from_str("finish the pending leaf first"));
            }
            let action = Action::Place(root_action);
            let edge = self
                .inner
                .root_edge(action)
                .ok_or_else(|| JsValue::from_str("root action is not legal"))?;
            match self
                .inner
                .start_simulation(Some(edge), self.parameters)
                .map_err(js_error)?
            {
                SimulationStart::Terminal { .. } => Ok(false),
                SimulationStart::NeedsEvaluation(request) => {
                    self.pending = Some(request);
                    Ok(true)
                }
            }
        }

        /// Pending leaf state.
        pub fn pending_state(&self) -> Result<WasmState, JsValue> {
            Ok(WasmState {
                inner: self
                    .pending
                    .as_ref()
                    .ok_or_else(|| JsValue::from_str("no leaf is pending"))?
                    .state
                    .clone(),
            })
        }

        /// Stable pending legal actions required by `finish`.
        pub fn pending_actions(&self) -> Result<Vec<u16>, JsValue> {
            Ok(placement_nodes(
                self.pending
                    .as_ref()
                    .ok_or_else(|| JsValue::from_str("no leaf is pending"))?
                    .legal_actions
                    .iter()
                    .copied(),
            ))
        }

        /// Token that must accompany the pending leaf response.
        pub fn pending_token(&self) -> Result<u64, JsValue> {
            Ok(self
                .pending
                .as_ref()
                .ok_or_else(|| JsValue::from_str("no leaf is pending"))?
                .token)
        }

        /// Supplies one pending leaf evaluation.
        pub fn finish(
            &mut self,
            token: u64,
            value: f32,
            policy_logits: Vec<f32>,
        ) -> Result<(), JsValue> {
            self.inner
                .finish_simulation(Evaluation {
                    token,
                    value,
                    policy_logits,
                })
                .map_err(js_error)?;
            self.pending = None;
            Ok(())
        }

        /// Root node ids in stable order.
        pub fn actions(&self) -> Vec<u16> {
            placement_nodes(
                self.inner
                    .root_stats()
                    .into_iter()
                    .map(|stats| stats.action),
            )
        }

        /// Root edge visit counts in stable order.
        pub fn visits(&self) -> Vec<u32> {
            self.inner
                .root_stats()
                .into_iter()
                .map(|stats| stats.visits)
                .collect()
        }

        /// Completed root Q values in stable order.
        pub fn completed_q(&self) -> Vec<f32> {
            self.inner.root_completed_q()
        }

        /// Visit-weighted root value for the player to move, once any
        /// simulation has completed. The pie swap is recommended for the
        /// responder exactly when this value is negative.
        pub fn root_value(&self) -> Option<f32> {
            self.inner.root_value()
        }

        /// Completed-Q policy target in stable order.
        pub fn policy_target(&self) -> Vec<f32> {
            self.inner
                .completed_q_target(self.parameters)
                .into_iter()
                .map(|(_, probability)| probability)
                .collect()
        }
    }

    /// Browser-side deterministic Gumbel top-k/sequential-halving scheduler.
    #[wasm_bindgen]
    pub struct WasmGumbel {
        inner: GumbelSequentialHalving,
    }

    #[wasm_bindgen]
    impl WasmGumbel {
        /// Creates a root scheduler over the supplied stable logit order.
        #[wasm_bindgen(constructor)]
        pub fn new(
            logits: Vec<f32>,
            simulations: u32,
            max_considered: usize,
            c_visit: f32,
            c_scale: f32,
            seed: u64,
        ) -> Result<WasmGumbel, JsValue> {
            Ok(Self {
                inner: GumbelSequentialHalving::new(
                    &logits,
                    simulations,
                    max_considered,
                    GumbelParameters { c_visit, c_scale },
                    seed,
                )
                .map_err(js_error)?,
            })
        }

        /// Next forced root edge, or `None` when complete.
        pub fn next(
            &mut self,
            completed_q: Vec<f32>,
            visits: Vec<u32>,
        ) -> Result<Option<usize>, JsValue> {
            self.inner
                .next_candidate(&completed_q, &visits)
                .map_err(js_error)
        }

        /// Records one completed root-edge simulation.
        pub fn record(&mut self, candidate: usize) -> Result<(), JsValue> {
            self.inner.record_simulation(candidate).map_err(js_error)
        }

        /// Whether the exact simulation budget has been consumed.
        pub fn done(&self) -> bool {
            self.inner.is_done()
        }

        /// Selected root edge after search.
        pub fn selected(&self, completed_q: Vec<f32>, visits: Vec<u32>) -> Result<usize, JsValue> {
            self.inner.selected(&completed_q, &visits).map_err(js_error)
        }
    }

    fn js_error(error: impl std::fmt::Display) -> JsValue {
        JsValue::from_str(&error.to_string())
    }
}

#[cfg(target_arch = "wasm32")]
pub use bindings::{WasmGumbel, WasmSearchTree, WasmState};
