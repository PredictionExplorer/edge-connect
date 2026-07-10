//! Lightweight native actor throughput smoke benchmark.
//!
//! Run with, for example:
//! `RAYON_NUM_THREADS=32 cargo run -p star-py --example actor_throughput --release`.

use std::convert::Infallible;
use std::sync::Arc;
use std::time::Instant;

use rayon::prelude::*;
use star_engine::{Action, Board, GameState, ScoringScratch};
use star_search::{
    BatchEvaluator, Evaluation, EvaluationRequest, GumbelParameters, RootSearchConfig,
    gumbel_search_batch,
};

const POSITION_COUNT: usize = 200_000;
const SEARCH_ROOTS: usize = 128;
const SEARCH_SIMULATIONS: u32 = 32;

struct UniformEvaluator;

impl BatchEvaluator for UniformEvaluator {
    type Error = Infallible;

    fn evaluate_batch(
        &mut self,
        requests: &[EvaluationRequest],
    ) -> Result<Vec<Evaluation>, Self::Error> {
        Ok(requests
            .iter()
            .map(|request| Evaluation {
                token: request.token,
                value: 0.0,
                policy_logits: vec![0.0; request.legal_actions.len()],
            })
            .collect())
    }
}

fn main() {
    let board = Arc::new(Board::new(6).expect("six-ring board"));
    let states: Vec<_> = (0..256)
        .map(|index| {
            let mut state = GameState::new(Arc::clone(&board));
            state
                .apply(Action::Place(
                    (index % usize::from(board.node_count())) as u16,
                ))
                .expect("opening");
            state
        })
        .collect();

    let score_started = Instant::now();
    let score_sink: i64 = (0..POSITION_COUNT)
        .into_par_iter()
        .map_init(ScoringScratch::default, |scratch, index| {
            i64::from(scratch.score_state(&states[index % states.len()]).players[0].total)
        })
        .sum();
    let score_elapsed = score_started.elapsed().as_secs_f64();

    let search_started = Instant::now();
    let search_sink: i64 = (0..SEARCH_ROOTS)
        .into_par_iter()
        .map(|index| {
            let mut evaluator = UniformEvaluator;
            let result = gumbel_search_batch(
                vec![states[index % states.len()].clone()],
                RootSearchConfig::deterministic(
                    SEARCH_SIMULATIONS,
                    8,
                    GumbelParameters::PAPER,
                    index as u64,
                ),
                &mut evaluator,
            )
            .expect("search");
            i64::from(result[0].selected_action.expect("active root").code())
        })
        .sum();
    let search_elapsed = search_started.elapsed().as_secs_f64();
    let simulations = f64::from(SEARCH_SIMULATIONS) * SEARCH_ROOTS as f64;

    println!(
        "threads={} positions/s={:.0} search-simulations/s={:.0} sinks=({}, {})",
        rayon::current_num_threads(),
        POSITION_COUNT as f64 / score_elapsed,
        simulations / search_elapsed,
        score_sink,
        search_sink
    );
}
