//! Deterministic CPU search benchmark with an optional complete comparison trace.
//!
//! Run release builds from each revision with identical arguments. `--trace PATH`
//! writes every requested semantic state/legal ordering and every result float's
//! bits; compare those files before comparing timings. The cheap synthetic
//! evaluator deliberately isolates search overhead and does not measure neural
//! inference, playing strength, or end-to-end self-play throughput.

use std::convert::Infallible;
use std::fs::File;
use std::hint::black_box;
use std::io::{BufWriter, Write};
use std::sync::Arc;
use std::time::Instant;

use star_engine::{Action, Board, GameState, Mode, Variant};
use star_search::{
    BatchEvaluator, Evaluation, EvaluationRequest, GumbelParameters, RootSearchConfig,
    SearchResult, gumbel_search_batch,
};

struct Evaluator<'a> {
    trace: Option<&'a mut BufWriter<File>>,
    concentrated: bool,
    calls: usize,
    rows: usize,
}

fn mix(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

impl BatchEvaluator for Evaluator<'_> {
    type Error = Infallible;

    fn evaluate_batch(
        &mut self,
        requests: &[EvaluationRequest],
    ) -> Result<Vec<Evaluation>, Self::Error> {
        self.calls += 1;
        self.rows += requests.len();
        if let Some(trace) = self.trace.as_mut() {
            writeln!(trace, "batch {}", requests.len()).unwrap();
        }
        Ok(requests
            .iter()
            .map(|request| {
                if let Some(trace) = self.trace.as_mut() {
                    writeln!(
                        trace,
                        "leaf {:?} {:?}",
                        request.state.key(),
                        request.legal_actions
                    )
                    .unwrap();
                }
                let hash = request.state.hash64();
                let scale = if self.concentrated { 0.05 } else { 0.4 };
                let value = ((mix(hash) & 0xffff) as f32 / 32768.0 - 1.0) * scale;
                let preferred = hash as usize % request.legal_actions.len();
                let policy_logits = request
                    .legal_actions
                    .iter()
                    .enumerate()
                    .map(|(index, action)| {
                        if self.concentrated {
                            if index == preferred { 8.0 } else { -2.0 }
                        } else {
                            (mix(hash ^ u64::from(action.node().unwrap())) & 1023) as f32 / 256.0
                                - 2.0
                        }
                    })
                    .collect();
                Evaluation {
                    token: request.token,
                    value,
                    policy_logits,
                }
            })
            .collect())
    }
}

fn roots(ring: u8, stage: &str) -> Vec<GameState> {
    let board = Arc::new(Board::new(ring).unwrap());
    [
        (Mode::Classic, 1, false),
        (Mode::Double, 1, false),
        (Mode::Classic, 4, false),
        (Mode::Double, 4, false),
        (Mode::Classic, 1, true),
        (Mode::Double, 1, true),
    ]
    .into_iter()
    .enumerate()
    .map(|(index, (mode, handicap, pie))| {
        let mut state = GameState::with_variant(
            Arc::clone(&board),
            Variant::new(mode, handicap, pie).unwrap(),
        );
        let placements = match stage {
            "opening" => usize::from(pie),
            "middle" => usize::from(board.node_count()) / 2,
            "tail" => usize::from(board.node_count()) - 4,
            _ => unreachable!(),
        };
        let mut seed = 0x3141_5926_5358_9793 ^ index as u64;
        for _ in 0..placements {
            seed = mix(seed.wrapping_add(0x9e37_79b9_7f4a_7c15));
            let actions = state.legal_actions().to_vec();
            state.apply(actions[seed as usize % actions.len()]).unwrap();
            if state.swap_available() && stage != "opening" && index % 2 == 0 {
                state.apply(Action::Swap).unwrap();
            }
        }
        state
    })
    .collect()
}

fn write_results(trace: &mut BufWriter<File>, results: &[SearchResult]) {
    for result in results {
        writeln!(
            trace,
            "result {:?} {:?} {:?} {:?}",
            result.selected_action,
            result.terminal_value.map(f32::to_bits),
            result.root_value.map(f32::to_bits),
            result.selected_action_value.map(f32::to_bits),
        )
        .unwrap();
        for stats in &result.root_stats {
            writeln!(
                trace,
                "edge {:?} {} {} {} {}",
                stats.action,
                stats.visits,
                stats.prior.to_bits(),
                stats.logit.to_bits(),
                stats.q.to_bits()
            )
            .unwrap();
        }
        for (action, probability) in &result.policy_target {
            writeln!(trace, "target {:?} {}", action, probability.to_bits()).unwrap();
        }
    }
}

fn main() {
    let mut repeats = 5_usize;
    let mut trace = None;
    let mut args = std::env::args().skip(1);
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--repeats" => repeats = args.next().expect("repeat count").parse().unwrap(),
            "--trace" => {
                trace = Some(BufWriter::new(
                    File::create(args.next().expect("trace path")).unwrap(),
                ))
            }
            _ => panic!("unknown argument {argument}"),
        }
    }
    assert!(repeats > 0);
    if trace.is_some() {
        repeats = 1;
    }
    // Production-scaled fast/full budgets, online and arena budgets, and
    // deliberately small/irregular schedules to exercise boundary behavior.
    let cases = [
        (4, 21, 21),
        (4, 256, 32),
        (6, 32, 32),
        (6, 384, 32),
        (8, 43, 43),
        (8, 512, 43),
        (10, 53, 53),
        (10, 640, 53),
        (10, 64, 16),
        (10, 512, 64),
        (10, 256, 64),
        (10, 1024, 64),
        (4, 1, 1),
        (4, 5, 3),
        (6, 17, 7),
    ];
    for (ring, simulations, candidates) in cases {
        for stage in ["opening", "middle", "tail"] {
            for concentrated in [false, true] {
                let roots = roots(ring, stage);
                let pattern = if concentrated {
                    "concentrated"
                } else {
                    "balanced"
                };
                if let Some(trace) = trace.as_mut() {
                    writeln!(
                        trace,
                        "case {ring} {simulations} {candidates} {stage} {pattern}"
                    )
                    .unwrap();
                }
                // A complete untimed warmup for each shape/pattern.
                if trace.is_none() {
                    let mut evaluator = Evaluator {
                        trace: None,
                        concentrated,
                        calls: 0,
                        rows: 0,
                    };
                    black_box(
                        gumbel_search_batch(
                            roots.clone(),
                            RootSearchConfig::deterministic(
                                simulations,
                                candidates,
                                GumbelParameters::PAPER,
                                1729,
                            ),
                            &mut evaluator,
                        )
                        .unwrap(),
                    );
                }
                for repeat in 0..repeats {
                    let mut evaluator = Evaluator {
                        trace: trace.as_mut(),
                        concentrated,
                        calls: 0,
                        rows: 0,
                    };
                    let started = Instant::now();
                    let results = gumbel_search_batch(
                        roots.clone(),
                        RootSearchConfig::deterministic(
                            simulations,
                            candidates,
                            GumbelParameters::PAPER,
                            1729,
                        ),
                        &mut evaluator,
                    )
                    .unwrap();
                    let seconds = started.elapsed().as_secs_f64();
                    let (calls, rows) = (evaluator.calls, evaluator.rows);
                    if let Some(trace) = trace.as_mut() {
                        write_results(trace, &results);
                    }
                    black_box(&results);
                    println!(
                        "{{\"ring\":{ring},\"simulations\":{simulations},\"candidates\":{candidates},\"stage\":\"{stage}\",\"pattern\":\"{pattern}\",\"repeat\":{repeat},\"roots\":{},\"seconds\":{seconds},\"evaluator_calls\":{calls},\"evaluator_rows\":{rows}}}",
                        roots.len()
                    );
                }
            }
        }
    }
    if let Some(trace) = trace.as_mut() {
        trace.flush().unwrap();
    }
}
