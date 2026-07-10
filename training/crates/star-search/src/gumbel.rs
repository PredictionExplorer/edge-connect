use std::error::Error;
use std::fmt;

/// Positive scaling parameters from Gumbel AlphaZero Equation 8.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GumbelParameters {
    /// Visit offset controlling the initial Q influence.
    pub c_visit: f32,
    /// Positive multiplier applied after the visit-dependent scale.
    pub c_scale: f32,
}

impl GumbelParameters {
    /// Parameters used by the paper for Go and chess.
    pub const PAPER: Self = Self {
        c_visit: 50.0,
        c_scale: 1.0,
    };

    /// Validates that both parameters are finite and strictly positive.
    pub fn validate(self) -> Result<(), GumbelError> {
        if self.c_visit.is_finite()
            && self.c_scale.is_finite()
            && self.c_visit > 0.0
            && self.c_scale > 0.0
        {
            Ok(())
        } else {
            Err(GumbelError::InvalidParameters)
        }
    }

    /// Visit-dependent linear multiplier in
    /// `sigma(Q) = (c_visit + max_visits) * c_scale * Q`.
    #[must_use]
    pub fn sigma_scale(self, max_visits: u32) -> f32 {
        (self.c_visit + max_visits as f32) * self.c_scale
    }
}

impl Default for GumbelParameters {
    fn default() -> Self {
        Self::PAPER
    }
}

/// Invalid Gumbel root-search configuration or scheduler use.
#[derive(Clone, Debug, PartialEq)]
pub enum GumbelError {
    /// Root has no legal actions.
    NoActions,
    /// Simulation budget must be positive.
    ZeroBudget,
    /// Candidate cap must be positive.
    ZeroCandidates,
    /// Gumbel constants must be finite and strictly positive.
    InvalidParameters,
    /// Root logit or completed Q is non-finite.
    NonFiniteInput,
    /// Completed-Q vector has the wrong size.
    QLength {
        /// Root action count.
        expected: usize,
        /// Submitted Q count.
        actual: usize,
    },
    /// Visit vector has the wrong size.
    VisitLength {
        /// Root action count.
        expected: usize,
        /// Submitted visit count.
        actual: usize,
    },
    /// A recorded candidate was not the one requested.
    UnexpectedCandidate {
        /// Most recently requested candidate.
        expected: usize,
        /// Candidate supplied by the caller.
        actual: usize,
    },
}

impl fmt::Display for GumbelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoActions => f.write_str("Gumbel root has no legal actions"),
            Self::ZeroBudget => f.write_str("Gumbel simulation budget must be positive"),
            Self::ZeroCandidates => f.write_str("Gumbel candidate cap must be positive"),
            Self::InvalidParameters => {
                f.write_str("c_visit and c_scale must be finite and strictly positive")
            }
            Self::NonFiniteInput => f.write_str("Gumbel input contains a non-finite number"),
            Self::QLength { expected, actual } => {
                write!(f, "expected {expected} completed Q values, got {actual}")
            }
            Self::VisitLength { expected, actual } => {
                write!(f, "expected {expected} visit counts, got {actual}")
            }
            Self::UnexpectedCandidate { expected, actual } => {
                write!(f, "expected candidate {expected}, got {actual}")
            }
        }
    }
}

impl Error for GumbelError {}

/// Seeded Gumbel top-k and exact-budget sequential-halving scheduler.
#[derive(Clone, Debug)]
pub struct GumbelSequentialHalving {
    logits: Vec<f32>,
    gumbels: Vec<f32>,
    tie_breakers: Vec<u64>,
    initial_candidates: Vec<usize>,
    active: Vec<usize>,
    phase_schedule: Vec<usize>,
    phase_cursor: usize,
    remaining: u32,
    initial_budget: u32,
    halving_phases: usize,
    parameters: GumbelParameters,
    requested: Option<usize>,
}

impl GumbelSequentialHalving {
    /// Samples root candidates and initializes an exact-budget schedule.
    ///
    /// The candidate count is `min(actions, max_considered, budget)`. If a
    /// rounded phase exhausts the budget, final selection is restricted to
    /// the most-visited candidates.
    pub fn new(
        logits: &[f32],
        budget: u32,
        max_considered: usize,
        parameters: GumbelParameters,
        seed: u64,
    ) -> Result<Self, GumbelError> {
        if logits.is_empty() {
            return Err(GumbelError::NoActions);
        }
        if budget == 0 {
            return Err(GumbelError::ZeroBudget);
        }
        if max_considered == 0 {
            return Err(GumbelError::ZeroCandidates);
        }
        parameters.validate()?;
        if logits.iter().any(|logit| !logit.is_finite()) {
            return Err(GumbelError::NonFiniteInput);
        }

        let mut rng = SplitMix64::new(seed);
        let gumbels: Vec<f32> = logits
            .iter()
            .map(|_| {
                let uniform = rng.open_unit_interval();
                -(-uniform.ln()).ln()
            })
            .collect();
        let tie_breakers: Vec<u64> = logits.iter().map(|_| rng.next()).collect();
        let mut ranked: Vec<usize> = (0..logits.len()).collect();
        ranked.sort_by(|left, right| {
            let left_score = logits[*left] + gumbels[*left];
            let right_score = logits[*right] + gumbels[*right];
            right_score
                .total_cmp(&left_score)
                .then_with(|| tie_breakers[*right].cmp(&tie_breakers[*left]))
        });

        let candidate_count = logits.len().min(max_considered).min(budget as usize);
        ranked.truncate(candidate_count);
        let halving_phases = ceil_log2(candidate_count);

        let mut scheduler = Self {
            logits: logits.to_vec(),
            gumbels,
            tie_breakers,
            initial_candidates: ranked.clone(),
            active: ranked,
            phase_schedule: Vec::new(),
            phase_cursor: 0,
            remaining: budget,
            initial_budget: budget,
            halving_phases,
            parameters,
            requested: None,
        };
        scheduler.configure_phase();
        Ok(scheduler)
    }

    /// Initial Gumbel top-k candidate edge indexes in perturbed-score order.
    #[must_use]
    pub fn candidates(&self) -> &[usize] {
        &self.initial_candidates
    }

    /// Number of simulations already recorded.
    #[must_use]
    pub const fn simulations(&self) -> u32 {
        self.initial_budget - self.remaining
    }

    /// Whether the exact configured budget has been consumed.
    #[must_use]
    pub const fn is_done(&self) -> bool {
        self.remaining == 0
    }

    /// Returns the next forced root edge.
    pub fn next_candidate(
        &mut self,
        completed_q: &[f32],
        visits: &[u32],
    ) -> Result<Option<usize>, GumbelError> {
        self.validate_statistics(completed_q, visits)?;
        if self.remaining == 0 {
            return Ok(None);
        }
        if let Some(candidate) = self.requested {
            return Ok(Some(candidate));
        }

        if self.phase_cursor == self.phase_schedule.len() {
            self.finish_phase(completed_q, visits);
            self.configure_phase();
        }
        let candidate = self.phase_schedule[self.phase_cursor];
        self.requested = Some(candidate);
        Ok(Some(candidate))
    }

    /// Records completion of the candidate most recently requested.
    pub fn record_simulation(&mut self, candidate: usize) -> Result<(), GumbelError> {
        let expected = self.requested.ok_or(GumbelError::UnexpectedCandidate {
            expected: usize::MAX,
            actual: candidate,
        })?;
        if candidate != expected {
            return Err(GumbelError::UnexpectedCandidate {
                expected,
                actual: candidate,
            });
        }
        self.phase_cursor += 1;
        self.remaining -= 1;
        self.requested = None;
        Ok(())
    }

    /// Selects from the most-visited original candidates as prescribed by the
    /// paper's finite-budget rounding rule.
    pub fn selected(&self, completed_q: &[f32], visits: &[u32]) -> Result<usize, GumbelError> {
        self.validate_statistics(completed_q, visits)?;
        let max_visits = self
            .initial_candidates
            .iter()
            .map(|candidate| visits[*candidate])
            .max()
            .expect("the candidate set is non-empty");
        Ok(self
            .initial_candidates
            .iter()
            .copied()
            .filter(|candidate| visits[*candidate] == max_visits)
            .max_by(|left, right| {
                self.score(*left, completed_q, visits)
                    .total_cmp(&self.score(*right, completed_q, visits))
                    .then_with(|| self.tie_breakers[*left].cmp(&self.tie_breakers[*right]))
            })
            .expect("at least one candidate has maximal visits"))
    }

    fn configure_phase(&mut self) {
        debug_assert!(self.remaining > 0);
        let active_count = self.active.len();
        let phase_budget = if active_count == 1 {
            self.remaining
        } else {
            let per_action =
                (self.initial_budget / (self.halving_phases * active_count) as u32).max(1);
            self.remaining.min(per_action * active_count as u32)
        };
        let base = phase_budget / active_count as u32;
        let remainder = phase_budget % active_count as u32;
        let max_target = base + u32::from(remainder > 0);

        self.phase_schedule.clear();
        self.phase_schedule.reserve(phase_budget as usize);
        for visit_round in 0..max_target {
            for (rank, &candidate) in self.active.iter().enumerate() {
                let target = base + u32::from((rank as u32) < remainder);
                if visit_round < target {
                    self.phase_schedule.push(candidate);
                }
            }
        }
        self.phase_cursor = 0;
        debug_assert_eq!(self.phase_schedule.len(), phase_budget as usize);
    }

    fn finish_phase(&mut self, completed_q: &[f32], visits: &[u32]) {
        if self.active.len() <= 1 {
            return;
        }
        let logits = &self.logits;
        let gumbels = &self.gumbels;
        let tie_breakers = &self.tie_breakers;
        let scale = self.parameters.sigma_scale(max_visit_count(visits));
        self.active.sort_by(|left, right| {
            let left_score = gumbels[*left] + logits[*left] + scale * completed_q[*left];
            let right_score = gumbels[*right] + logits[*right] + scale * completed_q[*right];
            right_score
                .total_cmp(&left_score)
                .then_with(|| tie_breakers[*right].cmp(&tie_breakers[*left]))
        });
        self.active.truncate(self.active.len().div_ceil(2));
    }

    fn score(&self, candidate: usize, completed_q: &[f32], visits: &[u32]) -> f32 {
        self.gumbels[candidate]
            + self.logits[candidate]
            + self.parameters.sigma_scale(max_visit_count(visits)) * completed_q[candidate]
    }

    fn validate_statistics(&self, completed_q: &[f32], visits: &[u32]) -> Result<(), GumbelError> {
        if completed_q.len() != self.logits.len() {
            return Err(GumbelError::QLength {
                expected: self.logits.len(),
                actual: completed_q.len(),
            });
        }
        if visits.len() != self.logits.len() {
            return Err(GumbelError::VisitLength {
                expected: self.logits.len(),
                actual: visits.len(),
            });
        }
        if completed_q.iter().any(|value| !value.is_finite()) {
            return Err(GumbelError::NonFiniteInput);
        }
        Ok(())
    }
}

fn max_visit_count(visits: &[u32]) -> u32 {
    visits.iter().copied().max().unwrap_or(0)
}

fn ceil_log2(count: usize) -> usize {
    if count <= 1 {
        1
    } else {
        (usize::BITS - (count - 1).leading_zeros()) as usize
    }
}

#[derive(Clone, Copy, Debug)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn open_unit_interval(&mut self) -> f32 {
        let mantissa = (self.next() >> 40) as u32;
        (mantissa as f32 + 0.5) / 16_777_216.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigma_uses_visit_dependent_linear_scale() {
        let parameters = GumbelParameters {
            c_visit: 3.0,
            c_scale: 0.5,
        };
        assert_eq!(parameters.sigma_scale(5), 4.0);
        assert!(
            GumbelParameters {
                c_visit: 0.0,
                c_scale: 1.0
            }
            .validate()
            .is_err()
        );
    }

    fn run(scheduler: &mut GumbelSequentialHalving, action_count: usize) -> Vec<u32> {
        let q = vec![0.0; action_count];
        let mut visits = vec![0_u32; action_count];
        while !scheduler.is_done() {
            let candidate = scheduler.next_candidate(&q, &visits).unwrap().unwrap();
            visits[candidate] += 1;
            scheduler.record_simulation(candidate).unwrap();
        }
        visits
    }

    #[test]
    fn seeded_top_k_is_deterministic_and_budget_exact() {
        let logits = vec![0.0; 20];
        let mut left =
            GumbelSequentialHalving::new(&logits, 17, 8, GumbelParameters::PAPER, 0xdecafbad)
                .unwrap();
        let mut right = left.clone();
        assert_eq!(left.candidates(), right.candidates());
        let left_visits = run(&mut left, logits.len());
        let right_visits = run(&mut right, logits.len());
        assert_eq!(left_visits, right_visits);
        assert_eq!(left.simulations(), 17);
        assert_eq!(
            left.selected(&vec![0.0; logits.len()], &left_visits),
            right.selected(&vec![0.0; logits.len()], &right_visits)
        );
    }

    #[test]
    fn non_divisible_budget_uses_every_simulation_and_most_visited_finalist() {
        let logits = vec![0.0; 12];
        for budget in [5, 11, 17, 31] {
            let mut scheduler = GumbelSequentialHalving::new(
                &logits,
                budget,
                12,
                GumbelParameters::PAPER,
                u64::from(budget),
            )
            .unwrap();
            let visits = run(&mut scheduler, logits.len());
            assert_eq!(visits.iter().sum::<u32>(), budget);
            let selected = scheduler
                .selected(&vec![0.0; logits.len()], &visits)
                .unwrap();
            let candidate_max = scheduler
                .candidates()
                .iter()
                .map(|candidate| visits[*candidate])
                .max()
                .unwrap();
            assert_eq!(visits[selected], candidate_max);
        }
    }

    #[test]
    fn budget_limited_search_visits_every_sampled_candidate_once() {
        let logits = vec![0.0; 20];
        let mut scheduler =
            GumbelSequentialHalving::new(&logits, 4, 16, GumbelParameters::PAPER, 11).unwrap();
        assert_eq!(scheduler.candidates().len(), 4);
        let visits = run(&mut scheduler, logits.len());
        assert!(
            scheduler
                .candidates()
                .iter()
                .all(|candidate| visits[*candidate] == 1)
        );
    }

    #[test]
    fn remainder_follows_gumbel_rank_not_action_index() {
        let logits = [0.0; 4];
        let (mut scheduler, top) = (0_u64..)
            .map(|seed| {
                let scheduler =
                    GumbelSequentialHalving::new(&logits, 5, 2, GumbelParameters::PAPER, seed)
                        .unwrap();
                let top = scheduler.candidates()[0];
                (scheduler, top)
            })
            .find(|(_, top)| *top != 0)
            .unwrap();
        let visits = run(&mut scheduler, logits.len());
        assert_eq!(visits[top], 3);
        assert_eq!(visits.iter().sum::<u32>(), 5);
    }

    #[test]
    fn completed_q_changes_halving_rank() {
        let logits = [0.0, 0.0];
        let mut scheduler = GumbelSequentialHalving::new(
            &logits,
            4,
            2,
            GumbelParameters {
                c_visit: 100.0,
                c_scale: 1.0,
            },
            7,
        )
        .unwrap();
        let q = [-1.0, 1.0];
        let mut visits = [0_u32; 2];
        while !scheduler.is_done() {
            let candidate = scheduler.next_candidate(&q, &visits).unwrap().unwrap();
            visits[candidate] += 1;
            scheduler.record_simulation(candidate).unwrap();
        }
        assert_eq!(scheduler.selected(&q, &visits).unwrap(), 1);
    }
}
