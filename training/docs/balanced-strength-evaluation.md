# Balanced variant strength

When `arena.balanced_cells` is enabled, promotion measures 24 equally weighted
cells: classic and double, each with standard, pie, and handicap rules, on rings
4, 6, 8, and 10. The model architecture and its weights are unchanged by enabling
this objective. Legacy arena behavior remains the default.

Each cell receives role-reversed game pairs. A pair is one observation with a
score of 0, 0.5, or 1; its two games can be arbitrarily correlated. Handicap
severity follows the frozen `handicap_severity_cycle`, normally 2, 4, 6, 9, in
absolute pair-index order. Both handicap modes receive every severity equally.
SHA-256 seed streams separate cells and pair indices while preserving the same
opening within a seat-reversed pair. Every search root receives an explicit seed
derived from that pair, seat, and its own move count; native search does not mix
batch membership, batch order, or a batch nonce into this stream. Native tests
compare complete, reordered, and singleton batches for all six categories.
Standard-only lineage and raw/EMA diagnostics can opt into the same streams with
`ArenaRunner(stable_pair_seeds=True)`; the default retains legacy seeds. The result
records `search.seed_stream_policy`, so transfer certification can require the
explicit independent-stream protocol without claiming 24-cell coverage.
Continuations retain their absolute indices
and skip existing cell pairs. A pair's durable key includes board, mode/rule cell,
and index, so cells cannot overwrite each other's evidence.

For real graph adapters, balanced evaluation runs at most twelve variant groups
per board concurrently, each with at most two native seat workers. Their requests
share one bounded inference owner that batches compatible model/board requests
across groups (512 rows, 24 pending requests, 2 ms maximum batching wait).
All producers drain before inference shutdown; failures stop sibling groups and
completed pairs remain role-reversed. Result metrics expose shared inference
requests, neural batches, queue wait, and failures. Custom/fake evaluators and
`parallel_variant_groups=1` retain the serialized path.

The statistical boundary is a complete severity cycle across all cells: normally
96 pairs, or 192 games. Incomplete cells, interrupted cycles, and oversampled
cells do not enter the promotion objective. Four initial pairs per cell and
four-pair continuation waves align with this cycle. Budget fields retain their
legacy names but mean pairs **per cell** when balanced evaluation is enabled.

Promotion uses a mixture of Hoeffding exponential e-processes at complete-cycle
boundaries. With N independent bounded pair scores and centered sum S, each fixed
positive lambda contributes `exp(lambda*S - lambda^2*N/8)`. A fixed equal mixture
over the declared lambda grid remains an e-process. This is valid for different
means across cells/severities because each complete cycle has the same fixed
allocation. It assumes independent seeded pairs; it does not assume independence
between the two games in a pair. This assumption and the seed scheme are part of
the immutable evaluation contract. Confidence sequences invert these e-processes;
the construction follows the exponential-supermartingale framework described by
[Howard et al.](https://arxiv.org/abs/1810.08240).

The candidate promotes on proven improvement in the balanced objective. Each
cell separately vetoes a proven regression below `cell_regression_floor_elo`,
using family-wise error allocation across cells. A cell need not prove a positive
gain before the candidate can promote. An exhausted budget is an inconclusive
rejection; small pilots are not strength certificates.

The promotion screen may use 256 simulations. The separate strength ladder uses
`strength_simulations`, normally 1024, and direct-predecessor historical crossplay.
The result contract pins cell weights, severity and PDA schedules, search budget,
opening protocol, seed scheme, and statistical method. Contract-specific result
paths and continuation checks prevent a legacy single-mode result, another search
budget, or another allocation from entering the same evaluation. Changing the
contract starts a distinct measurement epoch.

`strength_efficiency_report.py` adds `balanced_strength`. Its frontier is the
persisted champion, never the most recently rejected candidate. It accepts only
complete separate strength measurements with matching 1024-simulation contracts,
revalidates the raw paired outcomes, and reports missing cells, disconnected
graphs, and one-sided saturation explicitly. A saturated score has no fabricated
finite Elo. Cheap promotion screens remain available as screen evidence and do
not become strength-ladder edges.

The reported rating sums balanced-score Elo contrasts along a connected path to
the champion and therefore makes an explicit additive-Elo approximation. It is a
descriptive relative measure, not absolute Elo and not a promotion test. Per-cell
contrasts retain their paired uncertainty. Each chronological measurement edge
receives its own summable uncertainty budget, which is not reused when a new
shortcut changes the graph path. Elo per wall-hour and per provisioned
GPU-hour use total run wall time, including Stage A, waiting, pauses, evaluation,
and rejected candidates; the numerator's balanced epoch and anchor are explicit.
