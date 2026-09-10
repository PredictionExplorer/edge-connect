# MCTS corrections and deployment

The September 10 audit found two reproducible search defects: FP32 rounding in
the Gumbel uniform sampler could produce infinite noise, and the pie decision
used an exploratory root average even when search selected a winning keep move.

Commit `730328da07220b1984044a4a5278d9fd8015c10f` fixes both. Sampling and the
logarithms now use FP64 before the finished noise is converted to FP32. Native
results expose the selected placement value; self-play, arena, serving and the
browser use that value for the pie decision. The existing root average remains
a diagnostic. Regression tests cover the exact failing RNG seed and the
128-simulation/16-candidate pie example, plus classic/double keep, swap and dead
zone behavior and the resulting replay labels.

Search identity is now `gumbel-completed-q-v2-finite-noise-selected-keep`.
Python rejects older native extensions, and the browser checks the WASM identity
and versions both asset URLs. Balanced evaluation receives a new evidence
namespace; old reports and resume files remain available without contributing
to the corrected search's evaluation. New replay records carry the search ID.

Interactive inference now reuses predictions. The browser retains at most 1,024
entries and includes semantic history, legal ordering and immutable model
identity in its keys. Serving defaults to 4,096 entries within 64 MiB, with one
model-owned inference worker and bounded batching across concurrent requests.
A lone request adds no batching delay. Tests verify ownership, model changes,
PDA separation, eviction, cancellation and shutdown.

The rollout exposed a separate replay-backup stall. The database copy repeatedly
restarted while its temporary destination remained 4 MiB. The first cutover was
aborted before migration; the previous runtime resumed from the exact saved
step 162,142. Commit `5b4de8dc6e947fbf8bc96aed8bde035176035f3b` pins a read snapshot,
adds progress and total-duration bounds, closes connections explicitly, and
cleans only the attempted copy's temporary files. It preserves the last good
backup on failure. Concurrent-writer and stalled-copy tests cover the behavior.

The complete corrected disaster backup passed under restricted filesystem
permissions, covering 19,795,923,725 bytes and 12,479 catalog entries. Its snapshot
SHA-256 is `7ae07421b0eaac99c0c791dffa6550f1471f1fc17e59ca6f18ab9563d89bf4c0`.
The corrected database-copy stage also completed during the second stopped
boundary, where learner heartbeat and recovery checkpoint both recorded 162,153.

Validation includes the 1,917-test Python suite, thirteen additional teacher-loss
contract tests, 224 frontend tests and the production web build, Rust workspace
tests and lint, a rebuilt WASM regression check, and target-host native/CUDA
checks. The all-board CPU arena check exceeded its original 60-second limit on
the shared host, then passed in 66.06 seconds with a 180-second bound. Additional
backup/disaster-recovery tests passed locally and on the server. All 28 Python
coverage floors pass; overall coverage is 82.46%.

The first migration attempt after repairing the backup exposed a code-only
upgrade restriction: the migrator rejected identical profiles. No migration
was applied, and the old runtime resumed. Commit
`f41482963e373c1f2644881158171c90ce00a6ac` adds explicit source-only upgrades,
requiring recorded current-source authority and a distinct explicit target
commit. Configuration hashes, checkpoints, locks and UTD validation remain
mandatory. Fifty-five migration tests passed, including startup preflight and
verified disaster snapshots of a source-only upgraded run.
Subsequent profile migrations must use the updated migrator, which understands
the source-only records in the existing migration history.

The final immutable runtime is
`/home/ubuntu/edgeconnect-releases/variant-mcts-corrections-f414829`.
It launched at 04:41:45 UTC on September 10 and passed the sustained readiness
check at 04:46:58 UTC. Its frozen profile is
`/home/ubuntu/edgeconnect-runs/variant-network/profile-mcts-corrections-20260910.yaml`.
The stop and recovery checkpoint both recorded step 162,155; no uncheckpointed
learner updates were discarded. All 54 captured learner/arena control files
retained identical hashes through migration. The configuration hash remains
`cef8e21f826199dfb00f06c292c10fb09d2e4f874743c73d2de4b8407310c78b`.

The learner advanced to 162,156 and all ten workers passed fresh-heartbeat checks
with zero restarts. GPU 7's actor was normally paused while the arena used that
GPU. The final cutover used a separate fully verified snapshot namespace,
reusing immutable objects from the prior verified backup to avoid repeatedly
scanning the historical snapshot archive. Its directory is
`/lambda/nfs/texas-north-fs/edgeconnect-dr/manual/mcts-cutover-20260910`, and its
snapshot SHA-256 is
`f08dd040d937933fd72603ed4e6a062c9de8b8ba7ef1c9f5b1b9741f349e60d6`.
Regular backup timers retain their original production namespace.
Fresh local replay-backup and strength-report runs completed successfully after
activation. Training, monitoring and all three protection/report timers are
active; temporary deployment recovery timers are stopped.
A subsequent check reached learner step 162,157 with all ten workers running,
including GPU 7 after its normal evaluation pause. Every actor GPU had completed
neural requests with zero failed requests and zero inference-worker failures;
the captured evidence is `final-health.json` in the rollout directory.

Operational evidence, prior units, failed-attempt records and rollback material
are retained under `/home/ubuntu/edgeconnect-rollouts/mcts-corrections-20260910`.

Subtree visit reuse, multiple outstanding leaves in one tree, alternative graph
backups, Q normalization and evaluation-noise tuning remain research comparisons.
They were not established defects and are not changed by this release. There is
no claim of optimal search settings or measured Elo-per-hour improvement.
