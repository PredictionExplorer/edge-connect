# Gradient diagnostics and AdaGC — September 8, 2026

The user authorized implementation of named-gradient diagnostics, correction of
any demonstrated cause of gradient spikes, and a controlled AdaGC trial. The
17,402,775-parameter model, optimizer learning rates and state, 85/5/5/5 board
allocation, equal weighting of six modes, and largest-board promotion contract
are preserved.

## Implementation

`train.gradient_diagnostics` enables sampled, pre-clipping attribution alongside
the existing learner diagnostic cadence. Reports contain named parameter norms,
their shares of the total squared gradient norm, optimizer routing, parameter
norms, current batch board/mode/severity counts, and label availability. Original
variant labels travel with replay batches without changing features or the
stored replay schema; resolved pie positions remain identifiable. Older manually
constructed batches disclose ambiguity instead of inventing a mode.

`train.gradient_clipping.mode` defaults to `global`, preserving the existing
training calculation. Opt-in `adagc` follows
[AdaGC Algorithm 1, version 2](https://arxiv.org/html/2502.11034v2): global-clipping
warmup initializes per-tensor running minima, then each tensor is bounded using
its previous exponential moving average and the average is updated with the
clipped norm. Defaults are beta 0.99, multiplier 1.04, and 100 warmup steps.
Zero and missing gradients are treated as inactive to avoid an absorbing zero
threshold; newly active tensors bootstrap from a globally clipped observation.
This extension is explicit and tested.

Adaptive history is part of recovery and published checkpoints, bound to exact
parameter names, shapes, dtypes, and clipping settings. Missing or corrupt
adaptive history fails closed. A legacy global checkpoint can enter adaptive
warmup only with explicit opt-in. Distributed finite checks precede adaptive
history mutation. Clearing optimizer history also resets adaptive history.

## Controlled screening protocol

The trial restores raw weights, the full optimizer, scheduler age/rates, and EMA
from recovery step 126,309. It copies and hashes 84 replay shards (about 307 MiB),
covering all 24 board/mode cells with game-disjoint training and validation.
The prepared manifest SHA-256 is
`52eb795c052cef84b82cb39906ff1c19e02ecc686f829880cd7d12b45f956c49`.
Each cell contains 2,048 training and 512 validation positions.

A 24-batch diagnostic pass identifies gradient contributors without changing
training state. Global and AdaGC training arms receive identical deterministic
batches, augmentation, initial states, and random-generator states. Each
600-step arm contains five 120-step blocks with the exact requested mixture.
Both raw and EMA models are evaluated on the same held-out games. Source, model,
optimizer, scheduler, EMA, RNG, data, and batch schedule pins must match for a
comparison. GPU ownership is checked; bounded subprocess cleanup owns only the
trial process group. Compilation and the initial adaptation window are reported
separately from steady training time.

This is a frozen-replay screening experiment: largest-board training examples
repeat about 21 times over 600 steps. It does not measure self-play feedback or
establish an Elo/hour gain. A lower held-out loss is evidence for further
evaluation, not authorization to promote a model. Existing global clipping stays
available as the control. Experimental evidence and deployment records belong
under `/home/ubuntu/edgeconnect-rollouts/gradient-clipping-20260908`.

## Validation and outcome

The initial full suite passed 1,588 tests with seven hardware-dependent skips.
Additional focused tests cover exact adaptive continuation, explicit cold start,
corrupt history rejection, diagnostic equivalence, and safe profile migration.
Live trial and deployment results will be recorded after verification.

## Demonstrated underlying defect

The first 24-cell diagnostic localized the outliers to the six-ring board:
global norms were 8.88–127.97, with 97–99.54% of the squared norm in one late
relation-bias embedding. The other boards measured 1.18–2.54. All labels were
available and no teacher targets were present.

An independent check restored the same raw checkpoint and exactly the same
512-row six-ring double-standard inputs/targets. Compiled BF16 produced a
block-six relation-bias gradient norm of 8.69780; eager BF16 measured 0.00047717,
and a full-FP32 explicit-attention reference measured 0.00045633. The compiled
gradient also violated softmax's uniform-logit-shift invariant by a large margin.
This establishes corruption in the compiled backward path, not ordinary large
gradients that should merely be clipped more aggressively.

The correction preserves fused SDPA's forward values and Q/K/V gradients. Only
the additive-mask vector-Jacobian product runs through opaque custom operators,
using an explicit FP32 softmax derivative outside Inductor lowering. Operator
schema, fake-tensor, autograd, and AOT checks pass. No parameters are added or
removed, and inference does not execute this training-only path.

The corrected compiled model passed all 24 production-sized diagnostic batches.
Six-ring norms became 1.16–1.81, and the maximum across all boards/modes was 2.54.
Largest-board forward losses remained bitwise identical. The exact failing
batch's corrected bias norm was 0.00047746, matching the eager/reference scale.
Warm largest-board backward computation cost about 22% more; this is a learner
compute cost, not an actor-inference or measured wall-clock Elo penalty.

The original four 600-step clipping arms ran on the defective backward path and
cannot select a production clipping policy. Their strict comparison gate also
rejected unmatched Python/NumPy RNG fingerprints: the older shared seed helper
only seeded Torch. The harness now seeds and records all RNG families, touching
only its owned CUDA generator. A fresh 240-step screen on corrected code uses
two complete allocation blocks, identical full starting states, and explicit RNG
reset immediately before training. This remains a short frozen-replay screen,
not a claim of better Elo/hour.

Monitoring now distinguishes measured severe global clipping (retaining less
than 10% of the original norm) from routine clipping frequency. Older metrics
without severity retain a clearly labeled fallback warning. Nonfinite failures
remain errors, and the underlying diagnostic values stay visible.
