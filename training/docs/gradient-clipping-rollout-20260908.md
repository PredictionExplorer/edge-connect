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
