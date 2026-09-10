# Optional search execution controls

These experiments are disabled by default. Omitting `search_execution` keeps
one requested prediction per tree at a time, fresh root statistics for each
move, and the existing fixed full-search budget. Model precision is unchanged.

**First-visit batching** requests predictions for several unvisited root
children together. Simulations still traverse and back up in the original
Gumbel order; this is prediction prefetching, without concurrent tree updates
or virtual loss. The configured width is `1..64`, subject to the inference
broker's row limit. Browser batches stack all model inputs and validate all
output rows before caching predictions or submitting responses.

**Subtree reuse** retains an expanded, nonterminal proper descendant with an
exact semantic key, including observable history. Repeating the same root,
taking an unsearched swap, exceeding the retained-node cap, or changing model,
feature/value context or per-seat playout-doubling advantage (PDA) starts a
fresh search. Every root still requests network predictions, which may come
from the prediction cache. Only completed sessions are retained; a session is
removed from its cache while in use and discarded on failure or cancellation.

A requested budget always means **new simulations**. `visits` sums to that
budget; `inherited_visits` and `total_visits` report prior and combined work.
Q estimates and the diagnostic root mean include retained evidence, while the
scheduler and policy-target visit scale use new visits. Pie decisions compare
the selected keep continuation with swapping, rather than using the mean of
explored alternatives.

## Configuration

These fragments belong in staged configuration files. Each option can be
enabled independently.

For self-play, in the training profile:

```yaml
selfplay:
  search_execution:
    first_visit_batch_size: 4
    subtree_reuse: true
    subtree_reuse_max_nodes: 4096
    full_budget:
      mode: root-entropy
      minimum_fraction: 0.5
      entropy_threshold: 0.35
```

`root-entropy` applies only to planned full searches. It measures legal-policy
entropy divided by `log(number of legal actions)` before search. Below the
threshold, it reduces the full cap toward `minimum_fraction`; at or above the
threshold, it retains the original cap. Reductions preserve the fast-search
floor and PDA budget ratio, including the required rounding. Planned fast
searches remain fast. Actual budgets, entropy, execution settings and inherited
work are recorded in replay provenance. `mode: fixed` is the default.

For serving, in the server YAML:

```yaml
search_execution:
  first_visit_batch_size: 4
  subtree_reuse: true
  subtree_reuse_max_nodes: 4096
inference:
  max_batch_rows: 16
  search_cache_entries: 8
```

With shared inference enabled, the effective server batch limit is the smaller
of `inference.max_batch_rows` and
`limits.max_concurrency * first_visit_batch_size`. Serving honors the exact
requested budget and rejects adaptive full-budget policies.

For arena evaluation, in the training profile:

```yaml
arena:
  search_execution:
    first_visit_batch_size: 4
    subtree_reuse: true
    subtree_reuse_max_nodes: 4096
```

Arena also rejects adaptive budgets. Nondefault execution settings enter the
evaluation contract. Reuse depends on available session history and concurrent
group scheduling, so arena reports mark reuse runs nondeterministic; resumed
evaluations start with fresh tree statistics.

For the browser, the existing manifest's `recommended_local_search` object
accepts these optional fields:

```json
{
  "simulations": 64,
  "max_considered": 16,
  "c_visit": 50.0,
  "c_scale": 1.0,
  "swap_dead_zone": 0.02,
  "first_visit_batch_size": 4,
  "subtree_reuse": true,
  "subtree_reuse_max_nodes": 2048
}
```

Distillation accepts the same fields under `export.recommended_search` and
omits default execution fields from exported manifests. Browser adaptive
budgets are unsupported. Enabled experiments require execution API version 1
and `WasmSearchSession`; rebuild assets from the repository root with
`npm run build:star-wasm`. The implementation URL revision prevents stale
WASM assets from silently supplying the older API.

The retained-node limit defaults to 4,096 and accepts `1..65536`. It applies to
each reused self-play tree, to the aggregate completed-session pool in serving
and arena, and to the single completed browser session. New simulations can
grow active trees. Node limits are not byte budgets for active inference;
prediction-cache and broker limits remain separate. Serving retains at most
eight completed sessions by default (`inference.search_cache_entries`, `1..64`).

## Measure before enabling broadly

From `training/`, compare fixed caps and the entropy policy on frozen positions:

```bash
.venv/bin/python scripts/benchmark_search_budgets.py \
  --config /absolute/path/frozen-profile.yaml \
  --checkpoint /absolute/path/champion.json \
  --positions /absolute/path/frozen-positions.json \
  --caps 128 256 384 640 --reference-cap 1024 \
  --first-visit-batch-size 4 --device cpu --repeats 3 \
  --timeout-seconds 300 --output /absolute/path/new-search-sweep.json --execute
```

`--checkpoint` accepts an immutable model manifest or publication pointer.
Omit `--execute` to print the validated, hash-pinned plan without running
inference. The output path must be new when executing. A minimal position file
is `{"schema_version":1,"positions":[{"id":"opening-r6","rings":6,"actions":[],"seed":17}]}`;
optional position fields are `mode`, `handicap`, `pie` and `pda`. Actions are
native placement codes; the board's node count is the swap code.

The harness runs eager inference, clears prediction storage between arms,
enforces a process deadline, and reports cost, action disagreement, policy L1
difference and selected-value differences against the deeper reference. It
does not measure subtree reuse across games or production compiled throughput.
For CPU bookkeeping parity, use
`cargo run --release -p star-search --example benchmark_search -- --repeats 5`;
`--trace /absolute/path/search.trace` writes a complete comparison trace and
forces one repeat.

Changing physical neural batch shapes can change floating-point rounding.
Subtree reuse and reduced budgets change available search evidence. Protocol
tests and synthetic trace parity do not establish stronger play. No H100
throughput or Elo/hour gain is established for these opt-in controls.

### Harness execution smoke

On September 10, 2026, the real `--execute` subprocess path completed with an
untrained, 5,851-parameter GraphResTNet (width 4), FP32 CPU inference, thread
limits set to one, and two frozen ring-4 positions: the opening and a mid-turn
state after placements `[0, 1]`. Caps were 4 and 8, the reference cap was 16,
first-visit width was 4, and there was one repeat.

The output contained eight search records and exactly 64 new simulations;
every visit sum matched its recorded budget. Setting the entropy threshold to
1.0 deliberately exercised reduction: both entropy arms changed cap 8 to 4.
The searches processed 72 neural rows in 48 forward calls. Their reported arm
intervals summed to 20.8 ms, excluding setup and process/model initialization.
This is a small execution and accounting smoke test with an untrained model,
not a throughput comparison, recommended entropy threshold, or strength result.
