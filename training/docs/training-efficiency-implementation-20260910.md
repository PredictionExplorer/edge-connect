# Training efficiency implementation

The first implementation batch repairs the producer–consumer mismatch identified
in the [performance audit](training-performance-opportunities-20260910.md).
Selected short shards and partial tails now participate in full, ring-homogeneous
optimizer batches. Packing keeps unique row identities and disjoint distributed
ranks while combining as many files as necessary. Readiness and allocation use
the same complete-row capacity. The previous additional four-shard subsampling
loss is removed.

Replay GC now protects the learner's eligible per-ring sample window, including
its segment quotas and model-age limits, beyond the existing file-count minimum
and active reader watermarks. Shortfall redistribution is proportional with
deterministic ties. Window diagnostics expose usable rows, source-age/mode
distributions, and newly committed data waiting outside the active selection.

Native inference has exact compact state descriptors and lazy per-row feature
storage. A read-only cache peek precomputes expected misses on CPU producers,
preserving overlap with the inference worker. Complete hits avoid scoring and
feature construction. Misses racing with cache eviction safely materialize their
features. General Python inputs retain complete-byte validation and tensor-version
checks. Rebuild the native module and check native_inference_key_version()==1.

A focused CPU comparison of the trusted native path with the retained general
fallback reduced keys from 54,682 to 389 bytes. With a tiny test network, median
cold-path overhead changed from 2.389 to 1.829 ms and warm-path overhead from 2.273
to 1.021 ms. These are CPU overhead measurements, not H100 or Elo multipliers.

The bounded exact endgame solver now memoizes exact semantic states and prunes
with alpha-beta. It preserves first-tie choices, terminal board and ordered
history, and charges visited states against the hard budget. Incomplete or cutoff
results are never cached as exact. Across 144 reference comparisons, visited
nodes fell from 19,536 to 9,474 with identical results.

Two opt-in execution controls are exposed under
orchestration.model_refresh.inference: compact_inference_gather and
small_batch_graph_buckets. Both default to false. The compact-gather option moves
the same autocast conversion ahead of repeated neighbor gathering while keeping
normalization and residual arithmetic unchanged. The H100 whole-model comparison
at ring 10 and batch sizes 32/64/128 preserved all six outputs exactly but measured
median ratios of 1.007/1.021/1.000. That does not justify enabling it by default.
Raw measurements are in [the H100 record](compact-gather-h100-20260910.json).

Strength accounting reconstructs nondefault search contracts and uses an explicit
strength-epoch.json marker for current rates. Its schema is:

    {"schema_version":1,"evaluation_contract_identity":"sha256-...",
     "anchor_identity":"sha256-...","started_ns":123,
     "minimum_candidate_step":123,"source_commit":"optional exact commit"}

The marker must describe the strength-measurement contract, not the cheaper
promotion gate. Pre-boundary evaluations and candidates are excluded. Missing or
inadequate evidence yields an unavailable current rate; historical diagnostics
remain explicitly lifetime-normalized. This measures realized champion-frontier
change, not causal release efficiency. Source-role telemetry now receives the
actual candidate/champion/history role.

Local validation included the broad Python suite and targeted reruns after fixing
test doubles and strict additive-default compatibility, 128 rebuilt-native/model
checks, Rust workspace tests and strict Clippy, six WASM bridge tests, TypeScript,
and all 234 frontend tests. Runtime activation and checkpoint evidence are recorded
after the graceful rollout. No 10x Elo/hour gain has been established.
