# Target-host H100 benchmark status

The July 10–11, 2026 measurements were produced by the retired rules-v1 stack.
They are not certification evidence for the clean v2 migration and must not be
mixed with v2 replay, checkpoints, manifests, or serving results.

## Required v2 rerun

The next target-host run must use:

- rules `edgeconnect.star.rules.v2`,
  `fnv1a64:2da3783519381453`;
- supported rings exactly 4, 6, 8, and 10;
- feature schema v3 and node-only action tensors;
- replay schema v4 with binary loss/win outcomes;
- fresh run, replay, checkpoint, model-manifest, and browser-manifest roots; and
- a rebuilt rules-v2 `star_native` extension.

Run the bounded inference sweep on representative rings 6 and 10:

```bash
python scripts/h100_system_benchmark.py \
  --config configs/h100-8gpu-optimized.yaml \
  --output-dir "$RUN_ROOT/system-benchmark-v2" \
  --rings 6 10 \
  --batch-sizes 64 128 256 \
  --repeats 3 \
  --metrics-root "$RUN_ROOT"
```

Then record:

- leaf evaluations/s and latency for every ring/batch case;
- peak allocated and reserved HBM;
- learner examples/s, step time, and replay wait fraction;
- actor games/s, samples/s, game length, and policy supervision;
- replay decode/materialization throughput;
- arena binary wins/losses and pair-level confidence bounds; and
- recovery, graceful shutdown, and immutable publication evidence.

Until this rerun is complete, no H100 throughput or strength claim is certified
for v2.
