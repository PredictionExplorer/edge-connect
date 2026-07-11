# Target-host H100 benchmark results

Measured on `ubuntu@192.222.54.207` on 10–11 July 2026.

## Host and evidence

- 8 NVIDIA H100 80 GB HBM3 GPUs with all-to-all NV18 links
- 208 logical Xeon Platinum 8480+ CPUs and 1.7 TiB RAM
- 22 TB ext4 run volume; the Lambda virtiofs mount was not used for replay
- PyTorch 2.13.0 + CUDA 13.0, Python 3.11, Rust 1.93
- pinned base revision `a1198370490d9b9f6cec0791d7044a924d3c4363`
- complete source snapshot, patch, profile, environment and hardware inventory
  stored under each target-host run root

The local volume measured 7.97 GiB/s sequential writes and 1,693
fsync-bound 4 KiB writes/s.

## Certification

The following target-host gates passed:

- Rust formatting, Clippy and workspace tests
- Ruff, Pyright and 122 non-CUDA Python/native tests
- all ten static BF16 CUDA forward/backward shapes and reverse cache reuse
- exact native/Python feature parity
- rank-shifted ten-ring NCCL BF16 optimizer steps with matching parameters
- replay generation, checkpoint publication and resume contracts
- coordinator-owned actor/arena GPU handoff with token-matched acknowledgement
- graceful actor and DataLoader shutdown
- a complete 10-step learner run with two terminal candidate arena decisions

The production ring-12 batch-512 backward gate used 50.8 GB allocated and
52.0 GB reserved memory, leaving safe headroom on an 80 GB H100.

## Inference boundary

The original boundary converted and validated every legal policy scalar in
Python. On a serial NUMA-pinned run it produced:

- ring 6, batch 64: 2,329 leaves/s
- ring 12, batch 64: 758 leaves/s

Vectorized legal-policy extraction, trusted native-list handling and cached
immutable topology tensors produced these three-run medians:

- ring 6: 12,342 / 19,450 / 24,101 leaves/s at batches 64 / 128 / 256
- ring 12: 5,454 / 6,148 / 6,646 leaves/s at batches 64 / 128 / 256

All 18 formal benchmark cases passed the 5,000 leaves/s production floor.
Raw results are in
`/home/ubuntu/edgeconnect-runs/system-benchmark-optimized-a119837`.

## End-to-end optimization

The unchanged six-actor control completed 13,568 games, 772,930 replay
samples and 53.0 million search simulations. Its aggregate rate was 58.2k
search simulations/s.

Increasing actor cohorts from 64 to 128 raised six-actor throughput to 86.1k
simulations/s. The first treatment exposed excessive stop latency on a ring-12
cohort; incomplete cohorts now abort at a search-wave boundary without writing
partial replay.

The initial treatment moved promotion to the learner GPU with pause sharing
and assigned GPU 7 as a seventh actor. It raised aggregate throughput to
100.9k simulations/s, 73% above control, but production-sized arenas would
pause the learner for hours.

The selected reusable profile is
[`../configs/h100-8gpu-optimized.yaml`](../configs/h100-8gpu-optimized.yaml).
The reusable profile now keeps learner GPU 0 continuous and
coordinator-pauses `actor-gpu-7` before arena work on GPU 7. The token-matched
request/ready/release protocol was exercised twice in a complete learner and
arena cycle. Actor handoff took 2.01 and 1.01 seconds, released GPU memory
before arena allocation, preserved a zero restart count, and restored the
actor only after durable result/champion persistence.

## Progressive board curriculum

The historical profile's actor curriculum did not cause early learner
training: the stratified learner still waited for replay on every ring 3–12.
The new opt-in `learner.use_ring_mixture_curriculum` aligns learner replay with
the actor unlock schedule.

On the target host, the corrected curriculum:

- trained first on rings 3–6 while larger rings remained locked;
- completed 220 measured learner steps with no restart;
- sustained a median 6,419 examples/s and 79.8 ms measured step time;
- unlocked the large-board actor stage after one million replay samples.

Learner compilation is static per homogeneous ring. Dynamic backward
compilation was rejected after reproducing a PyTorch Inductor `CantSplit`
failure on the target host. Static compilation now uses an isolated
ten-specialization budget: the original eight-entry Dynamo limit caused the
failed preflight learner to restart nine times after reaching step 360.

## Prelaunch learning-efficiency experiment

Three matched-compute seeds compared champion-only self-play with an 80%
candidate / 20% champion mixture on rings 3–6. Direct paired evaluation at
matched learner steps produced:

- seed 101: +107.5 Elo;
- seed 202: +8.7 Elo;
- seed 303: -26.1 Elo.

The aggregate point estimate was +29.0 Elo over 60 pairs, but its paired
bootstrap interval was approximately -37.8 to +98.1 Elo and its anytime lower
bound was -80.6 Elo. Two seeds were positive, but the preregistered lower-bound
gate did not pass. The production run therefore retains champion-only
self-play rather than spending a multi-day run on an inconclusive treatment.

## Active production run

- run ID: `star-maxlearn-20260711T0500Z`
- run root: `/home/ubuntu/edgeconnect-runs/star-maxlearn-20260711T0500Z`
- service: `edgeconnect-startrain-star-maxlearn-20260711T0500Z.service`
- profile and source tree: frozen with SHA-256 records
- retention: enabled in dry-run mode

The clean run started with seven healthy actors, one continuous learner and
actor-pause-shared promotion. The learner passed the replay gate, began
training, and reached measured steps with zero worker or systemd restarts.

Arena calibration doubled a round from 25 to 50 pairs per ring while increasing
wall time only from 71.2 to 84.8 seconds in the reduced-search calibration.
Evaluator throughput rose from 1,788 to 2,992 rows/s. The production profile
uses 50-pair rounds, 15k-step candidate publication, candidate backlog
coalescing, candidate-specific seed blocks, a 45k plateau limit and a
worst-case final drain allowance.

## Time estimate

The 22-billion-leaf planning budget is bounded by:

- about 2.5 days at the observed small/mid-board cluster rate;
- about 5.9 days if all seven actors continuously ran the measured ring-12
  batch-128 rate.

Allowing for the all-ring mix, curriculum transitions, promotion pauses and
checkpoint overhead gives a provisional **4–7 day** full-budget estimate for
this host.

The agreed internal +400 Elo target is not yet estimable from a strength
curve. A step-10 integration checkpoint scored 43.75% over eight paired games
against the frozen shallow-search baseline, with an anytime Elo interval of
approximately -326 to +228. It did not reach the target. A defensible target
forecast requires the 100k- and 500k-game strength measurements; the first
useful reforecast should be available after roughly 12–24 hours.

The +400 target is an internal non-human benchmark and must not be described
as evidence of superhuman human-relative play.
