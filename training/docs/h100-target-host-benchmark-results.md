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
- Ruff, Pyright and 107 non-CUDA Python/native tests
- compiled BF16 CUDA forward/backward and repeated-inference soak
- exact native/Python feature parity
- real two-rank NCCL BF16 optimizer step with matching parameters
- replay generation, checkpoint publication and resume contracts
- learner/arena pause sharing
- graceful actor and DataLoader shutdown
- a complete 10-step learner run with two terminal candidate arena decisions

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

Moving promotion to the learner GPU with pause sharing and assigning GPU 7 as
a seventh actor raised aggregate throughput to 100.9k simulations/s, 73% above
control. The run stopped within seven seconds with every worker exiting zero.

The selected reusable profile is
[`../configs/h100-8gpu-optimized.yaml`](../configs/h100-8gpu-optimized.yaml).

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
failure on the target host.

## Active production run

- run ID: `star-opt-20260711T0030Z`
- run root: `/home/ubuntu/edgeconnect-runs/star-opt-20260711T0030Z`
- service: `edgeconnect-startrain-star-opt-20260711T0030Z.service`
- profile: frozen, SHA-256 recorded
- retention: enabled in dry-run mode

The service is enabled across reboot and started with seven healthy actors,
one learner and pause-shared promotion. No worker or service restart was
observed during startup.

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
