"""Deterministic ring-stratified replay sampling."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Sequence

import torch
from torch.utils.data import Sampler

from .topology import MAX_RINGS, MIN_RINGS


class RingStratifiedSampler(Sampler[int]):
    """Balance non-empty ring strata, cycling smaller strata as needed.

    ``rank`` and ``world_size`` partition one deterministic global stream, so
    distributed workers neither duplicate indices at the same stream position
    nor lose the ring balance.
    """

    def __init__(
        self,
        rings: Sequence[int],
        *,
        num_samples: int | None = None,
        seed: int = 0,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        if not rings:
            raise ValueError("rings cannot be empty")
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("rank must be in 0..world_size-1")
        self.rings = [int(ring) for ring in rings]
        if any(not MIN_RINGS <= ring <= MAX_RINGS for ring in self.rings):
            raise ValueError(f"rings must be in {MIN_RINGS}..{MAX_RINGS}")
        self.num_samples = int(num_samples if num_samples is not None else len(rings))
        if self.num_samples < 1:
            raise ValueError("num_samples must be positive")
        self.seed = int(seed)
        self.rank = rank
        self.world_size = world_size
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = epoch

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[int]:
        buckets: dict[int, list[int]] = defaultdict(list)
        for index, ring in enumerate(self.rings):
            buckets[ring].append(index)
        active_rings = sorted(buckets)
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch * 1_000_003)

        shuffled: dict[int, list[int]] = {}
        cursors = {ring: 0 for ring in active_rings}

        def reshuffle(ring: int) -> None:
            order = torch.randperm(len(buckets[ring]), generator=generator).tolist()
            shuffled[ring] = [buckets[ring][offset] for offset in order]
            cursors[ring] = 0

        for ring in active_rings:
            reshuffle(ring)

        total = self.num_samples * self.world_size
        global_stream: list[int] = []
        while len(global_stream) < total:
            ring_order = torch.randperm(len(active_rings), generator=generator).tolist()
            for ring_offset in ring_order:
                ring = active_rings[ring_offset]
                if cursors[ring] >= len(shuffled[ring]):
                    reshuffle(ring)
                global_stream.append(shuffled[ring][cursors[ring]])
                cursors[ring] += 1
                if len(global_stream) == total:
                    break
        return iter(global_stream[self.rank :: self.world_size])
