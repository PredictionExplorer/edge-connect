"""Bounded exact-input neural prediction cache and reusable pinned staging."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import sys
import threading
from typing import Mapping, Sequence

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class RawPrediction:
    """CPU float32 bytes: node policy, two outcome logits, then score logits.

    Values are deliberately stored before score utility or any caller-specific
    postprocessing. No GPU tensors or views of a larger batch are retained.
    """

    packed: bytes
    nodes: int


class BoundedPredictionCache:
    """LRU keyed by the complete input bytes, with entry and memory bounds."""

    def __init__(self, *, max_entries: int, max_bytes: int) -> None:
        if type(max_entries) is not int or max_entries < 0:
            raise ValueError("cache max_entries must be a non-negative integer")
        if type(max_bytes) is not int or max_bytes < 0:
            raise ValueError("cache max_bytes must be a non-negative integer")
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self._entries: OrderedDict[bytes, tuple[RawPrediction, int]] = OrderedDict()
        self.bytes = 0
        self.evictions = 0
        self._lock = threading.RLock()

    @property
    def enabled(self) -> bool:
        return self.max_entries > 0 and self.max_bytes > 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def peek_many(self, keys: Sequence[bytes]) -> tuple[RawPrediction | None, ...]:
        """Read immutable producer hints without changing LRU order or counters.

        Hints may be evicted before inference; the owner always looks up again.
        """
        with self._lock:
            return tuple(self._entries[key][0] if key in self._entries else None for key in keys)

    def get(self, key: bytes) -> RawPrediction | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            self._entries.move_to_end(key)
            return entry[0]

    def put(self, key: bytes, value: RawPrediction) -> None:
        with self._lock:
            self._put_locked(key, value)

    def _put_locked(self, key: bytes, value: RawPrediction) -> None:
        if not self.enabled:
            return
        # Include Python objects and conservative map/link accounting, not just
        # tensor payload. The advertised bound is this charged resident size.
        charge = (
            sys.getsizeof(key)
            + sys.getsizeof(value)
            + sys.getsizeof(value.packed)
            + sys.getsizeof(value.nodes)
            + 256
        )
        previous = self._entries.pop(key, None)
        if previous is not None:
            self.bytes -= previous[1]
        if charge > self.max_bytes:
            return
        while self._entries and (
            len(self._entries) >= self.max_entries
            or self.bytes + charge > self.max_bytes
        ):
            _, (_, removed_bytes) = self._entries.popitem(last=False)
            self.bytes -= removed_bytes
            self.evictions += 1
        self._entries[key] = (value, charge)
        self.bytes += charge

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self.bytes = 0


@dataclass(slots=True)
class _PinnedSlot:
    buffers: dict[str, Tensor]
    ready: torch.cuda.Event | None = None


class PinnedTransferPool:
    """Small ring of staging buffers, reused only after their DMA completes.

    Calls must be serialized by the inference owner. Slots grow to the largest
    field seen; an inference broker bounds the size of each incoming batch.
    This avoids allocating and pinning tensors anew for each search wave.
    """

    def __init__(self, *, slots: int, device: torch.device) -> None:
        if type(slots) is not int or not 1 <= slots <= 8:
            raise ValueError("pinned transfer slots must be in [1, 8]")
        if device.type != "cuda":
            raise ValueError("pinned transfers require a CUDA device")
        self.device = device
        self._slots = [_PinnedSlot({}) for _ in range(slots)]
        self._next = 0
        self.allocations = 0
        self.reuses = 0
        self.waits = 0
        self.transferred_bytes = 0

    def transfer(self, fields: Mapping[str, Tensor]) -> dict[str, Tensor]:
        slot = self._slots[self._next]
        self._next = (self._next + 1) % len(self._slots)
        if slot.ready is not None and not slot.ready.query():
            self.waits += 1
            slot.ready.synchronize()
        output: dict[str, Tensor] = {}
        try:
            for name, tensor in fields.items():
                if tensor.device.type != "cpu":
                    raise ValueError("pinned staging requires host tensors")
                buffer = slot.buffers.get(name)
                if (
                    buffer is None
                    or buffer.dtype != tensor.dtype
                    or buffer.numel() < tensor.numel()
                ):
                    buffer = torch.empty(
                        tensor.numel(),
                        dtype=tensor.dtype,
                        device="cpu",
                        pin_memory=True,
                    )
                    slot.buffers[name] = buffer
                    self.allocations += 1
                else:
                    self.reuses += 1
                staging = buffer[: tensor.numel()].view(tensor.shape)
                staging.copy_(tensor)
                output[name] = staging.to(self.device, non_blocking=True)
                self.transferred_bytes += tensor.numel() * tensor.element_size()
        finally:
            # Even a failed transfer may have queued a DMA reading this slot.
            slot.ready = torch.cuda.Event()
            slot.ready.record(torch.cuda.current_stream(self.device))
        return output

    @property
    def allocated_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for slot in self._slots
            for tensor in slot.buffers.values()
        )

    def close(self) -> None:
        for slot in self._slots:
            if slot.ready is not None:
                slot.ready.synchronize()
            slot.buffers.clear()
