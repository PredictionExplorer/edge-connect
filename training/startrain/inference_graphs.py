"""Bounded, single-owner CUDA graphs for immutable inference adapters.

Returned tensors are borrowed graph outputs and must be consumed before the
next call. The inference adapter copies predictions to owned CPU bytes while
still holding its evaluation lock. Limits cover retained graph storage;
capture can temporarily require additional warmup/workspace memory.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import logging
import threading
from typing import Any, Callable, Mapping, Protocol, Sequence

import torch
from torch import Tensor


Forward = Callable[..., Tensor]


class CaptureUnavailable(RuntimeError):
    """A recoverable capture limitation, after capture resources are cleaned up."""

    def __init__(self, message: str, *, warmup_calls: int = 0) -> None:
        super().__init__(message)
        self.warmup_calls = warmup_calls


def _signature(value: object) -> tuple[object, ...]:
    if isinstance(value, Tensor):
        return (
            "tensor",
            tuple(value.shape),
            tuple(value.stride()),
            str(value.dtype),
            str(value.device),
        )
    if isinstance(value, (tuple, list)):
        return (type(value).__name__, *(_signature(item) for item in value))
    if value is None or type(value) in (bool, int, float, str):
        return (type(value).__name__, value)
    raise TypeError(f"unsupported CUDA graph argument: {type(value).__name__}")


def _clone(value: object) -> object:
    if isinstance(value, Tensor):
        return value.detach().clone(memory_format=torch.preserve_format)
    if isinstance(value, tuple):
        return tuple(_clone(item) for item in value)
    if isinstance(value, list):
        return [_clone(item) for item in value]
    return value


def _copy(destination: object, source: object) -> None:
    if isinstance(destination, Tensor) and isinstance(source, Tensor):
        destination.copy_(source, non_blocking=True)
    elif isinstance(destination, (tuple, list)) and isinstance(source, (tuple, list)):
        for left, right in zip(destination, source, strict=True):
            _copy(left, right)


def _tensors(value: object) -> list[Tensor]:
    if isinstance(value, Tensor):
        return [value]
    if isinstance(value, (tuple, list)):
        return [tensor for item in value for tensor in _tensors(item)]
    return []


def _account_graph_storage(
    snapshot: Sequence[Mapping[str, Any]],
    *,
    pool: tuple[int, int],
    device_index: int,
    storages: Sequence[tuple[int, int]],
) -> int:
    """Count private-pool segments plus owned blocks outside that pool.

    Unlike a global reserved-memory delta, this remains conservative when
    unrelated cached blocks are released during capture. Block sizes include
    allocator rounding. A shared default-pool block is charged only once.
    """

    segments = [
        segment for segment in snapshot if segment.get("device") == device_index
    ]
    if any("segment_pool_id" not in segment for segment in segments):
        raise CaptureUnavailable(
            "CUDA allocator snapshot has no graph-pool ownership tags"
        )
    private = 0
    for segment in segments:
        if tuple(segment["segment_pool_id"]) == pool:
            size = segment.get("total_size")
            if type(size) is not int or size < 0:
                raise CaptureUnavailable(
                    "CUDA allocator graph-pool size is unavailable"
                )
            private += size
    outside: dict[int, int] = {}
    for address, size in storages:
        if size == 0:
            continue
        found = False
        for segment in segments:
            start, length = segment.get("address"), segment.get("total_size")
            if type(start) is not int or type(length) is not int:
                raise CaptureUnavailable(
                    "CUDA allocator segment address is unavailable"
                )
            if not start <= address or address + size > start + length:
                continue
            if tuple(segment["segment_pool_id"]) == pool:
                found = True
                break
            for block in segment.get("blocks", ()):
                block_start, block_size = block.get("address"), block.get("size")
                if type(block_start) is not int or type(block_size) is not int:
                    raise CaptureUnavailable(
                        "CUDA allocator block ownership is unavailable"
                    )
                if (
                    block_start <= address
                    and address + size <= block_start + block_size
                ):
                    outside[block_start] = block_size
                    found = True
                    break
            break
        if not found:
            raise CaptureUnavailable(
                "CUDA graph buffer is absent from allocator ownership snapshot"
            )
    return private + sum(outside.values())


def _bitwise_equal(left: Tensor, right: Tensor) -> bool:
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and torch.equal(
            left.detach().contiguous().reshape(-1).view(torch.uint8),
            right.detach().contiguous().reshape(-1).view(torch.uint8),
        )
    )


def _uncached_autocast(device: torch.device) -> torch.autocast:
    # Autocast's temporary weight cache may disappear after this request.
    # Capture casts inside the graph instead of retaining addresses owned by
    # that outer context. Dtype and enabled state remain exactly the caller's.
    return torch.autocast(
        device_type=device.type,
        dtype=torch.get_autocast_dtype(device.type),
        enabled=torch.is_autocast_enabled(device.type),
        cache_enabled=False,
    )


class Executable(Protocol):
    retained_bytes: int
    warmup_calls: int
    reference_output: Tensor | None

    def replay(
        self, args: Sequence[Tensor], kwargs: Mapping[str, object]
    ) -> Tensor: ...
    def close(self) -> None: ...


class Backend(Protocol):
    def capture(
        self, forward: Forward, args: Sequence[Tensor], kwargs: Mapping[str, object]
    ) -> Executable: ...


@dataclass
class _CudaExecutable:
    graph: torch.cuda.CUDAGraph
    args: tuple[Tensor, ...]
    kwargs: dict[str, object]
    output: Tensor
    device: torch.device
    retained_bytes: int
    warmup_calls: int
    reference_output: Tensor | None

    def replay(self, args: Sequence[Tensor], kwargs: Mapping[str, object]) -> Tensor:
        for destination, source in zip(self.args, args, strict=True):
            _copy(destination, source)
        for name, destination in self.kwargs.items():
            _copy(destination, kwargs[name])
        self.graph.replay()
        return self.output

    def close(self) -> None:
        # Releasing an executable must never invalidate an outstanding replay.
        torch.cuda.synchronize(self.device)
        self.graph.reset()
        self.args = ()
        self.kwargs.clear()
        self.output = torch.empty(0, device="cpu")
        self.reference_output = None


class CudaBackend:
    def __init__(self, device: torch.device, *, warmup_calls: int = 3) -> None:
        if device.type != "cuda":
            raise ValueError("CUDA graph backend requires a CUDA device")
        if type(warmup_calls) is not int or warmup_calls < 1:
            raise ValueError("CUDA graph warmup_calls must be positive")
        self.device = device
        self.warmup_calls = warmup_calls

    def capture(
        self, forward: Forward, args: Sequence[Tensor], kwargs: Mapping[str, object]
    ) -> Executable:
        device_index = (
            self.device.index
            if self.device.index is not None
            else torch.cuda.current_device()
        )
        if any(
            tensor.device.type != "cuda" or tensor.device.index != device_index
            for tensor in (*_tensors(args), *_tensors(tuple(kwargs.values())))
        ):
            raise CaptureUnavailable(
                "CUDA graph tensor arguments must be on the capture device"
            )
        graph = torch.cuda.CUDAGraph()
        static_args: tuple[Tensor, ...] = ()
        static_kwargs: dict[str, object] = {}
        output: Tensor | None = None
        reference: Tensor | None = None
        warmup_output: Tensor | None = None
        completed_warmups = 0
        try:
            static_args = tuple(
                value.detach().clone(memory_format=torch.preserve_format)
                for value in args
            )
            static_kwargs = {name: _clone(value) for name, value in kwargs.items()}
            stream = torch.cuda.Stream(device=self.device)
            stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(stream), _uncached_autocast(self.device):
                for index in range(self.warmup_calls):
                    warmup_output = forward(*static_args, **static_kwargs)
                    completed_warmups += 1
                    if index == self.warmup_calls - 1:
                        reference = warmup_output.detach().clone()
                    warmup_output = None
            stream.synchronize()
            with (
                _uncached_autocast(self.device),
                torch.cuda.graph(graph, stream=stream),
            ):
                output = forward(*static_args, **static_kwargs)
            if output.device.type != "cuda" or output.device.index != device_index:
                raise CaptureUnavailable(
                    "CUDA graph output must remain on the capture device"
                )
            try:
                snapshot = torch.cuda.memory_snapshot(include_traces=False)
            except TypeError:
                raise CaptureUnavailable(
                    "CUDA allocator does not support graph-pool accounting"
                ) from None
            owned = [
                *_tensors(static_args),
                *_tensors(tuple(static_kwargs.values())),
                output,
            ]
            retained = _account_graph_storage(
                snapshot,
                pool=(int(graph.pool()[0]), int(graph.pool()[1])),
                device_index=device_index,
                storages=[
                    (
                        tensor.untyped_storage().data_ptr(),
                        tensor.untyped_storage().nbytes(),
                    )
                    for tensor in owned
                ],
            )
            return _CudaExecutable(
                graph,
                static_args,
                static_kwargs,
                output,
                self.device,
                retained,
                self.warmup_calls,
                reference,
            )
        except BaseException as error:
            # A failed capture must be fully ended before eager inference is
            # safe. Cleanup failures propagate; they are not silent fallbacks.
            graph.reset()
            resource_failure = isinstance(error, torch.cuda.OutOfMemoryError)
            unsupported = isinstance(error, RuntimeError) and any(
                message in str(error).lower()
                for message in (
                    "operation not permitted when stream is capturing",
                    "operation not permitted on a capturing stream",
                    "cuda graph capture is not supported",
                )
            )
            if resource_failure or unsupported or isinstance(error, CaptureUnavailable):
                reason = (
                    "CUDA graph allocation exhausted memory"
                    if resource_failure
                    else str(error)
                )
                # Tracebacks can retain failed forward frames and their large
                # static buffers. Release those before attempting eager work.
                static_args = ()
                static_kwargs.clear()
                output = None
                reference = None
                warmup_output = None
                owned = []
                error.__traceback__ = None
                torch.cuda.synchronize(self.device)
                if resource_failure:
                    torch.cuda.empty_cache()
                raise CaptureUnavailable(
                    reason, warmup_calls=completed_warmups
                ) from None
            raise


class BoundedInferenceGraphs:
    """LRU graphs and bounded negative caching, called by one serialized owner."""

    def __init__(self, *, backend: Backend, max_entries: int, max_bytes: int) -> None:
        if type(max_entries) is not int or not 1 <= max_entries <= 64:
            raise ValueError("CUDA graph max_entries must be in [1, 64]")
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("CUDA graph max_bytes must be positive")
        self.backend = backend
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self._entries: OrderedDict[tuple[object, ...], Executable] = OrderedDict()
        self._pins: dict[tuple[object, ...], tuple[Tensor, ...]] = {}
        self._unsupported: OrderedDict[tuple[object, ...], str] = OrderedDict()
        self.retained_bytes = 0
        self.captures = 0
        self.replays = 0
        self.evictions = 0
        self.fallbacks = 0
        self.warmup_calls = 0
        self.warmup_rows = 0
        self.validation_failures = 0
        self.validation_replays = 0
        self._owner_thread: int | None = None

    def _evict(self) -> None:
        key, entry = self._entries.popitem(last=False)
        entry.close()
        self._pins.pop(key, None)
        self.retained_bytes -= entry.retained_bytes
        self.evictions += 1

    def _reject(self, key: tuple[object, ...], reason: str) -> None:
        logging.getLogger(__name__).warning("CUDA graph inference fallback: %s", reason)
        self._unsupported[key] = reason
        self._unsupported.move_to_end(key)
        while len(self._unsupported) > self.max_entries * 8:
            self._unsupported.popitem(last=False)

    def run(
        self,
        namespace: tuple[object, ...],
        forward: Forward,
        args: Sequence[Tensor],
        kwargs: Mapping[str, object],
        *,
        lifetime_pins: Sequence[Tensor] = (),
    ) -> Tensor:
        owner = threading.get_ident()
        if self._owner_thread is not None and owner != self._owner_thread:
            raise RuntimeError(
                "CUDA graph execution requires its single inference owner"
            )
        self._owner_thread = owner
        key = (
            namespace,
            tuple(_signature(value) for value in args),
            tuple((name, _signature(value)) for name, value in sorted(kwargs.items())),
        )
        if key in self._unsupported:
            self._unsupported.move_to_end(key)
            self.fallbacks += 1
            return forward(*args, **kwargs)
        entry = self._entries.get(key)
        if entry is None:
            while len(self._entries) >= self.max_entries:
                self._evict()
            try:
                entry = self.backend.capture(forward, args, kwargs)
            except CaptureUnavailable as error:
                self.warmup_calls += error.warmup_calls
                self.warmup_rows += error.warmup_calls * int(args[0].shape[0])
                self._reject(key, str(error))
                self.fallbacks += 1
                return forward(*args, **kwargs)
            self.captures += 1
            self.warmup_calls += entry.warmup_calls
            self.warmup_rows += entry.warmup_calls * int(args[0].shape[0])
            if entry.retained_bytes > self.max_bytes:
                entry.close()
                self._reject(key, "captured graph exceeds retained-memory budget")
                self.fallbacks += 1
                return forward(*args, **kwargs)
            try:
                reference = entry.reference_output
                if reference is None:
                    raise CaptureUnavailable("CUDA graph has no owned warmup reference")
                checked = entry.replay(args, kwargs)
                self.validation_replays += 1
                self.warmup_calls += 1
                self.warmup_rows += int(args[0].shape[0])
                if not _bitwise_equal(checked, reference):
                    raise CaptureUnavailable(
                        "CUDA graph failed bitwise output validation"
                    )
                entry.reference_output = None
                reference = None
                checked = None
            except (CaptureUnavailable, torch.cuda.OutOfMemoryError) as error:
                entry.close()
                reference = None
                checked = None
                reason = (
                    "CUDA graph validation exhausted memory"
                    if isinstance(error, torch.cuda.OutOfMemoryError)
                    else str(error)
                )
                error.__traceback__ = None
                if isinstance(error, torch.cuda.OutOfMemoryError) and isinstance(
                    self.backend, CudaBackend
                ):
                    torch.cuda.empty_cache()
                self.validation_failures += 1
                self._reject(key, reason)
                self.fallbacks += 1
                return forward(*args, **kwargs)
            except BaseException:
                entry.close()
                raise
            while (
                self._entries
                and self.retained_bytes + entry.retained_bytes > self.max_bytes
            ):
                self._evict()
            self._entries[key] = entry
            self._pins[key] = tuple(lifetime_pins)
            self.retained_bytes += entry.retained_bytes
        self._entries.move_to_end(key)
        output = entry.replay(args, kwargs)
        self.replays += 1
        return output

    def snapshot(self) -> dict[str, int]:
        return {
            "graph_captures": self.captures,
            "graph_replays": self.replays,
            "graph_evictions": self.evictions,
            "graph_fallbacks": self.fallbacks,
            "graph_warmup_calls": self.warmup_calls,
            "graph_warmup_rows": self.warmup_rows,
            "graph_validation_failures": self.validation_failures,
            "graph_validation_replays": self.validation_replays,
            "graph_entries": len(self._entries),
            "graph_retained_bytes": self.retained_bytes,
            "graph_negative_entries": len(self._unsupported),
        }

    def clear(self) -> None:
        while self._entries:
            self._evict()
        self._unsupported.clear()
        self._owner_thread = None
