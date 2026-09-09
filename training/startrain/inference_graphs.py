"""Bounded, single-owner CUDA graphs for immutable inference adapters.

Returned tensors are borrowed graph outputs and must be consumed before the
next call. The inference adapter copies predictions to owned CPU bytes while
still holding its evaluation lock. Limits cover retained graph storage;
capture can temporarily require additional warmup/workspace memory.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import importlib
import logging
import threading
import traceback
from typing import Any, Callable, Mapping, Protocol, Sequence

import torch
from torch import Tensor


Forward = Callable[..., Tensor]


class CaptureUnavailable(RuntimeError):
    """A recoverable capture limitation, after capture resources are cleaned up."""

    def __init__(self, message: str, *, warmup_calls: int = 0) -> None:
        super().__init__(message)
        self.warmup_calls = warmup_calls


def _clear_completed_error_frames(error: BaseException) -> None:
    """Drop retained graph/context locals while preserving traceback locations."""

    pending = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        traceback.clear_frames(current.__traceback__)
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)


@dataclass
class _OwnedCaptureStream:
    stream: torch.cuda.Stream
    destroy: Callable[[], None] | None = None

    def close(self) -> None:
        if self.destroy is not None:
            self.destroy()
            self.destroy = None


def _new_owned_capture_stream(device_index: int) -> _OwnedCaptureStream:
    """Allocate outside PyTorch's pool using NVIDIA's documented runtime API."""

    try:
        runtime = importlib.import_module("cuda.bindings.runtime")
    except ImportError:
        raise CaptureUnavailable(
            "owned CUDA capture streams require cuda.bindings.runtime"
        ) from None
    if not all(
        hasattr(runtime, name)
        for name in (
            "cudaStreamCreateWithFlags",
            "cudaStreamDestroy",
            "cudaStreamNonBlocking",
        )
    ):
        raise CaptureUnavailable(
            "CUDA runtime bindings do not expose owned nonblocking streams"
        )
    with torch.cuda.device(device_index):
        status, raw_stream = runtime.cudaStreamCreateWithFlags(
            runtime.cudaStreamNonBlocking
        )
        if int(status) != 0:
            if int(status) in (
                2,
                801,
            ):  # cudaErrorMemoryAllocation / cudaErrorNotSupported
                raise CaptureUnavailable(
                    f"owned CUDA stream allocation unavailable: {status}"
                )
            raise RuntimeError(f"owned CUDA stream allocation failed: {status}")

    def destroy() -> None:
        with torch.cuda.device(device_index):
            (status,) = runtime.cudaStreamDestroy(raw_stream)
        if int(status) != 0:
            raise RuntimeError(f"owned CUDA stream destruction failed: {status}")

    try:
        wrapped = torch.cuda.ExternalStream(int(raw_stream), device=device_index)
    except BaseException:
        destroy()
        raise
    return _OwnedCaptureStream(wrapped, destroy)


@dataclass
class _CaptureStreamLease:
    pool: "_CaptureStreamPool"
    device_index: int
    stream: torch.cuda.Stream
    released: bool = False

    def release(self) -> None:
        self.pool.release(self)


class _CaptureStreamPool:
    """Exclusive live graph streams shared by every adapter in this process.

    Production streams are owned CUDA-runtime allocations, not PyTorch pooled
    handles: unrelated compiler/autotune captures cannot alias them. PyTorch
    2.13 reset clears stream-keyed cuBLAS workspaces (pytorch/pytorch#193402).
    Live leases are bounded by the configured graph caches; only a bounded
    number of released streams are retained, and clear destroys idle handles.
    """

    def __init__(
        self, *, max_streams_per_device: int | None = None, max_idle_per_device: int = 8
    ) -> None:
        if max_streams_per_device is not None and (
            type(max_streams_per_device) is not int or max_streams_per_device <= 0
        ):
            raise ValueError("capture stream pool bound must be positive")
        if type(max_idle_per_device) is not int or not 0 <= max_idle_per_device <= 64:
            raise ValueError("capture stream idle bound must be in [0, 64]")
        self.max_streams_per_device = max_streams_per_device
        self.max_idle_per_device = max_idle_per_device
        self._lock = threading.Lock()
        self._known: dict[tuple[int, int], _OwnedCaptureStream] = {}
        self._leased: dict[tuple[int, int], _CaptureStreamLease] = {}
        self._idle: dict[int, list[torch.cuda.Stream]] = {}

    def acquire(
        self,
        device_index: int,
        factory: Callable[[], torch.cuda.Stream | _OwnedCaptureStream],
    ) -> _CaptureStreamLease:
        with self._lock:
            idle = self._idle.setdefault(device_index, [])
            if idle:
                stream = idle.pop()
            else:
                count = sum(key[0] == device_index for key in self._known)
                if (
                    self.max_streams_per_device is not None
                    and count >= self.max_streams_per_device
                ):
                    raise CaptureUnavailable(
                        "all CUDA capture streams are exclusively leased"
                    )
                stream = None
                for _ in range(self.max_streams_per_device or 1):
                    created = factory()
                    resource = (
                        created
                        if isinstance(created, _OwnedCaptureStream)
                        else _OwnedCaptureStream(created)
                    )
                    candidate = resource.stream
                    key = (device_index, int(candidate.cuda_stream))
                    if key not in self._known:
                        self._known[key] = resource
                        stream = candidate
                        break
                    if isinstance(created, _OwnedCaptureStream):
                        raise RuntimeError(
                            "CUDA runtime returned an already owned capture stream"
                        )
                if stream is None:
                    raise CaptureUnavailable(
                        "CUDA stream pool returned only live leased streams"
                    )
            key = (device_index, int(stream.cuda_stream))
            if key in self._leased:
                raise RuntimeError("capture stream lease would alias a live graph")
            lease = _CaptureStreamLease(self, device_index, stream)
            self._leased[key] = lease
            return lease

    def release(self, lease: _CaptureStreamLease) -> None:
        with self._lock:
            if lease.released:
                return
            key = (lease.device_index, int(lease.stream.cuda_stream))
            if self._leased.get(key) is not lease:
                raise RuntimeError("capture stream lease does not own this stream")
            idle = self._idle.setdefault(lease.device_index, [])
            if len(idle) >= self.max_idle_per_device:
                # On destruction failure the live lease remains quarantined.
                self._known[key].close()
                del self._known[key]
            else:
                idle.append(lease.stream)
            del self._leased[key]
            lease.released = True

    def release_idle(self, device_index: int) -> None:
        with self._lock:
            idle = self._idle.setdefault(device_index, [])
            while idle:
                stream = idle.pop()
                key = (device_index, int(stream.cuda_stream))
                # Remove from the reusable queue first. Failed destruction
                # keeps the resource known but quarantined until process exit.
                self._known[key].close()
                del self._known[key]


_CAPTURE_STREAM_POOL = _CaptureStreamPool()


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
    """Return the conservative retained charge used by admission and probes."""

    private, external = _graph_storage_breakdown(
        snapshot, pool=pool, device_index=device_index, storages=storages
    )
    return private + external


def _graph_storage_breakdown(
    snapshot: Sequence[Mapping[str, Any]],
    *,
    pool: tuple[int, int],
    device_index: int,
    storages: Sequence[tuple[int, int]],
) -> tuple[int, int]:
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
    return private, sum(outside.values())


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
    graph: torch.cuda.CUDAGraph | None
    args: tuple[Tensor, ...]
    kwargs: dict[str, object]
    output: Tensor
    device: torch.device
    retained_bytes: int
    warmup_calls: int
    reference_output: Tensor | None
    stream_lease: _CaptureStreamLease | None = None
    private_pool_bytes: int | None = None
    external_static_bytes: int | None = None

    def replay(self, args: Sequence[Tensor], kwargs: Mapping[str, object]) -> Tensor:
        if self.graph is None:
            raise RuntimeError("cannot replay a closed CUDA graph")
        for destination, source in zip(self.args, args, strict=True):
            _copy(destination, source)
        for name, destination in self.kwargs.items():
            _copy(destination, kwargs[name])
        self.graph.replay()
        return self.output

    def close(self) -> None:
        if self.graph is None:
            return
        # Releasing an executable must never invalidate an outstanding replay.
        torch.cuda.synchronize(self.device)
        self.graph.reset()
        # Destroy the wrapper before making this stream available to a new
        # graph; delayed destruction must not touch another graph's workspace.
        self.graph = None
        self.args = ()
        self.kwargs.clear()
        self.output = torch.empty(0, device="cpu")
        self.reference_output = None
        lease, self.stream_lease = self.stream_lease, None
        if lease is not None:
            lease.release()


class CudaBackend:
    def __init__(
        self,
        device: torch.device,
        *,
        warmup_calls: int = 3,
        stream_pool: _CaptureStreamPool | None = None,
    ) -> None:
        if device.type != "cuda":
            raise ValueError("CUDA graph backend requires a CUDA device")
        if type(warmup_calls) is not int or warmup_calls < 1:
            raise ValueError("CUDA graph warmup_calls must be positive")
        self.device = device
        self.warmup_calls = warmup_calls
        self.stream_pool = (
            stream_pool if stream_pool is not None else _CAPTURE_STREAM_POOL
        )

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
        lease = self.stream_pool.acquire(
            device_index, lambda: _new_owned_capture_stream(device_index)
        )
        stream = lease.stream
        graph: torch.cuda.CUDAGraph | None = None
        static_args: tuple[Tensor, ...] = ()
        static_kwargs: dict[str, object] = {}
        output: Tensor | None = None
        reference: Tensor | None = None
        warmup_output: Tensor | None = None
        completed_warmups = 0
        try:
            graph = torch.cuda.CUDAGraph()
            static_args = tuple(
                value.detach().clone(memory_format=torch.preserve_format)
                for value in args
            )
            static_kwargs = {name: _clone(value) for name, value in kwargs.items()}
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
            private_bytes, external_bytes = _graph_storage_breakdown(
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
                private_bytes + external_bytes,
                self.warmup_calls,
                reference,
                lease,
                private_bytes,
                external_bytes,
            )
        except BaseException as error:
            # A failed capture must be fully ended before eager inference is
            # safe. Cleanup failures propagate; they are not silent fallbacks.
            # In particular, graph.__exit__ may itself raise and its completed
            # traceback frame can own the graph wrapper. Destroy that ownership
            # before making the external stream available to another graph.
            _clear_completed_error_frames(error)
            if graph is not None:
                graph.reset()
                graph = None
            # Warmup may have failed after submitting work. Do not reuse its
            # stream until that work is done. A poisoned CUDA context keeps
            # the lease reserved and propagates the synchronization failure.
            stream.synchronize()
            static_args = ()
            static_kwargs.clear()
            output = None
            reference = None
            warmup_output = None
            owned = []
            lease.release()
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
                torch.cuda.synchronize(self.device)
                if resource_failure:
                    torch.cuda.empty_cache()
                raise CaptureUnavailable(
                    reason, warmup_calls=completed_warmups
                ) from None
            raise

    def release_idle_streams(self) -> None:
        device_index = (
            self.device.index
            if self.device.index is not None
            else torch.cuda.current_device()
        )
        self.stream_pool.release_idle(device_index)


@dataclass(frozen=True, slots=True)
class GraphResidency:
    """Capture-time metadata only; no tensors, executable or allocator access."""

    rows: int
    nodes: int
    key_sha256: str
    kwarg_names: tuple[str, ...]
    charged_bytes: int
    private_pool_bytes: int | None
    external_static_bytes: int | None


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
        self._residency: dict[tuple[object, ...], GraphResidency] = {}
        self._residency_lock = threading.Lock()
        self._published_residency: tuple[GraphResidency, ...] = ()
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
        self._residency.pop(key, None)
        self.retained_bytes -= entry.retained_bytes
        self.evictions += 1
        self._publish_residency()

    def _publish_residency(self) -> None:
        # Only the inference owner (or its quiescent cleanup) reads the mutable
        # inventory. Heartbeats receive an independently immutable tuple.
        records = tuple(
            sorted(
                self._residency.values(),
                key=lambda record: (record.rows, record.nodes, record.key_sha256),
            )
        )
        with self._residency_lock:
            self._published_residency = records

    def residency_snapshot(self) -> tuple[GraphResidency, ...]:
        """Read cached CPU metadata without GPU locks or live-cache iteration."""

        with self._residency_lock:
            return self._published_residency

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
                _clear_completed_error_frames(error)
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
            except BaseException as error:
                _clear_completed_error_frames(error)
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
            self._residency[key] = GraphResidency(
                rows=int(args[0].shape[0]),
                nodes=int(args[0].shape[1]) if args[0].ndim > 1 else 0,
                key_sha256=hashlib.sha256(repr(key).encode("utf-8")).hexdigest(),
                kwarg_names=tuple(sorted(kwargs)),
                charged_bytes=entry.retained_bytes,
                private_pool_bytes=getattr(entry, "private_pool_bytes", None),
                external_static_bytes=getattr(entry, "external_static_bytes", None),
            )
            self._publish_residency()
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
        release_idle = getattr(self.backend, "release_idle_streams", None)
        if callable(release_idle):
            release_idle()
