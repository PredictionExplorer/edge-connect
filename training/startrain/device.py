"""Device resolution, capability policy, and backend-dispatch helpers.

This module is the single source of truth for "what compute is available and
what does that imply". Everything here is host-deterministic: the same host
always resolves the same devices, precisions and topologies, so orchestrator
and worker processes independently reach identical decisions.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch

AUTO = "auto"
_KNOWN_DEVICE_TYPES = ("cuda", "mps", "cpu")


class DeviceResolutionError(ValueError):
    """A requested device cannot be used on this host."""


@dataclass(frozen=True, slots=True)
class AcceleratorInventory:
    """Deterministic snapshot of the host's usable compute."""

    cuda_device_count: int
    mps_available: bool
    cpu_count: int

    def __post_init__(self) -> None:
        if self.cuda_device_count < 0:
            raise ValueError("cuda_device_count must be non-negative")
        if self.cpu_count <= 0:
            raise ValueError("cpu_count must be positive")

    @property
    def preferred_device_type(self) -> str:
        if self.cuda_device_count > 0:
            return "cuda"
        if self.mps_available:
            return "mps"
        return "cpu"


def detect_accelerators() -> AcceleratorInventory:
    cuda_devices = torch.cuda.device_count() if torch.cuda.is_available() else 0
    mps = getattr(torch.backends, "mps", None)
    mps_available = bool(mps is not None and mps.is_available())
    return AcceleratorInventory(
        cuda_device_count=cuda_devices,
        mps_available=mps_available,
        cpu_count=os.cpu_count() or 1,
    )


def resolve_device_string(
    requested: str | None,
    *,
    inventory: AcceleratorInventory | None = None,
) -> str:
    """Return a concrete, validated device string for this host.

    ``None`` and ``"auto"`` select the best available backend
    (cuda > mps > cpu). Explicit requests are validated against the host and
    rejected with an actionable error instead of failing later inside torch.
    """

    hardware = detect_accelerators() if inventory is None else inventory
    if requested is None or requested == AUTO:
        return hardware.preferred_device_type
    if not isinstance(requested, str) or not requested:
        raise DeviceResolutionError("device must be a non-empty string or 'auto'")
    device_type = requested.split(":", 1)[0]
    if device_type not in _KNOWN_DEVICE_TYPES:
        raise DeviceResolutionError(
            f"unknown device {requested!r}; expected 'auto', 'cuda', 'cuda:N', "
            "'mps', or 'cpu'"
        )
    if device_type == "cuda":
        if hardware.cuda_device_count == 0:
            raise DeviceResolutionError(
                f"device {requested!r} was requested but CUDA is unavailable; "
                f"this host offers {hardware.preferred_device_type!r} "
                "(use --device auto to select it automatically)"
            )
        index = _device_index(requested)
        if index is not None and index >= hardware.cuda_device_count:
            raise DeviceResolutionError(
                f"device {requested!r} was requested but only "
                f"{hardware.cuda_device_count} CUDA device(s) are visible"
            )
    if device_type == "mps" and not hardware.mps_available:
        raise DeviceResolutionError(
            f"device {requested!r} was requested but Apple Metal (MPS) is "
            "unavailable; use 'cpu' or 'auto'"
        )
    return requested


def _device_index(device: str) -> int | None:
    if ":" not in device:
        return None
    _, _, raw = device.partition(":")
    if not raw.isdigit():
        raise DeviceResolutionError(f"device index in {device!r} must be an integer")
    return int(raw)


def normalize_device_string(
    requested: str | None,
    *,
    inventory: AcceleratorInventory | None = None,
) -> str:
    """Resolve ``auto`` without validating explicit devices against the host.

    Command construction (the orchestrator building worker argv) must not
    depend on the builder host's hardware; each worker validates its own
    device with :func:`resolve_device_string` before allocating.
    """

    if requested is None or requested == AUTO:
        hardware = detect_accelerators() if inventory is None else inventory
        return hardware.preferred_device_type
    if not isinstance(requested, str) or not requested:
        raise DeviceResolutionError("device must be a non-empty string or 'auto'")
    device_type = requested.split(":", 1)[0]
    if device_type not in _KNOWN_DEVICE_TYPES:
        raise DeviceResolutionError(
            f"unknown device {requested!r}; expected 'auto', 'cuda', 'cuda:N', "
            "'mps', or 'cpu'"
        )
    _device_index(requested)
    return requested


@dataclass(frozen=True, slots=True)
class DeviceCapabilities:
    device_type: str
    supports_bf16_autocast: bool
    supports_pin_memory: bool
    supports_cuda_streams: bool
    supports_compile: bool


def capabilities_for(device: torch.device | str) -> DeviceCapabilities:
    device_type = torch.device(device).type
    return DeviceCapabilities(
        device_type=device_type,
        # torch.autocast supports bf16 on CUDA and CPU; MPS autocast is
        # fp16-oriented and bf16 support is chip/OS dependent, so we refuse it.
        supports_bf16_autocast=device_type in ("cuda", "cpu"),
        supports_pin_memory=device_type == "cuda",
        supports_cuda_streams=device_type == "cuda",
        # Inductor is production-quality on CUDA only for this workload.
        supports_compile=device_type == "cuda",
    )


def resolve_precision(requested: str, device: torch.device | str) -> str:
    """Map a configured precision (possibly ``auto``) to a concrete one."""

    if requested not in ("fp32", "bf16", AUTO):
        raise DeviceResolutionError("precision must be fp32, bf16, or auto")
    capabilities = capabilities_for(device)
    if requested == AUTO:
        return "bf16" if capabilities.device_type == "cuda" else "fp32"
    if requested == "bf16" and not capabilities.supports_bf16_autocast:
        raise DeviceResolutionError(
            f"precision 'bf16' is unsupported on {capabilities.device_type!r} "
            "devices; set precision to 'auto' or 'fp32'"
        )
    return requested


def resolve_compile(requested: bool | str, device: torch.device | str) -> bool:
    """Map a configured compile setting (possibly ``auto``) to a boolean."""

    if isinstance(requested, bool):
        return requested
    if requested != AUTO:
        raise DeviceResolutionError("compile must be a boolean or 'auto'")
    return capabilities_for(device).supports_compile


def resolve_pin_memory(requested: bool, device: torch.device | str) -> bool:
    """Pin host batches only when a CUDA copy engine can benefit from it."""

    return bool(requested) and capabilities_for(device).supports_pin_memory


def resolve_loader_workers(requested: int) -> int:
    """Clamp configured DataLoader workers to the host CPU count."""

    if requested <= 0:
        return 0
    return min(requested, os.cpu_count() or 1)


def enable_fast_math(device: torch.device | str) -> None:
    """Allow TF32 tensor-core matmuls for the fp32 paths on CUDA hosts.

    The Muon optimizer runs fp32 matmuls every step; without TF32 those run
    far below the hardware's throughput on H100-class GPUs. bf16 autocast
    regions are unaffected.
    """

    if torch.device(device).type != "cuda":
        return
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def seed_all(seed: int) -> None:
    """Seed every available backend deterministically."""

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    mps = getattr(torch, "mps", None)
    if (
        mps is not None
        and getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
        and hasattr(mps, "manual_seed")
    ):
        mps.manual_seed(seed)


def synchronize_device(device: torch.device | str) -> None:
    target = torch.device(device)
    if target.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(target)
    elif target.type == "mps":
        torch.mps.synchronize()


def empty_device_cache(device: torch.device | str) -> None:
    target = torch.device(device)
    if target.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif target.type == "mps":
        torch.mps.empty_cache()


def device_memory_snapshot(device: torch.device | str) -> tuple[int, int]:
    """Return (allocated, reserved) bytes; zeros where unavailable."""

    target = torch.device(device)
    if target.type == "cuda" and torch.cuda.is_available():
        return (
            int(torch.cuda.memory_allocated(target)),
            int(torch.cuda.memory_reserved(target)),
        )
    if target.type == "mps":
        return (
            int(torch.mps.current_allocated_memory()),
            int(torch.mps.driver_allocated_memory()),
        )
    return 0, 0


def reset_peak_memory_stats(device: torch.device | str) -> None:
    target = torch.device(device)
    if target.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(target)


def peak_memory_stats(device: torch.device | str) -> tuple[int | None, int | None]:
    """Return (peak allocated, peak reserved) bytes, or Nones off CUDA."""

    target = torch.device(device)
    if target.type != "cuda" or not torch.cuda.is_available():
        return None, None
    return (
        int(torch.cuda.max_memory_allocated(target)),
        int(torch.cuda.max_memory_reserved(target)),
    )


def generate_auto_topology(
    inventory: AcceleratorInventory,
) -> dict[str, Any]:
    """Build an orchestration-section fragment matched to this host.

    Returns plain mappings so ``config.load_config`` can merge them under any
    explicit operator overrides before dataclass construction. Layout tiers:

    - 4+ CUDA GPUs: learner on GPU 0, dedicated arena on the last GPU,
      actors on everything between (the shipped H100 shape).
    - 2-3 CUDA GPUs: learner on GPU 0, actors on the rest, arena
      pause-sharing the last actor GPU.
    - 1 CUDA GPU: learner and actor colocated on GPU 0, arena pause-sharing.
    - MPS or CPU-only hosts: one learner slot, one actor slot, and a
      dedicated arena slot; slot ids are logical (no CUDA pinning applies).
    """

    gpus = inventory.cuda_device_count
    if gpus > 0:
        return _cuda_topology(inventory)
    device = "mps" if inventory.mps_available else "cpu"
    learner_threads, actor_threads, promotion_threads = _split_cpu_threads(
        inventory.cpu_count, actor_count=1
    )
    actor_batch = 16 if device == "mps" else 4
    return {
        "device": device,
        "gpus": (
            {"gpu_id": 0, "role": "learner", "cpu_threads": learner_threads},
            {
                "gpu_id": 1,
                "role": "actor",
                "cpu_threads": actor_threads,
                "actor_batch_size": actor_batch,
            },
        ),
        "actor_games_per_batch": actor_batch,
        "promotion": {
            "enabled": True,
            "gpu_id": 2,
            "cpu_threads": promotion_threads,
            "device": device,
            "pause_sharing_mode": False,
            "bootstrap_initial_champion": True,
        },
    }


def _cuda_topology(inventory: AcceleratorInventory) -> dict[str, Any]:
    gpus = inventory.cuda_device_count
    actor_batch = 64
    if gpus == 1:
        # The learner shares the only GPU with one actor; halve the actor's
        # inference batch to leave headroom for the training step.
        actor_batch = 32
        learner_gpu, actor_gpus, arena_gpu, pause_sharing = 0, (0,), 0, True
        colocated = True
    elif gpus == 2:
        learner_gpu, actor_gpus, arena_gpu, pause_sharing = 0, (1,), 1, True
        colocated = False
    elif gpus == 3:
        learner_gpu, actor_gpus, arena_gpu, pause_sharing = 0, (1, 2), 2, True
        colocated = False
    else:
        learner_gpu = 0
        actor_gpus = tuple(range(1, gpus - 1))
        arena_gpu = gpus - 1
        pause_sharing = False
        colocated = False
    learner_threads, actor_threads, promotion_threads = _split_cpu_threads(
        inventory.cpu_count, actor_count=len(actor_gpus)
    )
    worker_entries: list[dict[str, Any]] = [
        {"gpu_id": learner_gpu, "role": "learner", "cpu_threads": learner_threads}
    ]
    for gpu_id in actor_gpus:
        worker_entries.append(
            {
                "gpu_id": gpu_id,
                "role": "actor",
                "cpu_threads": actor_threads,
                "actor_batch_size": actor_batch,
            }
        )
    fragment: dict[str, Any] = {
        "device": "cuda",
        "gpus": tuple(worker_entries),
        "actor_games_per_batch": actor_batch,
        "promotion": {
            "enabled": True,
            "gpu_id": arena_gpu,
            "cpu_threads": promotion_threads,
            "device": "cuda",
            "pause_sharing_mode": pause_sharing,
            "bootstrap_initial_champion": True,
        },
    }
    if colocated:
        fragment["allow_colocated_workers"] = True
    return fragment


def _split_cpu_threads(cpu_count: int, *, actor_count: int) -> tuple[int, int, int]:
    """Split host CPUs into (learner, per-actor, promotion) budgets."""

    if actor_count <= 0:
        raise ValueError("actor_count must be positive")
    learner = max(2, min(16, cpu_count // 4))
    promotion = max(1, min(8, cpu_count // 8))
    remaining = max(actor_count, cpu_count - learner - promotion)
    per_actor = max(1, min(8, remaining // actor_count))
    return learner, per_actor, promotion
