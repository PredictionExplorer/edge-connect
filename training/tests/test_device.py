from __future__ import annotations

import pytest
import torch

from startrain.device import (
    AcceleratorInventory,
    DeviceResolutionError,
    capabilities_for,
    detect_accelerators,
    generate_auto_topology,
    normalize_device_string,
    resolve_compile,
    resolve_device_string,
    resolve_loader_workers,
    resolve_pin_memory,
    resolve_precision,
    seed_all,
)

CUDA_8 = AcceleratorInventory(cuda_device_count=8, mps_available=False, cpu_count=208)
CUDA_4 = AcceleratorInventory(cuda_device_count=4, mps_available=False, cpu_count=64)
CUDA_2 = AcceleratorInventory(cuda_device_count=2, mps_available=False, cpu_count=32)
CUDA_1 = AcceleratorInventory(cuda_device_count=1, mps_available=False, cpu_count=16)
MPS = AcceleratorInventory(cuda_device_count=0, mps_available=True, cpu_count=10)
CPU = AcceleratorInventory(cuda_device_count=0, mps_available=False, cpu_count=8)


def test_auto_prefers_cuda_then_mps_then_cpu() -> None:
    assert resolve_device_string("auto", inventory=CUDA_8) == "cuda"
    assert resolve_device_string("auto", inventory=MPS) == "mps"
    assert resolve_device_string("auto", inventory=CPU) == "cpu"
    assert resolve_device_string(None, inventory=CPU) == "cpu"


def test_explicit_devices_are_validated_against_the_host() -> None:
    assert resolve_device_string("cuda", inventory=CUDA_1) == "cuda"
    assert resolve_device_string("cuda:0", inventory=CUDA_1) == "cuda:0"
    with pytest.raises(DeviceResolutionError, match="CUDA is unavailable"):
        resolve_device_string("cuda", inventory=MPS)
    with pytest.raises(DeviceResolutionError, match="only 1 CUDA device"):
        resolve_device_string("cuda:1", inventory=CUDA_1)
    with pytest.raises(DeviceResolutionError, match="MPS.*unavailable"):
        resolve_device_string("mps", inventory=CPU)
    with pytest.raises(DeviceResolutionError, match="unknown device"):
        resolve_device_string("tpu", inventory=CPU)
    with pytest.raises(DeviceResolutionError, match="non-empty"):
        resolve_device_string("", inventory=CPU)


def test_normalize_passes_explicit_devices_without_host_validation() -> None:
    assert normalize_device_string("cuda", inventory=CPU) == "cuda"
    assert normalize_device_string("auto", inventory=MPS) == "mps"
    with pytest.raises(DeviceResolutionError, match="unknown device"):
        normalize_device_string("tpu", inventory=CPU)


def test_precision_policy_per_device() -> None:
    assert resolve_precision("auto", "cuda") == "bf16"
    assert resolve_precision("auto", "mps") == "fp32"
    assert resolve_precision("auto", "cpu") == "fp32"
    assert resolve_precision("bf16", "cpu") == "bf16"
    assert resolve_precision("fp32", "cuda") == "fp32"
    with pytest.raises(DeviceResolutionError, match="bf16.*unsupported"):
        resolve_precision("bf16", "mps")
    with pytest.raises(DeviceResolutionError, match="precision"):
        resolve_precision("fp16", "cuda")


def test_compile_and_pin_memory_policy_per_device() -> None:
    assert resolve_compile("auto", "cuda") is True
    assert resolve_compile("auto", "mps") is False
    assert resolve_compile("auto", "cpu") is False
    assert resolve_compile(True, "cpu") is True
    assert resolve_compile(False, "cuda") is False
    assert resolve_pin_memory(True, "cuda") is True
    assert resolve_pin_memory(True, "mps") is False
    assert resolve_pin_memory(True, "cpu") is False
    assert resolve_pin_memory(False, "cuda") is False


def test_loader_workers_clamped_to_host_cpus() -> None:
    assert resolve_loader_workers(0) == 0
    assert resolve_loader_workers(-3) == 0
    assert resolve_loader_workers(2) == 2
    assert resolve_loader_workers(100_000) <= 100_000


def test_capabilities_reflect_device_type() -> None:
    cuda = capabilities_for("cuda:3")
    assert cuda.supports_bf16_autocast
    assert cuda.supports_pin_memory
    assert cuda.supports_cuda_streams
    assert cuda.supports_compile
    mps = capabilities_for("mps")
    assert not mps.supports_bf16_autocast
    assert not mps.supports_pin_memory
    cpu = capabilities_for("cpu")
    assert cpu.supports_bf16_autocast
    assert not cpu.supports_compile


def test_detect_accelerators_matches_torch() -> None:
    inventory = detect_accelerators()
    expected_cuda = torch.cuda.device_count() if torch.cuda.is_available() else 0
    assert inventory.cuda_device_count == expected_cuda
    assert inventory.cpu_count >= 1


def test_seed_all_is_safe_on_any_host() -> None:
    seed_all(17)


def _topology_summary(
    inventory: AcceleratorInventory,
) -> tuple[str, list[tuple[int, str]], int, bool, bool]:
    fragment = generate_auto_topology(inventory)
    workers = [(entry["gpu_id"], entry["role"]) for entry in fragment["gpus"]]
    promotion = fragment["promotion"]
    return (
        fragment["device"],
        workers,
        promotion["gpu_id"],
        promotion["pause_sharing_mode"],
        bool(fragment.get("allow_colocated_workers", False)),
    )


def test_auto_topology_eight_gpus_matches_shipped_shape() -> None:
    device, workers, arena_gpu, pause_sharing, colocated = _topology_summary(CUDA_8)
    assert device == "cuda"
    assert workers == [(0, "learner")] + [(gpu, "actor") for gpu in range(1, 7)]
    assert arena_gpu == 7
    assert not pause_sharing
    assert not colocated


def test_auto_topology_four_gpus() -> None:
    device, workers, arena_gpu, pause_sharing, colocated = _topology_summary(CUDA_4)
    assert device == "cuda"
    assert workers == [(0, "learner"), (1, "actor"), (2, "actor")]
    assert arena_gpu == 3
    assert not pause_sharing
    assert not colocated


def test_auto_topology_two_gpus_pause_shares_arena() -> None:
    device, workers, arena_gpu, pause_sharing, colocated = _topology_summary(CUDA_2)
    assert device == "cuda"
    assert workers == [(0, "learner"), (1, "actor")]
    assert arena_gpu == 1
    assert pause_sharing
    assert not colocated


def test_auto_topology_single_gpu_colocates_learner_and_actor() -> None:
    device, workers, arena_gpu, pause_sharing, colocated = _topology_summary(CUDA_1)
    assert device == "cuda"
    assert workers == [(0, "learner"), (0, "actor")]
    assert arena_gpu == 0
    assert pause_sharing
    assert colocated


@pytest.mark.parametrize("inventory,device", [(MPS, "mps"), (CPU, "cpu")])
def test_auto_topology_non_cuda_hosts_use_logical_slots(
    inventory: AcceleratorInventory, device: str
) -> None:
    fragment = generate_auto_topology(inventory)
    assert fragment["device"] == device
    assert [entry["role"] for entry in fragment["gpus"]] == ["learner", "actor"]
    promotion = fragment["promotion"]
    assert promotion["device"] == device
    assert promotion["bootstrap_initial_champion"] is True
    assert not promotion["pause_sharing_mode"]


def test_auto_topology_thread_budgets_are_positive_and_bounded() -> None:
    for inventory in (CUDA_8, CUDA_4, CUDA_2, CUDA_1, MPS, CPU):
        fragment = generate_auto_topology(inventory)
        for entry in fragment["gpus"]:
            assert 1 <= entry["cpu_threads"] <= inventory.cpu_count
        assert 1 <= fragment["promotion"]["cpu_threads"] <= inventory.cpu_count
        for entry in fragment["gpus"]:
            if entry["role"] == "actor":
                assert entry["actor_batch_size"] <= fragment["actor_games_per_batch"]
