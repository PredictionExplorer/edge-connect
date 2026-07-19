"""Host capability preflight: what was detected, what a config resolves to.

``startrain-preflight`` answers "will this profile run here, and how?"
without touching any run state. It reports detected accelerators, the
worker topology and device policy a config resolves to on this host, and
optionally proves each configured device with a tiny forward/backward pass.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, replace

import torch

from .config import ExperimentConfig, load_config
from .device import (
    capabilities_for,
    detect_accelerators,
    resolve_compile,
    resolve_device_string,
    resolve_loader_workers,
    resolve_pin_memory,
    resolve_precision,
)
from .model import GraphResTNet, ModelConfig

PREFLIGHT_SCHEMA_VERSION = 1


def _resolved_policy(experiment: ExperimentConfig, device: str) -> dict[str, object]:
    return {
        "device": device,
        "precision": resolve_precision(experiment.train.precision, device),
        "compile": resolve_compile(experiment.train.compile, device),
        "pin_memory": resolve_pin_memory(experiment.data.pin_memory, device),
        "loader_workers": resolve_loader_workers(experiment.data.workers),
        "capabilities": asdict(capabilities_for(device)),
    }


def _topology_report(experiment: ExperimentConfig) -> dict[str, object]:
    orchestration = experiment.orchestration
    if not orchestration.enabled:
        return {"enabled": False}
    worker_device = resolve_device_string(orchestration.device)
    promotion_device = resolve_device_string(orchestration.promotion.device)
    return {
        "enabled": True,
        "worker_device": worker_device,
        "workers": [
            {
                "gpu_id": gpu.gpu_id,
                "role": gpu.role,
                "cpu_threads": gpu.cpu_threads,
                "actor_batch_size": gpu.actor_batch_size,
                "actor_lanes": gpu.actor_lanes,
            }
            for gpu in orchestration.gpus
        ],
        "promotion": {
            "enabled": orchestration.promotion.enabled,
            "gpu_id": orchestration.promotion.gpu_id,
            "device": promotion_device,
            "pause_sharing_mode": orchestration.promotion.pause_sharing_mode,
        },
        "allow_colocated_workers": orchestration.allow_colocated_workers,
        "distributed": {
            "enabled": orchestration.distributed.enabled,
            "backend": orchestration.distributed.backend,
        },
        "hardware_health": {
            "require_gpu_model": orchestration.hardware_health.require_gpu_model,
        },
    }


def _exercise_device(device: str, *, precision: str) -> dict[str, object]:
    """Prove one tiny forward/backward pass on the requested device."""

    started = time.perf_counter()
    model_config = replace(
        ModelConfig(),
        width=32,
        rrt_groups=1,
        attention_heads=4,
        kv_heads=1,
    )
    try:
        model = GraphResTNet(model_config).to(device)
        batch, nodes, neighbors = 2, 8, 3
        node_features = torch.randn(
            batch, nodes, model_config.node_feature_dim, device=device
        )
        global_features = torch.randn(
            batch, model_config.global_feature_dim, device=device
        )
        neighbor_index = torch.zeros(
            batch, nodes, neighbors, dtype=torch.long, device=device
        )
        neighbor_mask = torch.ones(
            batch, nodes, neighbors, dtype=torch.bool, device=device
        )
        neighbor_edge_type = torch.zeros(
            batch, nodes, neighbors, dtype=torch.long, device=device
        )
        node_mask = torch.ones(batch, nodes, dtype=torch.bool, device=device)
        legal_action_mask = torch.ones(
            batch, nodes, dtype=torch.bool, device=device
        )
        autocast_enabled = precision == "bf16"
        with torch.autocast(
            device_type=torch.device(device).type,
            dtype=torch.bfloat16,
            enabled=autocast_enabled,
        ):
            output = model(
                node_features,
                global_features,
                neighbor_index,
                neighbor_mask,
                neighbor_edge_type,
                node_mask,
                legal_action_mask,
            )
            loss = (
                output.policy_logits.float().sum()
                + output.outcome_logits.float().sum()
            )
        loss.backward()
    except Exception as exc:  # fail-closed: any device error is the finding
        return {
            "device": device,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "device": device,
        "ok": True,
        "precision": precision,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def preflight_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Report detected hardware and a config's resolution on it"
    )
    parser.add_argument("--config", help="optional experiment YAML to resolve")
    parser.add_argument(
        "--exercise",
        action="store_true",
        help="run a tiny forward/backward on each resolved device",
    )
    arguments = parser.parse_args(argv)

    inventory = detect_accelerators()
    report: dict[str, object] = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "torch_version": torch.__version__,
        "detected": {
            **asdict(inventory),
            "preferred_device": inventory.preferred_device_type,
            "cuda_device_names": [
                torch.cuda.get_device_name(index)
                for index in range(inventory.cuda_device_count)
            ],
        },
    }
    devices: list[str] = []
    if arguments.config:
        experiment = load_config(arguments.config)
        learner_device = resolve_device_string(experiment.learner.device)
        report["config"] = str(arguments.config)
        report["learner"] = _resolved_policy(experiment, learner_device)
        report["orchestration"] = _topology_report(experiment)
        devices.append(learner_device)
        if experiment.orchestration.enabled:
            devices.append(resolve_device_string(experiment.orchestration.device))
            devices.append(
                resolve_device_string(experiment.orchestration.promotion.device)
            )
    else:
        devices.append(inventory.preferred_device_type)
    unique_devices = list(dict.fromkeys(devices))
    if arguments.exercise:
        report["exercise"] = [
            _exercise_device(
                device,
                precision=resolve_precision("auto", device),
            )
            for device in unique_devices
        ]
    print(json.dumps(report, sort_keys=True))
    exercised = report.get("exercise")
    if isinstance(exercised, list) and any(
        not entry.get("ok") for entry in exercised
    ):
        sys.exit(2)


if __name__ == "__main__":
    preflight_main()
