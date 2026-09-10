#!/usr/bin/env python3
"""Bounded, offline H100 shared-geometry accuracy and training-step benchmark.

Without --execute, print a plan without importing Torch or touching CUDA.
Example during an isolated maintenance window (never stops other processes):

    PYTHONPATH=. .venv/bin/python scripts/benchmark_training_shared_geometry.py \
      --config configs/h100-8gpu-largest-board-priority.yaml \
      --checkpoint /readonly/checkpoint.pt --device cuda:2 --execute --compile \
      --rings 6 10 --accuracy-batch-size 16 --timing-batch-sizes 128 512 \
      --load-context isolated --output /tmp/shared-geometry.json

Fixtures are deterministic synthetic semantic positions, not an Elo evaluation.
Checkpoint weights use EMA, as in actor benchmarks; optimizer state starts fresh.
Each subprocess has a private compile cache, allocator cap and hard deadline.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, cast


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, required=True)
    result.add_argument("--checkpoint", type=Path)
    result.add_argument("--device", default="cuda:2")
    result.add_argument("--rings", type=int, nargs="+", default=[6, 10])
    result.add_argument("--accuracy-batch-size", type=int, default=16)
    result.add_argument("--timing-batch-sizes", type=int, nargs="*", default=[128, 512])
    result.add_argument("--warmups", type=int, default=2)
    result.add_argument("--repeats", type=int, default=5)
    result.add_argument("--seed", type=int, default=19091)
    result.add_argument("--max-memory-gib", type=float, default=68.0)
    result.add_argument("--case-timeout-seconds", type=float, default=540.0)
    result.add_argument("--timeout-seconds", type=float, default=540.0)
    result.add_argument(
        "--cutover",
        action="store_true",
        help="One ring, one reusable compiled model, total deadline at most600s",
    )
    result.add_argument(
        "--load-context",
        choices=["isolated", "shared", "unverified"],
        default="isolated",
    )
    result.add_argument("--compile", action="store_true", dest="compile_model")
    result.add_argument("--execute", action="store_true")
    result.add_argument("--output", type=Path)
    result.add_argument("--child-case", help=argparse.SUPPRESS)
    result.add_argument("--child-progress", type=Path, help=argparse.SUPPRESS)
    return result


def validate(args: argparse.Namespace) -> None:
    if args.cutover and (
        len(args.rings) != 1 or not args.compile_model or args.timeout_seconds > 600
    ):
        raise ValueError(
            "cutover requires one ring, --compile and total deadline<=600s"
        )
    if (
        not args.rings
        or len(set(args.rings)) != len(args.rings)
        or any(r not in (4, 6, 8, 10) for r in args.rings)
    ):
        raise ValueError("rings must be unique values from4,6,8,10")
    if not 1 <= args.accuracy_batch_size <= 32:
        raise ValueError("accuracy batch size must be1..32")
    if len(set(args.timing_batch_sizes)) != len(args.timing_batch_sizes) or any(
        not 1 <= b <= 512 for b in args.timing_batch_sizes
    ):
        raise ValueError("timing batch sizes must be unique values1..512")
    if not 1 <= args.warmups <= 10 or not 2 <= args.repeats <= 30:
        raise ValueError("warmups must be1..10 and repeats2..30")
    if not math.isfinite(args.max_memory_gib) or not 1 <= args.max_memory_gib <= 76:
        raise ValueError("memory cap must be1..76GiB (also capped at95%ofdevice)")
    if (
        not 1 <= args.case_timeout_seconds <= 900
        or not 1 <= args.timeout_seconds <= 3600
    ):
        raise ValueError("case timeout must be1..900s and total timeout1..3600s")


def gpu_ownership(gpu_uuid: str, *, own_pid: int | None = None) -> dict[str, Any]:
    """Read-only process inventory. Failure is unverified, never isolated."""

    own_pid = os.getpid() if own_pid is None else own_pid
    try:
        process = subprocess.run(
            [
                "nvidia-smi",
                "--id",
                gpu_uuid,
                "--query-compute-apps=gpu_uuid,pid",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if process.returncode:
            raise RuntimeError(process.stderr.strip())
        owners = []
        for line in process.stdout.splitlines():
            if not line.strip():
                continue
            uuid, pid = (part.strip() for part in line.split(",", 1))
            if uuid != gpu_uuid:
                raise ValueError("process inventory returned another GPU")
            owners.append(int(pid))
        return {
            "verified": set(owners) == {own_pid},
            "gpu_uuid": gpu_uuid,
            "owner_pids": sorted(set(owners)),
            "observed_ns": time.time_ns(),
        }
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        return {"verified": False, "error": str(error), "observed_ns": time.time_ns()}


def compare_tensors(
    candidate: dict[str, Any], reference: dict[str, Any]
) -> dict[str, Any]:
    """Global L2/cosine and worst tensor error; excludes no trainable tensors."""

    import torch

    if candidate.keys() != reference.keys() or not candidate:
        raise ValueError("tensor names differ or are empty")
    totals = [0.0, 0.0, 0.0, 0.0]
    maximum = 0.0
    worst = (0.0, "")
    finite = True
    for name, expected in reference.items():
        actual = candidate[name]
        if actual.shape != expected.shape:
            raise ValueError(f"tensor shape differs: {name}")
        a, b = actual.detach().cpu().double(), expected.detach().cpu().double()
        finite = finite and bool(torch.isfinite(a).all() and torch.isfinite(b).all())
        if not finite:
            return {
                "finite": False,
                "relative_l2": None,
                "cosine": None,
                "max_absolute": None,
                "worst_tensor": name,
            }
        delta = float((a - b).square().sum())
        aa, bb, ab = (
            float(a.square().sum()),
            float(b.square().sum()),
            float((a * b).sum()),
        )
        totals = [totals[0] + delta, totals[1] + aa, totals[2] + bb, totals[3] + ab]
        maximum = max(maximum, float((a - b).abs().max()))
        # Report tiny gradients, but avoid dividing an exact-zero reference by0.
        relative = math.sqrt(delta) / max(math.sqrt(bb), 1e-12)
        if relative > worst[0]:
            worst = relative, name
    delta, aa, bb, ab = totals
    cosine = ab / math.sqrt(aa * bb) if aa and bb else (1.0 if aa == bb else 0.0)
    return {
        "finite": True,
        "relative_l2": math.sqrt(delta) / max(math.sqrt(bb), 1e-12),
        "cosine": max(-1.0, min(1.0, cosine)),
        "max_absolute": maximum,
        "reference_l2": math.sqrt(bb),
        "worst_tensor_relative_l2": worst[0],
        "worst_tensor": worst[1],
    }


def numerical_gate(comparisons: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Conservative execution gate, not a training-strength acceptance test.

    FP32 tolerates changed reduction order. BF16 permits its expected rounding
    noise but checks bias separately so large trunk gradients cannot hide a bad
    bias VJP. CPU tiny-model bias differences reached3.4%; cap at5%, and also
    require both BF16 paths to track the FP32 oracle within10%.
    """

    expected_names = {
        f"{label}_{kind}"
        for label in ("fp32_shared", "bf16_shared", "bf16_oracle", "bf16_shared_oracle")
        for kind in ("outputs", "gradients", "bias_gradients")
    }
    failures = sorted(expected_names - comparisons.keys())
    for name, values in comparisons.items():
        fp32 = name.startswith("fp32_shared")
        oracle = "oracle" in name
        output = name.endswith("outputs")
        relative_limit = (
            (1e-5 if output else 1e-3) if fp32 else (0.10 if oracle else 0.05)
        )
        cosine_limit = 0.99999 if fp32 else (0.99 if oracle else 0.998)
        if (
            not values.get("finite")
            or values["relative_l2"] > relative_limit
            or values["cosine"] < cosine_limit
        ):
            failures.append(name)
    return {
        "passed": not failures,
        "failures": failures,
        "fp32_output_relative_l2_limit": 1e-5,
        "fp32_gradient_relative_l2_limit": 1e-3,
        "bf16_pair_relative_l2_limit": 0.05,
        "bf16_oracle_relative_l2_limit": 0.10,
        "bf16_pair_min_cosine": 0.998,
        "note": "Global gradients and relation-bias gradients are gated separately; worst-tensor errors are reported. Numerical pass alone does not authorize activation.",
    }


def make_batch(ring: int, rows: int, seed: int) -> Any:
    import numpy as np
    import torch
    from startrain.features import DoubleStarPosition
    from startrain.replay import ReplaySample, collate_replay_samples
    from startrain.scoring import score_position
    from startrain.topology import get_topology

    topology = get_topology(ring)
    random = np.random.default_rng(seed + ring)
    samples = []
    for row in range(rows):
        stones = np.full(topology.n, -1, dtype=np.int8)
        occupied = 1 + row % max(2, topology.n * 3 // 4)
        chosen = random.permutation(topology.n)[:occupied]
        stones[chosen] = random.integers(0, 2, size=occupied, dtype=np.int8)
        mode = "classic" if row % 2 == 0 else "double"
        position = DoubleStarPosition.from_sequence(
            rings=ring,
            stones=stones,
            to_move=row % 2,
            moves_left=1 if mode == "classic" else 2,
            opening=False,
            terminal=False,
            mode=mode,
            handicap=2 if row % 6 in (2, 3) else 1,
            pie=row % 6 in (4, 5),
            history_known=False,
        )
        legal = stones == -1
        policy = random.random(topology.n).astype(np.float32) * legal
        policy /= policy.sum()
        completed = stones.copy()
        completed[legal] = row % 2
        score = score_position(topology, torch.from_numpy(completed))
        if score.leader == -1:
            # A full-board deterministic auxiliary fixture must have an outcome.
            score = score_position(topology, torch.zeros(topology.n, dtype=torch.int8))
        samples.append(
            ReplaySample.from_position(
                position,
                policy=policy,
                final_score=score,
                search_provenance="synthetic:shared-geometry-benchmark",
                policy_provenance="synthetic:random-legal",
                game_id=f"fixture-{row}",
            )
        )
    batch = collate_replay_samples(samples)
    if batch.homogeneous_ring != ring:
        raise RuntimeError("CPU geometry certification failed")
    return batch


def _capture(
    model: Any, batch: Any, config: Any, *, shared: bool, precision: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch
    from startrain.losses import compute_losses
    from startrain.training import unwrap_model

    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16"):
        output = model(
            *batch.inputs.model_args(),
            **({"homogeneous_ring": batch.homogeneous_ring} if shared else {}),
        )
        losses = compute_losses(
            output,
            batch.targets,
            legal_action_mask=batch.inputs.legal_action_mask,
            node_mask=batch.inputs.node_mask,
            weights=config.loss,
        )
    losses["total"].backward()
    outputs = {}
    for name, tensor in output._asdict().items():
        if "policy" in name:
            tensor = tensor[batch.inputs.legal_action_mask]
        elif name in ("ownership_logits", "alive_logits"):
            tensor = tensor[batch.inputs.node_mask]
        outputs[name] = tensor.detach().float().cpu()
    gradients = {
        name: parameter.grad.detach().float().cpu()
        for name, parameter in unwrap_model(model).named_parameters()
        if parameter.grad is not None
    }
    return outputs, gradients


def _execution(args: argparse.Namespace, cpu_model: Any) -> tuple[Any, Any]:
    from copy import deepcopy
    from startrain.training import maybe_compile_model

    configure_math(oracle=False, device=args.device)
    model = deepcopy(cpu_model).to(args.device).train()
    runner = maybe_compile_model(
        model, enabled=args.compile_model, **compile_settings(args)
    )
    return model, runner


def compile_settings(args: argparse.Namespace) -> dict[str, Any]:
    # Production needs four entries for one arm across four rings. This
    # benchmark needs baseline/shared across accuracy and requested timing sizes.
    shapes = set([args.accuracy_batch_size, *args.timing_batch_sizes])
    return {
        "dynamic": False,
        "fullgraph": True,
        "isolate_recompiles": True,
        "recompile_limit": max(4, 2 * len(shapes)),
    }


def math_settings() -> dict[str, Any]:
    import torch

    return {
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
    }


def configure_math(*, oracle: bool, device: str) -> dict[str, Any]:
    import torch
    from startrain.device import enable_fast_math

    if oracle:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:
        enable_fast_math(device)
    return math_settings()


def production_execution_matches(result: dict[str, Any]) -> bool:
    compilation = result.get("compile_settings", {})
    math = result.get("measured_math", {})
    return (
        compilation.get("dynamic") is False
        and compilation.get("isolate_recompiles") is True
        and compilation.get("fullgraph") is True
        and math.get("float32_matmul_precision") == "high"
        and math.get("cuda_matmul_allow_tf32") is True
        and math.get("cudnn_allow_tf32") is True
    )


def _accuracy(
    args: argparse.Namespace,
    cpu_model: Any,
    config: Any,
    batch: Any,
    execution: tuple[Any, Any] | None = None,
) -> dict[str, Any]:
    import torch
    from startrain.optim import build_optimizer

    captures = {}
    finite_steps = {}
    capture_math = {}
    model, compiled = execution or _execution(args, cpu_model)
    for label, shared, precision in (
        ("fp32", False, "fp32"),
        ("fp32_shared", True, "fp32"),
        ("bf16", False, "bf16"),
        ("bf16_shared", True, "bf16"),
    ):
        capture_math[label] = configure_math(
            oracle=precision == "fp32", device=args.device
        )
        model.load_state_dict(cpu_model.state_dict())
        runner = compiled if precision == "bf16" else model
        captures[label] = _capture(
            runner, batch, config, shared=shared, precision=precision
        )
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.train.gradient_clip_norm, error_if_nonfinite=True
        )
        optimizer = build_optimizer(model, config.optimizer)
        optimizer.step()
        finite_steps[label] = bool(
            torch.stack([torch.isfinite(p).all() for p in model.parameters()]).all()
        )
        if not finite_steps[label]:
            raise FloatingPointError(f"nonfinite optimizer step: {label}")
        del optimizer
        model.zero_grad(set_to_none=True)
    comparisons = {}
    for label, reference in (
        ("fp32_shared", "fp32"),
        ("bf16_shared", "bf16"),
        ("bf16_oracle", "fp32"),
        ("bf16_shared_oracle", "fp32"),
    ):
        actual = label.removesuffix("_oracle")
        for index, kind in enumerate(("outputs", "gradients")):
            comparisons[f"{label}_{kind}"] = compare_tensors(
                captures[actual][index], captures[reference][index]
            )
        comparisons[f"{label}_bias_gradients"] = compare_tensors(
            {n: t for n, t in captures[actual][1].items() if "relation_bias" in n},
            {n: t for n, t in captures[reference][1].items() if "relation_bias" in n},
        )
    return {
        "comparisons": comparisons,
        "numerical_gate": numerical_gate(comparisons),
        "finite_optimizer_steps": finite_steps,
        "math_by_capture": capture_math,
    }


def _timing(
    args: argparse.Namespace,
    cpu_model: Any,
    config: Any,
    batch: Any,
    shared: bool,
    execution: tuple[Any, Any] | None = None,
) -> dict[str, Any]:
    import torch
    from startrain.optim import build_optimizer
    from startrain.training import train_step

    measured_math = configure_math(oracle=False, device=args.device)
    model, runner = execution or _execution(args, cpu_model)
    model.load_state_dict(cpu_model.state_dict())
    optimizer = build_optimizer(model, config.optimizer)

    def step() -> None:
        train_step(
            cast(torch.nn.Module, runner),
            batch,
            optimizer,
            precision="bf16",
            loss_weights=config.loss,
            gradient_clip_norm=config.train.gradient_clip_norm,
            trusted_batch=True,
            share_homogeneous_geometry=shared,
        )

    start = time.perf_counter()
    for _ in range(args.warmups):
        step()
    torch.cuda.synchronize(args.device)
    warmup_seconds = time.perf_counter() - start
    warmup_peak = torch.cuda.max_memory_allocated(args.device)
    torch.cuda.reset_peak_memory_stats(args.device)
    elapsed = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        step()
        torch.cuda.synchronize(args.device)
        elapsed.append(time.perf_counter() - start)
    finite = bool(
        torch.stack([torch.isfinite(p).all() for p in model.parameters()]).all()
    )
    if not finite:
        raise FloatingPointError("timing produced nonfinite model parameters")
    median = statistics.median(elapsed)
    return {
        "warmup_seconds": warmup_seconds,
        "measured_math": measured_math,
        "warmup_peak_allocated_bytes": warmup_peak,
        "step_seconds": elapsed,
        "median_step_seconds": median,
        "rows_per_second": batch.inputs.batch_size / median,
        "finite_parameters": finite,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(args.device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(args.device),
        "scope": "resident-batch forward/backward/production optimizer; excludes loader, EMA and checkpoint I/O",
    }


def _cutover(
    args: argparse.Namespace, cpu_model: Any, config: Any, batch: Any, ring: int
) -> dict[str, Any]:
    import torch

    execution = _execution(args, cpu_model)
    result = _accuracy(args, cpu_model, config, batch, execution)
    result["timing_results"] = []
    _save(result, args.child_progress)
    if not result["numerical_gate"]["passed"]:
        return result
    for size in args.timing_batch_sizes:
        timing_batch = make_batch(ring, size, args.seed).to(args.device)
        for shared in (False, True):
            record = _timing(args, cpu_model, config, timing_batch, shared, execution)
            result["timing_results"].append(
                record
                | {
                    "kind": "timing",
                    "ring": ring,
                    "batch_size": size,
                    "shared": shared,
                    "status": "ok",
                }
            )
            _save(result, args.child_progress)
        execution[0].zero_grad(set_to_none=True)
        del timing_batch
        gc.collect()
        torch.cuda.empty_cache()
    return result


def load_weights(
    source: Path, model: Any, config: Any
) -> tuple[dict[str, Any], dict[str, Any]]:
    from startrain.checkpoint import (
        load_ema_checkpoint,
        load_model_manifest,
        sha256_file,
    )

    manifest = load_model_manifest(source) if source.suffix.lower() == ".json" else None
    checkpoint = manifest.checkpoint if manifest else source
    digest = manifest.checkpoint_sha256 if manifest else sha256_file(checkpoint)
    metadata = load_ema_checkpoint(
        checkpoint,
        model=model,
        map_location="cpu",
        expected_sha256=digest,
        expected_bytes=manifest.checkpoint_bytes if manifest else None,
        expected_model_config=asdict(config.model),
        expected_game_config=asdict(config.game),
        expected_run_id=manifest.run_id if manifest else None,
        expected_generation_family=manifest.generation_family if manifest else None,
    )
    if manifest is not None and int(metadata["step"]) != manifest.model_step:
        raise ValueError("manifest and checkpoint step disagree")
    return metadata, {
        "checkpoint_sha256": digest,
        "checkpoint_path": str(checkpoint),
        "manifest_sha256": manifest.manifest_sha256 if manifest else None,
        "model_identity": manifest.model_identity if manifest else None,
    }


def _child(args: argparse.Namespace, case: dict[str, Any]) -> dict[str, Any]:
    import torch
    from startrain.checkpoint import sha256_file
    from startrain.config import load_config
    from startrain.model import GraphResTNet

    started = time.monotonic()
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise ValueError("execution requires CUDA")
    torch.set_num_threads(2)
    torch.cuda.set_device(args.device)
    # Full FP32 oracle only. Measured BF16 paths enable production fast math.
    oracle_math = configure_math(oracle=True, device=args.device)
    torch.manual_seed(args.seed)
    # Create our own CUDA context before asking nvidia-smi to identify its PID.
    free, total = torch.cuda.mem_get_info(args.device)
    properties = torch.cuda.get_device_properties(args.device)
    uuid = str(properties.uuid)
    if not uuid.startswith("GPU-"):
        uuid = "GPU-" + uuid
    before = gpu_ownership(uuid)
    if args.load_context == "isolated" and not before["verified"]:
        raise RuntimeError(f"isolated GPU preflight failed: {before}")
    cap = min(int(args.max_memory_gib * 1024**3), int(properties.total_memory * 0.95))
    torch.cuda.set_per_process_memory_fraction(
        cap / properties.total_memory, args.device
    )
    config = load_config(args.config)
    cpu_model = GraphResTNet(config.model)
    if not config.model.relational_bias:
        raise ValueError("benchmark requires trainable relational bias")
    metadata = None
    identity: dict[str, Any] = {"checkpoint_sha256": None}
    if args.checkpoint:
        metadata, identity = load_weights(args.checkpoint, cpu_model, config)
    batch = make_batch(case["ring"], case["batch_size"], args.seed).to(args.device)
    observations = [before]
    stopped = threading.Event()

    def inspect_ownership() -> None:
        while not stopped.wait(2):
            observations.append(gpu_ownership(uuid))

    monitor = threading.Thread(target=inspect_ownership, daemon=True)
    monitor.start()
    try:
        if case["kind"] == "cutover":
            measured = _cutover(args, cpu_model, config, batch, case["ring"])
        elif case["kind"] == "accuracy":
            measured = _accuracy(args, cpu_model, config, batch)
        else:
            measured = _timing(args, cpu_model, config, batch, case["shared"])
    finally:
        stopped.set()
        monitor.join(timeout=6)
    after = gpu_ownership(uuid)
    isolated = (
        args.load_context == "isolated"
        and all(o["verified"] for o in observations)
        and after["verified"]
    )
    return (
        case
        | measured
        | {
            "status": "ok",
            "elapsed_seconds": time.monotonic() - started,
            "gpu": properties.name,
            "gpu_uuid": uuid,
            "total_memory_bytes": total,
            "preflight_free_bytes": free,
            "allocator_cap_bytes": cap,
            "ownership_before": before,
            "ownership_after": after,
            "ownership_samples": observations,
            "isolation_verified": isolated,
            "load_context": args.load_context,
            "parameter_count": cpu_model.parameter_count(),
            "model_config": asdict(config.model),
            **identity,
            "config_sha256": sha256_file(args.config),
            "seed": args.seed,
            "checkpoint_step": metadata["step"] if metadata else None,
            "weights": "ema" if args.checkpoint else "seeded-random",
            "fixture": "synthetic semantic states with random legal policy and synthetic completed-board auxiliary targets",
            "optimizer": asdict(config.optimizer),
            "optimizer_state": "fresh",
            "torch_version": torch.__version__,
            "compile": args.compile_model,
            "compile_settings": compile_settings(args),
            "oracle_math": oracle_math,
            "measured_math": math_settings(),
        }
    )


def _run_process(
    command: list[str], *, env: dict[str, str], timeout: float
) -> subprocess.CompletedProcess[str]:
    process = subprocess.Popen(
        command,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    except BaseException:
        # Only this benchmark's own process group, including compile workers.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.communicate()
        raise


def _save(report: dict[str, Any], output: Path | None) -> None:
    if output:
        temporary = output.with_name(output.name + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(output)


def main(argv: list[str] | None = None) -> int:
    argument_parser = parser()
    args = argument_parser.parse_args(argv)
    try:
        validate(args)
    except ValueError as error:
        argument_parser.error(str(error))
    if args.child_case:
        print(json.dumps(_child(args, json.loads(args.child_case)), allow_nan=False))
        return 0
    cases = [
        {"kind": "accuracy", "ring": ring, "batch_size": args.accuracy_batch_size}
        for ring in args.rings
    ]
    cases += [
        {"kind": "timing", "ring": ring, "batch_size": size, "shared": shared}
        for ring in args.rings
        for size in args.timing_batch_sizes
        for shared in (False, True)
    ]
    if args.cutover:
        cases = [
            {
                "kind": "cutover",
                "ring": args.rings[0],
                "batch_size": args.accuracy_batch_size,
            }
        ]
    report: dict[str, Any] = {
        "benchmark": "training-shared-homogeneous-geometry",
        "schema_version": 1,
        "cases": cases,
        "results": [],
        "adoptable": False,
        "execute": args.execute,
        "load_context": args.load_context,
        "limitation": "Synthetic resident-batch execution benchmark; no Elo or end-to-end learner throughput claim.",
    }
    if not args.execute:
        print(json.dumps(report, indent=2))
        return 0
    deadline = time.monotonic() + args.timeout_seconds
    if args.output and (not args.output.parent.is_dir() or args.output.exists()):
        argument_parser.error("output parent must exist and output must be new")
    with tempfile.TemporaryDirectory(prefix="startrain-shared-geometry-") as directory:
        for index, case in enumerate(cases):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                report["results"].append(case | {"status": "total-timeout"})
                break
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--config",
                str(args.config.resolve()),
                "--device",
                args.device,
                "--seed",
                str(args.seed),
                "--accuracy-batch-size",
                str(args.accuracy_batch_size),
                "--warmups",
                str(args.warmups),
                "--repeats",
                str(args.repeats),
                "--max-memory-gib",
                str(args.max_memory_gib),
                "--load-context",
                args.load_context,
                "--timing-batch-sizes",
                *[str(size) for size in args.timing_batch_sizes],
                "--child-case",
                json.dumps(case),
                "--child-progress",
                str(Path(directory) / f"progress-{index}.json"),
            ]
            if args.checkpoint:
                command += ["--checkpoint", str(args.checkpoint.resolve())]
            if args.compile_model:
                command.append("--compile")
            cache = Path(directory) / str(index)
            env = dict(
                os.environ,
                OMP_NUM_THREADS="2",
                MKL_NUM_THREADS="2",
                OPENBLAS_NUM_THREADS="2",
                TORCHINDUCTOR_CACHE_DIR=str(cache / "inductor"),
                TORCHINDUCTOR_PERSISTENT_AUTOTUNE_DIR=str(cache / "autotune"),
                TORCHINDUCTOR_COMPILE_THREADS="2",
                TRITON_HOME=str(cache / "triton-home"),
                TRITON_CACHE_DIR=str(cache / "triton"),
                TRITON_DUMP_DIR=str(cache / "triton-dump"),
                TRITON_OVERRIDE_DIR=str(cache / "triton-override"),
                XDG_CACHE_HOME=str(cache / "xdg"),
                CUDA_CACHE_PATH=str(cache / "cuda"),
            )
            result: dict[str, Any]
            try:
                process = _run_process(
                    command,
                    env=env,
                    timeout=max(0.01, min(remaining, args.case_timeout_seconds) - 5),
                )
                if process.returncode:
                    result = case | {
                        "status": "failed",
                        "returncode": process.returncode,
                        "stderr": process.stderr[-12000:],
                    }
                else:
                    result = json.loads(process.stdout.strip().splitlines()[-1])
            except subprocess.TimeoutExpired:
                result = case | {"status": "case-timeout"}
            progress = Path(directory) / f"progress-{index}.json"
            if result["status"] != "ok" and progress.exists():
                result = json.loads(progress.read_text()) | result
            report["results"].append(result)
            _save(report, args.output)
            print(
                json.dumps({"case": case, "status": result["status"]}),
                file=sys.stderr,
                flush=True,
            )
            if result["status"] != "ok" or (
                case["kind"] != "timing" and not result["numerical_gate"]["passed"]
            ):
                break
    results = report["results"]
    complete = len(results) == len(cases) and all(r["status"] == "ok" for r in results)
    accurate = complete and all(
        r.get("numerical_gate", {"passed": True})["passed"] for r in results
    )
    report["execution_gate_passed"] = accurate
    report["comparisons"] = []
    timing_results = [r for r in results if r["kind"] == "timing"]
    timing_results += [
        timing for result in results for timing in result.get("timing_results", [])
    ]
    for ring in args.rings:
        for size in args.timing_batch_sizes:
            pair = [
                r
                for r in timing_results
                if r["kind"] == "timing"
                and r["ring"] == ring
                and r["batch_size"] == size
                and r["status"] == "ok"
            ]
            if len(pair) == 2:
                base, shared = sorted(pair, key=lambda r: r["shared"])
                report["comparisons"].append(
                    {
                        "ring": ring,
                        "batch_size": size,
                        "speedup": base["median_step_seconds"]
                        / shared["median_step_seconds"],
                        "peak_allocated_reduction_bytes": base["peak_allocated_bytes"]
                        - shared["peak_allocated_bytes"],
                    }
                )
    report["adoptable"] = bool(
        accurate
        and args.load_context == "isolated"
        and args.checkpoint
        and args.compile_model
        and args.timing_batch_sizes
        and len(report["comparisons"]) == len(args.rings) * len(args.timing_batch_sizes)
        and len({r["checkpoint_sha256"] for r in results}) == 1
        and len({r["config_sha256"] for r in results}) == 1
        and all(r["parameter_count"] == 17_402_775 for r in results)
        and all(production_execution_matches(r) for r in results)
        and all(r["isolation_verified"] for r in results)
        and all(c["speedup"] >= 1.05 for c in report["comparisons"])
    )
    report["adoption_note"] = (
        "Eligibility for controlled activation only: requires checkpoint, compiled isolated complete run, numerical gate and>=5% warm-step improvement in every requested timing case; does not establish Elo or long-run stability."
    )
    _save(report, args.output)
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if accurate else 1


if __name__ == "__main__":
    raise SystemExit(main())
