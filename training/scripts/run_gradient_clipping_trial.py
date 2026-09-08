#!/usr/bin/env python3
"""Isolated, full-state-restored gradient clipping trials on pinned replay.

Prepare while production runs:
  --prepare-only --config PROFILE --recovery recovery.json --replay-root REPLAY
  --output-dir NEW_EXPERIMENT --train-samples-per-cell 2048
  --validation-samples-per-cell 512

After isolating the selected GPU:
  --replay-manifest NEW_EXPERIMENT/frozen-replay.json --output-dir NEW_ARM
  --arms global --steps 600 --device cuda:1 --cpu-affinity 32-63
  --timeout-seconds 1800

Use --diagnostic-only for backward passes without changing model, optimizer,
scheduler or EMA. Loss comparisons are diagnostic; playing strength requires arena
evaluation. Trial checkpoints never update production pointers or replay.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
from typing import Any

FORMAT = "startrain.gradient-clipping-trial-replay"
RINGS = (4, 6, 8, 10)
MODES = (
    "classic-standard",
    "double-standard",
    "classic-pie",
    "double-pie",
    "classic-handicap",
    "double-handicap",
)
CELLS = tuple(f"r{ring}/{mode}" for ring in RINGS for mode in MODES)


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _helpers():
    if __package__:
        from . import run_frozen_replay_optimizer_calibration as helpers
    else:
        import run_frozen_replay_optimizer_calibration as helpers
    return helpers


def _control():
    if __package__:
        from . import benchmark_actor_throughput as control
    else:
        import benchmark_actor_throughput as control
    return control


def _disjoint_output(output: Path, sources: list[Path]) -> Path:
    target = output.resolve()
    for source in sources:
        source = source.resolve()
        if target == source or target in source.parents or source in target.parents:
            raise ValueError(f"output overlaps protected source: {source}")
    if output.exists():
        raise ValueError("trial output must be a new directory")
    target.mkdir(parents=True, exist_ok=False)
    return target


def _copy_pin(
    source: Path, target: Path, expected: str | None = None
) -> dict[str, object]:
    from startrain.checkpoint import sha256_file

    if source.is_symlink() or not source.is_file():
        raise ValueError(f"unsafe source artifact: {source}")
    checksum = sha256_file(source)
    if expected is not None and checksum != expected:
        raise ValueError(f"source artifact checksum mismatch: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    if sha256_file(target) != checksum:
        raise ValueError(f"artifact changed during copy: {source}")
    target.chmod(0o444)
    return {
        "path": str(target.resolve()),
        "sha256": checksum,
        "bytes": target.stat().st_size,
    }


def prepare(args) -> dict[str, object]:
    from startrain.checkpoint import load_recovery_pointer, sha256_file
    from startrain.config import load_config
    from startrain.contracts import FEATURE_SCHEMA_HASH, RULES_HASH
    from startrain.replay import decode_replay_shard
    from startrain.selfplay import GameVariant

    config = load_config(args.config)
    _validate_production_config(config)
    raw = json.loads(args.recovery.read_text())
    recovery = load_recovery_pointer(
        args.recovery,
        expected_run_id=raw["run_id"],
        expected_generation_family=raw["generation_family"],
    )
    protected_run = (
        args.replay_root.parent
        if args.replay_root.name == "replay"
        else args.replay_root
    )
    output = _disjoint_output(
        args.output_dir, [protected_run, args.config, args.recovery.parent]
    )
    pinned_config = _copy_pin(args.config.resolve(), output / "profile.yaml")
    pinned_recovery = _copy_pin(
        recovery.checkpoint, output / "source-recovery.pt", recovery.checkpoint_sha256
    )
    helpers = _helpers()
    with helpers.open_replay_read_only(args.replay_root) as connection:
        helpers._replay_metadata(connection)
        cutoff = args.replay_cutoff or int(
            connection.execute("SELECT MAX(id) FROM shards").fetchone()[0]
        )
        rows = connection.execute(
            "SELECT * FROM shards WHERE state='ready' AND id<=? AND run_id=? "
            "AND generation_family=? AND rules_hash=? AND feature_schema_hash=? ORDER BY id DESC",
            (
                cutoff,
                recovery.run_id,
                recovery.generation_family,
                f"{RULES_HASH:016x}",
                f"{FEATURE_SCHEMA_HASH:016x}",
            ),
        ).fetchall()
    partitions: dict[str, dict[str, list]] = {
        cell: {"train": [], "validation": []} for cell in CELLS
    }
    pinned_shards = []
    for row in rows:
        variant = GameVariant.parse(str(row["variant"]))
        mode = f"{variant.mode}-{'pie' if variant.pie else 'handicap' if variant.handicap > 1 else 'standard'}"
        cell = f"r{int(row['ring'])}/{mode}"
        if cell not in partitions:
            continue
        selected = partitions[cell]
        if (
            len(selected["train"]) >= args.train_samples_per_cell
            and len(selected["validation"]) >= args.validation_samples_per_cell
        ):
            continue
        shard = helpers._frozen_shard(args.replay_root.resolve(), row)
        if sha256_file(shard.path) != shard.checksum_sha256:
            raise ValueError("replay shard failed integrity verification")
        decoded = decode_replay_shard(shard.path)
        if len(decoded) != shard.sample_count:
            raise ValueError("replay shard sample count differs from ledger")
        added = False
        references = [
            helpers._stable_reference(shard, index, args.seed)
            for index in range(len(decoded))
        ]
        references.sort(key=lambda ref: ref.order_sha256)
        for ref in references:
            game = str(decoded.arrays["game_id"][ref.sample_index])
            validation = (
                int(digest(["holdout-game-v1", args.seed, game])[:16], 16) % 5 == 0
            )
            split = "validation" if validation else "train"
            limit = (
                args.validation_samples_per_cell
                if validation
                else args.train_samples_per_cell
            )
            if len(selected[split]) >= limit:
                continue
            sample = decoded.sample(ref.sample_index)
            if (
                sample.rings != int(row["ring"])
                or sample.variant_label != variant.label
                or sample.run_id != recovery.run_id
                or sample.generation_family != recovery.generation_family
                or sample.model_identity != shard.model_identity
            ):
                raise ValueError("replay cell metadata differs from sample")
            selected[split].append(ref.as_hash_record() | {"game_id": game})
            added = True
        if added:
            pin = _copy_pin(
                shard.path,
                output / "shards" / f"shard-{shard.shard_id}.npz",
                shard.checksum_sha256,
            )
            pinned_shards.append(shard.hash_record() | pin)
    missing = {
        cell: {split: len(refs) for split, refs in partitions[cell].items()}
        for cell in CELLS
        if len(partitions[cell]["train"]) < args.train_samples_per_cell
        or len(partitions[cell]["validation"]) < args.validation_samples_per_cell
    }
    if missing:
        raise ValueError(
            f"insufficient frozen replay for all board/mode cells: {missing}"
        )
    for values in partitions.values():
        for refs in values.values():
            refs.sort(key=lambda ref: digest([args.seed, ref["stable_id"]]))
    train_games = {
        ref["game_id"] for values in partitions.values() for ref in values["train"]
    }
    validation_games = {
        ref["game_id"] for values in partitions.values() for ref in values["validation"]
    }
    if train_games & validation_games:
        raise ValueError("training and validation game identities overlap")
    document = {
        "format": FORMAT,
        "schema_version": 1,
        "seed": args.seed,
        "source_replay_root": str(args.replay_root.resolve()),
        "cutoff": cutoff,
        "run_id": recovery.run_id,
        "generation_family": recovery.generation_family,
        "source_step": recovery.step,
        "source_epoch": recovery.epoch,
        "config": pinned_config,
        "recovery": pinned_recovery,
        "shards": pinned_shards,
        "cells": partitions,
        "split_method": "game-hash-20-percent-holdout-v1",
        "schedule": "120-step-block-85-percent-r10-equal-six-modes-v1",
    }
    path = output / "frozen-replay.json"
    path.write_text(json.dumps(document, indent=2))
    path.chmod(0o444)
    return {
        "status": "prepared",
        "manifest": str(path),
        "manifest_sha256": sha256_file(path),
        "source_step": recovery.step,
        "shards": len(pinned_shards),
        "cells": len(partitions),
    }


def _validate_production_config(config) -> None:
    from startrain.model import model_parameter_count

    if model_parameter_count(config.model) != 17_402_775:
        raise ValueError("trial requires the unchanged 17,402,775-parameter model")
    if config.model.dropout != 0 or config.train.precision != "bf16":
        raise ValueError("trial requires production BF16 and zero dropout")
    weights = config.orchestration.ring_mixture.weights_for_step(10**12)
    if tuple(config.orchestration.ring_mixture.rings) != RINGS or tuple(
        weights or ()
    ) != (0.05, 0.05, 0.05, 0.85):
        raise ValueError("trial requires the 85/5 largest-board allocation")


def cell_schedule(*, seed: int, steps: int) -> list[str]:
    result = []
    while len(result) < steps:
        block = [
            cell for cell in CELLS for _ in range(17 if cell.startswith("r10/") else 1)
        ]
        random.Random(seed + len(result) // 120).shuffle(block)
        result.extend(block)
    return result[:steps]


def load_frozen(path: Path):
    from startrain.checkpoint import verify_file
    from startrain.replay import decode_replay_shard

    helpers = _helpers()
    document = json.loads(path.read_text())
    if document.get("format") != FORMAT or document.get("schema_version") != 1:
        raise ValueError("invalid frozen clipping trial manifest")
    if set(document["cells"]) != set(CELLS):
        raise ValueError("frozen replay must cover all24 board/mode cells")
    for pin in (document["config"], document["recovery"], *document["shards"]):
        artifact = Path(pin["path"])
        if (
            artifact.is_symlink()
            or path.resolve().parent not in artifact.resolve().parents
        ):
            raise ValueError("frozen artifact escapes the experiment directory")
        verify_file(
            artifact, expected_sha256=pin["sha256"], expected_bytes=pin["bytes"]
        )
    shards = {
        record["id"]: helpers.FrozenShard(
            record["id"],
            Path(record["path"]),
            record["relative_path"],
            record["sample_count"],
            record["checksum_sha256"],
            record["model_step"],
            record["model_identity"],
        )
        for record in document["shards"]
    }
    decoded = {key: decode_replay_shard(shard.path) for key, shard in shards.items()}
    cells = {}
    games = {"train": set(), "validation": set()}
    for cell, values in document["cells"].items():
        cells[cell] = {}
        for split, records in values.items():
            if split not in games or not records:
                raise ValueError("invalid or empty frozen partition")
            references = []
            for record in records:
                shard = shards[record["shard_id"]]
                ref = helpers._stable_reference(
                    shard, record["sample_index"], document["seed"]
                )
                sample = decoded[shard.shard_id].sample(ref.sample_index)
                mode = f"{sample.mode}-{'pie' if sample.pie else 'handicap' if sample.handicap > 1 else 'standard'}"
                if (
                    ref.stable_id != record["stable_id"]
                    or sample.game_id != record["game_id"]
                    or cell != f"r{sample.rings}/{mode}"
                ):
                    raise ValueError("frozen reference identity or cell mismatch")
                games[split].add(sample.game_id)
                references.append(ref)
            cells[cell][split] = tuple(references)
    if games["train"] & games["validation"]:
        raise ValueError("frozen validation leaks training games")
    replay = helpers.FrozenReplay(
        document["cutoff"],
        digest(document["shards"]),
        tuple(shards.values()),
        (),
        (),
        "",
        "",
        digest(document["cells"]),
        decoded,
    )
    return document, replay, cells


def state_fingerprint(value: Any) -> str:
    import torch

    h = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            h.update(json.dumps([str(item.dtype), list(item.shape)]).encode())
            h.update(
                item.detach()
                .contiguous()
                .reshape(-1)
                .view(torch.uint8)
                .cpu()
                .numpy()
                .tobytes()
            )
        elif isinstance(item, dict):
            for key in sorted(item, key=lambda key: (str(type(key)), str(key))):
                h.update(repr(key).encode())
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            h.update(str(type(item)).encode())
            for part in item:
                visit(part)
        else:
            h.update(repr(item).encode())

    visit(value)
    return h.hexdigest()


def rng_fingerprints(device) -> dict[str, str]:
    import numpy as np
    import torch

    numpy_state: Any = np.random.get_state()
    result = {
        "python": state_fingerprint(random.getstate()),
        "numpy": state_fingerprint(
            [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]]
        ),
        "torch_cpu": state_fingerprint(torch.get_rng_state()),
    }
    if device.type == "cuda":
        result["cuda_selected"] = state_fingerprint(torch.cuda.get_rng_state(device))
    return result


def restore_training_state(config, document, *, device, arm: str):
    from startrain.checkpoint import ExponentialMovingAverage, load_checkpoint
    from startrain.gradient_clipping import GradientClipper
    from startrain.model import GraphResTNet
    from startrain.optim import build_optimizer
    from startrain.training import build_scheduler

    model = GraphResTNet(config.model).to(device)
    optimizer = build_optimizer(model, config.optimizer)
    scheduler = build_scheduler(optimizer, config.train.scheduler)
    ema = ExponentialMovingAverage(model, decay=config.train.resolved_ema_decay(1))
    clipper = (
        GradientClipper(
            model.named_parameters(),
            config=replace(config.train.gradient_clipping, mode="adagc"),
            max_norm=config.train.gradient_clip_norm,
        )
        if arm == "adagc"
        else None
    )

    def validate(metadata):
        source = metadata["config"]
        for field in ("model", "game", "optimizer", "loss"):
            if digest(source[field]) != digest(config.as_dict()[field]):
                raise ValueError(f"recovery {field} differs from trial configuration")
        for field in (
            "per_rank_batch_size",
            "precision",
            "compile",
            "ema_decay",
            "ema_half_life_examples",
            "gradient_clip_norm",
            "scheduler",
        ):
            if digest(source["train"].get(field)) != digest(
                config.as_dict()["train"].get(field)
            ):
                raise ValueError(f"recovery training setting differs: {field}")
        if metadata["step"] != document["source_step"]:
            raise ValueError("recovery step differs from frozen pin")

    metadata = load_checkpoint(
        document["recovery"]["path"],
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema,
        map_location=device,
        use_ema_weights=False,
        require_ema=True,
        expected_model_config=asdict(config.model),
        expected_game_config=asdict(config.game),
        expected_run_id=document["run_id"],
        expected_generation_family=document["generation_family"],
        expected_sha256=document["recovery"]["sha256"],
        expected_bytes=document["recovery"]["bytes"],
        metadata_validator=validate,
        gradient_clipper=clipper,
        allow_gradient_clipping_cold_start=arm == "adagc",
    )
    initial = {
        "model": state_fingerprint(model.state_dict()),
        "optimizer": state_fingerprint(optimizer.state_dict()),
        "scheduler": state_fingerprint(scheduler.state_dict()),
        "ema": state_fingerprint(ema.state_dict()),
        "learning_rates": [g["lr"] for g in optimizer.param_groups],
        "ema_updates": ema.num_updates,
        "parameter_identity": digest(
            [
                (name, list(p.shape), str(p.dtype))
                for name, p in model.named_parameters()
            ]
        ),
    }
    return model, optimizer, scheduler, ema, clipper, metadata, initial


def _diagnostic_backward(model, batch, optimizer, config, *, forward_model=None):
    import torch
    from startrain.gradient_diagnostics import collect_gradient_diagnostics
    from startrain.losses import compute_losses

    device = next(model.parameters()).device
    moved = batch.to(device)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=config.train.precision == "bf16",
    ):
        output = (forward_model if forward_model is not None else model)(
            *moved.inputs.model_args()
        )
        losses = compute_losses(
            output,
            moved.targets,
            legal_action_mask=moved.inputs.legal_action_mask,
            node_mask=moved.inputs.node_mask,
            weights=config.loss,
        )
    losses["total"].backward()
    diagnostics = collect_gradient_diagnostics(model, batch, optimizer).to_host()
    raw_norm = diagnostics["pre_clip_global_norm"]
    norm = float(raw_norm) if isinstance(raw_norm, int | float) else None
    coefficient = (
        min(1.0, config.train.gradient_clip_norm / (norm + 1e-6))
        if norm is not None
        else None
    )
    return {
        "losses": {k: float(v.detach().float().cpu()) for k, v in losses.items()},
        "gradient_diagnostics": diagnostics,
        "global_clip_if_applied": {
            "threshold": config.train.gradient_clip_norm,
            "coefficient": coefficient,
            "severity": 1 - coefficient if coefficient is not None else None,
        },
    }


def evaluate_models(model, ema, config, replay, cells, *, device, batches: int):
    from startrain.model import GraphResTNet

    helpers = _helpers()
    shadow = GraphResTNet(config.model).to(device)
    shadow.load_state_dict(model.state_dict())
    ema.copy_to(shadow)
    model.eval()
    shadow.eval()
    result = {}
    for name, evaluated in (("raw", model), ("ema", shadow)):
        per_cell = {}
        for cell in CELLS:
            references = cells[cell]["validation"]
            total: dict[str, float] = {}
            count = 0
            for index in range(batches):
                size = min(config.train.per_rank_batch_size, len(references))
                batch = helpers._materialize(
                    replay,
                    references,
                    start=index * size,
                    batch_size=size,
                    seed=config.train.seed,
                    augment=False,
                )
                losses = helpers._component_losses(
                    evaluated,
                    batch,
                    config,
                    device=device,
                    precision=config.train.precision,
                )
                for key, value in losses.items():
                    total[key] = total.get(key, 0.0) + value * size
                count += size
            per_cell[cell] = {key: value / count for key, value in total.items()} | {
                "samples": count
            }
        weighted = {
            key: sum(
                (0.85 if cell.startswith("r10/") else 0.05) / 6 * values[key]
                for cell, values in per_cell.items()
            )
            for key in ("policy", "value", "composite")
        }
        result[name] = {"per_cell": per_cell, "objective_weighted": weighted}
    model.train()
    return result


def _runtime_source_pin() -> dict[str, str]:
    from startrain.checkpoint import sha256_file

    root = Path(__file__).resolve().parents[1]
    paths = [
        Path(__file__).resolve(),
        *sorted((root / "startrain").glob("*.py")),
        Path(__file__).with_name("run_frozen_replay_optimizer_calibration.py"),
    ]
    return {str(path): sha256_file(path) for path in paths}


def run_child(args) -> dict[str, object]:
    started = time.monotonic()
    cpus = set()
    for part in args.cpu_affinity.split(","):
        ends = part.split("-")
        cpus.update(range(int(ends[0]), int(ends[-1]) + 1))
    getattr(os, "sched_setaffinity")(0, cpus)
    import torch
    from startrain.checkpoint import save_checkpoint, sha256_file
    from startrain.config import load_config
    from startrain.device import enable_fast_math, seed_all
    from startrain.training import maybe_compile_model, train_step
    from startrain.runtime import atomic_json

    if sha256_file(args.replay_manifest) != args.manifest_sha256:
        raise ValueError("frozen manifest changed after parent pinned it")

    if not args.device.startswith("cuda:") or not torch.cuda.is_available():
        raise ValueError("trial execution requires an explicitly isolated CUDA GPU")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    sentinel = torch.empty(1, device=device)
    uuid = str(torch.cuda.get_device_properties(device).uuid)
    uuid = uuid if uuid.startswith("GPU-") else "GPU-" + uuid
    ownership_before = _control()._gpu_ownership(uuid)
    if not ownership_before["verified"]:
        raise RuntimeError(f"GPU is not isolated: {ownership_before}")
    document, replay, cells = load_frozen(args.replay_manifest)
    protected_replay = Path(document["source_replay_root"]).resolve()
    protected_run = (
        protected_replay.parent
        if protected_replay.name == "replay"
        else protected_replay
    )
    if (
        args.output_dir.resolve() == protected_run
        or protected_run in args.output_dir.resolve().parents
    ):
        raise ValueError("child output overlaps production replay/control directory")
    config = load_config(document["config"]["path"])
    _validate_production_config(config)
    learner_gpu = next(g for g in config.orchestration.gpus if g.role == "learner")
    threads = learner_gpu.blas_threads or learner_gpu.cpu_threads
    if threads > len(cpus):
        raise ValueError(
            "reserved CPU affinity is smaller than production learner thread budget"
        )
    torch.set_num_threads(threads)
    enable_fast_math(device)
    seed_all(document["seed"])
    source_pin = _runtime_source_pin()
    arm = args.arms[0]
    model, optimizer, scheduler, ema, clipper, metadata, initial = (
        restore_training_state(config, document, device=device, arm=arm)
    )
    compiled = maybe_compile_model(
        model,
        enabled=config.train.compile,
        dynamic=False,
        recompile_limit=len(RINGS),
        isolate_recompiles=True,
    )
    compiled.train()
    schedule = (
        [CELLS[i % len(CELLS)] for i in range(args.steps)]
        if args.diagnostic_only
        else cell_schedule(seed=document["seed"], steps=args.steps)
    )
    counts = Counter()
    batch_pins = []
    for cell in schedule:
        start = counts[cell] * config.train.per_rank_batch_size
        refs = cells[cell]["train"]
        selected = [
            refs[(start + i) % len(refs)].stable_id
            for i in range(config.train.per_rank_batch_size)
        ]
        batch_pins.append(
            digest(
                {
                    "cell": cell,
                    "start": start,
                    "references": selected,
                    "seed": document["seed"],
                    "augmentation": config.data.d5_augmentation,
                }
            )
        )
        counts[cell] += 1
    report = {
        "format": "startrain.gradient-clipping-trial",
        "schema_version": 1,
        "arm": arm,
        "diagnostic_only": args.diagnostic_only,
        "source_step": metadata["step"],
        "frozen_manifest_sha256": sha256_file(args.replay_manifest),
        "source_recovery": document["recovery"],
        "source_config": document["config"],
        "runtime_source_sha256": digest(source_pin),
        "runtime_source_files": source_pin,
        "initial_state": initial,
        "batch_schedule_sha256": digest(batch_pins),
        "batch_pins": batch_pins,
        "cell_schedule": schedule,
        "cell_batch_counts": dict(counts),
        "precision": config.train.precision,
        "compile": config.train.compile,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cpu_affinity": sorted(cpus),
        "blas_threads": torch.get_num_threads(),
        "gpu_uuid": uuid,
        "ownership_before": ownership_before,
        "scope": "loss and gradient diagnostics only; no Elo inference or production publication",
        "training_includes_first_use_compilation": True,
    }
    report["startup_seconds"] = time.monotonic() - started
    validation_started = time.monotonic()
    if not args.diagnostic_only:
        report["initial_validation"] = evaluate_models(
            model,
            ema,
            config,
            replay,
            cells,
            device=device,
            batches=args.validation_batches,
        )
    report["validation_seconds"] = time.monotonic() - validation_started
    report["rng_before_training"] = rng_fingerprints(device)
    report["gradient_clipping_config"] = (
        asdict(clipper.config)
        if clipper is not None
        else {"mode": "global", "max_norm": config.train.gradient_clip_norm}
    )
    counts.clear()
    step_records = []
    train_seconds = 0.0
    seen_rings = set()
    for index, cell in enumerate(schedule):
        if time.monotonic() - started > args.timeout_seconds:
            raise TimeoutError("trial total wall budget exhausted")
        batch = _helpers()._materialize(
            replay,
            cells[cell]["train"],
            start=counts[cell] * config.train.per_rank_batch_size,
            batch_size=config.train.per_rank_batch_size,
            seed=document["seed"],
            augment=config.data.d5_augmentation,
        )
        torch.cuda.synchronize(device)
        step_started = time.monotonic()
        if args.diagnostic_only:
            values = _diagnostic_backward(
                model, batch, optimizer, config, forward_model=compiled
            )
        else:
            collect = (
                metadata["step"] + index + 1
            ) % config.learner.metrics_interval == 0
            result = train_step(
                compiled,
                batch,
                optimizer,
                loss_weights=config.loss,
                precision=config.train.precision,
                gradient_clip_norm=config.train.gradient_clip_norm,
                scheduler=scheduler,
                ema=ema,
                trusted_batch=True,
                gradient_clipper=clipper,
                collect_diagnostics=collect,
                collect_gradient_diagnostics=collect,
            )
            values = asdict(result.to_host())
        torch.cuda.synchronize(device)
        elapsed = time.monotonic() - step_started
        train_seconds += elapsed
        ring = cell.split("/")[0]
        first_shape = ring not in seen_rings
        seen_rings.add(ring)
        record = {
            "trial_step": index + 1,
            "absolute_step": metadata["step"]
            + (0 if args.diagnostic_only else index + 1),
            "cell": cell,
            "batch_recipe_sha256": batch_pins[index],
            "seconds": elapsed,
            **values,
        }
        with (args.output_dir / "steps.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        step_records.append(
            {
                "trial_step": index + 1,
                "cell": cell,
                "seconds": elapsed,
                "total_loss": values["losses"]["total"],
                "first_training_shape": first_shape,
                "adagc_warmup_window": index
                < config.train.gradient_clipping.warmup_steps,
            }
        )
        counts[cell] += 1
    report["training_seconds"] = train_seconds
    report["steps"] = step_records
    report["completed_steps"] = len(schedule)
    steady = [
        s
        for s in step_records
        if not s["first_training_shape"] and not s["adagc_warmup_window"]
    ]
    report["timing"] = {
        "training_seconds": train_seconds,
        "first_shape_seconds": sum(
            s["seconds"] for s in step_records if s["first_training_shape"]
        ),
        "first_shape_steps": [
            s["trial_step"] for s in step_records if s["first_training_shape"]
        ],
        "warmup_window_steps": config.train.gradient_clipping.warmup_steps,
        "warmup_window_seconds": sum(
            s["seconds"] for s in step_records if s["adagc_warmup_window"]
        ),
        "steady_steps": len(steady),
        "steady_seconds": sum(s["seconds"] for s in steady),
        "note": "steady excludes first use of each board shape and the configured adaptive warmup window in both arms",
    }
    if args.diagnostic_only:
        final = {
            "model": state_fingerprint(model.state_dict()),
            "optimizer": state_fingerprint(optimizer.state_dict()),
            "scheduler": state_fingerprint(scheduler.state_dict()),
            "ema": state_fingerprint(ema.state_dict()),
        }
        if any(initial[key] != value for key, value in final.items()):
            raise RuntimeError("diagnostic-only run changed training state")
        report["diagnostic_state_unchanged"] = True
    else:
        evaluation_started = time.monotonic()
        report["final_validation"] = evaluate_models(
            model,
            ema,
            config,
            replay,
            cells,
            device=device,
            batches=args.validation_batches,
        )
        report["validation_seconds"] += time.monotonic() - evaluation_started
        trial_config = config.as_dict()
        if clipper is not None:
            trial_config["train"]["gradient_clipping"] = asdict(clipper.config)
        path = args.output_dir / "trial-recovery.pt"
        save_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            ema=ema,
            gradient_clipper=clipper,
            step=metadata["step"] + len(schedule),
            epoch=metadata["epoch"],
            config=trial_config,
            extra={
                "run_id": document["run_id"],
                "generation_family": document["generation_family"],
                "gradient_clipping_trial": True,
                "source_recovery_sha256": document["recovery"]["sha256"],
                "frozen_manifest_sha256": report["frozen_manifest_sha256"],
                "arm": arm,
                "batch_schedule_sha256": report["batch_schedule_sha256"],
            },
        )
        report["trial_checkpoint"] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    report["ownership_after"] = _control()._gpu_ownership(uuid)
    if not report["ownership_after"]["verified"]:
        raise RuntimeError("GPU isolation changed during trial")
    if _runtime_source_pin() != source_pin:
        raise RuntimeError("trial runtime source changed during execution")
    if sha256_file(args.replay_manifest) != args.manifest_sha256:
        raise ValueError("frozen manifest changed during trial")
    report["total_seconds"] = time.monotonic() - started
    report["status"] = "complete"
    atomic_json(args.output_dir / "result.json", report)
    del sentinel
    return report


def validate_result(
    path: Path, *, arm: str, manifest_sha256: str, steps: int, diagnostic: bool
) -> dict:
    from startrain.checkpoint import verify_file

    result = json.loads(path.read_text())
    if (
        result.get("format") != "startrain.gradient-clipping-trial"
        or result.get("schema_version") != 1
        or result.get("status") != "complete"
        or result.get("arm") != arm
        or result.get("diagnostic_only") is not diagnostic
        or result.get("frozen_manifest_sha256") != manifest_sha256
        or result.get("completed_steps") != steps
        or len(result.get("steps", [])) != steps
    ):
        raise ValueError("child result does not match completed pinned trial")
    pins = result.get("batch_pins", [])
    if len(pins) != steps or result.get("batch_schedule_sha256") != digest(pins):
        raise ValueError("child result batch schedule pin is incomplete")
    for name in ("model", "optimizer", "scheduler", "ema", "parameter_identity"):
        fingerprint = result.get("initial_state", {}).get(name)
        if not isinstance(fingerprint, str) or len(fingerprint) != 64:
            raise ValueError(f"child result lacks initial {name} fingerprint")
    for side in ("ownership_before", "ownership_after"):
        if result.get(side, {}).get("verified") is not True:
            raise ValueError("child result lacks verified GPU isolation")
    if diagnostic:
        if result.get("diagnostic_state_unchanged") is not True:
            raise ValueError("diagnostic did not prove unchanged training state")
    else:
        pin = result["trial_checkpoint"]
        artifact = Path(pin["path"]).resolve()
        if path.resolve().parent not in artifact.parents:
            raise ValueError("trial checkpoint escaped its arm directory")
        verify_file(
            artifact, expected_sha256=pin["sha256"], expected_bytes=pin["bytes"]
        )
    return result


def compare_trial_results(paths: list[Path]) -> dict[str, object]:
    records = [json.loads(path.read_text()) for path in paths]
    if len(records) != 2 or {r.get("arm") for r in records} != {"global", "adagc"}:
        raise ValueError("comparison requires exactly global and adagc results")
    records.sort(key=lambda record: record["arm"] != "global")
    baseline, treatment = records
    fields = (
        "source_step",
        "frozen_manifest_sha256",
        "source_recovery",
        "source_config",
        "runtime_source_sha256",
        "initial_state",
        "rng_before_training",
        "batch_schedule_sha256",
        "batch_pins",
        "completed_steps",
        "cell_schedule",
        "precision",
        "compile",
        "float32_matmul_precision",
    )
    mismatches = [
        field
        for field in fields
        if field not in baseline or baseline.get(field) != treatment.get(field)
    ]
    for path in paths:
        record = json.loads(path.read_text())
        validate_result(
            path,
            arm=record["arm"],
            manifest_sha256=baseline["frozen_manifest_sha256"],
            steps=baseline["completed_steps"],
            diagnostic=False,
        )
    if mismatches:
        raise ValueError(f"trial arms are not comparable: {mismatches}")
    improvements = {}
    for model in ("raw", "ema"):
        improvements[model] = {
            cell: {
                key: baseline["final_validation"][model]["per_cell"][cell][key]
                - treatment["final_validation"][model]["per_cell"][cell][key]
                for key in ("policy", "value", "composite")
            }
            for cell in CELLS
        }
    return {
        "format": "startrain.gradient-clipping-trial-comparison",
        "comparable": True,
        "frozen_manifest_sha256": baseline["frozen_manifest_sha256"],
        "source_step": baseline["source_step"],
        "steps": baseline["completed_steps"],
        "adagc_loss_improvement_vs_global": improvements,
        "timing": {r["arm"]: r["timing"] for r in records},
        "scope": "held-out loss comparison only; no Elo or automatic production-adoption claim",
    }


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--recovery", type=Path)
    parser.add_argument("--replay-root", type=Path)
    parser.add_argument("--replay-cutoff", type=int)
    parser.add_argument("--train-samples-per-cell", type=int, default=2048)
    parser.add_argument("--validation-samples-per-cell", type=int, default=512)
    parser.add_argument("--replay-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--arms", nargs="+", choices=("global", "adagc"), default=["global", "adagc"]
    )
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--validation-batches", type=int, default=1)
    parser.add_argument("--diagnostic-only", action="store_true")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--cpu-affinity")
    parser.add_argument("--timeout-seconds", type=float, default=1800)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--manifest-sha256", help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if not 1 <= args.steps <= 2000 or not 1 <= args.validation_batches <= 4:
        parser.error("steps must be1..2000 and validation batches1..4")
    if not 1 <= args.timeout_seconds <= 1800 or not math.isfinite(args.timeout_seconds):
        parser.error("timeout must be finite and1..1800 seconds")
    if (
        not 1 <= args.train_samples_per_cell <= 32768
        or not 1 <= args.validation_samples_per_cell <= 4096
    ):
        parser.error("frozen sample bounds exceeded")
    if args.prepare_only:
        if not all((args.config, args.recovery, args.replay_root)):
            parser.error(
                "preparation requires config, recovery pointer and replay root"
            )
        print(json.dumps(prepare(args)))
        return 0
    if args.replay_manifest is None or not args.cpu_affinity:
        parser.error(
            "execution requires frozen replay manifest and reserved CPU affinity"
        )
    if args.child:
        if len(args.arms) != 1:
            parser.error("child requires one arm")
        run_child(args)
        return 0
    control = _control()
    from startrain.training import isolated_compile_cache
    from startrain.checkpoint import sha256_file

    manifest_sha256 = sha256_file(args.replay_manifest)
    frozen = json.loads(args.replay_manifest.read_text())
    source_replay = Path(frozen["source_replay_root"]).resolve()
    source_run = (
        source_replay.parent if source_replay.name == "replay" else source_replay
    )
    output = _disjoint_output(args.output_dir, [args.replay_manifest, source_run])
    statuses = []
    with control._controller_signals():
        for arm in ["global"] if args.diagnostic_only else args.arms:
            arm_dir = output / arm
            arm_dir.mkdir()
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--child",
                "--replay-manifest",
                str(args.replay_manifest.resolve()),
                "--manifest-sha256",
                manifest_sha256,
                "--output-dir",
                str(arm_dir),
                "--arms",
                arm,
                "--steps",
                str(args.steps),
                "--validation-batches",
                str(args.validation_batches),
                "--device",
                args.device,
                "--cpu-affinity",
                args.cpu_affinity,
                "--timeout-seconds",
                str(args.timeout_seconds),
            ]
            if args.diagnostic_only:
                command.append("--diagnostic-only")
            try:
                with isolated_compile_cache(arm_dir / "compile-cache") as cache:
                    process = control._run_owned(
                        command,
                        env=dict(os.environ, **cache.environment),
                        timeout=args.timeout_seconds,
                    )
                (arm_dir / "stdout.log").write_text(process.stdout)
                (arm_dir / "stderr.log").write_text(process.stderr)
                if process.returncode == 0:
                    validate_result(
                        arm_dir / "result.json",
                        arm=arm,
                        manifest_sha256=manifest_sha256,
                        steps=args.steps,
                        diagnostic=args.diagnostic_only,
                    )
                statuses.append(
                    {
                        "arm": arm,
                        "returncode": process.returncode,
                        "status": "complete" if process.returncode == 0 else "failed",
                    }
                )
            except (
                control.BenchmarkInterrupted,
                subprocess.TimeoutExpired,
                TimeoutError,
                ValueError,
                OSError,
            ) as error:
                statuses.append(
                    {
                        "arm": arm,
                        "status": "interrupted"
                        if isinstance(error, control.BenchmarkInterrupted)
                        else "timeout"
                        if isinstance(error, (subprocess.TimeoutExpired, TimeoutError))
                        else "failed",
                        "error": str(error),
                    }
                )
                for stream in ("stdout", "stderr"):
                    value = getattr(error, stream, "") or ""
                    if isinstance(value, bytes):
                        value = value.decode(errors="replace")
                    (arm_dir / f"{stream}.log").write_text(value)
                break
    (output / "execution.json").write_text(json.dumps(statuses, indent=2))
    if len(statuses) == 2 and all(item["status"] == "complete" for item in statuses):
        comparison = compare_trial_results(
            [output / item["arm"] / "result.json" for item in statuses]
        )
        (output / "comparison.json").write_text(json.dumps(comparison, indent=2))
    return int(any(item["status"] != "complete" for item in statuses))


if __name__ == "__main__":
    raise SystemExit(main())
