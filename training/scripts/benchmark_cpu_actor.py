#!/usr/bin/env python3
"""Bounded CPU self-play sweep, each case in a fresh independently pinned process.

The default prints the matrix. --execute measures real native games with BF16
inference; generated replay is temporary diagnostic data and is never imported.
No learner, production actor, or service is launched or modified.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, replace
from contextlib import nullcontext
from itertools import product
from pathlib import Path


def _checkpoint_record(manifest) -> dict[str, object]:
    return {
        "model_identity": manifest.model_identity,
        "model_version": manifest.model_version,
        "model_step": manifest.model_step,
        "checkpoint": str(manifest.checkpoint.resolve()),
        "checkpoint_sha256": manifest.checkpoint_sha256,
        "checkpoint_bytes": manifest.checkpoint_bytes,
        "manifest": str((manifest.artifact_manifest or manifest.path).resolve()),
        "manifest_sha256": manifest.manifest_sha256,
        "run_id": manifest.run_id,
        "generation_family": manifest.generation_family,
        "random_initialization": False,
    }


def _load_actor_weights(config, model, checkpoint: Path, manifest_sha256: str | None):
    from startrain.checkpoint import load_ema_checkpoint, load_model_manifest

    manifest = load_model_manifest(checkpoint)
    if manifest_sha256 is not None and manifest.manifest_sha256 != manifest_sha256:
        raise ValueError(
            "benchmark checkpoint manifest changed after the parent pinned it"
        )
    metadata = load_ema_checkpoint(
        manifest.checkpoint,
        model=model,
        expected_model_config=asdict(config.model),
        expected_game_config=asdict(config.game),
        map_location="cpu",
        expected_run_id=manifest.run_id,
        expected_generation_family=manifest.generation_family,
        expected_sha256=manifest.checkpoint_sha256,
        expected_bytes=manifest.checkpoint_bytes,
    )
    if metadata["step"] != manifest.model_step:
        raise ValueError("benchmark checkpoint step differs from the manifest")
    return _checkpoint_record(manifest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    weights = parser.add_mutually_exclusive_group()
    weights.add_argument(
        "--checkpoint",
        type=Path,
        help="verified immutable model manifest or champion/candidate pointer JSON",
    )
    weights.add_argument(
        "--random-initialization",
        action="store_true",
        help="explicitly benchmark deterministic random weights instead of a checkpoint",
    )
    parser.add_argument("--checkpoint-manifest-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--work-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--cpu-affinity", required=True)
    parser.add_argument("--native-threads", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--blas-threads", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[8, 16])
    parser.add_argument("--rings", type=int, nargs="+", choices=[4, 6], default=[4])
    parser.add_argument("--timeout-seconds", type=float, default=300)
    parser.add_argument(
        "--exact-endgame-max-empty", type=int, choices=range(9), default=0
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    from startrain.config import parse_cpu_affinity

    cpus = parse_cpu_affinity(args.cpu_affinity)
    if (
        (args.execute or args.child)
        and args.checkpoint is None
        and not args.random_initialization
    ):
        parser.error(
            "--execute requires --checkpoint or explicit --random-initialization"
        )
    if any(
        value <= 0 or value > len(cpus)
        for value in args.native_threads + args.blas_threads
    ):
        parser.error("thread budgets must fit the reserved CPU list")
    if any(value <= 0 or value > 32 for value in args.batch_sizes):
        parser.error("bounded diagnostic batches must be in 1..32")
    if not 1 <= args.timeout_seconds <= 3600:
        parser.error("timeout must be in 1..3600 seconds")
    if args.child:
        if hasattr(os, "sched_setaffinity"):
            getattr(os, "sched_setaffinity")(0, set(cpus))
        else:
            raise RuntimeError("CPU affinity benchmarking requires a Linux target host")
        import torch
        import star_native
        from startrain.config import load_config
        from startrain.inference import GraphInferenceAdapter, InferenceConfig
        from startrain.model import GraphResTNet, model_parameter_count
        from startrain.replay_store import ReplayStore
        from startrain.runtime import RunIdentity
        from startrain.selfplay import SelfPlayActor, SelfPlayIdentity

        torch.set_num_threads(args.blas_threads[0])
        torch.set_num_interop_threads(1)
        config = load_config(args.config)
        torch.manual_seed(config.train.seed)
        model = GraphResTNet(config.model).eval()
        checkpoint_record = (
            _load_actor_weights(
                config, model, args.checkpoint, args.checkpoint_manifest_sha256
            )
            if args.checkpoint is not None
            else {
                "model_identity": "sha256-" + "0" * 64,
                "model_version": "cpu-benchmark-random",
                "model_step": 0,
                "random_initialization": True,
            }
        )
        model_step = checkpoint_record["model_step"]
        assert isinstance(model_step, int)
        evaluator = GraphInferenceAdapter(
            model,
            device="cpu",
            config=InferenceConfig(precision="bf16"),
            model_version=str(checkpoint_record["model_version"]),
            model_step=model_step,
            model_identity=str(checkpoint_record["model_identity"]),
        )
        work_directory = (
            nullcontext(str(args.work_dir))
            if args.work_dir is not None
            else tempfile.TemporaryDirectory(prefix="startrain-cpu-benchmark-")
        )
        with work_directory as root:
            identity = RunIdentity(
                Path(root) / "run.json",
                "cpu-benchmark",
                "cpu-benchmark",
                time.time_ns(),
            )
            with ReplayStore(Path(root) / "replay") as store:
                generation = store.lease_generation(identity, "actor-benchmark")
                actor = SelfPlayActor(
                    star_native,
                    evaluator,
                    store,
                    replace(
                        config.selfplay,
                        rings=args.rings[0],
                        games=args.batch_sizes[0],
                        batch_size=args.batch_sizes[0],
                        exact_endgame_max_empty=args.exact_endgame_max_empty,
                    ),
                    SelfPlayIdentity(
                        identity.run_id,
                        identity.generation_family,
                        "actor-benchmark",
                        generation,
                    ),
                )
                started = time.monotonic()
                summaries = actor.run()
                elapsed = time.monotonic() - started
                print(
                    json.dumps(
                        dict(
                            status="measured",
                            elapsed_seconds=elapsed,
                            games=len(summaries),
                            samples=sum(item.samples for item in summaries),
                            samples_per_second=sum(item.samples for item in summaries)
                            / elapsed,
                            parameters=model_parameter_count(config.model),
                            precision="bf16",
                            mode=config.selfplay.mode,
                            variant=config.selfplay.variant.label,
                            checkpoint=checkpoint_record,
                            native_threads=getattr(star_native, "rayon_num_threads")(),
                            blas_threads=torch.get_num_threads(),
                            cpu_affinity=sorted(getattr(os, "sched_getaffinity")(0)),
                            selfplay=asdict(actor.metrics_snapshot()),
                        )
                    )
                )
        return 0
    pinned_checkpoint = None
    if args.checkpoint is not None:
        from startrain.checkpoint import load_model_manifest

        pinned_checkpoint = _checkpoint_record(load_model_manifest(args.checkpoint))
    results = []
    for native, blas, batch, ring in product(
        args.native_threads, args.blas_threads, args.batch_sizes, args.rings
    ):
        case = dict(
            native_threads=native, blas_threads=blas, batch_size=batch, ring=ring
        )
        if args.execute:
            environment = dict(
                os.environ,
                RAYON_NUM_THREADS=str(native),
                OMP_NUM_THREADS=str(blas),
                MKL_NUM_THREADS=str(blas),
                OPENBLAS_NUM_THREADS=str(blas),
                CUDA_VISIBLE_DEVICES="",
            )
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--child",
                "--config",
                str(args.config.resolve()),
                "--cpu-affinity",
                args.cpu_affinity,
                "--native-threads",
                str(native),
                "--blas-threads",
                str(blas),
                "--batch-sizes",
                str(batch),
                "--rings",
                str(ring),
                "--exact-endgame-max-empty",
                str(args.exact_endgame_max_empty),
            ]
            if pinned_checkpoint is not None:
                command.extend(
                    [
                        "--checkpoint",
                        str(pinned_checkpoint["manifest"]),
                        "--checkpoint-manifest-sha256",
                        str(pinned_checkpoint["manifest_sha256"]),
                    ]
                )
            else:
                command.append("--random-initialization")
            try:
                with tempfile.TemporaryDirectory(
                    prefix="startrain-cpu-benchmark-"
                ) as work_dir:
                    run = subprocess.run(
                        [*command, "--work-dir", work_dir],
                        env=environment,
                        capture_output=True,
                        text=True,
                        timeout=args.timeout_seconds,
                    )
                case.update(
                    json.loads(run.stdout.splitlines()[-1])
                    if run.returncode == 0
                    else dict(status="failed", error=run.stderr[-4000:])
                )
            except subprocess.TimeoutExpired:
                case.update(status="timeout")
        else:
            case.update(status="planned")
        results.append(case)
    report = dict(
        schema_version=1,
        report="bounded-cpu-actor-throughput",
        config=str(args.config.resolve()),
        cpu_affinity=list(cpus),
        cases=results,
        diagnostic_only=True,
        checkpoint=pinned_checkpoint,
        random_initialization=args.random_initialization,
    )
    encoded = json.dumps(report, indent=2)
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(encoded + "\n")
    print(encoded)
    return int(any(case["status"] in ("failed", "timeout") for case in results))


if __name__ == "__main__":
    raise SystemExit(main())
