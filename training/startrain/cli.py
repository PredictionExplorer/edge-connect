"""Console entry points for single-machine self-play and learning."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict, replace
from pathlib import Path

import torch

from .actor import ActorSupervisor
from .arena import ArenaRunner, internal_elo_target_assessment
from .baselines import FROZEN_BASELINE_CHOICES, create_frozen_baseline
from .checkpoint import (
    load_ema_checkpoint,
    load_model_manifest,
)
from .config import load_config
from .distill import distill_main
from .inference import GraphInferenceAdapter, InferenceConfig
from .learner import LearnerLoop
from .model import GraphResTNet
from .native import load_star_native
from .promotion import load_manifest_evaluator, promotion_main
from .publish import publish_browser_main
from .replay_store import ReplayStore
from .runtime import (
    HeartbeatReporter,
    SignalLatch,
    atomic_json,
    load_run_identity,
)
from .selfplay import SelfPlayActor, SelfPlayConfig, SelfPlayIdentity


def selfplay_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run local Double *Star self-play")
    parser.add_argument("--config", required=True)
    parser.add_argument("--replay-store", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--games", type=int)
    parser.add_argument("--rings", type=int)
    parser.add_argument("--cpu-smoke", action="store_true")
    parser.add_argument("--run-identity", required=True)
    parser.add_argument("--actor-id", default="standalone-selfplay")
    arguments = parser.parse_args(argv)

    experiment = load_config(arguments.config)
    selfplay_config = experiment.selfplay
    model_config = experiment.model
    if arguments.cpu_smoke:
        selfplay_config = SelfPlayConfig.cpu_smoke(seed=selfplay_config.seed)
        model_config = replace(model_config, width=16, attention_heads=4, kv_heads=1)
    if arguments.games is not None:
        selfplay_config = replace(selfplay_config, games=arguments.games)
    if arguments.rings is not None:
        selfplay_config = replace(selfplay_config, rings=arguments.rings)

    model = GraphResTNet(model_config).to(arguments.device)
    model_version = _model_state_identity(model)
    model_step = 0
    model_identity = model_version
    manifest = None
    if arguments.checkpoint:
        manifest = load_model_manifest(arguments.checkpoint)
        checkpoint = manifest.checkpoint
        metadata = load_ema_checkpoint(
            checkpoint,
            model=model,
            expected_model_config=asdict(model_config),
            expected_game_config=asdict(experiment.game),
            expected_run_id=manifest.run_id,
            expected_generation_family=manifest.generation_family,
            expected_sha256=manifest.checkpoint_sha256,
            expected_bytes=manifest.checkpoint_bytes,
            map_location=arguments.device,
        )
        model_step = int(metadata["step"])
        model_version = manifest.model_version
        model_identity = manifest.model_identity
    elif not arguments.cpu_smoke:
        raise ValueError("self-play requires an immutable champion manifest")
    evaluator = GraphInferenceAdapter(
        model,
        device=arguments.device,
        config=InferenceConfig(
            precision="fp32" if arguments.cpu_smoke else experiment.train.precision,
            score_utility_weight=selfplay_config.score_utility_weight,
            initial_pass_logit_penalty=selfplay_config.initial_pass_logit_penalty,
        ),
        model_version=model_version,
        model_step=model_step,
        model_identity=model_identity,
    )
    native = load_star_native(required=True)
    assert native is not None
    run_identity = load_run_identity(arguments.run_identity)
    if manifest is not None and (
        manifest.run_id != run_identity.run_id
        or manifest.generation_family != run_identity.generation_family
    ):
        raise ValueError("champion manifest does not belong to the active run")
    with ReplayStore(arguments.replay_store) as store:
        generation = store.lease_generation(run_identity, arguments.actor_id)
        summaries = SelfPlayActor(
            native,
            evaluator,
            store,
            selfplay_config,
            SelfPlayIdentity(
                run_identity.run_id,
                run_identity.generation_family,
                arguments.actor_id,
                generation,
            ),
        ).run()
    print(json.dumps([asdict(summary) for summary in summaries], sort_keys=True))


def train_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the local replay learner")
    parser.add_argument("--config", required=True)
    parser.add_argument("--replay-store", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--device")
    parser.add_argument("--resume-latest", action="store_true")
    parser.add_argument("--distributed-backend", choices=("nccl", "gloo"))
    parser.add_argument("--local-rank", "--local_rank", type=int)
    parser.add_argument("--heartbeat")
    parser.add_argument("--run-identity", required=True)
    parser.add_argument("--promotion-status")
    parser.add_argument("--gpu-pause")
    arguments = parser.parse_args(argv)

    experiment = load_config(arguments.config)
    run_identity = load_run_identity(arguments.run_identity)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = (
        arguments.local_rank
        if arguments.local_rank is not None
        else int(os.environ.get("LOCAL_RANK", "0"))
    )
    if world_size > 1:
        distributed = experiment.orchestration.distributed
        if (
            not distributed.enabled
            or arguments.distributed_backend != distributed.backend
        ):
            raise RuntimeError(
                "DDP requires explicitly enabled matching distributed configuration"
            )
        if arguments.device not in (None, "cuda"):
            raise RuntimeError("DDP learner device must be cuda")
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(
            backend=arguments.distributed_backend,
        )
    else:
        if arguments.distributed_backend is not None:
            raise RuntimeError("distributed backend was set without a DDP launch")
        if (
            experiment.orchestration.distributed.enabled
            and len(experiment.orchestration.learner_gpus) > 1
        ):
            raise RuntimeError(
                "configured DDP learner must be launched through torchrun"
            )
        device = arguments.device or experiment.learner.device
    if device:
        experiment = replace(
            experiment,
            learner=replace(experiment.learner, device=device),
        )
    stop = SignalLatch()
    stop.install()
    heartbeat = (
        HeartbeatReporter(
            arguments.heartbeat,
            worker="learner",
            interval_seconds=(
                experiment.orchestration.shutdown.heartbeat_interval_seconds
            ),
        )
        if arguments.heartbeat and rank == 0
        else None
    )
    if heartbeat is not None:
        heartbeat.start()
        heartbeat.update(phase="initializing", rank=rank, world_size=world_size)
    try:
        with ReplayStore(arguments.replay_store) as store:
            learner = LearnerLoop.from_experiment(
                experiment,
                store=store,
                output_directory=arguments.output,
                run_identity=run_identity,
                promotion_status_path=arguments.promotion_status,
                gpu_pause_path=arguments.gpu_pause,
                rank=rank,
                world_size=world_size,
            )
            resume_manifest = (
                load_model_manifest(arguments.resume) if arguments.resume else None
            )
            if (
                resume_manifest is None
                and (arguments.resume_latest or experiment.learner.resume_latest)
                and learner.publisher.candidate_path.is_file()
            ):
                resume_manifest = load_model_manifest(learner.publisher.candidate_path)
            if resume_manifest is not None:
                learner.resume(
                    resume_manifest.checkpoint,
                    expected_sha256=resume_manifest.checkpoint_sha256,
                    expected_bytes=resume_manifest.checkpoint_bytes,
                )
            if heartbeat is not None:
                heartbeat.update(
                    phase="training",
                    step=learner.step,
                    resumed_from=(
                        str(resume_manifest.path)
                        if resume_manifest is not None
                        else None
                    ),
                )
            final_step = learner.run(
                steps=arguments.steps,
                stop_requested=stop.is_set,
                progress=heartbeat.advance if heartbeat is not None else None,
            )
        if rank == 0:
            print(json.dumps({"step": final_step}, sort_keys=True))
    finally:
        if heartbeat is not None:
            heartbeat.close(final_phase="stopped" if stop.is_set() else "completed")
        if world_size > 1 and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def actor_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run one long-lived self-play actor supervisor"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpu-id", required=True, type=int)
    parser.add_argument("--lane-id", type=int, default=0)
    parser.add_argument("--replay-store", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--candidate-manifest", required=True)
    parser.add_argument("--run-identity", required=True)
    parser.add_argument("--heartbeat", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args(argv)

    experiment = load_config(arguments.config)
    matches = [
        gpu
        for gpu in experiment.orchestration.actor_gpus
        if gpu.gpu_id == arguments.gpu_id
    ]
    if len(matches) != 1:
        raise ValueError("gpu-id is not a unique configured actor GPU")
    if arguments.lane_id < 0 or arguments.lane_id >= matches[0].actor_lanes:
        raise ValueError("lane-id is outside the configured actor lane range")
    native = load_star_native(required=True)
    assert native is not None
    stop = SignalLatch()
    stop.install()
    run_identity = load_run_identity(arguments.run_identity)
    batches = ActorSupervisor(
        native_module=native,
        experiment=experiment,
        gpu=matches[0],
        replay_directory=arguments.replay_store,
        manifest_path=arguments.manifest,
        candidate_manifest_path=arguments.candidate_manifest,
        run_identity=run_identity,
        heartbeat_path=arguments.heartbeat,
        metrics_path=arguments.metrics,
        device=arguments.device,
        lane_id=arguments.lane_id,
    ).run(stop_requested=stop.is_set)
    print(json.dumps({"batches": batches}, sort_keys=True))


def arena_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run paired deterministic arena evaluation"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument(
        "--baseline",
        help="immutable checkpoint manifest (required for checkpoint baseline)",
    )
    parser.add_argument(
        "--baseline-kind",
        choices=("checkpoint", *FROZEN_BASELINE_CHOICES),
        default="checkpoint",
        help="checkpoint (default) or a versioned frozen non-human opponent",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--target-elo-lcb",
        type=float,
        help="optional internal Elo target for the paired anytime-valid lower bound",
    )
    parser.add_argument(
        "--target-rings",
        type=int,
        nargs="+",
        help="rings assessed by --target-elo-lcb (defaults to configured arena rings)",
    )
    arguments = parser.parse_args(argv)
    if arguments.baseline_kind == "checkpoint" and not arguments.baseline:
        parser.error("--baseline is required when --baseline-kind=checkpoint")
    if arguments.baseline_kind != "checkpoint" and arguments.baseline:
        parser.error("--baseline cannot be combined with a frozen --baseline-kind")
    if arguments.target_rings and arguments.target_elo_lcb is None:
        parser.error("--target-rings requires --target-elo-lcb")

    experiment = load_config(arguments.config)
    candidate_manifest = load_model_manifest(arguments.candidate)
    baseline_manifest = (
        load_model_manifest(arguments.baseline)
        if arguments.baseline_kind == "checkpoint"
        else None
    )
    candidate = load_manifest_evaluator(
        experiment, candidate_manifest, device=arguments.device
    )
    checkpoint_baseline = (
        load_manifest_evaluator(experiment, baseline_manifest, device=arguments.device)
        if baseline_manifest is not None
        else None
    )
    native = load_star_native(required=True)
    assert native is not None
    if arguments.baseline_kind == "checkpoint":
        assert checkpoint_baseline is not None
        result = ArenaRunner(
            native_module=native,
            candidate=candidate,
            baseline=checkpoint_baseline,
            config=experiment.arena,
        ).run()
    else:
        baseline = create_frozen_baseline(
            arguments.baseline_kind,
            native_module=native,
        )
        result = ArenaRunner(
            native_module=native,
            candidate=candidate,
            baseline=baseline,
            config=experiment.arena,
            baseline_search=baseline.search_budget,
            baseline_metadata=baseline.result_metadata(),
        ).run()
    if arguments.target_elo_lcb is not None:
        target_rings = tuple(arguments.target_rings or experiment.arena.rings)
        unavailable = sorted(set(target_rings) - set(experiment.arena.rings))
        if unavailable:
            parser.error(
                "--target-rings contains rings outside the arena config: "
                + ", ".join(str(ring) for ring in unavailable)
            )
        result["internal_elo_target"] = internal_elo_target_assessment(
            result,
            rings=target_rings,
            target_elo=arguments.target_elo_lcb,
        )
    atomic_json(arguments.output, result)
    promotion = result.get("promotion")
    if not isinstance(promotion, dict) or not isinstance(
        promotion.get("decision"), str
    ):
        raise RuntimeError("arena result omitted a promotion decision")
    print(
        json.dumps(
            {
                "output": str(Path(arguments.output).resolve()),
                "decision": promotion["decision"],
                "aggregate": result["aggregate"],
                "baseline": result["baseline_metadata"],
                "internal_elo_target": result.get("internal_elo_target"),
            },
            sort_keys=True,
        )
    )


def main(argv: list[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        raise SystemExit(
            "expected one of: selfplay, train, actor, arena, distill, "
            "promote, publish-browser"
        )
    command, command_arguments = arguments[0], arguments[1:]
    commands = {
        "selfplay": selfplay_main,
        "train": train_main,
        "actor": actor_main,
        "arena": arena_main,
        "distill": distill_main,
        "promote": promotion_main,
        "publish-browser": publish_browser_main,
    }
    try:
        entrypoint = commands[command]
    except KeyError:
        raise SystemExit(f"unknown startrain command: {command}") from None
    entrypoint(command_arguments)


def _model_state_identity(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return f"sha256-{digest.hexdigest()}"


if __name__ == "__main__":
    main()
