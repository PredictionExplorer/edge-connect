from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from starserve.config import load_server_config
from starserve.snapshot import ChampionSnapshotError, export_champion_snapshot
from startrain.checkpoint import (
    ExponentialMovingAverage,
    load_model_manifest,
    write_model_pointer,
)
from startrain.config import load_config
from startrain.learner import ImmutableModelPublisher
from startrain.model import GraphResTNet
from startrain.optim import build_optimizer
from startrain.runtime import RunIdentity
from startrain.training import build_scheduler


def _published_champion(tmp_path: Path) -> tuple[Path, Path]:
    base_profile = Path(__file__).parents[1] / "configs" / "small.yaml"
    profile_payload = yaml.safe_load(base_profile.read_text(encoding="utf-8"))
    profile_payload["model"].update(
        {
            "width": 8,
            "rrt_groups": 1,
            "attention_heads": 2,
            "kv_heads": 1,
        }
    )
    profile_payload["train"]["precision"] = "bf16"
    profile_payload["train"]["compile"] = True
    profile = tmp_path / "source-profile.yaml"
    profile.write_text(
        yaml.safe_dump(profile_payload, sort_keys=False),
        encoding="utf-8",
    )
    experiment = load_config(profile)

    model = GraphResTNet(experiment.model)
    optimizer = build_optimizer(model, experiment.optimizer)
    scheduler = build_scheduler(optimizer, experiment.train.scheduler)
    ema = ExponentialMovingAverage(model, decay=experiment.train.ema_decay)
    publisher = ImmutableModelPublisher(
        tmp_path / "publication",
        RunIdentity(
            tmp_path / "run.json",
            "snapshot-run",
            "snapshot-family",
            1,
        ),
    )
    candidate = publisher.publish(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema,
        step=7,
        epoch=1,
        config=experiment.as_dict(),
    )
    champion = publisher.root / "champion.json"
    write_model_pointer(
        champion, candidate, role="champion", promotion_result="bootstrap"
    )
    return champion, profile


def test_champion_snapshot_relocates_verifies_hash_and_refuses_unsafe_inputs(
    tmp_path: Path,
) -> None:
    source_pointer, profile = _published_champion(tmp_path)
    source_pointer_bytes = source_pointer.read_bytes()
    source_pointer_payload = json.loads(source_pointer_bytes)
    source_manifest = load_model_manifest(source_pointer)

    destination = tmp_path / "champion-export"
    result = export_champion_snapshot(source_pointer, profile, destination)

    assert result["role"] == "champion"
    assert result["model_identity"] == source_manifest.model_identity
    assert result["device"] == "mps"
    assert (destination / "champion.json").read_bytes() == source_pointer_bytes
    assert json.loads((destination / "champion.json").read_bytes()) == (
        source_pointer_payload
    )

    copied_config = load_server_config(destination / "starserve-mac.yaml")
    assert copied_config.host == "127.0.0.1"
    assert copied_config.device == "mps"
    assert copied_config.limits.max_concurrency == 1
    copied_experiment = load_config(copied_config.experiment_config)
    assert copied_experiment.profile == "standalone-smoke"
    assert copied_experiment.train.precision == "fp32"
    assert copied_experiment.train.compile is False

    relocated = tmp_path / "relocated"
    destination.rename(relocated)
    relocated_config = load_server_config(relocated / "starserve-mac.yaml")
    relocated_manifest = load_model_manifest(relocated_config.model_manifest)
    assert relocated_config.experiment_config.is_relative_to(relocated)
    assert relocated_config.model_manifest == relocated / "champion.json"
    assert relocated_manifest.model_identity == source_manifest.model_identity
    assert relocated_manifest.role == "champion"
    assert relocated_manifest.checkpoint.is_relative_to(relocated)

    checkpoint_bytes = bytearray(relocated_manifest.checkpoint.read_bytes())
    checkpoint_bytes[0] ^= 0xFF
    relocated_manifest.checkpoint.write_bytes(checkpoint_bytes)
    with pytest.raises(ValueError, match="SHA-256"):
        load_model_manifest(relocated_config.model_manifest)

    candidate_pointer = source_pointer.with_name("candidate.json")
    candidate_target = tmp_path / "candidate-export"
    with pytest.raises(ChampionSnapshotError, match="champion role"):
        export_champion_snapshot(candidate_pointer, profile, candidate_target)
    assert not candidate_target.exists()

    existing_target = tmp_path / "existing"
    existing_target.mkdir()
    sentinel = existing_target / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        export_champion_snapshot(source_pointer, profile, existing_target)
    assert sentinel.read_text(encoding="utf-8") == "keep"

    source_checkpoint = source_manifest.checkpoint
    damaged_source = bytearray(source_checkpoint.read_bytes())
    damaged_source[-1] ^= 0xFF
    source_checkpoint.write_bytes(damaged_source)
    corrupt_target = tmp_path / "corrupt-export"
    with pytest.raises(ChampionSnapshotError, match="SHA-256"):
        export_champion_snapshot(source_pointer, profile, corrupt_target)
    assert not corrupt_target.exists()
