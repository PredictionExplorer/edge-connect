"""Lineage transfer from synthetic previous-lineage artifacts."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from startrain.checkpoint import (
    CHECKPOINT_FORMAT,
    CHECKPOINT_VERSION,
    EMA_VERSION,
    LEGACY_MODEL_SCHEMA_VERSION,
    ExponentialMovingAverage,
    legacy_model_config,
    load_legacy_ema_checkpoint,
    save_checkpoint,
)
from startrain.contracts import (
    ACTION_LAYOUT_SCHEMA_ID,
    ACTION_LAYOUT_VERSION,
    FEATURE_SCHEMA_HASH,
    LEGACY_FEATURE_SCHEMA_HASH,
    LEGACY_RULES_HASH,
    LEGACY_RULES_HASH_WIRE,
    LEGACY_RULES_SCHEMA_ID,
    RULES_HASH,
    TARGET_TEACHER,
)
from startrain.lineage import (
    LINEAGE_TRANSFER_REPORT,
    TRANSFER_ACTOR_ID,
    LineageTransferError,
    list_legacy_shards,
    load_legacy_teacher,
    new_run_identity,
    partition_legacy_shards,
    select_recent_legacy_shards,
    transfer_lineage,
    transfer_lineage_parallel,
    write_transfer_report,
)
from startrain.model import GraphResTNet, ModelConfig
from startrain.replay import collate_replay_samples
from startrain.replay_store import ReplayStore
from startrain.runtime import RunIdentity, load_run_identity

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_replay import legacy_v4_shard, sample_for  # noqa: E402

LEGACY_MODEL = ModelConfig.legacy(
    width=8, rrt_groups=1, attention_heads=2, kv_heads=1, bottleneck_ratio=0.5
)


def _legacy_model_mapping() -> dict[str, object]:
    """The previous lineage's ModelConfig fields exactly as it stored them."""

    return {
        "node_feature_dim": 15,
        "global_feature_dim": 17,
        "width": 8,
        "rrt_groups": 1,
        "attention_heads": 2,
        "kv_heads": 1,
        "bottleneck_ratio": 0.5,
        "ff_multiplier": 2.0,
        "dropout": 0.0,
        "rms_norm_eps": 1e-6,
        "score_margin_min": LEGACY_MODEL.score_margin_min,
        "score_margin_max": LEGACY_MODEL.score_margin_max,
        "soft_policy_temperature": LEGACY_MODEL.soft_policy_temperature,
        "local_operator": "mean",
        "local_blocks_per_group": 2,
    }


def write_legacy_checkpoint(path: Path, *, seed: int = 3) -> Path:
    torch.manual_seed(seed)
    model = GraphResTNet(LEGACY_MODEL)
    ema = ExponentialMovingAverage(model)
    ema.update(model)
    save_checkpoint(
        path,
        model=model,
        step=1234,
        ema=ema,
        config={
            "model": _legacy_model_mapping(),
            "game": {"mode": "double", "pie_rule": False, "rings": [4, 6, 8, 10]},
        },
        extra={"run_id": "legacy-run", "generation_family": "legacy-family"},
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload.update(
        {
            "rules_schema": LEGACY_RULES_SCHEMA_ID,
            "rules_hash": LEGACY_RULES_HASH,
            "rules_hash_wire": LEGACY_RULES_HASH_WIRE,
            "feature_schema_hash": LEGACY_FEATURE_SCHEMA_HASH,
            "model_schema_version": LEGACY_MODEL_SCHEMA_VERSION,
        }
    )
    torch.save(payload, path)
    return path


def write_legacy_store(root: Path, games: int) -> list[Path]:
    """A previous-lineage replay root: manifest schema 4 plus v4 shards."""

    shards = root / "shards"
    shards.mkdir(parents=True)
    connection = sqlite3.connect(root / "manifest.sqlite3")
    connection.executescript(
        """
        CREATE TABLE store_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE shards (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            relative_path TEXT NOT NULL UNIQUE,
            created_ns INTEGER NOT NULL,
            sample_count INTEGER NOT NULL,
            ring INTEGER NOT NULL,
            phase_min INTEGER NOT NULL,
            phase_max INTEGER NOT NULL,
            model_version TEXT NOT NULL,
            model_step INTEGER NOT NULL,
            model_identity TEXT NOT NULL,
            run_id TEXT NOT NULL,
            generation_family TEXT NOT NULL,
            actor_id TEXT NOT NULL,
            generation INTEGER NOT NULL,
            game_count INTEGER NOT NULL,
            state TEXT NOT NULL DEFAULT 'ready',
            quarantine_reason TEXT,
            rules_hash TEXT NOT NULL,
            feature_schema_hash TEXT NOT NULL,
            checksum_sha256 TEXT NOT NULL
        );
        """
    )
    connection.executemany(
        "INSERT INTO store_metadata(key, value) VALUES (?, ?)",
        (
            ("manifest_schema_version", "4"),
            ("rules_hash", LEGACY_RULES_HASH_WIRE),
            ("feature_schema_hash", f"{LEGACY_FEATURE_SCHEMA_HASH:016x}"),
        ),
    )
    paths: list[Path] = []
    for index in range(games):
        ring = 4 if index % 2 == 0 else 6
        legacy_identity = "sha256-" + "1" * 64
        from dataclasses import replace

        sample = replace(
            sample_for(ring),
            game_id=f"legacy-game-{index}",
            ply=0,
            run_id="legacy-run",
            generation_family="legacy-family",
            actor_id="legacy-actor",
            model_identity=legacy_identity,
        )
        path = legacy_v4_shard(shards / f"shard-{index}.npz", [sample])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        connection.execute(
            """
            INSERT INTO shards(
                relative_path, created_ns, sample_count, ring, phase_min,
                phase_max, model_version, model_step, model_identity, run_id,
                generation_family, actor_id, generation, game_count, state,
                rules_hash, feature_schema_hash, checksum_sha256
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'ready', ?, ?, ?)
            """,
            (
                str(path.relative_to(root)),
                1_000 + index,
                1,
                ring,
                2,
                2,
                legacy_identity,
                500 + index,
                legacy_identity,
                "legacy-run",
                "legacy-family",
                "legacy-actor",
                0,
                1,
                f"{LEGACY_RULES_HASH:016x}",
                f"{LEGACY_FEATURE_SCHEMA_HASH:016x}",
                digest,
            ),
        )
        paths.append(path)
    connection.commit()
    connection.close()
    return paths


def test_legacy_checkpoint_loader_accepts_only_the_previous_lineage(tmp_path) -> None:
    checkpoint = write_legacy_checkpoint(tmp_path / "legacy.pt")
    assert legacy_model_config(_legacy_model_mapping()) == LEGACY_MODEL
    model = GraphResTNet(LEGACY_MODEL)
    metadata = load_legacy_ema_checkpoint(
        checkpoint, model=model, expected_model_config=LEGACY_MODEL
    )
    assert metadata["step"] == 1234
    assert metadata["model_schema_version"] == LEGACY_MODEL_SCHEMA_VERSION
    with pytest.raises(ValueError, match="incompatible"):
        load_legacy_ema_checkpoint(
            checkpoint,
            model=GraphResTNet(ModelConfig.legacy(width=16, rrt_groups=1)),
            expected_model_config=ModelConfig.legacy(
                width=16, rrt_groups=1, attention_heads=2, kv_heads=1
            ),
        )
    with pytest.raises(ValueError, match="previous feature schema"):
        load_legacy_ema_checkpoint(
            checkpoint,
            model=model,
            expected_model_config=ModelConfig(
                width=8, rrt_groups=1, attention_heads=2, kv_heads=1
            ),
        )
    # A current-lineage checkpoint is not a teacher.
    current = GraphResTNet(
        ModelConfig(width=8, rrt_groups=1, attention_heads=2, kv_heads=1)
    )
    ema = ExponentialMovingAverage(current)
    ema.update(current)
    save_checkpoint(
        tmp_path / "current.pt",
        model=current,
        step=1,
        ema=ema,
        config={"model": {"width": 8}, "game": {}},
    )
    with pytest.raises(ValueError, match="previous lineage"):
        load_legacy_ema_checkpoint(
            tmp_path / "current.pt", model=model, expected_model_config=LEGACY_MODEL
        )
    with pytest.raises(ValueError, match="relational_bias is invalid"):
        legacy_model_config({**_legacy_model_mapping(), "relational_bias": True})
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert payload["format"] == CHECKPOINT_FORMAT
    assert payload["version"] == CHECKPOINT_VERSION
    assert payload["ema"]["version"] == EMA_VERSION
    assert payload.get("action_layout_schema", ACTION_LAYOUT_SCHEMA_ID) == (
        ACTION_LAYOUT_SCHEMA_ID
    )
    assert payload.get("action_layout_version", ACTION_LAYOUT_VERSION) == (
        ACTION_LAYOUT_VERSION
    )


def test_lineage_transfer_labels_and_rehomes_legacy_replay(tmp_path) -> None:
    checkpoint = write_legacy_checkpoint(tmp_path / "legacy.pt")
    legacy_root = tmp_path / "legacy-replay"
    write_legacy_store(legacy_root, games=5)
    teacher = load_legacy_teacher(checkpoint)
    assert teacher.config == LEGACY_MODEL
    assert (
        teacher.identity
        == "sha256-" + hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    )
    assert not any(parameter.requires_grad for parameter in teacher.model.parameters())

    shards = list_legacy_shards(legacy_root)
    assert [shard.shard_id for shard in shards] == [1, 2, 3, 4, 5]
    assert list_legacy_shards(legacy_root, rings=(4,)) == [
        shard for shard in shards if shard.ring == 4
    ]
    recent = select_recent_legacy_shards(shards, max_samples=2)
    assert [shard.shard_id for shard in recent] == [4, 5]
    with pytest.raises(LineageTransferError, match="max_samples"):
        select_recent_legacy_shards(shards, max_samples=0)
    # Shards alternate rings 4, 6, 4, 6, 4: one per ring keeps the newest of each.
    per_ring = select_recent_legacy_shards(
        shards, max_samples=None, max_samples_per_ring=1
    )
    assert [(shard.shard_id, shard.ring) for shard in per_ring] == [(4, 6), (5, 4)]
    both = select_recent_legacy_shards(shards, max_samples=1, max_samples_per_ring=1)
    assert [shard.shard_id for shard in both] == [5]
    with pytest.raises(LineageTransferError, match="max_samples_per_ring"):
        select_recent_legacy_shards(shards, max_samples=None, max_samples_per_ring=0)

    output_root = tmp_path / "new-run"
    output_root.mkdir()
    identity = new_run_identity(
        output_root, run_id="variant-run", generation_family="variant-family"
    )
    assert load_run_identity(output_root / "run.json") == identity
    with pytest.raises(LineageTransferError, match="already exists"):
        new_run_identity(output_root, run_id="x", generation_family="y")

    report = transfer_lineage(
        teacher=teacher,
        legacy_shards=shards,
        replay_root=output_root / "replay",
        identity=identity,
        batch_size=2,
    )
    assert report["report"] == LINEAGE_TRANSFER_REPORT
    assert report["committed_shards"] == 5
    assert report["committed_samples"] == 5
    assert report["committed_games"] == 5
    assert report["samples_by_ring"] == {"4": 3, "6": 2}
    assert report["teacher"]["identity"] == teacher.identity
    written = write_transfer_report(output_root / "lineage-transfer.json", report)
    stored = json.loads(written.read_text(encoding="utf-8"))
    assert stored["digest"] and stored["actor_id"] == TRANSFER_ACTOR_ID

    with ReplayStore(output_root / "replay") as store:
        samples = store.load_recent_samples(
            sample_window=64,
            run_id=identity.run_id,
            generation_family=identity.generation_family,
            current_model_step=0,
            max_model_lag_steps=0,
        )
        assert len(samples) == 5
        for sample in samples:
            assert sample.rules_hash == RULES_HASH
            assert sample.feature_schema_hash == FEATURE_SCHEMA_HASH
            assert sample.run_id == identity.run_id
            assert sample.actor_id == TRANSFER_ACTOR_ID
            assert sample.model_identity == teacher.identity
            assert sample.has_teacher and sample.target_mask & TARGET_TEACHER
            assert not sample.history_known and sample.segment == "standard"
            assert sample.search_provenance.startswith("lineage-transfer:teacher=")
            assert sample.teacher_policy is not None
            legal = sample.stones == -1
            policy = sample.teacher_policy.astype(np.float32)
            assert np.isclose(policy.sum(), 1.0, atol=5e-3)
            assert not policy[~legal].any()
            assert sample.teacher_outcome is not None
            assert np.isclose(
                sample.teacher_outcome.astype(np.float32).sum(), 1.0, atol=5e-3
            )
            assert sample.teacher_score_margin is not None
            assert sample.teacher_score_margin.shape == (
                LEGACY_MODEL.score_margin_bins,
            )
        records = store.recent_shards(
            sample_window=64,
            run_id=identity.run_id,
            generation_family=identity.generation_family,
        )
        assert {record.model_step for record in records} == {0}
        assert {record.segment for record in records} == {"standard"}
        # Teacher targets collate into the KL branch of the loss.
        batch = collate_replay_samples(samples[:3])
        assert batch.targets.teacher_mask is not None
        assert bool(batch.targets.teacher_mask.all())

    # The legacy store was never modified.
    connection = sqlite3.connect(
        f"file:{legacy_root / 'manifest.sqlite3'}?mode=ro", uri=True
    )
    try:
        assert connection.execute("SELECT COUNT(*) FROM shards").fetchone()[0] == 5
        metadata = dict(connection.execute("SELECT key, value FROM store_metadata"))
        assert metadata["manifest_schema_version"] == "4"
    finally:
        connection.close()


def test_lineage_transfer_rejects_foreign_stores_and_tampered_shards(tmp_path) -> None:
    legacy_root = tmp_path / "legacy-replay"
    paths = write_legacy_store(legacy_root, games=2)
    with pytest.raises(LineageTransferError, match="missing"):
        list_legacy_shards(tmp_path / "nowhere")
    # A current-lineage store is not a transfer source.
    identity = RunIdentity(tmp_path / "run.json", "run-current", "family-current", 1)
    with ReplayStore(tmp_path / "current-replay") as store:
        store.register_run(identity)
    with pytest.raises(LineageTransferError, match="not the previous lineage"):
        list_legacy_shards(tmp_path / "current-replay")
    # Tampering with a shard is detected before any labelling.
    paths[0].write_bytes(paths[0].read_bytes() + b"\0")
    teacher = load_legacy_teacher(write_legacy_checkpoint(tmp_path / "legacy.pt"))
    shards = list_legacy_shards(legacy_root)
    output_root = tmp_path / "new-run"
    output_root.mkdir()
    run_identity = new_run_identity(output_root, run_id="r", generation_family="f")
    with pytest.raises(LineageTransferError, match="checksum"):
        transfer_lineage(
            teacher=teacher,
            legacy_shards=shards,
            replay_root=output_root / "replay",
            identity=run_identity,
        )


def test_prepare_lineage_transfer_cli(tmp_path, capsys) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import prepare_lineage_transfer

    checkpoint = write_legacy_checkpoint(tmp_path / "legacy.pt")
    legacy_root = tmp_path / "legacy-replay"
    write_legacy_store(legacy_root, games=3)
    output_root = tmp_path / "new-run"
    code = prepare_lineage_transfer.main(
        [
            "--legacy-checkpoint",
            str(checkpoint),
            "--legacy-replay-root",
            str(legacy_root),
            "--output-root",
            str(output_root),
            "--run-id",
            "variant-run",
            "--generation-family",
            "variant-family",
            "--max-samples",
            "2",
            "--batch-size",
            "1",
        ]
    )
    captured = capsys.readouterr()
    assert code == 0, captured.err
    summary = json.loads(captured.out.strip().splitlines()[-1])
    assert summary["committed_shards"] == 2
    assert summary["committed_samples"] == 2
    report = json.loads((output_root / "lineage-transfer.json").read_text())
    assert report["committed_samples"] == 2
    assert [shard["shard_id"] for shard in report["legacy_shards"]] == [2, 3]
    # A second run refuses to clobber the new identity.
    code = prepare_lineage_transfer.main(
        [
            "--legacy-checkpoint",
            str(checkpoint),
            "--legacy-replay-root",
            str(legacy_root),
            "--output-root",
            str(output_root),
            "--run-id",
            "variant-run",
            "--generation-family",
            "variant-family",
        ]
    )
    assert code == 1
    assert "already exists" in capsys.readouterr().err


def test_parallel_lineage_transfer_matches_the_serial_store(tmp_path) -> None:
    checkpoint = write_legacy_checkpoint(tmp_path / "legacy.pt")
    legacy_root = tmp_path / "legacy-replay"
    write_legacy_store(legacy_root, games=3)
    shards = list_legacy_shards(legacy_root)
    groups = partition_legacy_shards(shards, 2)
    assert sorted(shard.shard_id for group in groups for shard in group) == [
        shard.shard_id for shard in shards
    ]
    assert all(
        group == sorted(group, key=lambda shard: shard.shard_id) for group in groups
    )
    assert abs(
        sum(shard.sample_count for shard in groups[0])
        - sum(shard.sample_count for shard in groups[1])
    ) <= max(shard.sample_count for shard in shards)
    with pytest.raises(LineageTransferError, match="workers"):
        partition_legacy_shards(shards, 0)

    serial_root = tmp_path / "serial"
    serial_root.mkdir()
    serial_identity = new_run_identity(
        serial_root, run_id="variant-run", generation_family="variant-family"
    )
    teacher = load_legacy_teacher(checkpoint, device=torch.device("cpu"))
    serial = transfer_lineage(
        teacher=teacher,
        legacy_shards=shards,
        replay_root=serial_root / "replay",
        identity=serial_identity,
        batch_size=4,
    )
    parallel_root = tmp_path / "parallel"
    parallel_root.mkdir()
    parallel_identity = new_run_identity(
        parallel_root, run_id="variant-run", generation_family="variant-family"
    )
    parallel = transfer_lineage_parallel(
        checkpoint=checkpoint,
        device="cpu,cpu",
        legacy_shards=shards,
        replay_root=parallel_root / "replay",
        identity=parallel_identity,
        workers=2,
        batch_size=4,
    )
    for key in (
        "committed_shards",
        "committed_samples",
        "committed_games",
        "samples_by_ring",
    ):
        assert parallel[key] == serial[key], key
    assert parallel["legacy_shards"] == serial["legacy_shards"]
    assert parallel["actor_id"] == TRANSFER_ACTOR_ID and parallel["generation"] is None
    assert sorted(worker["actor_id"] for worker in parallel["workers"]) == [
        f"{TRANSFER_ACTOR_ID}-0",
        f"{TRANSFER_ACTOR_ID}-1",
    ]
    assert (
        sum(worker["committed_samples"] for worker in parallel["workers"])
        == (parallel["committed_samples"])
    )
    # Both reports pin the same evidence.
    write_transfer_report(parallel_root / "lineage-transfer.json", parallel)
    write_transfer_report(serial_root / "lineage-transfer.json", serial)
    assert (
        json.loads((parallel_root / "lineage-transfer.json").read_text())["digest"]
        == (json.loads((serial_root / "lineage-transfer.json").read_text())["digest"])
    )
    with sqlite3.connect(parallel_root / "replay" / "manifest.sqlite3") as connection:
        rows = connection.execute(
            "SELECT actor_id, sample_count FROM shards WHERE state = 'ready'"
        ).fetchall()
    assert sum(count for _, count in rows) == serial["committed_samples"]
    assert {actor for actor, _ in rows} == {
        f"{TRANSFER_ACTOR_ID}-0",
        f"{TRANSFER_ACTOR_ID}-1",
    }


@pytest.mark.native
def test_lineage_arena_pits_the_candidate_against_the_legacy_teacher(
    tmp_path, capsys
) -> None:
    pytest.importorskip("star_native")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_lineage_arena

    legacy = write_legacy_checkpoint(tmp_path / "legacy.pt")
    candidate_config = ModelConfig(width=8, rrt_groups=1, attention_heads=2, kv_heads=1)
    candidate_model = GraphResTNet(candidate_config)
    ema = ExponentialMovingAverage(candidate_model)
    ema.update(candidate_model)
    from dataclasses import asdict

    from startrain.config import GameConfig

    candidate = tmp_path / "candidate.pt"
    save_checkpoint(
        candidate,
        model=candidate_model,
        step=7,
        ema=ema,
        config={"model": asdict(candidate_config), "game": asdict(GameConfig())},
        extra={"run_id": "variant-run", "generation_family": "variant-family"},
    )
    output = tmp_path / "lineage-arena.json"
    code = run_lineage_arena.main(
        [
            "--candidate-checkpoint",
            str(candidate),
            "--legacy-checkpoint",
            str(legacy),
            "--output",
            str(output),
            "--rings",
            "4",
            "--pairs-per-ring",
            "2",
            "--simulations",
            "2",
            "--max-considered",
            "2",
        ]
    )
    captured = capsys.readouterr()
    assert code == 0, captured.err
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["result_kind"] == run_lineage_arena.RESULT_KIND
    assert result["evaluation_mode"] == run_lineage_arena.EVALUATION_MODE
    assert result["aggregate"]["games"] == 4
    assert result["baseline_metadata"]["kind"] == "legacy_champion"
    assert result["baseline_metadata"]["feature_schema_version"] == 3
    assert result["candidate_metadata"]["feature_schema_version"] == 4
    assert {pair["segment"] for pair in result["pairs"]} == {"standard"}
    # The legacy checkpoint is not accepted as a candidate.
    code = run_lineage_arena.main(
        [
            "--candidate-checkpoint",
            str(legacy),
            "--legacy-checkpoint",
            str(legacy),
            "--output",
            str(tmp_path / "other.json"),
            "--rings",
            "4",
            "--pairs-per-ring",
            "2",
            "--simulations",
            "1",
        ]
    )
    assert code == 1
    assert "lineage arena failed" in capsys.readouterr().err


def write_legacy_publication(root: Path) -> Path:
    """A previous-lineage learner directory: champion pointer -> manifest -> checkpoint."""

    from startrain.checkpoint import sha256_file

    learner = root / "learner"
    (learner / "checkpoints").mkdir(parents=True)
    (learner / "manifests").mkdir(parents=True)
    staged = write_legacy_checkpoint(root / "staged.pt")
    digest = sha256_file(staged)
    checkpoint = learner / "checkpoints" / f"sha256-{digest}.pt"
    staged.rename(checkpoint)
    manifest_payload = {
        "format": "startrain.model-manifest",
        "schema_version": 3,
        "rules_hash": LEGACY_RULES_HASH_WIRE,
        "feature_schema_hash": f"{LEGACY_FEATURE_SCHEMA_HASH:016x}",
        "model_schema_version": LEGACY_MODEL_SCHEMA_VERSION,
        "weights": "ema",
        "model_identity": f"sha256-{digest}",
        "model_version": f"sha256-{digest}",
        "model_step": 1234,
        "created_ns": 1,
        "run_id": "legacy-run",
        "generation_family": "legacy-family",
        "checkpoint": f"../checkpoints/sha256-{digest}.pt",
        "checkpoint_sha256": digest,
        "checkpoint_bytes": checkpoint.stat().st_size,
    }
    manifest_text = json.dumps(manifest_payload, sort_keys=True)
    manifest_digest = hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()
    manifest = learner / "manifests" / f"manifest-{manifest_digest}.json"
    manifest.write_text(manifest_text, encoding="utf-8")
    pointer = learner / "champion.json"
    pointer.write_text(
        json.dumps(
            {
                "format": "startrain.model-pointer",
                "schema_version": 2,
                "role": "champion",
                "manifest": f"manifests/{manifest.name}",
                "manifest_sha256": manifest_digest,
                "manifest_bytes": manifest.stat().st_size,
                "model_identity": f"sha256-{digest}",
                "model_step": 1234,
                "run_id": "legacy-run",
                "generation_family": "legacy-family",
            }
        ),
        encoding="utf-8",
    )
    return pointer


def test_legacy_champion_pointer_resolves_to_a_verified_checkpoint(tmp_path) -> None:
    from startrain.lineage import resolve_legacy_champion

    pointer = write_legacy_publication(tmp_path / "legacy")
    checkpoint = resolve_legacy_champion(pointer)
    assert checkpoint.is_file() and checkpoint.name.startswith("sha256-")
    teacher = load_legacy_teacher(checkpoint)
    assert teacher.step == 1234
    # Tampering with the checkpoint or pointing at a candidate is refused.
    checkpoint.write_bytes(checkpoint.read_bytes() + b"\0")
    with pytest.raises(LineageTransferError):
        resolve_legacy_champion(pointer)
    payload = json.loads(pointer.read_text(encoding="utf-8"))
    payload["role"] = "candidate"
    pointer.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(LineageTransferError, match="champion pointer"):
        resolve_legacy_champion(pointer)
