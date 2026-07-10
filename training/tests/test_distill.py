from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from startrain.distill import (
    BrowserSearchConfig,
    DistillationConfig,
    DistillationExportConfig,
    DistillationLossConfig,
    DistillationRunner,
    DistillationTrainConfig,
    ReplaySourceConfig,
    TeacherConfig,
    _teacher_kl_losses,
    sha256_file,
)
from startrain.features import DoubleStarPosition
from startrain.model import ModelConfig, StarModelOutput
from startrain.publish import publish_browser_artifacts
from startrain.replay import ReplaySample, write_replay_shard
from startrain.scoring import score_position
from startrain.topology import get_topology


def replay_sample() -> ReplaySample:
    topology = get_topology(3)
    stones = torch.full((topology.n,), -1, dtype=torch.int8)
    position = DoubleStarPosition(
        rings=3,
        stones=stones,
        to_move=0,
        moves_left=1,
        opening=True,
        pass_streak=0,
        terminal=False,
    )
    policy = np.full(topology.n + 1, 1.0 / (topology.n + 1), dtype=np.float32)
    return ReplaySample.from_position(
        position,
        policy=policy,
        final_score=score_position(topology, stones),
        search_provenance="distill-test-search",
        policy_provenance="completed-q",
    )


def test_distillation_smoke_emits_checksum_verified_browser_manifest(
    tmp_path,
    monkeypatch,
) -> None:
    shard = write_replay_shard(tmp_path / "replay.npz", [replay_sample()])
    output = tmp_path / "browser"
    config = DistillationConfig(
        replay=ReplaySourceConfig((shard,)),
        teacher=TeacherConfig(),
        student=ModelConfig(
            width=8,
            rrt_groups=1,
            attention_heads=2,
            kv_heads=1,
        ),
        train=DistillationTrainConfig(
            steps=1,
            batch_size=1,
            device="cpu",
            ema_decay=0.9,
            d5_augmentation=False,
        ),
        loss=DistillationLossConfig(
            policy=1.0,
            wdl=1.0,
            score_margin=0.1,
            ownership=0.1,
            alive=0.1,
        ),
        export=DistillationExportConfig(
            output_directory=output,
            model_version="browser-smoke-v1",
            recommended_search=BrowserSearchConfig(
                simulations=8,
                max_considered=4,
            ),
        ),
    )

    artifacts = DistillationRunner(config).run()
    manifest = json.loads(artifacts.manifest.read_text())

    assert artifacts.checkpoint.is_file()
    assert artifacts.onnx.is_file()
    assert artifacts.checkpoint_sha256 == sha256_file(artifacts.checkpoint)
    assert artifacts.onnx_sha256 == sha256_file(artifacts.onnx)
    assert manifest["artifacts"]["onnx"]["sha256"] == artifacts.onnx_sha256
    assert manifest["artifacts"]["checkpoint"]["sha256"] == artifacts.checkpoint_sha256
    assert manifest["precision"] == "float16"
    assert manifest["architecture"]["all_size"] is True
    assert manifest["tensors"]["inputs"]["node_features"]["shape"] == [
        "batch",
        "nodes",
        15,
    ]
    assert manifest["recommended_local_search"]["simulations"] == 8
    assert artifacts.final_losses["total"] >= 0
    wasm_source = tmp_path / "wasm-build"
    wasm_source.mkdir()
    (wasm_source / "star_wasm.js").write_text("export default async function init() {}")
    (wasm_source / "star_wasm_bg.wasm").write_bytes(b"\x00asm\x01\x00\x00\x00")
    replacements = []
    real_replace = os.replace

    def recording_replace(source, destination):
        replacements.append(str(destination))
        return real_replace(source, destination)

    monkeypatch.setattr("startrain.publish.os.replace", recording_replace)
    published = publish_browser_artifacts(
        artifacts.manifest,
        tmp_path / "public" / "models" / "star",
        wasm_source_directory=wasm_source,
    )
    canonical = tmp_path / "public" / "models" / "star" / "manifest.json"
    assert replacements[-1] == str(canonical)
    assert published["manifest"] == str(canonical)
    assert json.loads(canonical.read_text()) == manifest
    published_onnx = tmp_path / "public" / "models" / "star" / artifacts.onnx.name
    assert sha256_file(published_onnx) == artifacts.onnx_sha256
    assert (tmp_path / "public" / "models" / "star" / "wasm" / "star_wasm.js").is_file()
    assert (
        (tmp_path / "public" / "models" / "star" / "wasm" / "star_wasm_bg.wasm")
        .read_bytes()
        .startswith(b"\x00asm")
    )
    marker = tmp_path / "wasm-build.ok"
    publish_browser_artifacts(
        artifacts.manifest,
        tmp_path / "public-with-build",
        wasm_build_command=(
            sys.executable,
            "-c",
            f"from pathlib import Path; Path({str(marker)!r}).write_text('ok')",
        ),
        wasm_working_directory=tmp_path,
        wasm_source_directory=wasm_source,
    )
    assert marker.read_text() == "ok"
    failed_target = tmp_path / "failed-release"
    with pytest.raises(subprocess.CalledProcessError):
        publish_browser_artifacts(
            artifacts.manifest,
            failed_target,
            wasm_build_command=(sys.executable, "-c", "raise SystemExit(3)"),
            wasm_source_directory=wasm_source,
        )
    assert not (failed_target / "manifest.json").exists()

    with artifacts.onnx.open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="byte length|SHA-256"):
        publish_browser_artifacts(
            artifacts.manifest,
            tmp_path / "rejected-publication",
            wasm_source_directory=wasm_source,
        )


def test_teacher_logit_kl_covers_all_distilled_heads() -> None:
    output = StarModelOutput(
        policy_logits=torch.tensor([[2.0, 0.0, -1.0]]),
        wdl_logits=torch.tensor([[1.0, 0.0, -1.0]]),
        score_margin_logits=torch.zeros(1, 363),
        ownership_logits=torch.tensor([[[1.0, 0.0, -1.0], [0.0, 1.0, -1.0]]]),
        alive_logits=torch.tensor([[1.0, -1.0]]),
        soft_policy_logits=torch.zeros(1, 3),
    )
    losses = _teacher_kl_losses(
        output,
        output,
        legal_action_mask=torch.tensor([[True, False, True]]),
        node_mask=torch.tensor([[True, False]]),
        temperature=2.0,
    )
    assert set(losses) == {
        "policy_kl",
        "wdl_kl",
        "score_margin_kl",
        "ownership_kl",
        "alive_kl",
    }
    assert all(
        value.item() == pytest.approx(0.0, abs=1e-7) for value in losses.values()
    )
