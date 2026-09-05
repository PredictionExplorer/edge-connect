from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess

import pytest

from scripts import benchmark_cpu_actor as benchmark
import startrain.checkpoint as checkpoints
from startrain.config import load_config


def manifest(tmp_path):
    return SimpleNamespace(
        model_identity="sha256-" + "b" * 64,
        model_version="model-current",
        model_step=123,
        checkpoint=tmp_path / "current.pt",
        checkpoint_sha256="b" * 64,
        checkpoint_bytes=321,
        artifact_manifest=tmp_path / "immutable.json",
        path=tmp_path / "champion.json",
        manifest_sha256="a" * 64,
        run_id="actual-run",
        generation_family="actual-family",
    )


def arguments():
    return [
        "--config",
        "configs/small.yaml",
        "--cpu-affinity",
        "0-7",
        "--native-threads",
        "4",
        "--blas-threads",
        "2",
        "--batch-sizes",
        "8",
    ]


def test_execution_requires_explicit_checkpoint_or_random_initialization():
    with pytest.raises(SystemExit) as error:
        benchmark.main([*arguments(), "--execute"])
    assert error.value.code == 2


def test_checkpoint_loader_checks_contract_identity_integrity_and_step(
    tmp_path, monkeypatch
):
    published = manifest(tmp_path)
    config = load_config(Path(__file__).parents[1] / "configs/small.yaml")
    model = object()
    calls = []
    monkeypatch.setattr(checkpoints, "load_model_manifest", lambda _: published)

    def loader(path, **kwargs):
        calls.append((path, kwargs))
        return {"step": 123}

    monkeypatch.setattr(checkpoints, "load_ema_checkpoint", loader)
    result = benchmark._load_actor_weights(config, model, published.path, "a" * 64)
    assert (
        result["model_identity"] == published.model_identity
        and result["model_step"] == 123
    )
    path, kwargs = calls[0]
    assert path == published.checkpoint and kwargs["model"] is model
    assert kwargs["expected_model_config"] == asdict(config.model)
    assert kwargs["expected_game_config"] == asdict(config.game)
    assert kwargs["expected_sha256"] == published.checkpoint_sha256
    assert kwargs["expected_bytes"] == published.checkpoint_bytes
    assert kwargs["expected_run_id"] == published.run_id
    assert kwargs["expected_generation_family"] == published.generation_family
    with pytest.raises(ValueError, match="changed after"):
        benchmark._load_actor_weights(config, model, published.path, "c" * 64)
    monkeypatch.setattr(
        checkpoints, "load_ema_checkpoint", lambda *args, **kwargs: {"step": 122}
    )
    with pytest.raises(ValueError, match="step differs"):
        benchmark._load_actor_weights(config, model, published.path, "a" * 64)


@pytest.mark.parametrize("timed_out", [False, True])
def test_parent_pins_immutable_manifest_and_removes_child_replay_on_timeout(
    tmp_path, monkeypatch, capsys, timed_out
):
    published = manifest(tmp_path)
    monkeypatch.setattr(checkpoints, "load_model_manifest", lambda _: published)
    inspected = []

    def run(command, **kwargs):
        assert command[command.index("--checkpoint") + 1] == str(
            published.artifact_manifest
        )
        assert (
            command[command.index("--checkpoint-manifest-sha256") + 1]
            == published.manifest_sha256
        )
        work = Path(command[command.index("--work-dir") + 1])
        assert work.is_dir()
        (work / "partial-replay").write_text("diagnostic")
        inspected.append(work)
        if timed_out:
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return SimpleNamespace(
            returncode=0, stdout=json.dumps({"status": "measured"}), stderr=""
        )

    monkeypatch.setattr(benchmark.subprocess, "run", run)
    code = benchmark.main(
        [*arguments(), "--execute", "--checkpoint", str(published.path)]
    )
    assert code == int(timed_out) and all(not path.exists() for path in inspected)
    report = json.loads(capsys.readouterr().out)
    assert report["checkpoint"]["model_identity"] == published.model_identity
    assert report["cases"][0]["status"] == ("timeout" if timed_out else "measured")
