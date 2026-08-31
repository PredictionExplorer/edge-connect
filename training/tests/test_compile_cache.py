from __future__ import annotations

import os
from pathlib import Path

import pytest

from startrain.training import (
    COMPILE_CACHE_SCHEMA_VERSION,
    configure_isolated_compile_cache,
    isolated_compile_cache,
)

_ENVIRONMENT = (
    "HOME",
    "TORCHINDUCTOR_CACHE_DIR",
    "TORCHINDUCTOR_PERSISTENT_AUTOTUNE_DIR",
    "TRITON_HOME",
    "TRITON_CACHE_DIR",
    "TRITON_DUMP_DIR",
    "TRITON_OVERRIDE_DIR",
    "XDG_CACHE_HOME",
    "CUDA_CACHE_PATH",
)


def test_isolated_compile_cache_is_private_idempotent_and_restores_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in _ENVIRONMENT:
        monkeypatch.setenv(name, f"/before/{name.lower()}")
    output = tmp_path / "arm"

    with isolated_compile_cache(output) as provenance:
        document = provenance.as_dict()
        assert document["schema_version"] == COMPILE_CACHE_SCHEMA_VERSION
        assert provenance.root == output / "compile-cache" / "v1"
        assert all(Path(value).is_dir() for value in provenance.environment.values())
        assert all(
            os.environ[name] == value for name, value in provenance.environment.items()
        )
        assert not list(provenance.root.glob(".write-test-*"))

    assert {name: os.environ[name] for name in _ENVIRONMENT} == {
        name: f"/before/{name.lower()}" for name in _ENVIRONMENT
    }

    with isolated_compile_cache(output) as repeated:
        assert repeated.root == provenance.root
        assert repeated.environment == provenance.environment


def test_compile_cache_roots_are_isolated_per_arm(tmp_path: Path) -> None:
    with isolated_compile_cache(tmp_path / "control") as first:
        pass
    with isolated_compile_cache(tmp_path / "clip-2") as second:
        pass

    assert first.root != second.root
    assert first.root not in second.root.parents
    assert second.root not in first.root.parents


def test_compile_cache_rejects_symlink_or_non_directory_paths(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    symlink = tmp_path / "arm"
    symlink.symlink_to(target, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        configure_isolated_compile_cache(symlink)

    regular = tmp_path / "regular"
    regular.mkdir()
    (regular / "compile-cache").write_text("unsafe\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not a regular directory"):
        configure_isolated_compile_cache(regular)


@pytest.mark.parametrize(
    "name",
    (
        "TRITON_CACHE_MANAGER",
        "TRITON_REMOTE_CACHE_BACKEND",
        "TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE",
        "TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE",
        "TORCHINDUCTOR_AUTOGRAD_REMOTE_CACHE",
        "TORCHINDUCTOR_FORCE_DISABLE_CACHES",
        "TORCHINDUCTOR_BUNDLED_AUTOTUNE_REMOTE_CACHE",
        "TORCHINDUCTOR_FX_GRAPH_CACHE",
        "TORCHINDUCTOR_AUTOGRAD_CACHE",
        "TORCH_DYNAMO_AUTOMATIC_DYNAMIC_LOCAL_PGO",
        "TORCH_DYNAMO_AUTOMATIC_DYNAMIC_REMOTE_PGO",
        "TORCH_COMPILE_JOB_ID",
        "TORCH_COMPILE_FORCE_DISABLE_CACHES",
        "CUDA_CACHE_DISABLE",
        "TRITON_FUTURE_UNPINNED_CONTROL",
    ),
)
def test_compile_cache_rejects_inherited_remote_or_disabled_controls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    monkeypatch.setenv(name, "1")

    with pytest.raises(ValueError, match="unsupported inherited compiler controls"):
        configure_isolated_compile_cache(tmp_path / "arm")
