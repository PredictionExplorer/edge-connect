"""Verified, relocatable exports of one atomically published champion."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from startrain.checkpoint import (
    MODEL_POINTER_FORMAT,
    MODEL_POINTER_VERSION,
    load_model_manifest,
    verify_file,
)
from startrain.config import load_config
from startrain.contracts import (
    ACTION_LAYOUT_SCHEMA_ID,
    EXTERNAL_FEATURE_SCHEMA_ID,
    FEATURE_SCHEMA_HASH,
    RULES_HASH_WIRE,
    RULES_SCHEMA_ID,
)

from .config import (
    SERVER_CONFIG_SCHEMA_VERSION,
    LimitConfig,
    SearchConfig,
    load_server_config,
)
from .runtime import AtomicModelManager


class ChampionSnapshotError(ValueError):
    """The requested champion cannot form a safe, verified snapshot."""


def export_champion_snapshot(
    champion_pointer: str | Path,
    experiment_profile: str | Path,
    destination: str | Path,
) -> dict[str, object]:
    """Export exactly one champion publication and a Mac-local serving setup."""

    target = Path(destination).expanduser().resolve(strict=False)
    if os.path.lexists(target):
        raise FileExistsError(f"snapshot destination already exists: {target}")

    pointer = _resolve_file(champion_pointer, "champion pointer")
    source_root = pointer.parent
    pointer_bytes = _read_bytes(pointer, "champion pointer")
    pointer_payload = _json_object(pointer_bytes, "champion pointer")
    if (
        pointer_payload.get("format") != MODEL_POINTER_FORMAT
        or pointer_payload.get("schema_version") != MODEL_POINTER_VERSION
    ):
        raise ChampionSnapshotError("source is not a supported atomic model pointer")
    if pointer_payload.get("role") != "champion":
        raise ChampionSnapshotError("snapshot source must have champion role")
    manifest_source, manifest_relative = _artifact_path(
        source_root,
        source_root,
        pointer_payload.get("manifest"),
        "manifest",
    )

    try:
        champion = load_model_manifest(pointer)
    except (OSError, ValueError) as exc:
        raise ChampionSnapshotError(f"champion publication is invalid: {exc}") from exc
    _require_unchanged_pointer(pointer, pointer_bytes)
    if champion.role != "champion":
        raise ChampionSnapshotError("snapshot source must have champion role")
    loaded_manifest = champion.artifact_manifest or champion.path
    if loaded_manifest.resolve() != manifest_source:
        raise ChampionSnapshotError(
            "champion pointer resolved a different immutable manifest"
        )

    manifest_bytes = _read_bytes(manifest_source, "immutable model manifest")
    if (
        len(manifest_bytes) != champion.manifest_bytes
        or hashlib.sha256(manifest_bytes).hexdigest() != champion.manifest_sha256
    ):
        raise ChampionSnapshotError("immutable model manifest changed during export")
    manifest_payload = _json_object(manifest_bytes, "immutable model manifest")
    checkpoint_source, checkpoint_relative = _artifact_path(
        source_root,
        manifest_source.parent,
        manifest_payload.get("checkpoint"),
        "checkpoint",
    )
    if champion.checkpoint.resolve() != checkpoint_source:
        raise ChampionSnapshotError(
            "immutable manifest resolved a different checkpoint"
        )

    profile = _resolve_file(experiment_profile, "experiment profile")
    try:
        load_config(profile)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise ChampionSnapshotError(f"experiment profile is invalid: {exc}") from exc
    profile_bytes = _read_bytes(profile, "experiment profile")
    derived_profile_bytes = _derived_profile(profile_bytes)

    source_profile_relative = Path("profiles") / profile.name
    derived_profile_relative = Path("profiles") / f"{profile.stem}-mac-serving.yaml"
    server_config_relative = Path("starserve-mac.yaml")
    publication_pointer_relative = Path("champion.json")
    _require_distinct_paths(
        publication_pointer_relative,
        manifest_relative,
        checkpoint_relative,
        source_profile_relative,
        derived_profile_relative,
        server_config_relative,
    )

    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        target.mkdir()
    except FileExistsError:
        raise FileExistsError(
            f"snapshot destination already exists: {target}"
        ) from None

    try:
        checkpoint_target = target / checkpoint_relative
        _copy_verified(
            checkpoint_source,
            checkpoint_target,
            expected_sha256=champion.checkpoint_sha256,
            expected_bytes=champion.checkpoint_bytes,
        )

        manifest_target = target / manifest_relative
        _write_new(manifest_target, manifest_bytes)
        verify_file(
            manifest_target,
            expected_sha256=champion.manifest_sha256,
            expected_bytes=champion.manifest_bytes,
        )

        pointer_target = target / publication_pointer_relative
        _write_new(pointer_target, pointer_bytes)
        copied_champion = load_model_manifest(pointer_target)
        if (
            copied_champion.role != "champion"
            or copied_champion.model_identity != champion.model_identity
        ):
            raise ChampionSnapshotError(
                "copied champion identity or publication role changed"
            )

        source_profile_target = target / source_profile_relative
        _write_new(source_profile_target, profile_bytes)
        load_config(source_profile_target)

        derived_profile_target = target / derived_profile_relative
        _write_new(derived_profile_target, derived_profile_bytes)
        derived_experiment = load_config(derived_profile_target)
        if derived_experiment.train.precision != "fp32":
            raise ChampionSnapshotError("derived serving profile is not FP32")
        if derived_experiment.train.compile:
            raise ChampionSnapshotError("derived serving profile enables compilation")

        server_config_target = target / server_config_relative
        _write_new(
            server_config_target,
            yaml.safe_dump(
                _mac_server_config(
                    experiment_config=derived_profile_relative,
                    model_manifest=publication_pointer_relative,
                ),
                sort_keys=False,
            ).encode("utf-8"),
        )
        server_config = load_server_config(server_config_target)
        if (
            server_config.host != "127.0.0.1"
            or server_config.device != "mps"
            or server_config.limits.max_concurrency != 1
        ):
            raise ChampionSnapshotError("generated Mac server configuration is unsafe")

        # Verify that the copied EMA checkpoint, derived model profile, and champion
        # pointer load together. CPU verification is portable to the export host;
        # the generated runtime remains MPS-first.
        verifier = AtomicModelManager(
            replace(server_config, device="cpu"),
            experiment=derived_experiment,
        )
        verifier.startup()
        verified_health = verifier.health()
        if (
            verified_health.get("model_identity") != champion.model_identity
            or verified_health.get("role") != "champion"
        ):
            raise ChampionSnapshotError("verified model identity or role changed")
        del verifier

        _require_unchanged_pointer(pointer, pointer_bytes)
    except BaseException:
        shutil.rmtree(target, ignore_errors=True)
        raise

    return {
        "destination": str(target),
        "server_config": str(target / server_config_relative),
        "source_profile": str(target / source_profile_relative),
        "experiment_config": str(target / derived_profile_relative),
        "model_manifest": str(target / publication_pointer_relative),
        "checkpoint": str(target / checkpoint_relative),
        "model_identity": champion.model_identity,
        "model_step": champion.model_step,
        "role": champion.role,
        "device": "mps",
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=("Export one verified champion publication for Mac-local starserve")
    )
    parser.add_argument(
        "--champion", required=True, help="atomic champion.json pointer"
    )
    parser.add_argument(
        "--profile",
        required=True,
        help="experiment YAML used to create the champion",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="new destination directory; existing paths are never overwritten",
    )
    arguments = parser.parse_args(argv)
    result = export_champion_snapshot(
        arguments.champion,
        arguments.profile,
        arguments.output,
    )
    print(json.dumps(result, sort_keys=True))


def _resolve_file(path: str | Path, name: str) -> Path:
    try:
        resolved = Path(path).expanduser().resolve(strict=True)
    except OSError as exc:
        raise ChampionSnapshotError(f"cannot resolve {name}: {exc}") from exc
    if not resolved.is_file():
        raise ChampionSnapshotError(f"{name} is not a regular file: {resolved}")
    return resolved


def _read_bytes(path: Path, name: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ChampionSnapshotError(f"cannot read {name} {path}: {exc}") from exc


def _json_object(data: bytes, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ChampionSnapshotError(f"{name} is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ChampionSnapshotError(f"{name} must be a JSON object")
    return payload


def _artifact_path(
    root: Path,
    base: Path,
    value: object,
    name: str,
) -> tuple[Path, Path]:
    if not isinstance(value, str) or not value:
        raise ChampionSnapshotError(f"{name} path is missing")
    fragment = Path(value)
    if fragment.is_absolute():
        raise ChampionSnapshotError(
            f"{name} path must be relative for a relocatable snapshot"
        )
    logical = Path(os.path.abspath(base / fragment))
    try:
        relative = logical.relative_to(root)
    except ValueError as exc:
        raise ChampionSnapshotError(
            f"{name} path escapes the publication root"
        ) from exc
    try:
        resolved = logical.resolve(strict=True)
    except OSError as exc:
        raise ChampionSnapshotError(f"cannot resolve published {name}: {exc}") from exc
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ChampionSnapshotError(
            f"{name} symlink escapes the publication root"
        ) from exc
    if logical != resolved:
        raise ChampionSnapshotError(
            f"{name} path uses a symlink and cannot be preserved safely"
        )
    if not resolved.is_file():
        raise ChampionSnapshotError(f"published {name} is not a regular file")
    return resolved, relative


def _derived_profile(profile_bytes: bytes) -> bytes:
    try:
        raw = yaml.safe_load(profile_bytes)
    except yaml.YAMLError as exc:
        raise ChampionSnapshotError(
            f"experiment profile is invalid YAML: {exc}"
        ) from exc
    if not isinstance(raw, Mapping):
        raise ChampionSnapshotError("experiment profile root must be a mapping")
    output = dict(raw)
    train = output.get("train")
    if not isinstance(train, Mapping):
        raise ChampionSnapshotError("experiment profile train section is missing")
    serving_train = dict(train)
    serving_train["precision"] = "fp32"
    serving_train["compile"] = False
    output["train"] = serving_train
    return yaml.safe_dump(output, sort_keys=False).encode("utf-8")


def _mac_server_config(
    *,
    experiment_config: Path,
    model_manifest: Path,
) -> dict[str, object]:
    search = SearchConfig()
    limits = LimitConfig(max_concurrency=1)
    return {
        "schema_version": SERVER_CONFIG_SCHEMA_VERSION,
        "experiment_config": experiment_config.as_posix(),
        "model_manifest": model_manifest.as_posix(),
        "device": "mps",
        "host": "127.0.0.1",
        "port": 8080,
        "rules_schema_id": RULES_SCHEMA_ID,
        "rules_hash": RULES_HASH_WIRE,
        "feature_schema_id": EXTERNAL_FEATURE_SCHEMA_ID,
        "feature_schema_hash": f"{FEATURE_SCHEMA_HASH:016x}",
        "action_schema_id": ACTION_LAYOUT_SCHEMA_ID,
        "search": {
            "default_simulations": search.default_simulations,
            "maximum_simulations": search.maximum_simulations,
            "default_max_considered": search.default_max_considered,
            "maximum_max_considered": search.maximum_max_considered,
            "c_visit": search.c_visit,
            "c_scale": search.c_scale,
        },
        "limits": {
            "max_concurrency": limits.max_concurrency,
            "max_request_bytes": limits.max_request_bytes,
            "request_timeout_seconds": limits.request_timeout_seconds,
            "queue_timeout_seconds": limits.queue_timeout_seconds,
        },
        "security": {
            "cors_allow_origins": [],
            "bearer_token_env": None,
        },
    }


def _require_distinct_paths(*paths: Path) -> None:
    normalized = [path.as_posix() for path in paths]
    if len(set(normalized)) != len(normalized):
        raise ChampionSnapshotError("snapshot artifacts would overwrite each other")


def _write_new(destination: Path, data: bytes) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _copy_verified(
    source: Path,
    destination: Path,
    *,
    expected_sha256: str,
    expected_bytes: int,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as input_stream, destination.open("xb") as output_stream:
        shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)
        output_stream.flush()
        os.fsync(output_stream.fileno())
    verify_file(
        destination,
        expected_sha256=expected_sha256,
        expected_bytes=expected_bytes,
    )


def _require_unchanged_pointer(pointer: Path, expected: bytes) -> None:
    if _read_bytes(pointer, "champion pointer") != expected:
        raise ChampionSnapshotError(
            "champion pointer changed during export; retry the snapshot"
        )


if __name__ == "__main__":
    main()
