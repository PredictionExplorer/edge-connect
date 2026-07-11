"""Verified atomic publication of distilled browser artifacts."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

from .checkpoint import sha256_file, verify_file
from .contracts import RULES_HASH_HEX
from .distill import BROWSER_MANIFEST_FORMAT, BROWSER_MANIFEST_SCHEMA_VERSION

WASM_ASSET_DIRECTORY = f"wasm-{RULES_HASH_HEX}"


def publish_browser_artifacts(
    manifest_path: str | Path,
    target_directory: str | Path,
    *,
    wasm_build_command: Sequence[str] = (),
    wasm_working_directory: str | Path | None = None,
    wasm_source_directory: str | Path | None = None,
) -> dict[str, object]:
    source_manifest = Path(manifest_path).resolve()
    with source_manifest.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if (
        not isinstance(payload, dict)
        or payload.get("format") != BROWSER_MANIFEST_FORMAT
        or payload.get("schema_version") != BROWSER_MANIFEST_SCHEMA_VERSION
    ):
        raise ValueError("unsupported browser distillation manifest")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("browser manifest artifacts are missing")
    resolved: dict[str, Path] = {}
    for name in ("onnx", "checkpoint"):
        entry = artifacts.get(name)
        if not isinstance(entry, Mapping):
            raise ValueError(f"browser manifest {name} artifact is missing")
        filename = entry.get("file")
        if (
            not isinstance(filename, str)
            or not filename
            or Path(filename).name != filename
        ):
            raise ValueError(f"browser manifest {name} filename is unsafe")
        source = source_manifest.parent / filename
        verify_file(
            source,
            expected_sha256=_sha256(entry.get("sha256")),
            expected_bytes=_positive_int("bytes", entry.get("bytes")),
        )
        resolved[name] = source

    if wasm_build_command:
        subprocess.run(
            list(wasm_build_command),
            cwd=(
                str(Path(wasm_working_directory).resolve())
                if wasm_working_directory is not None
                else None
            ),
            check=True,
        )
    target = Path(target_directory).resolve()
    target.mkdir(parents=True, exist_ok=True)
    wasm_source = (
        Path(wasm_source_directory).resolve()
        if wasm_source_directory is not None
        else target / WASM_ASSET_DIRECTORY
    )
    wasm_sources = {
        "module": wasm_source / "star_wasm.js",
        "binary": wasm_source / "star_wasm_bg.wasm",
    }
    _verify_wasm(wasm_sources["module"], wasm_sources["binary"])

    wasm_target = target / WASM_ASSET_DIRECTORY
    wasm_target.mkdir(parents=True, exist_ok=True)
    destination_onnx = target / resolved["onnx"].name
    destination_manifest = target / "manifest.json"
    destinations = {
        "module": wasm_target / "star_wasm.js",
        "binary": wasm_target / "star_wasm_bg.wasm",
        "onnx": destination_onnx,
        "manifest": destination_manifest,
    }
    staged: dict[str, Path] = {}
    try:
        staged["module"] = _stage_copy(wasm_sources["module"], destinations["module"])
        staged["binary"] = _stage_copy(wasm_sources["binary"], destinations["binary"])
        staged["onnx"] = _stage_copy(resolved["onnx"], destination_onnx)
        staged["manifest"] = _stage_copy(source_manifest, destination_manifest)
        _verify_wasm(staged["module"], staged["binary"])
        onnx_entry = artifacts["onnx"]
        verify_file(
            staged["onnx"],
            expected_sha256=str(onnx_entry["sha256"]),
            expected_bytes=int(onnx_entry["bytes"]),
        )
        verify_file(
            staged["manifest"],
            expected_sha256=sha256_file(source_manifest),
            expected_bytes=source_manifest.stat().st_size,
        )
        # Runtime assets are committed first. The canonical model manifest is
        # the release pointer and is atomically replaced strictly last.
        for name in ("module", "binary", "onnx", "manifest"):
            os.replace(staged[name], destinations[name])
            staged.pop(name)
        _fsync_directory(target)
        _fsync_directory(wasm_target)
    finally:
        for temporary in staged.values():
            temporary.unlink(missing_ok=True)
    return {
        "manifest": str(destination_manifest),
        "onnx": str(destination_onnx),
        "model_version": payload.get("model_version"),
        "wasm_build_invoked": bool(wasm_build_command),
        "wasm_module": str(destinations["module"]),
        "wasm_binary": str(destinations["binary"]),
        "wasm_module_sha256": sha256_file(destinations["module"]),
        "wasm_binary_sha256": sha256_file(destinations["binary"]),
    }


def publish_browser_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Verify and atomically publish distilled browser artifacts"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--wasm-cwd")
    parser.add_argument("--wasm-source")
    parser.add_argument("--wasm-build", nargs=argparse.REMAINDER, default=())
    arguments = parser.parse_args(argv)
    result = publish_browser_artifacts(
        arguments.manifest,
        arguments.target,
        wasm_build_command=arguments.wasm_build,
        wasm_working_directory=arguments.wasm_cwd,
        wasm_source_directory=arguments.wasm_source,
    )
    print(json.dumps(result, sort_keys=True))


def _stage_copy(source: Path, destination: Path) -> Path:
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            with source.open("rb") as stream:
                shutil.copyfileobj(stream, temporary)
            temporary.flush()
            os.fsync(temporary.fileno())
        staged = Path(temporary_name)
        temporary_name = None
        return staged
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _verify_wasm(module: Path, binary: Path) -> None:
    if not module.is_file() or module.stat().st_size <= 0:
        raise ValueError("WASM JavaScript module is missing or empty")
    try:
        javascript = module.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("WASM JavaScript module is not UTF-8") from exc
    if not any(token in javascript for token in ("export", "WebAssembly")):
        raise ValueError("WASM JavaScript module lacks an export/runtime")
    if not binary.is_file() or binary.stat().st_size < 8:
        raise ValueError("WASM binary is missing or too small")
    with binary.open("rb") as stream:
        if stream.read(4) != b"\x00asm":
            raise ValueError("WASM binary magic is invalid")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sha256(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("artifact SHA-256 is invalid")
    return value


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"artifact {name} must be a positive integer")
    return value
