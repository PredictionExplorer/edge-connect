"""Strict configuration for the production inference service."""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, TypeVar, cast
from urllib.parse import urlsplit

import yaml

from startrain.contracts import (
    ACTION_LAYOUT_SCHEMA_ID,
    EXTERNAL_FEATURE_SCHEMA_ID,
    FEATURE_SCHEMA_HASH,
    RULES_HASH_WIRE,
    RULES_SCHEMA_ID,
)

SERVER_CONFIG_SCHEMA_VERSION = 2
_T = TypeVar("_T")


class ServerConfigError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class SearchConfig:
    default_simulations: int = 512
    maximum_simulations: int = 4_096
    default_max_considered: int = 16
    maximum_max_considered: int = 64
    c_visit: float = 50.0
    c_scale: float = 1.0

    def __post_init__(self) -> None:
        integers = (
            self.default_simulations,
            self.maximum_simulations,
            self.default_max_considered,
            self.maximum_max_considered,
        )
        if any(type(value) is not int or value <= 0 for value in integers):
            raise ServerConfigError("search budgets must be positive integers")
        if self.default_simulations > self.maximum_simulations:
            raise ServerConfigError("default simulations exceed the configured maximum")
        if self.default_max_considered > self.maximum_max_considered:
            raise ServerConfigError("default max_considered exceeds its maximum")
        if self.c_visit <= 0 or self.c_scale <= 0:
            raise ServerConfigError("Gumbel search constants must be positive")

    def named_presets(self) -> dict[str, dict[str, int]]:
        return {
            "quick": {
                "simulations": min(128, self.default_simulations),
                "max_considered": min(8, self.default_max_considered),
            },
            "strong": {
                "simulations": self.default_simulations,
                "max_considered": self.default_max_considered,
            },
            "maximum": {
                "simulations": self.maximum_simulations,
                "max_considered": self.maximum_max_considered,
            },
        }


@dataclass(frozen=True, slots=True)
class LimitConfig:
    max_concurrency: int = 2
    max_request_bytes: int = 1_048_576
    request_timeout_seconds: float = 60.0
    queue_timeout_seconds: float = 5.0

    def __post_init__(self) -> None:
        if type(self.max_concurrency) is not int or self.max_concurrency <= 0:
            raise ServerConfigError("max_concurrency must be a positive integer")
        if type(self.max_request_bytes) is not int or self.max_request_bytes <= 0:
            raise ServerConfigError("max_request_bytes must be a positive integer")
        if self.request_timeout_seconds <= 0 or self.queue_timeout_seconds <= 0:
            raise ServerConfigError("service timeouts must be positive")


@dataclass(frozen=True, slots=True)
class SecurityConfig:
    cors_allow_origins: tuple[str, ...] = ()
    bearer_token_env: str | None = None

    def __post_init__(self) -> None:
        if len(set(self.cors_allow_origins)) != len(self.cors_allow_origins):
            raise ServerConfigError("CORS origins must be unique")
        for origin in self.cors_allow_origins:
            try:
                parsed = urlsplit(origin) if isinstance(origin, str) else None
            except ValueError:
                parsed = None
            if (
                not isinstance(origin, str)
                or not origin
                or origin == "*"
                or parsed is None
                or parsed.scheme not in ("http", "https")
                or not parsed.hostname
                or parsed.username is not None
                or parsed.password is not None
                or parsed.path
                or parsed.query
                or parsed.fragment
            ):
                raise ServerConfigError(
                    "CORS origins must be explicit http(s) origins; wildcard is forbidden"
                )
        if self.bearer_token_env is not None and (
            not isinstance(self.bearer_token_env, str)
            or not self.bearer_token_env
            or "=" in self.bearer_token_env
        ):
            raise ServerConfigError(
                "bearer_token_env must name one environment variable"
            )

    def bearer_token(self) -> str | None:
        if self.bearer_token_env is None:
            return None
        token = os.environ.get(self.bearer_token_env)
        if token is None or not token.strip():
            raise ServerConfigError(
                f"bearer token environment variable {self.bearer_token_env!r} is unset"
            )
        return token


@dataclass(frozen=True, slots=True)
class ServerConfig:
    experiment_config: Path
    model_manifest: Path
    device: str = "cuda:0"
    host: str = "0.0.0.0"
    port: int = 8080
    schema_version: int = SERVER_CONFIG_SCHEMA_VERSION
    rules_schema_id: str = RULES_SCHEMA_ID
    rules_hash: str = RULES_HASH_WIRE
    feature_schema_id: str = EXTERNAL_FEATURE_SCHEMA_ID
    feature_schema_hash: str = f"{FEATURE_SCHEMA_HASH:016x}"
    action_schema_id: str = ACTION_LAYOUT_SCHEMA_ID
    search: SearchConfig = SearchConfig()
    limits: LimitConfig = LimitConfig()
    security: SecurityConfig = SecurityConfig()

    def __post_init__(self) -> None:
        if self.schema_version != SERVER_CONFIG_SCHEMA_VERSION:
            raise ServerConfigError("server configuration schema_version must be 2")
        expected = (
            (self.rules_schema_id, RULES_SCHEMA_ID, "rules schema"),
            (self.rules_hash, RULES_HASH_WIRE, "rules hash"),
            (self.feature_schema_id, EXTERNAL_FEATURE_SCHEMA_ID, "feature schema"),
            (
                self.feature_schema_hash,
                f"{FEATURE_SCHEMA_HASH:016x}",
                "feature schema hash",
            ),
            (self.action_schema_id, ACTION_LAYOUT_SCHEMA_ID, "action schema"),
        )
        for actual, required, name in expected:
            if actual != required:
                raise ServerConfigError(f"configured {name} is incompatible")
        if not isinstance(self.device, str) or not self.device:
            raise ServerConfigError("device must be non-empty")
        if not isinstance(self.host, str) or not self.host:
            raise ServerConfigError("host must be non-empty")
        if type(self.port) is not int or not 1 <= self.port <= 65_535:
            raise ServerConfigError("port must be in 1..65535")


def _mapping(name: str, value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ServerConfigError(f"{name} must be a mapping")
    return dict(value)


def _construct(cls: type[_T], values: object) -> _T:
    mapping = _mapping(cls.__name__, values)
    allowed = {field.name for field in fields(cast(Any, cls))}
    unknown = set(mapping) - allowed
    if unknown:
        raise ServerConfigError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    try:
        return cls(**mapping)
    except (TypeError, ValueError) as exc:
        raise ServerConfigError(f"invalid {cls.__name__}: {exc}") from exc


def load_server_config(path: str | Path) -> ServerConfig:
    source = Path(path)
    try:
        with source.open("r", encoding="utf-8") as stream:
            raw = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as exc:
        raise ServerConfigError(f"cannot read server configuration: {exc}") from exc
    values = _mapping("server configuration", raw)
    required = {"schema_version", "experiment_config", "model_manifest", "device"}
    allowed = {field.name for field in fields(ServerConfig)}
    missing = required - set(values)
    unknown = set(values) - allowed
    if missing or unknown:
        raise ServerConfigError(
            f"server configuration has missing keys {sorted(missing)} "
            f"and unknown keys {sorted(unknown)}"
        )
    for name, cls in (
        ("search", SearchConfig),
        ("limits", LimitConfig),
        ("security", SecurityConfig),
    ):
        nested = _mapping(name, values.get(name, {}))
        if name == "security" and "cors_allow_origins" in nested:
            origins = nested["cors_allow_origins"]
            if not isinstance(origins, list):
                raise ServerConfigError("cors_allow_origins must be a list")
            nested["cors_allow_origins"] = tuple(origins)
        values[name] = _construct(cls, nested)
    root = source.resolve().parent
    for name in ("experiment_config", "model_manifest"):
        value = values[name]
        if not isinstance(value, str) or not value:
            raise ServerConfigError(f"{name} must be a non-empty path")
        resolved = Path(value)
        if not resolved.is_absolute():
            resolved = root / resolved
        values[name] = resolved.resolve()
    return _construct(ServerConfig, values)
