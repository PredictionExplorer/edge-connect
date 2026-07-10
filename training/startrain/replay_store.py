"""Durable immutable replay shards with a SQLite WAL manifest."""

from __future__ import annotations

import hashlib
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

from .contracts import FEATURE_SCHEMA_HASH, RULES_HASH, RULES_HASH_WIRE
from .replay import ReplaySample, read_replay_shard, write_replay_shard
from .runtime import RunIdentity, validate_identifier

MANIFEST_SCHEMA_VERSION = 3


class DuplicateGameError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ShardRecord:
    shard_id: int
    path: Path
    created_ns: int
    sample_count: int
    ring: int
    phase_min: int
    phase_max: int
    model_version: str
    model_step: int
    model_identity: str
    run_id: str
    generation_family: str
    actor_id: str
    generation: int
    game_count: int
    checksum_sha256: str
    state: str
    quarantine_reason: str | None


@dataclass(frozen=True, slots=True)
class ReplaySpan:
    record: ShardRecord
    sample_start: int
    sample_count: int


@dataclass(frozen=True, slots=True)
class ReplaySelection:
    spans: tuple[ReplaySpan, ...]
    samples_by_ring: dict[int, int]
    max_shard_id: int

    @property
    def sample_count(self) -> int:
        return sum(self.samples_by_ring.values())


@dataclass(frozen=True, slots=True)
class ReplayCursor:
    shard_id: int = 0
    sample_offset: int = 0


class ReplayStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.shard_directory = self.root / "shards"
        self.quarantine_directory = self.root / "quarantine"
        self.root.mkdir(parents=True, exist_ok=True)
        self.shard_directory.mkdir(parents=True, exist_ok=True)
        self.quarantine_directory.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.sqlite3"
        self.connection = sqlite3.connect(
            self.manifest_path, timeout=30.0, isolation_level=None
        )
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.execute("PRAGMA busy_timeout=30000")
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self._initialize()

    def _initialize(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS store_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS shards (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                relative_path TEXT NOT NULL UNIQUE,
                created_ns INTEGER NOT NULL,
                sample_count INTEGER NOT NULL CHECK(sample_count > 0),
                ring INTEGER NOT NULL,
                phase_min INTEGER NOT NULL,
                phase_max INTEGER NOT NULL,
                model_version TEXT NOT NULL,
                model_step INTEGER NOT NULL,
                model_identity TEXT NOT NULL,
                run_id TEXT NOT NULL,
                generation_family TEXT NOT NULL,
                actor_id TEXT NOT NULL,
                generation INTEGER NOT NULL CHECK(generation >= 0),
                game_count INTEGER NOT NULL CHECK(game_count > 0),
                state TEXT NOT NULL DEFAULT 'ready',
                quarantine_reason TEXT,
                rules_hash TEXT NOT NULL,
                feature_schema_hash TEXT NOT NULL,
                checksum_sha256 TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS shards_ring_created
                ON shards(ring, created_ns DESC);
            CREATE INDEX IF NOT EXISTS shards_model_step
                ON shards(model_step DESC);
            CREATE INDEX IF NOT EXISTS shards_eligibility
                ON shards(run_id, generation_family, ring, model_step DESC, id DESC);
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                generation_family TEXT NOT NULL UNIQUE,
                created_ns INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS actor_generations (
                run_id TEXT NOT NULL,
                generation_family TEXT NOT NULL,
                actor_id TEXT NOT NULL,
                generation INTEGER NOT NULL CHECK(generation >= 0),
                leased_ns INTEGER NOT NULL,
                PRIMARY KEY(run_id, generation_family, actor_id),
                FOREIGN KEY(run_id) REFERENCES runs(run_id)
            );
            CREATE TABLE IF NOT EXISTS games (
                game_id TEXT PRIMARY KEY,
                shard_id INTEGER NOT NULL,
                run_id TEXT NOT NULL,
                generation_family TEXT NOT NULL,
                actor_id TEXT NOT NULL,
                generation INTEGER NOT NULL,
                ring INTEGER NOT NULL,
                model_identity TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS gc_watermarks (
                name TEXT PRIMARY KEY,
                minimum_shard_id INTEGER NOT NULL,
                maximum_shard_id INTEGER NOT NULL,
                updated_ns INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS cursors (
                name TEXT PRIMARY KEY,
                shard_id INTEGER NOT NULL,
                sample_offset INTEGER NOT NULL,
                updated_ns INTEGER NOT NULL
            );
            """
        )
        expected = {
            "manifest_schema_version": str(MANIFEST_SCHEMA_VERSION),
            "rules_hash": RULES_HASH_WIRE,
            "feature_schema_hash": f"{FEATURE_SCHEMA_HASH:016x}",
        }
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            for key, value in expected.items():
                row = self.connection.execute(
                    "SELECT value FROM store_metadata WHERE key = ?", (key,)
                ).fetchone()
                if row is None:
                    self.connection.execute(
                        "INSERT INTO store_metadata(key, value) VALUES (?, ?)",
                        (key, value),
                    )
                elif row["value"] != value:
                    raise ValueError(f"replay store {key} is incompatible")
            self.connection.execute("COMMIT")
        except Exception:
            self.connection.execute("ROLLBACK")
            raise
        self.reconciliation_metrics = self.reconcile_orphans()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "ReplayStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def register_run(self, identity: RunIdentity) -> None:
        self.connection.execute(
            """
            INSERT INTO runs(run_id, generation_family, created_ns)
            VALUES (?, ?, ?)
            ON CONFLICT(run_id) DO NOTHING
            """,
            (identity.run_id, identity.generation_family, identity.created_ns),
        )
        row = self.connection.execute(
            "SELECT generation_family FROM runs WHERE run_id = ?",
            (identity.run_id,),
        ).fetchone()
        if row is None or row["generation_family"] != identity.generation_family:
            raise ValueError("run_id is registered to another generation family")

    def lease_generation(self, identity: RunIdentity, actor_id: str) -> int:
        actor = validate_identifier("actor_id", actor_id)
        self.register_run(identity)
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            row = self.connection.execute(
                """
                SELECT generation FROM actor_generations
                WHERE run_id = ? AND generation_family = ? AND actor_id = ?
                """,
                (identity.run_id, identity.generation_family, actor),
            ).fetchone()
            generation = 0 if row is None else int(row["generation"]) + 1
            self.connection.execute(
                """
                INSERT INTO actor_generations(
                    run_id, generation_family, actor_id, generation, leased_ns
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(run_id, generation_family, actor_id) DO UPDATE SET
                    generation = excluded.generation,
                    leased_ns = excluded.leased_ns
                """,
                (
                    identity.run_id,
                    identity.generation_family,
                    actor,
                    generation,
                    time.time_ns(),
                ),
            )
            self.connection.execute("COMMIT")
        except Exception:
            self.connection.execute("ROLLBACK")
            raise
        return generation

    def reconcile_orphans(self) -> dict[str, int]:
        referenced = {
            str(row["relative_path"])
            for row in self.connection.execute("SELECT relative_path FROM shards")
        }
        removed_files = 0
        now = time.time()
        for path in self.shard_directory.glob("*"):
            relative = str(path.relative_to(self.root))
            if (
                path.is_file()
                and (path.suffix == ".npz" or path.name.endswith(".tmp"))
                and relative not in referenced
                and now - path.stat().st_mtime >= 300.0
            ):
                path.unlink(missing_ok=True)
                removed_files += 1
        missing: list[int] = []
        corrupt: list[tuple[int, str, str]] = []
        for row in self.connection.execute(
            """
            SELECT id, relative_path, checksum_sha256 FROM shards
            WHERE state = 'ready'
            """
        ):
            source = self.root / str(row["relative_path"])
            if not source.is_file():
                missing.append(int(row["id"]))
                continue
            if _sha256(source) != str(row["checksum_sha256"]):
                quarantine = self.quarantine_directory / (
                    f"corrupt-{int(row['id']):012d}-{source.name}"
                )
                os.replace(source, quarantine)
                corrupt.append(
                    (
                        int(row["id"]),
                        str(quarantine.relative_to(self.root)),
                        "checksum mismatch",
                    )
                )
        if missing or corrupt:
            self.connection.execute("BEGIN IMMEDIATE")
            try:
                self.connection.executemany(
                    """
                    UPDATE shards
                    SET state = 'quarantined', quarantine_reason = ?
                    WHERE id = ?
                    """,
                    (("committed file missing", shard_id) for shard_id in missing),
                )
                self.connection.executemany(
                    """
                    UPDATE shards
                    SET relative_path = ?, state = 'quarantined',
                        quarantine_reason = ?
                    WHERE id = ?
                    """,
                    (
                        (relative_path, reason, shard_id)
                        for shard_id, relative_path, reason in corrupt
                    ),
                )
                self.connection.execute("COMMIT")
            except Exception:
                self.connection.execute("ROLLBACK")
                raise
        return {
            "orphan_files": removed_files,
            "missing_committed": len(missing),
            "corrupt_committed": len(corrupt),
        }

    def set_gc_watermark(self, name: str, selection: ReplaySelection) -> None:
        if not name or not selection.spans:
            raise ValueError("GC watermark requires a name and replay spans")
        shard_ids = [span.record.shard_id for span in selection.spans]
        self.connection.execute(
            """
            INSERT INTO gc_watermarks(
                name, minimum_shard_id, maximum_shard_id, updated_ns
            ) VALUES (?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                minimum_shard_id = excluded.minimum_shard_id,
                maximum_shard_id = excluded.maximum_shard_id,
                updated_ns = excluded.updated_ns
            """,
            (name, min(shard_ids), max(shard_ids), time.time_ns()),
        )

    def clear_gc_watermark(self, name: str) -> None:
        self.connection.execute("DELETE FROM gc_watermarks WHERE name = ?", (name,))

    def collect_garbage(
        self,
        *,
        run_id: str,
        generation_family: str,
        retain_shards_per_ring: int,
        dry_run: bool,
    ) -> dict[str, int]:
        if retain_shards_per_ring <= 0:
            raise ValueError("retain_shards_per_ring must be positive")
        protected_ranges = [
            (int(row["minimum_shard_id"]), int(row["maximum_shard_id"]))
            for row in self.connection.execute(
                "SELECT minimum_shard_id, maximum_shard_id FROM gc_watermarks"
            )
        ]
        candidates: list[ShardRecord] = []
        for ring in range(3, 13):
            rows = self.connection.execute(
                """
                SELECT * FROM shards
                WHERE state = 'ready'
                  AND run_id = ?
                  AND generation_family = ?
                  AND ring = ?
                ORDER BY id DESC
                """,
                (run_id, generation_family, ring),
            )
            retained = 0
            for row in rows:
                record = self._record(row)
                is_protected = any(
                    lower <= record.shard_id <= upper
                    for lower, upper in protected_ranges
                )
                if is_protected or retained < retain_shards_per_ring:
                    retained += 1
                    continue
                candidates.append(record)
        bytes_reclaimable = sum(
            record.path.stat().st_size for record in candidates if record.path.is_file()
        )
        metrics = {
            "candidate_shards": len(candidates),
            "candidate_bytes": bytes_reclaimable,
            "deleted_shards": 0,
            "deleted_bytes": 0,
            "protected_watermarks": len(protected_ranges),
            "dry_run": int(dry_run),
        }
        if dry_run or not candidates:
            return metrics
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            self.connection.executemany(
                "DELETE FROM shards WHERE id = ?",
                ((record.shard_id,) for record in candidates),
            )
            self.connection.execute("COMMIT")
        except Exception:
            self.connection.execute("ROLLBACK")
            raise
        for record in candidates:
            record.path.unlink(missing_ok=True)
        metrics["deleted_shards"] = len(candidates)
        metrics["deleted_bytes"] = bytes_reclaimable
        return metrics

    def append(
        self,
        samples: Sequence[ReplaySample],
        *,
        phase_min: int,
        phase_max: int,
        model_version: str,
        model_step: int,
        model_identity: str,
        run_id: str,
        generation_family: str,
        actor_id: str,
        generation: int,
    ) -> ShardRecord:
        if not samples:
            raise ValueError("cannot append an empty replay shard")
        rings = {sample.rings for sample in samples}
        if len(rings) != 1:
            raise ValueError("replay shards must be ring-homogeneous")
        if phase_min < 0 or phase_max < phase_min:
            raise ValueError("invalid phase range")
        if not model_version:
            raise ValueError("model_version must be non-empty")
        if model_step < 0:
            raise ValueError("model_step must be non-negative")
        for name, value in (
            ("model_identity", model_identity),
            ("run_id", run_id),
            ("generation_family", generation_family),
            ("actor_id", actor_id),
        ):
            validate_identifier(name, value)
        if not model_identity.startswith("sha256-") or model_version != model_identity:
            raise ValueError(
                "replay model version must equal its content-addressed identity"
            )
        if (
            isinstance(generation, bool)
            or not isinstance(generation, int)
            or generation < 0
        ):
            raise ValueError("generation must be a non-negative integer")
        run = self.connection.execute(
            "SELECT generation_family FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if run is None or run["generation_family"] != generation_family:
            raise ValueError("replay run is not registered to this generation family")
        lease = self.connection.execute(
            """
            SELECT generation FROM actor_generations
            WHERE run_id = ? AND generation_family = ? AND actor_id = ?
            """,
            (run_id, generation_family, actor_id),
        ).fetchone()
        if lease is None or int(lease["generation"]) != generation:
            raise ValueError("replay generation is not the actor's active lease")
        if any(
            sample.rules_hash != RULES_HASH
            or sample.feature_schema_hash != FEATURE_SCHEMA_HASH
            for sample in samples
        ):
            raise ValueError("sample contract hashes are incompatible")
        if any(
            sample.run_id != run_id
            or sample.generation_family != generation_family
            or sample.actor_id != actor_id
            or sample.generation != generation
            or sample.model_identity != model_identity
            for sample in samples
        ):
            raise ValueError("sample provenance disagrees with shard provenance")
        game_plys: dict[str, set[int]] = {}
        game_samples: dict[str, int] = {}
        for sample in samples:
            game_plys.setdefault(sample.game_id, set()).add(sample.ply)
            game_samples[sample.game_id] = game_samples.get(sample.game_id, 0) + 1
        for game_id, plys in game_plys.items():
            if len(plys) != game_samples[game_id] or plys != set(range(len(plys))):
                raise ValueError(
                    f"game {game_id} has missing or duplicate ply provenance"
                )

        created_ns = time.time_ns()
        filename = f"shard-{created_ns}-{uuid.uuid4().hex}.npz"
        destination = self.shard_directory / filename
        write_replay_shard(destination, samples, compressed=True)
        checksum = _sha256(destination)
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            active_lease = self.connection.execute(
                """
                SELECT generation FROM actor_generations
                WHERE run_id = ? AND generation_family = ? AND actor_id = ?
                """,
                (run_id, generation_family, actor_id),
            ).fetchone()
            if active_lease is None or int(active_lease["generation"]) != generation:
                raise ValueError("actor generation lease changed before replay commit")
            cursor = self.connection.execute(
                """
                INSERT INTO shards(
                    relative_path, created_ns, sample_count, ring,
                    phase_min, phase_max, model_version, model_step,
                    model_identity, run_id, generation_family, actor_id,
                    generation, game_count,
                    rules_hash, feature_schema_hash, checksum_sha256
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(destination.relative_to(self.root)),
                    created_ns,
                    len(samples),
                    next(iter(rings)),
                    phase_min,
                    phase_max,
                    model_version,
                    model_step,
                    model_identity,
                    run_id,
                    generation_family,
                    actor_id,
                    generation,
                    len(game_plys),
                    f"{RULES_HASH:016x}",
                    f"{FEATURE_SCHEMA_HASH:016x}",
                    checksum,
                ),
            )
            if cursor.lastrowid is None:
                raise RuntimeError("replay shard insert did not return an identifier")
            shard_id = int(cursor.lastrowid)
            try:
                self.connection.executemany(
                    """
                    INSERT INTO games(
                        game_id, shard_id, run_id, generation_family,
                        actor_id, generation, ring, model_identity
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        (
                            game_id,
                            shard_id,
                            run_id,
                            generation_family,
                            actor_id,
                            generation,
                            next(iter(rings)),
                            model_identity,
                        )
                        for game_id in game_plys
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise DuplicateGameError(
                    "replay manifest already contains a completed game ID"
                ) from exc
            self.connection.execute("COMMIT")
        except Exception:
            self.connection.execute("ROLLBACK")
            destination.unlink(missing_ok=True)
            raise
        return ShardRecord(
            shard_id=shard_id,
            path=destination,
            created_ns=created_ns,
            sample_count=len(samples),
            ring=next(iter(rings)),
            phase_min=phase_min,
            phase_max=phase_max,
            model_version=model_version,
            model_step=model_step,
            model_identity=model_identity,
            run_id=run_id,
            generation_family=generation_family,
            actor_id=actor_id,
            generation=generation,
            game_count=len(game_plys),
            checksum_sha256=checksum,
            state="ready",
            quarantine_reason=None,
        )

    def recent_shards(
        self,
        *,
        sample_window: int,
        run_id: str,
        generation_family: str,
        rings: Sequence[int] | None = None,
        current_model_step: int | None = None,
        max_model_lag_steps: int | None = None,
        maximum_shard_id: int | None = None,
    ) -> list[ShardRecord]:
        if sample_window <= 0:
            raise ValueError("sample_window must be positive")
        clauses = [
            "state = 'ready'",
            "rules_hash = ?",
            "feature_schema_hash = ?",
            "run_id = ?",
            "generation_family = ?",
        ]
        parameters: list[object] = [
            f"{RULES_HASH:016x}",
            f"{FEATURE_SCHEMA_HASH:016x}",
            validate_identifier("run_id", run_id),
            validate_identifier("generation_family", generation_family),
        ]
        if rings:
            placeholders = ",".join("?" for _ in rings)
            clauses.append(f"ring IN ({placeholders})")
            parameters.extend(int(ring) for ring in rings)
        if max_model_lag_steps is not None:
            if current_model_step is None or max_model_lag_steps < 0:
                raise ValueError(
                    "bounded lag requires current step and non-negative lag"
                )
            clauses.append("model_step >= ?")
            parameters.append(max(0, current_model_step - max_model_lag_steps))
        if current_model_step is not None:
            if current_model_step < 0:
                raise ValueError("current_model_step must be non-negative")
            clauses.append("model_step <= ?")
            parameters.append(current_model_step)
        if maximum_shard_id is not None:
            if maximum_shard_id < 0:
                raise ValueError("maximum_shard_id must be non-negative")
            clauses.append("id <= ?")
            parameters.append(maximum_shard_id)
        rows = self.connection.execute(
            f"""
            SELECT * FROM shards
            WHERE {" AND ".join(clauses)}
            ORDER BY id DESC
            """,
            parameters,
        )
        selected: list[ShardRecord] = []
        samples = 0
        for row in rows:
            record = self._record(row)
            selected.append(record)
            samples += record.sample_count
            if samples >= sample_window:
                break
        selected.reverse()
        return selected

    def sample_counts_by_ring(
        self,
        rings: Sequence[int] = tuple(range(3, 13)),
        *,
        run_id: str,
        generation_family: str,
    ) -> dict[int, int]:
        requested = tuple(int(ring) for ring in rings)
        if not requested or len(set(requested)) != len(requested):
            raise ValueError("rings must be a non-empty unique sequence")
        placeholders = ",".join("?" for _ in requested)
        rows = self.connection.execute(
            f"""
            SELECT ring, COALESCE(SUM(sample_count), 0) AS samples
            FROM shards
            WHERE rules_hash = ?
              AND feature_schema_hash = ?
              AND state = 'ready'
              AND run_id = ?
              AND generation_family = ?
              AND ring IN ({placeholders})
            GROUP BY ring
            """,
            (
                f"{RULES_HASH:016x}",
                f"{FEATURE_SCHEMA_HASH:016x}",
                validate_identifier("run_id", run_id),
                validate_identifier("generation_family", generation_family),
                *requested,
            ),
        )
        counts = {ring: 0 for ring in requested}
        for row in rows:
            counts[int(row["ring"])] = int(row["samples"])
        return counts

    def available_sample_count(
        self,
        *,
        run_id: str,
        generation_family: str,
        current_model_step: int | None = None,
        max_model_lag_steps: int | None = None,
    ) -> int:
        clauses = [
            "state = 'ready'",
            "rules_hash = ?",
            "feature_schema_hash = ?",
            "run_id = ?",
            "generation_family = ?",
        ]
        parameters: list[object] = [
            f"{RULES_HASH:016x}",
            f"{FEATURE_SCHEMA_HASH:016x}",
            validate_identifier("run_id", run_id),
            validate_identifier("generation_family", generation_family),
        ]
        if max_model_lag_steps is not None:
            if current_model_step is None or max_model_lag_steps < 0:
                raise ValueError(
                    "bounded lag requires current step and non-negative lag"
                )
            clauses.append("model_step >= ?")
            parameters.append(max(0, current_model_step - max_model_lag_steps))
        if current_model_step is not None:
            if current_model_step < 0:
                raise ValueError("current_model_step must be non-negative")
            clauses.append("model_step <= ?")
            parameters.append(current_model_step)
        row = self.connection.execute(
            f"""
            SELECT COALESCE(SUM(sample_count), 0) AS samples
            FROM shards
            WHERE {" AND ".join(clauses)}
            """,
            parameters,
        ).fetchone()
        return int(row["samples"])

    def eligible_sample_counts(
        self,
        rings: Sequence[int],
        *,
        run_id: str,
        generation_family: str,
        current_model_step: int,
        max_model_lag_steps: int,
    ) -> dict[int, int]:
        requested = tuple(int(ring) for ring in rings)
        if not requested:
            raise ValueError("rings must be non-empty")
        lower = max(0, current_model_step - max_model_lag_steps)
        placeholders = ",".join("?" for _ in requested)
        rows = self.connection.execute(
            f"""
            SELECT ring, COALESCE(SUM(sample_count), 0) AS samples
            FROM shards
            WHERE rules_hash = ?
              AND feature_schema_hash = ?
              AND state = 'ready'
              AND run_id = ?
              AND generation_family = ?
              AND model_step BETWEEN ? AND ?
              AND ring IN ({placeholders})
            GROUP BY ring
            """,
            (
                f"{RULES_HASH:016x}",
                f"{FEATURE_SCHEMA_HASH:016x}",
                validate_identifier("run_id", run_id),
                validate_identifier("generation_family", generation_family),
                lower,
                current_model_step,
                *requested,
            ),
        )
        counts = {ring: 0 for ring in requested}
        for row in rows:
            counts[int(row["ring"])] = int(row["samples"])
        return counts

    def select_recent_spans(
        self,
        *,
        rings: Sequence[int],
        per_ring_quota: int,
        run_id: str,
        generation_family: str,
        current_model_step: int,
        max_model_lag_steps: int,
    ) -> ReplaySelection:
        if per_ring_quota <= 0:
            raise ValueError("per_ring_quota must be positive")
        row = self.connection.execute(
            """
            SELECT COALESCE(MAX(id), 0) AS max_id FROM shards
            WHERE state = 'ready'
              AND run_id = ?
              AND generation_family = ?
              AND model_step BETWEEN ? AND ?
            """,
            (
                run_id,
                generation_family,
                max(0, current_model_step - max_model_lag_steps),
                current_model_step,
            ),
        ).fetchone()
        maximum_shard_id = int(row["max_id"])
        spans: list[ReplaySpan] = []
        counts: dict[int, int] = {}
        for ring in rings:
            records = self.recent_shards(
                sample_window=per_ring_quota,
                run_id=run_id,
                generation_family=generation_family,
                rings=(ring,),
                current_model_step=current_model_step,
                max_model_lag_steps=max_model_lag_steps,
                maximum_shard_id=maximum_shard_id,
            )
            remaining = per_ring_quota
            selected: list[ReplaySpan] = []
            for record in reversed(records):
                take = min(record.sample_count, remaining)
                if take <= 0:
                    break
                selected.append(
                    ReplaySpan(
                        record=record,
                        sample_start=record.sample_count - take,
                        sample_count=take,
                    )
                )
                remaining -= take
            selected.reverse()
            spans.extend(selected)
            counts[int(ring)] = sum(span.sample_count for span in selected)
        spans.sort(key=lambda span: span.record.shard_id)
        return ReplaySelection(tuple(spans), counts, maximum_shard_id)

    def load_recent_samples(
        self,
        *,
        sample_window: int,
        run_id: str,
        generation_family: str,
        rings: Sequence[int] | None = None,
        current_model_step: int | None = None,
        max_model_lag_steps: int | None = None,
        verify_checksums: bool = True,
    ) -> list[ReplaySample]:
        output: list[ReplaySample] = []
        records = self.recent_shards(
            sample_window=sample_window,
            run_id=run_id,
            generation_family=generation_family,
            rings=rings,
            current_model_step=current_model_step,
            max_model_lag_steps=max_model_lag_steps,
        )
        for record in records:
            if verify_checksums and _sha256(record.path) != record.checksum_sha256:
                raise ValueError(f"replay shard checksum failed: {record.path}")
            output.extend(read_replay_shard(record.path))
        return output[-sample_window:]

    def get_cursor(self, name: str) -> ReplayCursor:
        row = self.connection.execute(
            "SELECT shard_id, sample_offset FROM cursors WHERE name = ?", (name,)
        ).fetchone()
        if row is None:
            return ReplayCursor()
        return ReplayCursor(
            shard_id=int(row["shard_id"]),
            sample_offset=int(row["sample_offset"]),
        )

    def set_cursor(self, name: str, cursor: ReplayCursor) -> None:
        if not name or cursor.shard_id < 0 or cursor.sample_offset < 0:
            raise ValueError("invalid replay cursor")
        self.connection.execute(
            """
            INSERT INTO cursors(name, shard_id, sample_offset, updated_ns)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                shard_id = excluded.shard_id,
                sample_offset = excluded.sample_offset,
                updated_ns = excluded.updated_ns
            """,
            (name, cursor.shard_id, cursor.sample_offset, time.time_ns()),
        )

    def iter_after_cursor(
        self, name: str, *, limit: int | None = None
    ) -> Iterator[tuple[ShardRecord, list[ReplaySample]]]:
        cursor = self.get_cursor(name)
        query = "SELECT * FROM shards WHERE id >= ? ORDER BY id"
        parameters: list[object] = [max(1, cursor.shard_id)]
        if limit is not None:
            if limit <= 0:
                raise ValueError("limit must be positive")
            query += " LIMIT ?"
            parameters.append(limit)
        for row in self.connection.execute(query, parameters):
            record = self._record(row)
            samples = read_replay_shard(record.path)
            start = cursor.sample_offset if record.shard_id == cursor.shard_id else 0
            if start < len(samples):
                yield record, samples[start:]

    def _record(self, row: sqlite3.Row) -> ShardRecord:
        return ShardRecord(
            shard_id=int(row["id"]),
            path=self.root / row["relative_path"],
            created_ns=int(row["created_ns"]),
            sample_count=int(row["sample_count"]),
            ring=int(row["ring"]),
            phase_min=int(row["phase_min"]),
            phase_max=int(row["phase_max"]),
            model_version=str(row["model_version"]),
            model_step=int(row["model_step"]),
            model_identity=str(row["model_identity"]),
            run_id=str(row["run_id"]),
            generation_family=str(row["generation_family"]),
            actor_id=str(row["actor_id"]),
            generation=int(row["generation"]),
            game_count=int(row["game_count"]),
            checksum_sha256=str(row["checksum_sha256"]),
            state=str(row["state"]),
            quarantine_reason=(
                str(row["quarantine_reason"])
                if row["quarantine_reason"] is not None
                else None
            ),
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
