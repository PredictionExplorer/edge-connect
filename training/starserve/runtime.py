"""EMA model lifecycle and native Gumbel analysis execution."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from startrain.checkpoint import ModelManifest, load_ema_checkpoint, load_model_manifest
from startrain.config import ExperimentConfig, load_config
from startrain.contracts import SCORE_MARGIN_MAX, SCORE_MARGIN_MIN
from startrain.features import GLOBAL_FEATURE_DIM, NODE_FEATURE_DIM
from startrain.inference import GraphInferenceAdapter, InferenceConfig
from startrain.model import GraphResTNet
from startrain.native import BITBOARD_WORDS, load_star_native, positions_from_native
from startrain.training import maybe_compile_model

from .config import ServerConfig
from .schemas import AnalyzeRequest


class AnalysisError(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        status_code: int = 500,
        details: object | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details


class SearchCancelled(AnalysisError):
    def __init__(self) -> None:
        super().__init__(
            "request_cancelled",
            "analysis was cancelled",
            status_code=499,
        )


@dataclass(frozen=True, slots=True)
class LoadedModel:
    manifest: ModelManifest
    evaluator: GraphInferenceAdapter


@dataclass(frozen=True, slots=True)
class ModelLease:
    model: LoadedModel
    reload_ms: float


class AtomicModelManager:
    """Shares one immutable evaluator and swaps it only while no request is active."""

    def __init__(
        self,
        config: ServerConfig,
        *,
        experiment: ExperimentConfig | None = None,
        manifest_reader: Callable[[str | Path], ModelManifest] = load_model_manifest,
        bundle_loader: Callable[[ModelManifest, ExperimentConfig, str], LoadedModel]
        | None = None,
    ) -> None:
        self.config = config
        self.experiment = experiment or load_config(config.experiment_config)
        if (
            self.experiment.model.node_feature_dim != NODE_FEATURE_DIM
            or self.experiment.model.global_feature_dim != GLOBAL_FEATURE_DIM
        ):
            raise ValueError(
                "experiment model dimensions do not match the finalized feature schema"
            )
        self._manifest_reader = manifest_reader
        self._bundle_loader = bundle_loader or self._load_bundle
        self._condition = threading.Condition()
        self._active = 0
        self._loading = False
        self._current: LoadedModel | None = None
        self._last_reload_error: str | None = None
        self._last_reload_ns: int | None = None
        self._pointer_signature: tuple[int, int, int] | None = None

    def startup(self) -> None:
        with self.lease():
            pass

    @contextmanager
    def lease(self) -> Iterator[ModelLease]:
        reload_ms = 0.0
        should_refresh = False
        current: LoadedModel | None = None
        with self._condition:
            while self._loading:
                self._condition.wait()
            if self._active == 0:
                self._loading = True
                should_refresh = True
            else:
                self._active += 1
                assert self._current is not None
                current = self._current

        if should_refresh:
            started = time.perf_counter()
            load_error: Exception | None = None
            candidate: LoadedModel | None = None
            try:
                signature = self._publication_signature()
                if not (
                    self._current is not None
                    and signature is not None
                    and signature == self._pointer_signature
                ):
                    manifest = self._manifest_reader(self.config.model_manifest)
                    if manifest.role != "champion":
                        raise ValueError(
                            "starserve accepts only a champion model pointer"
                        )
                    if not self._same_manifest(manifest):
                        candidate = self._bundle_loader(
                            manifest, self.experiment, self.config.device
                        )
                    self._pointer_signature = signature
            # A bad atomic publication must never evict a known-good EMA model.
            except Exception as exc:
                load_error = exc
            reload_ms = (time.perf_counter() - started) * 1_000.0
            with self._condition:
                if candidate is not None:
                    self._current = candidate
                    self._last_reload_error = None
                    self._last_reload_ns = time.time_ns()
                elif load_error is not None:
                    self._last_reload_error = (
                        f"{type(load_error).__name__}: {load_error}"
                    )
                self._loading = False
                if self._current is None:
                    self._condition.notify_all()
                    assert load_error is not None
                    raise load_error
                self._active += 1
                current = self._current
                self._condition.notify_all()

        assert current is not None
        try:
            yield ModelLease(current, reload_ms)
        finally:
            with self._condition:
                self._active -= 1
                if self._active < 0:
                    raise RuntimeError("model lease count became negative")
                self._condition.notify_all()

    def health(self) -> dict[str, object]:
        with self._condition:
            current = self._current
            return {
                "ready": current is not None,
                "model_version": (
                    current.manifest.model_version if current is not None else None
                ),
                "model_step": (
                    current.manifest.model_step if current is not None else None
                ),
                "model_identity": (
                    current.manifest.model_identity if current is not None else None
                ),
                "active_requests": self._active,
                "reload_in_progress": self._loading,
                "last_reload_ns": self._last_reload_ns,
                "last_reload_error": self._last_reload_error,
            }

    def _same_manifest(self, manifest: ModelManifest) -> bool:
        current = self._current
        return current is not None and (
            current.manifest.model_version == manifest.model_version
            and current.manifest.model_step == manifest.model_step
            and current.manifest.checkpoint == manifest.checkpoint
            and current.manifest.checkpoint_sha256 == manifest.checkpoint_sha256
        )

    def _publication_signature(self) -> tuple[int, int, int] | None:
        try:
            stat = self.config.model_manifest.stat()
        except OSError:
            return None
        return stat.st_mtime_ns, stat.st_size, stat.st_ino

    @staticmethod
    def _load_bundle(
        manifest: ModelManifest,
        experiment: ExperimentConfig,
        device: str,
    ) -> LoadedModel:
        target = torch.device(device)
        if target.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"configured CUDA device {device!r} is unavailable")
        model = GraphResTNet(experiment.model).to(target)
        metadata = load_ema_checkpoint(
            manifest.checkpoint,
            model=model,
            expected_model_config=asdict(experiment.model),
            expected_game_config=asdict(experiment.game),
            expected_run_id=manifest.run_id,
            expected_generation_family=manifest.generation_family,
            expected_sha256=manifest.checkpoint_sha256,
            expected_bytes=manifest.checkpoint_bytes,
            map_location=target,
        )
        if int(metadata["step"]) != manifest.model_step:
            raise ValueError("model manifest and EMA checkpoint identity disagree")
        model.eval()
        inference_model = maybe_compile_model(
            model,
            enabled=experiment.train.compile,
            dynamic=True,
            fullgraph=True,
        )
        evaluator = GraphInferenceAdapter(
            inference_model,
            device=target,
            config=InferenceConfig(
                precision=experiment.train.precision,
                score_utility_weight=experiment.selfplay.score_utility_weight,
                initial_pass_logit_penalty=(
                    experiment.selfplay.initial_pass_logit_penalty
                ),
            ),
            model_version=manifest.model_version,
            model_step=manifest.model_step,
            model_identity=manifest.model_identity,
        )
        return LoadedModel(manifest=manifest, evaluator=evaluator)


class NativeAnalysisService:
    def __init__(
        self,
        config: ServerConfig,
        *,
        native_module: object | None = None,
        model_manager: AtomicModelManager | None = None,
    ) -> None:
        self.config = config
        self.native: Any = native_module or load_star_native(required=True)
        assert self.native is not None
        self.models = model_manager or AtomicModelManager(config)

    def startup(self) -> None:
        self.models.startup()

    def health(self) -> dict[str, object]:
        return self.models.health()

    def analyze(
        self,
        request: AnalyzeRequest,
        cancellation: threading.Event,
    ) -> dict[str, object]:
        started = time.perf_counter()
        states = self._import_state(request)
        if cancellation.is_set():
            raise SearchCancelled()
        with self.models.lease() as lease:
            if cancellation.is_set():
                raise SearchCancelled()
            search_started = time.perf_counter()
            evaluator = lease.model.evaluator
            search = self.native.SearchBatch(
                states,
                simulations=request.search.simulations,
                max_considered=request.search.max_considered,
                c_visit=self.config.search.c_visit,
                c_scale=self.config.search.c_scale,
                deterministic_seed=request.search.seed,
            )
            roots = search.root_requests()
            if len(roots) != 1:
                raise AnalysisError(
                    "native_search_error",
                    "native search did not return exactly one active root",
                )
            detailed = evaluator.evaluate_detailed(roots)
            if cancellation.is_set():
                raise SearchCancelled()
            search.initialize_roots(*detailed.response.submit_args())
            guard = 0
            maximum_iterations = request.search.simulations * 4 + 16
            while not search.is_done():
                if cancellation.is_set():
                    raise SearchCancelled()
                guard += 1
                if guard > maximum_iterations:
                    raise AnalysisError(
                        "native_search_stalled",
                        "native Gumbel search failed to make progress",
                    )
                requests = search.next_requests()
                if len(requests) == 0:
                    continue
                response = evaluator.evaluate(requests)
                if cancellation.is_set():
                    raise SearchCancelled()
                search.submit(*response.submit_args())
            if cancellation.is_set():
                raise SearchCancelled()
            results = search.results()
            search_ms = (time.perf_counter() - search_started) * 1_000.0
            payload = self._response_payload(
                results,
                detailed,
                evaluator=evaluator,
                reload_ms=lease.reload_ms,
                search_ms=search_ms,
                total_ms=(time.perf_counter() - started) * 1_000.0,
                node_count=len(request.stones),
            )
        return payload

    def _import_state(self, request: AnalyzeRequest) -> Any:
        zero_bits = _pack_stones(request.stones, player=0)
        one_bits = _pack_stones(request.stones, player=1)
        state_batch = getattr(self.native, "StateBatch", None)
        importer = getattr(state_batch, "from_semantic", None)
        if not callable(importer):
            raise AnalysisError(
                "native_incompatible",
                "star_native lacks StateBatch.from_semantic",
            )
        try:
            states: Any = importer(
                request.rings,
                zero_bits,
                one_bits,
                [request.to_move],
                [request.moves_left],
                [request.opening],
                [request.pass_streak],
            )
            imported = positions_from_native(states.data())
        except (TypeError, ValueError) as exc:
            raise AnalysisError(
                "invalid_semantic_state",
                "state was rejected by finalized native rules",
                status_code=422,
                details={"reason": str(exc)},
            ) from exc
        if len(imported) != 1 or imported[0].terminal:
            raise AnalysisError(
                "invalid_semantic_state",
                "service accepts one active nonterminal state",
                status_code=422,
            )
        position = imported[0]
        if (
            position.rings != request.rings
            or position.stones.tolist() != request.stones
            or position.to_move != request.to_move
            or position.moves_left != request.moves_left
            or position.opening != request.opening
            or position.pass_streak != request.pass_streak
        ):
            raise AnalysisError(
                "native_incompatible",
                "native semantic import did not preserve the requested state",
            )
        return states

    @staticmethod
    def _response_payload(
        results: Any,
        detailed: Any,
        *,
        evaluator: GraphInferenceAdapter,
        reload_ms: float,
        search_ms: float,
        total_ms: float,
        node_count: int,
    ) -> dict[str, object]:
        offsets = [int(value) for value in results.action_offsets]
        actions = [int(value) for value in results.actions]
        policy = [float(value) for value in results.policy_target]
        q_values = [float(value) for value in results.q_values]
        visits = [int(value) for value in results.visits]
        selected = [int(value) for value in results.selected_actions]
        terminal = [bool(value) for value in results.terminal]
        if (
            offsets != [0, len(actions)]
            or not actions
            or len(policy) != len(actions)
            or len(q_values) != len(actions)
            or len(visits) != len(actions)
            or len(selected) != 1
            or terminal != [False]
            or selected[0] not in actions
            or any(action < -1 or action >= node_count for action in actions)
        ):
            raise AnalysisError(
                "native_search_error",
                "native root statistics are malformed",
            )
        numeric = [*policy, *q_values]
        if any(not math.isfinite(value) for value in numeric):
            raise AnalysisError(
                "native_search_error",
                "native root statistics contain non-finite values",
            )
        if (
            any(value < 0 for value in visits)
            or any(value < 0 for value in policy)
            or any(not -1.0 <= value <= 1.0 for value in q_values)
            or not math.isclose(sum(policy), 1.0, rel_tol=1e-4, abs_tol=1e-4)
        ):
            raise AnalysisError(
                "native_search_error",
                "native root policy or visits are invalid",
            )
        if (
            len(detailed.wdl_probabilities) != 1
            or len(detailed.score_probabilities) != 1
            or len(detailed.response.values) != 1
        ):
            raise AnalysisError(
                "model_output_error",
                "root model output has an invalid batch shape",
            )
        wdl = detailed.wdl_probabilities[0]
        scores = detailed.score_probabilities[0]
        if len(wdl) != 3 or len(scores) != SCORE_MARGIN_MAX - SCORE_MARGIN_MIN + 1:
            raise AnalysisError(
                "model_output_error",
                "root belief support has an invalid shape",
            )
        beliefs = [*wdl, *scores]
        values = [
            detailed.wdl_values[0],
            detailed.response.values[0],
            detailed.score_expectations[0],
        ]
        if (
            any(not math.isfinite(value) or value < 0 for value in beliefs)
            or not math.isclose(sum(wdl), 1.0, rel_tol=1e-5, abs_tol=1e-5)
            or not math.isclose(sum(scores), 1.0, rel_tol=1e-5, abs_tol=1e-5)
            or any(not math.isfinite(value) for value in values)
            or not -1.0 <= detailed.wdl_values[0] <= 1.0
            or not -1.0 <= detailed.response.values[0] <= 1.0
        ):
            raise AnalysisError(
                "model_output_error",
                "root model beliefs contain invalid probabilities or values",
            )
        return {
            "schema_version": 1,
            "action": _action_payload(selected[0]),
            "root_actions": [_action_payload(action) for action in actions],
            "root_policy": policy,
            "root_q": q_values,
            "root_visits": visits,
            "wdl": {"loss": wdl[0], "draw": wdl[1], "win": wdl[2]},
            "value": detailed.wdl_values[0],
            "search_value": detailed.response.values[0],
            "score_belief": {
                "support_min": SCORE_MARGIN_MIN,
                "support_max": SCORE_MARGIN_MAX,
                "expected_margin": detailed.score_expectations[0],
                "probabilities": scores,
            },
            "model_version": evaluator.model_version,
            "model_step": evaluator.model_step,
            "timing_ms": {
                "queue": 0.0,
                "model_reload": reload_ms,
                "inference_search": search_ms,
                "total": total_ms,
            },
        }


def _pack_stones(stones: Sequence[int], *, player: int) -> list[int]:
    words = [0] * BITBOARD_WORDS
    for node, stone in enumerate(stones):
        if stone == player:
            words[node // 64] |= 1 << (node % 64)
    return words


def _action_payload(code: int) -> dict[str, object]:
    if code == -1:
        return {"code": -1, "kind": "pass", "node": None}
    if code < 0:
        raise AnalysisError(
            "native_search_error",
            f"native search returned invalid action code {code}",
        )
    return {"code": code, "kind": "place", "node": code}
