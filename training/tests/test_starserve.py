from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from starserve.app import create_app
from starserve.config import (
    LimitConfig,
    SearchConfig,
    SecurityConfig,
    ServerConfig,
    ServerConfigError,
)
from starserve.runtime import (
    AnalysisError,
    AtomicModelManager,
    LoadedModel,
    ModelLease,
    NativeAnalysisService,
    SearchCancelled,
)
from starserve.schemas import AnalyzeRequest, AnalyzeResponse, AtomicAction, ScoreBelief
from startrain.checkpoint import ModelManifest
from startrain.contracts import (
    ACTION_LAYOUT_SCHEMA_ID,
    RULES_HASH_WIRE,
    SCORE_MARGIN_MAX,
    SCORE_MARGIN_MIN,
)
from startrain.features import GLOBAL_FEATURE_DIM, NODE_FEATURE_DIM
from startrain.inference import (
    DetailedInferenceResponse,
    GraphInferenceAdapter,
    InferenceResponse,
)
from startrain.model import GraphResTNet, ModelConfig
from startrain.native import BITBOARD_WORDS
from startrain.topology import get_topology


def server_config(tmp_path, **changes: object) -> ServerConfig:
    values: dict[str, object] = {
        "experiment_config": tmp_path / "experiment.yaml",
        "model_manifest": tmp_path / "champion.json",
        "device": "cpu",
        "search": SearchConfig(
            default_simulations=4,
            maximum_simulations=16,
            default_max_considered=2,
        ),
        "limits": LimitConfig(
            max_concurrency=1,
            max_request_bytes=4096,
            request_timeout_seconds=1.0,
            queue_timeout_seconds=0.2,
        ),
        "security": SecurityConfig(),
    }
    values.update(changes)
    return ServerConfig(**values)


def request_payload() -> dict[str, object]:
    return {
        "schema_version": 2,
        "rules_hash": RULES_HASH_WIRE,
        "rings": 4,
        "stones": [-1] * get_topology(4).n,
        "to_move": 0,
        "moves_left": 1,
        "opening": True,
        "terminal": False,
        "search": {"simulations": 4, "max_considered": 2, "seed": 7},
    }


def response_payload() -> dict[str, object]:
    score = [0.0] * (SCORE_MARGIN_MAX - SCORE_MARGIN_MIN + 1)
    score[-SCORE_MARGIN_MIN] = 1.0
    return {
        "schema_version": 2,
        "action": {"code": 0, "kind": "place", "node": 0},
        "root_actions": [
            {"code": 0, "kind": "place", "node": 0},
            {"code": 1, "kind": "place", "node": 1},
        ],
        "root_policy": [0.75, 0.25],
        "root_q": [0.2, -0.1],
        "root_visits": [3, 1],
        "outcome": {"loss": 0.2, "win": 0.8},
        "value": 0.6,
        "search_value": 0.3,
        "score_belief": {
            "support_min": SCORE_MARGIN_MIN,
            "support_max": SCORE_MARGIN_MAX,
            "expected_margin": 0.0,
            "probabilities": score,
        },
        "model_version": "fake-v2",
        "model_step": 5,
        "timing_ms": {
            "queue": 0.0,
            "model_reload": 0.0,
            "inference_search": 1.0,
            "total": 1.0,
        },
    }


class FakeService:
    def __init__(self) -> None:
        self.started = False

    def startup(self) -> None:
        self.started = True

    def health(self) -> dict[str, object]:
        return {
            "ready": self.started,
            "model_version": "fake-v2",
            "model_step": 5,
        }

    def analyze(
        self, request: AnalyzeRequest, cancellation: threading.Event
    ) -> dict[str, object]:
        assert request.rings == 4
        assert not cancellation.is_set()
        return response_payload()


def test_v2_api_health_auth_and_binary_response(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TEST_STARSERVE_TOKEN", "correct-secret")
    config = server_config(
        tmp_path,
        security=SecurityConfig(
            cors_allow_origins=("https://play.example",),
            bearer_token_env="TEST_STARSERVE_TOKEN",
        ),
    )
    with TestClient(create_app(config, service=FakeService())) as client:
        health = client.get("/v2/health")
        assert health.status_code == 200
        assert health.json()["api_schema_version"] == 2
        assert health.json()["server_config_schema_version"] == 2
        assert health.json()["model_schema_version"] == 2
        assert health.json()["actions"] == {
            "schema_id": ACTION_LAYOUT_SCHEMA_ID,
            "types": ["place"],
        }
        assert health.json()["outcomes"]["classes"] == ["loss", "win"]

        assert client.post("/v2/analyze", json=request_payload()).status_code == 401
        headers = {"Authorization": "Bearer correct-secret"}
        response = client.post("/v2/analyze", json=request_payload(), headers=headers)
        assert response.status_code == 200
        body = response.json()
        assert body["schema_version"] == 2
        assert body["outcome"] == {"loss": 0.2, "win": 0.8}
        assert body["value"] == pytest.approx(0.6)
        assert all(action["kind"] == "place" for action in body["root_actions"])
        assert client.post("/v1/analyze", json=request_payload()).status_code == 404


def test_request_rejects_old_schema_pass_fields_and_unsupported_rings() -> None:
    old = request_payload()
    old["schema_version"] = 1
    with pytest.raises(ValidationError):
        AnalyzeRequest.model_validate(old)

    legacy_action = request_payload()
    legacy_action["pass_streak"] = 0
    with pytest.raises(ValidationError, match="pass_streak"):
        AnalyzeRequest.model_validate(legacy_action)

    for rings in (3, 5, 7, 9, 11, 12):
        invalid = request_payload()
        invalid["rings"] = rings
        invalid["stones"] = []
        with pytest.raises(ValidationError):
            AnalyzeRequest.model_validate(invalid)


def test_response_schema_is_placement_only_and_binary() -> None:
    validated = AnalyzeResponse.model_validate(
        {**response_payload(), "request_id": "request-1"}
    )
    assert validated.outcome.loss + validated.outcome.win == pytest.approx(1.0)

    legacy = {**response_payload(), "request_id": "request-1"}
    legacy["root_actions"] = [{"code": -1, "kind": "pass", "node": None}]
    with pytest.raises(ValidationError):
        AnalyzeResponse.model_validate(legacy)

    wrong_outcome = {**response_payload(), "request_id": "request-1"}
    wrong_outcome["outcome"] = {"loss": 0.2, "win": 0.7}
    with pytest.raises(ValidationError, match="sum to one"):
        AnalyzeResponse.model_validate(wrong_outcome)

    wrong_score = {**response_payload(), "request_id": "request-1"}
    wrong_score["score_belief"] = {
        **wrong_score["score_belief"],
        "probabilities": [1.0],
    }
    with pytest.raises(ValidationError, match="303 bins"):
        AnalyzeResponse.model_validate(wrong_score)


def test_runtime_payload_rejects_negative_native_actions() -> None:
    score = [0.0] * (SCORE_MARGIN_MAX - SCORE_MARGIN_MIN + 1)
    score[-SCORE_MARGIN_MIN] = 1.0
    detailed = DetailedInferenceResponse(
        response=InferenceResponse([1], [0.6], [0, 1], [0.0]),
        outcome_probabilities=[[0.2, 0.8]],
        outcome_values=[0.6],
        score_expectations=[0.0],
        score_probabilities=[score],
    )
    evaluator = SimpleNamespace(model_version="v2", model_step=1)
    malformed = SimpleNamespace(
        action_offsets=[0, 1],
        actions=[-1],
        policy_target=[1.0],
        q_values=[0.0],
        visits=[1],
        selected_actions=[-1],
        terminal=[False],
    )
    with pytest.raises(AnalysisError, match="malformed"):
        NativeAnalysisService._response_payload(
            malformed,
            detailed,
            evaluator=evaluator,
            reload_ms=0.0,
            search_ms=1.0,
            total_ms=1.0,
            node_count=50,
        )


def test_server_config_and_request_size_fail_closed(tmp_path) -> None:
    with pytest.raises(ServerConfigError, match="schema_version"):
        server_config(tmp_path, schema_version=1)
    with pytest.raises(ServerConfigError, match="rules hash"):
        server_config(tmp_path, rules_hash="fnv1a64:0000000000000000")
    with pytest.raises(ServerConfigError, match="wildcard"):
        server_config(
            tmp_path,
            security=SecurityConfig(cors_allow_origins=("*",)),
        )

    config = server_config(
        tmp_path,
        limits=LimitConfig(
            max_concurrency=1,
            max_request_bytes=64,
            request_timeout_seconds=1,
            queue_timeout_seconds=1,
        ),
    )
    with TestClient(create_app(config, service=FakeService())) as client:
        response = client.post(
            "/v2/analyze",
            content=json.dumps(request_payload()),
            headers={"Content-Type": "application/json"},
        )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "request_too_large"


def test_v2_api_cors_validation_budgets_and_request_ids(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TEST_STARSERVE_TOKEN", "correct-secret")
    config = server_config(
        tmp_path,
        security=SecurityConfig(
            cors_allow_origins=("https://play.example",),
            bearer_token_env="TEST_STARSERVE_TOKEN",
        ),
    )
    headers = {
        "Authorization": "Bearer correct-secret",
        "X-Request-ID": "analysis-request.7",
    }
    with TestClient(create_app(config, service=FakeService())) as client:
        preflight = client.options(
            "/v2/analyze",
            headers={
                "Origin": "https://play.example",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert preflight.status_code == 200
        assert (
            preflight.headers["access-control-allow-origin"] == "https://play.example"
        )

        invalid = request_payload()
        invalid["rules_hash"] = "fnv1a64:0000000000000000"
        rejected = client.post("/v2/analyze", json=invalid, headers=headers)
        assert rejected.status_code == 422
        assert rejected.json()["error"]["code"] == "invalid_request"
        assert rejected.headers["x-request-id"] == "analysis-request.7"

        too_many_simulations = request_payload()
        too_many_simulations["search"] = {
            "simulations": 17,
            "max_considered": 2,
            "seed": 7,
        }
        rejected = client.post(
            "/v2/analyze", json=too_many_simulations, headers=headers
        )
        assert rejected.status_code == 422
        assert rejected.json()["error"] == {
            "code": "search_budget_exceeded",
            "message": "simulations exceed the configured maximum",
            "details": {"maximum_simulations": 16},
            "request_id": "analysis-request.7",
        }

        too_many_candidates = request_payload()
        too_many_candidates["search"] = {
            "simulations": 4,
            "max_considered": 65,
            "seed": 7,
        }
        rejected = client.post("/v2/move", json=too_many_candidates, headers=headers)
        assert rejected.status_code == 422
        assert rejected.json()["error"]["details"] == {
            "maximum_max_considered": config.search.maximum_max_considered
        }


def test_request_size_limit_rejects_streamed_body(tmp_path) -> None:
    config = server_config(
        tmp_path,
        limits=LimitConfig(
            max_concurrency=1,
            max_request_bytes=64,
            request_timeout_seconds=1,
            queue_timeout_seconds=1,
        ),
    )
    with TestClient(create_app(config, service=FakeService())) as client:
        response = client.post(
            "/v2/analyze",
            content=(chunk for chunk in (b"{" + b"x" * 40, b"x" * 40 + b"}")),
            headers={
                "Content-Type": "application/json",
                "X-Request-ID": "stream-too-large",
            },
        )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "request_too_large"
    assert response.json()["error"]["request_id"] == "stream-too-large"


class SlowService(FakeService):
    def __init__(self) -> None:
        super().__init__()
        self.cancelled = threading.Event()

    def analyze(
        self, request: AnalyzeRequest, cancellation: threading.Event
    ) -> dict[str, object]:
        while not cancellation.wait(0.005):
            pass
        self.cancelled.set()
        return response_payload()


def test_timeout_signals_cooperative_cancellation(tmp_path) -> None:
    service = SlowService()
    config = server_config(
        tmp_path,
        limits=LimitConfig(
            max_concurrency=1,
            max_request_bytes=4096,
            request_timeout_seconds=0.05,
            queue_timeout_seconds=0.05,
        ),
    )
    with TestClient(create_app(config, service=service)) as client:
        response = client.post("/v2/analyze", json=request_payload())
        assert response.status_code == 504
        assert response.json()["error"]["code"] == "analysis_timeout"
        assert service.cancelled.wait(1)


class BrokenService(FakeService):
    def analyze(
        self, request: AnalyzeRequest, cancellation: threading.Event
    ) -> dict[str, object]:
        raise RuntimeError("private failure detail")


def test_unexpected_service_errors_are_redacted(tmp_path) -> None:
    with TestClient(
        create_app(server_config(tmp_path), service=BrokenService()),
        raise_server_exceptions=False,
    ) as client:
        response = client.post("/v2/analyze", json=request_payload())
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal_error"
    assert "private failure detail" not in response.text


def test_atomic_model_manager_reloads_only_between_leases(tmp_path) -> None:
    manifests = [
        ModelManifest(tmp_path / "m", tmp_path / "one.pt", "v1", 1, 1, role="champion"),
        ModelManifest(tmp_path / "m", tmp_path / "two.pt", "v2", 2, 2, role="champion"),
        ValueError("bad publication"),
    ]
    calls = 0

    def reader(_path):
        nonlocal calls
        item = manifests[min(calls, len(manifests) - 1)]
        calls += 1
        if isinstance(item, Exception):
            raise item
        return item

    def loader(manifest, _experiment, _device):
        return LoadedModel(manifest, SimpleNamespace())

    manager = AtomicModelManager(
        server_config(tmp_path),
        experiment=SimpleNamespace(
            model=SimpleNamespace(
                node_feature_dim=NODE_FEATURE_DIM,
                global_feature_dim=GLOBAL_FEATURE_DIM,
            )
        ),
        manifest_reader=reader,
        bundle_loader=loader,
    )
    assert manager.health()["ready"] is False
    with manager.lease() as first:
        assert first.model.manifest.model_version == "v1"
        with manager.lease() as overlapping:
            assert overlapping.model.manifest.model_version == "v1"
            assert calls == 1
    with manager.lease() as second:
        assert second.model.manifest.model_version == "v2"
    with manager.lease() as retained:
        assert retained.model.manifest.model_version == "v2"
    health = manager.health()
    assert health["last_reload_error"] == "ValueError: bad publication"
    assert health["active_requests"] == 0


def test_atomic_model_manager_rejects_candidate_and_bad_dimensions(tmp_path) -> None:
    with pytest.raises(ValueError, match="feature schema"):
        AtomicModelManager(
            server_config(tmp_path),
            experiment=SimpleNamespace(
                model=SimpleNamespace(
                    node_feature_dim=NODE_FEATURE_DIM + 1,
                    global_feature_dim=GLOBAL_FEATURE_DIM,
                )
            ),
        )

    candidate = ModelManifest(
        tmp_path / "candidate.json",
        tmp_path / "model.pt",
        "sha256-" + "1" * 64,
        1,
        1,
        model_identity="sha256-" + "1" * 64,
        role="candidate",
    )
    manager = AtomicModelManager(
        server_config(tmp_path),
        experiment=SimpleNamespace(
            model=SimpleNamespace(
                node_feature_dim=NODE_FEATURE_DIM,
                global_feature_dim=GLOBAL_FEATURE_DIM,
            )
        ),
        manifest_reader=lambda _path: candidate,
        bundle_loader=lambda manifest, _experiment, _device: LoadedModel(
            manifest, SimpleNamespace()
        ),
    )
    with pytest.raises(ValueError, match="champion"):
        manager.startup()


class FakeStateBatch:
    def __init__(self, data) -> None:
        self._data = data

    @staticmethod
    def from_semantic(
        rings,
        zero_bits,
        one_bits,
        to_move,
        moves_left,
        opening,
    ):
        nodes = get_topology(rings).n
        occupied = sum(word.bit_count() for word in zero_bits + one_bits)
        legal_words = [0] * BITBOARD_WORDS
        for node in range(nodes):
            if not (
                zero_bits[node // 64] & (1 << (node % 64))
                or one_bits[node // 64] & (1 << (node % 64))
            ):
                legal_words[node // 64] |= 1 << (node % 64)
        data = SimpleNamespace(
            rings=rings,
            node_count=nodes,
            batch_size=1,
            zero_bits=zero_bits,
            one_bits=one_bits,
            legal_bits=legal_words,
            hashes=[1],
            stones_placed=[occupied],
            to_move=to_move,
            moves_left=moves_left,
            opening=opening,
            mid_turn=[not opening[0] and moves_left[0] == 1],
            terminal=[False],
        )
        return FakeStateBatch(data)

    def data(self):
        return self._data


class FakeRootBatch:
    def __init__(self, states) -> None:
        self.states = states.data()
        self.tokens = [1]
        self.legal_actions = list(range(self.states.node_count))
        self.legal_offsets = [0, len(self.legal_actions)]

    def __len__(self):
        return 1


class FakeSearchBatch:
    def __init__(self, states, **_options) -> None:
        self.states = states
        self.done = False

    def root_requests(self):
        return FakeRootBatch(self.states)

    def initialize_roots(self, *_buffers):
        self.done = True

    def is_done(self):
        return self.done

    def next_requests(self):
        raise AssertionError("fake search completes at the root")

    def submit(self, *_buffers):
        raise AssertionError("fake search has no leaves")

    def results(self):
        actions = [0, 1]
        return SimpleNamespace(
            action_offsets=[0, len(actions)],
            actions=actions,
            policy_target=[0.75, 0.25],
            q_values=[0.1, -0.1],
            visits=[3, 1],
            selected_actions=[0],
            terminal=[False],
        )


class FakeEvaluator:
    model_version = "fake-native-v2"
    model_step = 8

    def evaluate_detailed(self, roots):
        response = InferenceResponse(
            tokens=roots.tokens,
            values=[0.2],
            policy_offsets=roots.legal_offsets,
            policy_logits=[0.0] * len(roots.legal_actions),
        )
        score = [0.0] * (SCORE_MARGIN_MAX - SCORE_MARGIN_MIN + 1)
        score[-SCORE_MARGIN_MIN] = 1.0
        return DetailedInferenceResponse(
            response=response,
            outcome_probabilities=[[0.2, 0.8]],
            outcome_values=[0.6],
            score_expectations=[0.0],
            score_probabilities=[score],
        )

    def evaluate(self, _requests):
        raise AssertionError("fake search has no leaves")


class FakeManager:
    def startup(self):
        pass

    def health(self):
        return {"ready": True, "model_version": "fake-native-v2", "model_step": 8}

    @contextmanager
    def lease(self):
        manifest = ModelManifest(
            SimpleNamespace(),
            SimpleNamespace(),
            "fake-native-v2",
            8,
            time.time_ns(),
            role="champion",
        )
        yield ModelLease(LoadedModel(manifest, FakeEvaluator()), 0.0)


def test_native_analysis_imports_v2_state_and_returns_node_only_root(tmp_path) -> None:
    native = SimpleNamespace(StateBatch=FakeStateBatch, SearchBatch=FakeSearchBatch)
    service = NativeAnalysisService(
        server_config(tmp_path),
        native_module=native,
        model_manager=FakeManager(),
    )
    result = service.analyze(
        AnalyzeRequest.model_validate(request_payload()),
        threading.Event(),
    )
    assert result["schema_version"] == 2
    assert result["action"] == {"code": 0, "kind": "place", "node": 0}
    assert result["root_actions"] == [
        {"code": 0, "kind": "place", "node": 0},
        {"code": 1, "kind": "place", "node": 1},
    ]
    assert result["outcome"] == {"loss": 0.2, "win": 0.8}
    assert result["score_belief"]["support_min"] == -151
    assert result["score_belief"]["support_max"] == 151
    assert result["model_version"] == "fake-native-v2"


def test_native_analysis_rejects_incompatible_import_and_cancellation(tmp_path) -> None:
    request = AnalyzeRequest.model_validate(request_payload())
    missing_importer = NativeAnalysisService(
        server_config(tmp_path),
        native_module=SimpleNamespace(StateBatch=object, SearchBatch=FakeSearchBatch),
        model_manager=FakeManager(),
    )
    with pytest.raises(AnalysisError, match="from_semantic") as error:
        missing_importer.analyze(request, threading.Event())
    assert error.value.code == "native_incompatible"

    service = NativeAnalysisService(
        server_config(tmp_path),
        native_module=SimpleNamespace(
            StateBatch=FakeStateBatch,
            SearchBatch=FakeSearchBatch,
        ),
        model_manager=FakeManager(),
    )
    cancellation = threading.Event()
    cancellation.set()
    with pytest.raises(SearchCancelled) as error:
        service.analyze(request, cancellation)
    assert error.value.status_code == 499


@pytest.mark.native
def test_true_native_server_search_when_available(tmp_path) -> None:
    native = pytest.importorskip("star_native")
    evaluator = GraphInferenceAdapter(
        GraphResTNet(
            ModelConfig(
                width=8,
                rrt_groups=1,
                attention_heads=2,
                kv_heads=1,
            )
        ).eval(),
        model_version="native-server-v2",
        model_step=1,
    )

    class Manager:
        def startup(self):
            pass

        def health(self):
            return {
                "ready": True,
                "model_version": evaluator.model_version,
                "model_step": evaluator.model_step,
            }

        @contextmanager
        def lease(self):
            manifest = ModelManifest(
                tmp_path / "manifest.json",
                tmp_path / "model.pt",
                evaluator.model_version,
                evaluator.model_step,
                time.time_ns(),
                role="champion",
            )
            yield ModelLease(LoadedModel(manifest, evaluator), 0.0)

    service = NativeAnalysisService(
        server_config(tmp_path),
        native_module=native,
        model_manager=Manager(),
    )
    payload = request_payload()
    payload["search"] = {"simulations": 1, "max_considered": 2, "seed": 11}
    result = service.analyze(
        AnalyzeRequest.model_validate(payload),
        threading.Event(),
    )
    assert 0 <= result["action"]["code"] < get_topology(4).n
    assert sum(result["root_policy"]) == pytest.approx(1.0)
    assert result["model_version"] == "native-server-v2"


def test_v2_schema_cross_field_validation_is_strict() -> None:
    request = request_payload()
    request["stones"] = [0] * get_topology(4).n
    with pytest.raises(ValidationError, match="terminal must equal board-full"):
        AnalyzeRequest.model_validate(request)

    request = request_payload()
    request["stones"] = request["stones"][:-1]
    with pytest.raises(ValidationError, match="exactly 50"):
        AnalyzeRequest.model_validate(request)

    with pytest.raises(ValidationError, match="must match"):
        AtomicAction.model_validate({"code": 1, "kind": "place", "node": 2})

    score = response_payload()["score_belief"]
    assert isinstance(score, dict)
    with pytest.raises(ValidationError, match="sum to one"):
        ScoreBelief.model_validate(
            {
                **score,
                "probabilities": [0.0] * (SCORE_MARGIN_MAX - SCORE_MARGIN_MIN + 1),
            }
        )

    malformed = {**response_payload(), "request_id": "request-1"}
    malformed["root_visits"] = [1]
    with pytest.raises(ValidationError, match="inconsistent shapes"):
        AnalyzeResponse.model_validate(malformed)

    malformed = {**response_payload(), "request_id": "request-1"}
    malformed["action"] = {"code": 3, "kind": "place", "node": 3}
    with pytest.raises(ValidationError, match="selected action"):
        AnalyzeResponse.model_validate(malformed)

    malformed = {**response_payload(), "request_id": "request-1"}
    malformed["value"] = 0.5
    with pytest.raises(ValidationError, match=r"P\(win\)-P\(loss\)"):
        AnalyzeResponse.model_validate(malformed)
