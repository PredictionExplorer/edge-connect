from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from starserve.app import create_app
from starserve.config import (
    LimitConfig,
    SearchConfig,
    SecurityConfig,
    ServerConfig,
    ServerConfigError,
)
from starserve.runtime import (
    AtomicModelManager,
    LoadedModel,
    ModelLease,
    NativeAnalysisService,
)
from starserve.schemas import AnalyzeRequest
from startrain.checkpoint import ModelManifest
from startrain.contracts import RULES_HASH_WIRE
from startrain.inference import (
    DetailedInferenceResponse,
    GraphInferenceAdapter,
    InferenceResponse,
)
from startrain.model import GraphResTNet, ModelConfig
from startrain.native import BITBOARD_WORDS
from startrain.topology import get_topology


def server_config(tmp_path, **changes) -> ServerConfig:
    values = {
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
        "schema_version": 1,
        "rules_hash": RULES_HASH_WIRE,
        "rings": 3,
        "stones": [-1] * get_topology(3).n,
        "to_move": 0,
        "moves_left": 1,
        "opening": True,
        "pass_streak": 0,
        "terminal": False,
        "search": {"simulations": 4, "max_considered": 2, "seed": 7},
    }


def response_payload() -> dict[str, object]:
    score = [0.0] * 363
    score[181] = 1.0
    return {
        "schema_version": 1,
        "action": {"code": 0, "kind": "place", "node": 0},
        "root_actions": [
            {"code": 0, "kind": "place", "node": 0},
            {"code": -1, "kind": "pass", "node": None},
        ],
        "root_policy": [0.75, 0.25],
        "root_q": [0.2, -0.1],
        "root_visits": [3, 1],
        "wdl": {"loss": 0.2, "draw": 0.3, "win": 0.5},
        "value": 0.3,
        "search_value": 0.3,
        "score_belief": {
            "support_min": -181,
            "support_max": 181,
            "expected_margin": 0.0,
            "probabilities": score,
        },
        "model_version": "fake-v1",
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
            "model_version": "fake-v1",
            "model_step": 5,
        }

    def analyze(
        self, request: AnalyzeRequest, cancellation: threading.Event
    ) -> dict[str, object]:
        assert request.rings == 3
        assert not cancellation.is_set()
        return response_payload()


def test_fastapi_health_auth_validation_and_analysis(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TEST_STARSERVE_TOKEN", "correct-secret")
    config = server_config(
        tmp_path,
        security=SecurityConfig(
            cors_allow_origins=("https://play.example",),
            bearer_token_env="TEST_STARSERVE_TOKEN",
        ),
    )
    with TestClient(create_app(config, service=FakeService())) as client:
        health = client.get("/healthz")
        assert health.status_code == 200
        assert health.json()["rules"]["hash"] == RULES_HASH_WIRE
        assert health.json()["model"]["model_version"] == "fake-v1"
        preflight = client.options(
            "/v1/analyze",
            headers={
                "Origin": "https://play.example",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert preflight.status_code == 200
        assert (
            preflight.headers["access-control-allow-origin"] == "https://play.example"
        )

        assert client.post("/v1/analyze", json=request_payload()).status_code == 401
        headers = {"Authorization": "Bearer correct-secret"}
        response = client.post("/v1/analyze", json=request_payload(), headers=headers)
        assert response.status_code == 200
        assert response.json()["action"] == {"code": 0, "kind": "place", "node": 0}
        assert response.json()["model_version"] == "fake-v1"
        assert response.headers["cache-control"] == "no-store"

        invalid = request_payload()
        invalid["rules_hash"] = "fnv1a64:0000000000000000"
        rejected = client.post("/v1/analyze", json=invalid, headers=headers)
        assert rejected.status_code == 422
        assert rejected.json()["error"]["code"] == "invalid_request"

        over_budget = request_payload()
        over_budget["search"] = {
            "simulations": 17,
            "max_considered": 2,
            "seed": 7,
        }
        rejected = client.post("/v1/analyze", json=over_budget, headers=headers)
        assert rejected.status_code == 422
        assert rejected.json()["error"]["code"] == "search_budget_exceeded"


def test_server_config_rejects_contract_and_cors_drift(tmp_path) -> None:
    with pytest.raises(ServerConfigError, match="rules hash"):
        server_config(tmp_path, rules_hash="fnv1a64:0000000000000000")
    with pytest.raises(ServerConfigError, match="wildcard"):
        server_config(
            tmp_path,
            security=SecurityConfig(cors_allow_origins=("*",)),
        )


def test_request_size_limit_is_structured(tmp_path) -> None:
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
            "/v1/analyze",
            content=json.dumps(request_payload()),
            headers={"Content-Type": "application/json"},
        )
        chunked = client.post(
            "/v1/analyze",
            content=(chunk for chunk in (b"{" + b"x" * 40, b"x" * 40 + b"}")),
            headers={"Content-Type": "application/json"},
        )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "request_too_large"
    assert response.json()["error"]["request_id"]
    assert chunked.status_code == 413
    assert chunked.json()["error"]["code"] == "request_too_large"


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
        response = client.post("/v1/analyze", json=request_payload())
        assert response.status_code == 504
        assert response.json()["error"]["code"] == "analysis_timeout"
        assert service.cancelled.wait(1)


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
            model=SimpleNamespace(node_feature_dim=15, global_feature_dim=18)
        ),
        manifest_reader=reader,
        bundle_loader=loader,
    )
    with manager.lease() as first:
        assert first.model.manifest.model_version == "v1"
        with manager.lease() as overlapping:
            assert overlapping.model.manifest.model_version == "v1"
            assert calls == 1
    with manager.lease() as second:
        assert second.model.manifest.model_version == "v2"
    with manager.lease() as retained:
        assert retained.model.manifest.model_version == "v2"
    assert manager.health()["last_reload_error"] == "ValueError: bad publication"


def test_atomic_model_manager_rejects_candidate_pointer(tmp_path) -> None:
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
            model=SimpleNamespace(node_feature_dim=15, global_feature_dim=18)
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
        pass_streak,
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
            pass_streak=pass_streak,
            terminal=[False],
            pass_legal=[True],
        )
        return FakeStateBatch(data)

    def data(self):
        return self._data


class FakeRootBatch:
    def __init__(self, states) -> None:
        self.states = states.data()
        self.tokens = [1]
        self.legal_actions = list(range(self.states.node_count)) + [-1]
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
        actions = list(range(30)) + [-1]
        return SimpleNamespace(
            action_offsets=[0, len(actions)],
            actions=actions,
            policy_target=[1.0] + [0.0] * (len(actions) - 1),
            q_values=[0.1] * len(actions),
            visits=[4] + [0] * (len(actions) - 1),
            selected_actions=[0],
            terminal=[False],
        )


class FakeEvaluator:
    model_version = "fake-native-v1"
    model_step = 8

    def evaluate_detailed(self, roots):
        response = InferenceResponse(
            tokens=roots.tokens,
            values=[0.2],
            policy_offsets=roots.legal_offsets,
            policy_logits=[0.0] * len(roots.legal_actions),
        )
        score = [0.0] * 363
        score[181] = 1.0
        return DetailedInferenceResponse(
            response=response,
            wdl_probabilities=[[0.2, 0.4, 0.4]],
            wdl_values=[0.2],
            score_expectations=[0.0],
            score_probabilities=[score],
        )

    def evaluate(self, _requests):
        raise AssertionError("fake search has no leaves")


class FakeManager:
    def startup(self):
        pass

    def health(self):
        return {"ready": True, "model_version": "fake-native-v1", "model_step": 8}

    @contextmanager
    def lease(self):
        manifest = ModelManifest(
            SimpleNamespace(),
            SimpleNamespace(),
            "fake-native-v1",
            8,
            time.time_ns(),
            role="champion",
        )
        yield ModelLease(LoadedModel(manifest, FakeEvaluator()), 0.0)


def test_native_analysis_path_imports_semantic_state_and_returns_root(tmp_path) -> None:
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
    assert result["action"] == {"code": 0, "kind": "place", "node": 0}
    assert result["root_visits"][0] == 4
    assert result["model_version"] == "fake-native-v1"


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
        model_version="native-server-smoke",
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
    assert result["action"]["code"] in list(range(30)) + [-1]
    assert sum(result["root_policy"]) == pytest.approx(1.0)
    assert result["model_version"] == "native-server-smoke"
