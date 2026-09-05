from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

import startrain.promotion as promotion_module
from startrain.arena import ArenaRunner, summarize_completed_arena_pairs
from startrain.checkpoint import write_model_pointer
from startrain.config import HistoricalEvaluationConfig
from startrain.historical_evaluation import HistoricalEvaluationPlan
from startrain.promotion import PromotionSupervisor

from test_promotion import _promotion_wave_case


def test_balanced_time_slices_preserve_moves_before_first_complete_pair(
    tmp_path, monkeypatch
):
    case = _promotion_wave_case(tmp_path, monkeypatch)
    experiment = replace(
        case.experiment,
        arena=replace(case.experiment.arena, rings=(4, 6, 8, 10), balanced_cells=True),
        orchestration=replace(
            case.experiment.orchestration,
            promotion=replace(
                case.experiment.orchestration.promotion,
                session_seconds=5.0,
                max_waves_per_lease=1,
                inter_wave_cooldown_seconds=7.0,
                finish_inflight_candidate=True,
            ),
        ),
    )
    subject = case.supervisor
    subject.experiment = experiment
    clock = SimpleNamespace(now=0.0)
    subject.clock = lambda: clock.now
    subject.wall_clock_ns = lambda: int(clock.now * 1_000_000_000)
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        clock.now += seconds

    subject.sleep = sleep
    seen_states = []

    class InterruptedArena:
        def __init__(self, **options):
            self.config = options["config"]
            assert options["stable_pair_seeds"] is True

        def run(self, *, resume_state, checkpoint, stop_requested, **_options):
            seen_states.append(resume_state)
            moves = (resume_state or {}).get("progress", {}).get("completed_moves", 0)
            state = {"progress": {"completed_moves": moves + 1}, "game_states": []}
            checkpoint(state)
            clock.now += 5.0
            assert stop_requested()
            return {
                "candidate": case.candidate.model_identity,
                "baseline": case.champion.model_identity,
                "pairs": [],
                "games": [],
                "resume_state": state,
                **summarize_completed_arena_pairs([], self.config),
            }

    monkeypatch.setattr(promotion_module, "ArenaRunner", InterruptedArena)
    path = subject._result_path(case.candidate, case.champion)
    previous = None
    for iteration in range(2):
        evaluated, state = subject._evaluate_candidate_session(
            candidate=case.candidate,
            champion=case.champion,
            previous=previous,
            stop_requested=lambda: False,
            progress=None,
            once=True,
        )
        assert (evaluated, state) == (1, "lease_yield")
        previous = json.loads(path.read_text())
        assert previous["pairs"] == []
        assert previous["terminal"] is False
        saved = subject._load_resume_state(path, case.candidate, case.champion)
        assert saved["progress"]["completed_moves"] == iteration + 1
        assert subject._evaluation_started(case.candidate, case.champion, previous)
    assert seen_states == [
        None,
        {"progress": {"completed_moves": 1}, "game_states": []},
    ]
    assert sum(sleeps) == pytest.approx(7.0)
    assert case.state.lease_entries == 2
    events = [
        json.loads(line)
        for line in subject.session_events_path.read_text().splitlines()
    ]
    assert [event["reason"] for event in events] == ["time_budget", "time_budget"]


def test_saved_moves_keep_older_candidate_inflight_after_new_publication(
    tmp_path, monkeypatch
):
    case = _promotion_wave_case(tmp_path, monkeypatch)
    subject = case.supervisor
    write_model_pointer(
        case.publisher.champion_path,
        case.champion,
        role="champion",
        promotion_result="bootstrap",
    )
    newer = replace(case.candidate, model_step=2, model_identity="newer-candidate")
    monkeypatch.setattr(
        subject, "_candidate_manifests", lambda: [case.champion, case.candidate, newer]
    )
    path = subject._result_path(case.candidate, case.champion)
    subject._resume_writer(path, case.candidate, case.champion)(
        {"progress": {"completed_moves": 3}}
    )
    selected = []

    def evaluate(**options):
        selected.append(options["candidate"])
        assert options["previous"] is None
        return 1, "once"

    monkeypatch.setattr(subject, "_evaluate_candidate_session", evaluate)
    assert subject.run(stop_requested=lambda: False, once=True) == 1
    assert selected == [case.candidate]


def test_historical_candidate_preemption_stays_latched_and_records_cooldown(
    tmp_path, monkeypatch
):
    case = _promotion_wave_case(tmp_path, monkeypatch)
    subject = case.supervisor
    subject.experiment = replace(
        case.experiment,
        orchestration=replace(
            case.experiment.orchestration,
            historical_evaluation=HistoricalEvaluationConfig(enabled=True),
        ),
    )
    subject.clock = lambda: 0.0
    subject.wall_clock_ns = lambda: 1_000_000_000
    ready = SimpleNamespace(value=False)
    monkeypatch.setattr(subject, "_has_ready_candidate", lambda: ready.value)
    path = tmp_path / "arena" / "crossplay.json"
    plan = HistoricalEvaluationPlan(
        case.candidate, case.champion, path, None, "measurement"
    )
    monkeypatch.setattr(
        promotion_module, "select_historical_evaluation", lambda **_options: plan
    )
    calls = []

    def evaluate(*, stop_requested, **_options):
        assert stop_requested() is False
        # Move beyond the throttle window after a candidate arrives.
        ready.value = True
        subject.clock = lambda: 1.0
        calls.extend(stop_requested() for _ in range(3))
        return 0

    monkeypatch.setattr(subject, "_evaluate_historical_waves", evaluate)
    assert (
        subject._evaluate_historical_if_due(
            champion=case.champion,
            stop_requested=lambda: False,
            progress=None,
            once=True,
        )
        == 0
    )
    assert calls == [True, True, True]
    assert subject._historical_cooldown_ready() is False
    event = json.loads(subject.session_events_path.read_text())
    assert event["reason"] == "candidate_ready"


def test_historical_cooldown_does_not_delay_ready_promotion(tmp_path, monkeypatch):
    case = _promotion_wave_case(tmp_path, monkeypatch)
    subject = case.supervisor
    subject.wall_clock_ns = lambda: 1_000_000_000
    subject._record_historical_cooldown(case.champion)
    assert subject._historical_cooldown_ready() is False
    calls = []

    def evaluate(**options):
        calls.append(options["candidate"])
        return 1, "once"

    monkeypatch.setattr(subject, "_evaluate_candidate_session", evaluate)
    assert subject.run(stop_requested=lambda: False, once=True) == 1
    assert [item.model_identity for item in calls] == [case.candidate.model_identity]


def test_legacy_nonbalanced_evaluations_preserve_seed_policy(tmp_path, monkeypatch):
    case = _promotion_wave_case(tmp_path, monkeypatch)
    base = promotion_module.ArenaRunner
    seen = []

    class LegacyArena(base):
        def __init__(self, **options):
            seen.append(options["stable_pair_seeds"])
            super().__init__(**options)

        def run(self, **options):
            assert "resume_state" not in options
            assert "checkpoint" not in options
            return super().run(**options)

    monkeypatch.setattr(promotion_module, "ArenaRunner", LegacyArena)
    assert case.supervisor.run(stop_requested=lambda: False, once=True) == 1
    assert seen == [False]
    assert not case.supervisor._resume_path(case.result_path).exists()


@pytest.mark.native
def test_native_balanced_supervisor_resumes_saved_moves_after_restart(
    tmp_path, monkeypatch
):
    from test_arena_resume import Clock, Evaluator

    native = pytest.importorskip("star_native")
    case = _promotion_wave_case(tmp_path, monkeypatch)
    experiment = replace(
        case.experiment,
        arena=replace(case.experiment.arena, rings=(4, 6, 8, 10), balanced_cells=True),
        orchestration=replace(
            case.experiment.orchestration,
            promotion=replace(
                case.experiment.orchestration.promotion,
                session_seconds=8.0,
                max_waves_per_lease=1,
                finish_inflight_candidate=True,
            ),
        ),
    )
    clock = Clock()
    monkeypatch.setattr(promotion_module, "ArenaRunner", ArenaRunner)
    monkeypatch.setattr(
        promotion_module,
        "load_manifest_evaluator",
        lambda _experiment, manifest, **_options: Evaluator(
            manifest.model_version, clock
        ),
    )
    snapshots = []
    previous = None
    for _ in range(2):
        # A fresh supervisor and fresh evaluators must recover solely from the
        # persisted result and move sidecar, as a deployed process would.
        subject = PromotionSupervisor(
            experiment=experiment,
            run_identity=case.identity,
            candidate_path=case.publisher.candidate_path,
            champion_path=case.publisher.champion_path,
            results_directory=tmp_path / "arena",
            native_module=native,
            device="cpu",
            clock=lambda: float(clock.now),
            wall_clock_ns=lambda: clock.now * 1_000_000_000,
        )
        evaluated, state = subject._evaluate_candidate_session(
            candidate=case.candidate,
            champion=case.champion,
            previous=previous,
            stop_requested=lambda: False,
            progress=None,
            once=True,
        )
        assert (evaluated, state) == (1, "lease_yield")
        path = subject._result_path(case.candidate, case.champion)
        previous = json.loads(path.read_text())
        assert previous["pairs"] == []
        assert previous["terminal"] is False
        saved = subject._load_resume_state(path, case.candidate, case.champion)
        assert saved["config"]["balanced_cells"] is True
        assert saved["progress"]["completed_moves"] > 0
        snapshots.append(saved)
    assert (
        snapshots[1]["progress"]["completed_moves"]
        > snapshots[0]["progress"]["completed_moves"]
    )
    histories = {
        (item["ring"], item["variant"], item["pair"], item["candidate_player"]): item[
            "actions"
        ]
        for item in snapshots[1]["game_states"]
    }
    for item in snapshots[0]["game_states"]:
        key = (item["ring"], item["variant"], item["pair"], item["candidate_player"])
        assert histories[key][: len(item["actions"])] == item["actions"]


def test_deadline_during_model_loading_yields_without_requiring_pairs(
    tmp_path, monkeypatch
):
    case = _promotion_wave_case(tmp_path, monkeypatch)
    subject = case.supervisor
    subject.experiment = replace(
        case.experiment,
        arena=replace(case.experiment.arena, rings=(4, 6, 8, 10), balanced_cells=True),
        orchestration=replace(
            case.experiment.orchestration,
            promotion=replace(
                case.experiment.orchestration.promotion, session_seconds=5.0
            ),
        ),
    )
    clock = SimpleNamespace(now=0.0)
    subject.clock = lambda: clock.now
    original_loader = promotion_module.load_manifest_evaluator

    def load(*args, **options):
        clock.now += 6.0
        return original_loader(*args, **options)

    monkeypatch.setattr(promotion_module, "load_manifest_evaluator", load)
    assert subject._evaluate_candidate_session(
        candidate=case.candidate,
        champion=case.champion,
        previous=None,
        stop_requested=lambda: False,
        progress=None,
        once=True,
    ) == (0, "lease_yield")
    assert case.state.wave_calls == 0
    assert not subject._result_path(case.candidate, case.champion).exists()
    event = json.loads(subject.session_events_path.read_text())
    assert event["reason"] == "time_budget"
