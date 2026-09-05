from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from startrain.balanced_evaluation import evaluation_contract
from startrain.config import ArenaConfig, PlateauConfig, load_config
from startrain.learner import LearnerLoop
from startrain.promotion import PromotionSupervisor
from startrain.runtime import RunIdentity


def experiment():
    base = load_config(Path(__file__).parents[1] / "configs" / "small.yaml")
    return replace(base, arena=ArenaConfig(balanced_cells=True))


@pytest.mark.parametrize("prior_terminal", [False, True])
@pytest.mark.parametrize("prior_contract", [None, "sha256-retired-contract"])
def test_first_balanced_rejection_starts_a_new_contract_streak(
    tmp_path, prior_terminal, prior_contract
):
    supervisor = object.__new__(PromotionSupervisor)
    supervisor.experiment = experiment()
    supervisor.status_path = tmp_path / "promotion-status.json"
    supervisor._resume_cutover = lambda: None
    candidate = SimpleNamespace(model_identity="candidate-one", model_step=1)
    champion = SimpleNamespace(model_identity="champion", model_step=0)
    prior = {
        "candidate_identity": candidate.model_identity,
        "terminal": prior_terminal,
        "decision": "reject" if prior_terminal else "continue",
        "consecutive_terminal_rejections": 9,
        "consecutive_conclusive_rejections": 9,
        "cutover_created_ns": None,
    }
    if prior_contract is not None:
        prior["evaluation_contract_identity"] = prior_contract
    supervisor.status_path.write_text(json.dumps(prior))
    supervisor._write_status(
        candidate=candidate, champion=champion, decision="reject", terminal=True
    )
    first = json.loads(supervisor.status_path.read_text())
    assert (
        first["evaluation_contract_identity"]
        == evaluation_contract(supervisor.experiment.arena)["identity"]
    )
    assert (
        first["consecutive_terminal_rejections"]
        == first["consecutive_conclusive_rejections"]
        == 1
    )
    # The same new-contract result is idempotent.
    supervisor._write_status(
        candidate=candidate, champion=champion, decision="reject", terminal=True
    )
    assert (
        json.loads(supervisor.status_path.read_text())[
            "consecutive_conclusive_rejections"
        ]
        == 1
    )
    second_candidate = SimpleNamespace(model_identity="candidate-two", model_step=2)
    supervisor._write_status(
        candidate=second_candidate, champion=champion, decision="reject", terminal=True
    )
    assert (
        json.loads(supervisor.status_path.read_text())[
            "consecutive_conclusive_rejections"
        ]
        == 2
    )


def test_learner_ignores_retired_status_but_counts_new_contract_rejections(
    tmp_path, monkeypatch
):
    from test_pipeline_core import _plateau_policy_fixture

    learner, status, _ = _plateau_policy_fixture(tmp_path, monkeypatch)
    learner.step = 160_000
    expected = str(evaluation_contract(experiment().arena)["identity"])
    learner.expected_promotion_contract_identity = expected
    configured = PlateauConfig(
        enabled=True, action="reduce_lr_keep_weights", consecutive_terminal_rejections=2
    )
    verdict = dict(
        decision="reject",
        terminal=True,
        conclusive=True,
        consecutive_terminal_rejections=9,
        consecutive_conclusive_rejections=9,
    )
    status(**verdict)
    assert learner._rank_zero_plateau_action(configured) == {"kind": "proceed"}
    status(**verdict, evaluation_contract_identity="sha256-retired-contract")
    assert learner._rank_zero_plateau_action(configured) == {"kind": "proceed"}
    verdict.update(
        consecutive_terminal_rejections=1, consecutive_conclusive_rejections=1
    )
    status(**verdict, evaluation_contract_identity=expected)
    assert learner._rank_zero_plateau_action(configured) == {"kind": "proceed"}
    verdict.update(
        consecutive_terminal_rejections=2, consecutive_conclusive_rejections=2
    )
    status(**verdict, evaluation_contract_identity=expected)
    assert learner._rank_zero_plateau_action(configured)["kind"] == "recover"


@pytest.mark.parametrize("balanced", [False, True])
def test_learner_factory_derives_runtime_contract_without_changing_saved_config(
    tmp_path, balanced
):
    configured = experiment()
    configured = replace(
        configured, arena=replace(configured.arena, balanced_cells=balanced)
    )

    class CapturedLoop(LearnerLoop):
        def __init__(self, **options):
            self.options = options

    learner = CapturedLoop.from_experiment(
        configured,
        store=object(),
        output_directory=tmp_path,
        run_identity=RunIdentity(tmp_path / "run.json", "run", "family", 1),
    )
    assert learner.options["expected_promotion_contract_identity"] == (
        evaluation_contract(configured.arena)["identity"] if balanced else None
    )
    assert learner.options["serialized_config"] == configured.as_dict()
