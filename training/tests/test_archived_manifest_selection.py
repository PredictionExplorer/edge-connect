from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import pytest
import torch
import yaml

import scripts.evaluate_archived_manifests as archived_selection_script
from scripts.evaluate_archived_manifests import (
    canonicalize_device_lanes,
    evaluate_archived_manifests,
    plan_archived_manifest_evaluation,
    static_device_assignments,
)
from scripts.fork_elo_ablation import fork_elo_ablation
from scripts.prepare_champion_warm_start import (
    WarmStartError,
    prepare_champion_warm_start,
)
from scripts.prepare_elo_ablation import prepare_elo_ablation
from startrain.arena import ARENA_RESULT_SCHEMA_VERSION, ArenaPair, summarize_pairs
from startrain.checkpoint import (
    ExponentialMovingAverage,
    ModelManifest,
    load_model_manifest,
    write_model_pointer,
    write_recovery_checkpoint,
)
from startrain.config import ExperimentConfig, load_config
from startrain.learner import ImmutableModelPublisher
from startrain.manifest_selection import (
    ManifestSelectionError,
    SelectionEvidence,
    SelectionPlan,
    build_selection_evidence,
    candidate_seed,
    freeze_selection_evidence,
    freeze_selection_plan,
    load_persisted_selection_result,
    next_selection_pair_count,
    selection_opening,
    verify_selection_snapshot,
)
from startrain.model import GraphResTNet
from startrain.optim import build_optimizer
from startrain.replay_store import ReplayStore
from startrain.runtime import (
    RunIdentity,
    atomic_json,
    require_active_selection_cutover,
    require_launch_ready,
)
from startrain.training import build_scheduler

CONFIGS = Path(__file__).parents[1] / "configs"


@dataclass(frozen=True)
class RunFixture:
    root: Path
    profile: Path
    experiment: ExperimentConfig
    identity: RunIdentity
    champion: ModelManifest
    archived: tuple[ModelManifest, ModelManifest]


def _run_fixture(tmp_path: Path, *, name: str = "source") -> RunFixture:
    directory = tmp_path / name
    directory.mkdir(parents=True)
    root = (directory / "run").resolve()
    root.mkdir()
    raw = yaml.safe_load(
        (CONFIGS / "h100-8gpu-throughput.yaml").read_text(encoding="utf-8")
    )
    raw["model"].update(
        {
            "width": 8,
            "rrt_groups": 1,
            "attention_heads": 2,
            "kv_heads": 1,
        }
    )
    raw["train"].update(
        {
            "per_rank_batch_size": 1,
            "precision": "fp32",
            "compile": False,
        }
    )
    raw["data"].update({"workers": 0, "pin_memory": False})
    raw["learner"]["device"] = "cpu"
    run_id = f"run-{name}"
    raw["orchestration"]["run_id"] = run_id
    raw["orchestration"]["directories"]["root"] = str(root)
    profile = directory / "profile.yaml"
    profile.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    experiment = load_config(profile)
    identity = RunIdentity(
        root / "run.json",
        run_id,
        f"family-{name}",
        1,
    )
    atomic_json(
        identity.path,
        {
            "schema_version": 1,
            "run_id": identity.run_id,
            "generation_family": identity.generation_family,
            "created_ns": identity.created_ns,
        },
    )
    (root / "source-commit.txt").write_text(f"{'a' * 40}\n", encoding="utf-8")
    with ReplayStore(root / "replay") as store:
        store.register_run(identity)

    model = GraphResTNet(experiment.model)
    optimizer = build_optimizer(model, experiment.optimizer)
    scheduler = build_scheduler(optimizer, experiment.train.scheduler)
    ema = ExponentialMovingAverage(model, decay=experiment.train.ema_decay)
    publisher = ImmutableModelPublisher(root / "learner", identity)

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.1)
    ema.update(model)
    champion = publisher.publish(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema,
        step=10,
        epoch=0,
        config=experiment.as_dict(),
        examples_consumed=100,
        global_batch_size=1,
    )
    write_model_pointer(
        root / "learner" / "champion.json",
        champion,
        role="champion",
        promotion_result="bootstrap",
    )

    archived = []
    for step, examples, increment in ((20, 200, 0.2), (30, 300, 0.3)):
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(increment)
        ema.update(model)
        archived.append(
            publisher.publish(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                ema=ema,
                step=step,
                epoch=1,
                config=experiment.as_dict(),
                examples_consumed=examples,
                global_batch_size=1,
            )
        )
    write_recovery_checkpoint(
        root / "learner",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema,
        step=30,
        epoch=1,
        config=experiment.as_dict(),
        run_id=identity.run_id,
        generation_family=identity.generation_family,
        examples_consumed=300,
        global_batch_size=1,
    )
    return RunFixture(
        root=root,
        profile=profile,
        experiment=experiment,
        identity=identity,
        champion=champion,
        archived=(archived[0], archived[1]),
    )


def _manifest_path(manifest: ModelManifest) -> Path:
    return (manifest.artifact_manifest or manifest.path).resolve()


def _plan(fixture: RunFixture, output: Path) -> tuple[SelectionPlan, Path]:
    plan = plan_archived_manifest_evaluation(
        source_run_root=fixture.root,
        profile=fixture.profile,
        candidate_manifest_paths=tuple(
            _manifest_path(manifest) for manifest in reversed(fixture.archived)
        ),
    )
    output.mkdir(parents=True)
    path = output / "selection-plan.json"
    freeze_selection_plan(path, plan)
    return plan, path


def _result(
    plan: SelectionPlan,
    candidate_index: int,
    path: Path,
    *,
    lower_elo: float,
    completed_pairs: int | None = None,
) -> Path:
    candidate = plan.candidates[candidate_index]
    requested_proven = lower_elo > 0
    completed = (
        completed_pairs
        if completed_pairs is not None
        else (
            plan.contract.initial_pairs if requested_proven else plan.contract.max_pairs
        )
    )

    def outcomes(pair: int) -> tuple[int, int]:
        if lower_elo >= 8:
            return (1, 1)
        if lower_elo > 0:
            return (1, 1) if pair < int(completed * 0.9) else (1, -1)
        if lower_elo < 0 and pair < int(completed * 0.1):
            return (-1, -1)
        return (1, -1)

    pairs = [
        ArenaPair(
            ring=10,
            pair=pair,
            opening_seed=selection_opening(plan, candidate, pair)[0],
            opening_action=selection_opening(plan, candidate, pair)[2],
            forced_opening=selection_opening(plan, candidate, pair)[1],
            outcomes=outcomes(pair),
        )
        for pair in range(completed)
    ]
    summary = summarize_pairs(
        pairs,
        confidence=plan.contract.confidence,
        bootstrap_samples=plan.contract.bootstrap_samples,
        seed=candidate_seed(plan, candidate.model_identity) + 10 * 1_000_003,
    )
    interval = summary["anytime_elo_interval"]
    assert isinstance(interval, list)
    proven = float(interval[0]) > 0
    assert proven is requested_proven
    terminal = (proven and completed >= plan.contract.minimum_pairs) or (
        completed == plan.contract.max_pairs
    )
    budget = {
        "simulations": plan.contract.simulations,
        "max_considered": plan.contract.max_considered,
        "c_visit": plan.contract.c_visit,
        "c_scale": plan.contract.c_scale,
    }
    search = {
        "deterministic": True,
        **budget,
        "pie_rule": False,
        "search_workers": 2,
        "inference_workers": 1,
        "pair_chunk_size": plan.contract.pair_chunk_size,
        "effective_pair_chunking": (
            "configured"
            if plan.contract.pair_chunk_size is not None
            else "full_requested_ring_batch"
        ),
    }
    wave_history = []
    wave_start = 0
    while wave_start < completed:
        pair_count = next_selection_pair_count(plan.contract, wave_start)
        wave_start += pair_count
        wave_history.append(
            {
                "wave_index": len(wave_history),
                "pair_start": wave_start - pair_count,
                "pair_count": pair_count,
                "completed_pairs": wave_start,
                "evaluation_metrics": {},
            }
        )
    atomic_json(
        path,
        {
            "schema_version": ARENA_RESULT_SCHEMA_VERSION,
            "result_kind": "archived_manifest_selection",
            "selection_plan_digest": plan.plan_digest,
            "selection_contract": plan.contract.as_dict(),
            "candidate": candidate.model_identity,
            "baseline": plan.source_champion.manifest.model_identity,
            "candidate_manifest": candidate.manifest.path,
            "baseline_manifest": plan.source_champion.manifest.manifest.path,
            "arena_seed_block": candidate_seed(plan, candidate.model_identity),
            "evaluation_started_ns": 1,
            "search": search,
            "baseline_metadata": {
                "kind": "checkpoint",
                "identity": plan.source_champion.manifest.model_identity,
                "search_budget": budget,
                "deterministic": True,
                "seed_schedule": "arena-runner-v2-pair-chunks",
            },
            "pairs": [asdict(pair) for pair in pairs],
            "games": [{} for _ in range(completed * 2)],
            "wave_history": wave_history,
            "per_ring": {"10": summary},
            "statistically_proven_improvement": proven,
            "selection_decision": ("proven_improvement" if proven else "not_proven"),
            "terminal": terminal,
        },
    )
    return path


def _selection(
    fixture: RunFixture,
    output: Path,
    *,
    lower_elos: tuple[float, float],
) -> tuple[SelectionPlan, SelectionEvidence, Path]:
    plan, plan_path = _plan(fixture, output)
    results = {
        candidate.model_identity: _result(
            plan,
            index,
            output / f"result-{index}.json",
            lower_elo=lower_elos[index],
        )
        for index, candidate in enumerate(plan.candidates)
    }
    evidence = build_selection_evidence(
        plan_path=plan_path,
        result_paths=results,
        generated_ns=123,
    )
    snapshot = output / "selection-snapshot.json"
    freeze_selection_evidence(snapshot, evidence)
    return plan, evidence, snapshot


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def _fork(
    fixture: RunFixture,
    tmp_path: Path,
    snapshot: Path,
    *,
    prefix: str,
    treatment: str = "control",
) -> tuple[Path, dict[str, object]]:
    profiles = tmp_path / f"{prefix}-profiles"
    prepare_elo_ablation(
        base_config=fixture.profile,
        output_dir=profiles,
        run_root_parent=tmp_path / f"{prefix}-runs",
        run_id=fixture.identity.run_id,
        source_run_root=fixture.root,
        prefix=prefix,
        seed=17,
        wall_budget_hours=1,
        leaf_budget=10,
        guard_floor_elo=-35,
        treatments=(treatment,),
    )
    metadata = fork_elo_ablation(
        source_run_root=fixture.root,
        plan_path=profiles / "ablation-plan.json",
        treatment=treatment,
        selection_snapshot=snapshot,
    )
    return tmp_path / f"{prefix}-runs" / f"{prefix}-{treatment}-seed17", metadata


def test_plan_digest_seed_and_pair_contract_are_deterministic(tmp_path: Path) -> None:
    fixture = _run_fixture(tmp_path)

    first = plan_archived_manifest_evaluation(
        source_run_root=fixture.root,
        profile=fixture.profile,
        candidate_manifest_paths=tuple(
            _manifest_path(manifest) for manifest in fixture.archived
        ),
    )
    second = plan_archived_manifest_evaluation(
        source_run_root=fixture.root,
        profile=fixture.profile,
        candidate_manifest_paths=tuple(
            _manifest_path(manifest) for manifest in reversed(fixture.archived)
        ),
    )

    assert first.as_dict() == second.as_dict()
    assert first.plan_digest == second.plan_digest
    assert first.evaluation_seed == second.evaluation_seed
    assert first.contract.initial_pairs == 50
    assert first.contract.continuation_pairs == 25
    assert first.contract.max_pairs == 200
    assert first.contract.ring == 10


def test_planning_and_evidence_do_not_mutate_parent(tmp_path: Path) -> None:
    fixture = _run_fixture(tmp_path)
    before = _tree_bytes(fixture.root)

    planned = evaluate_archived_manifests(
        source_run_root=fixture.root,
        profile=fixture.profile,
        output_directory=tmp_path / "plan-only",
        candidate_manifest_paths=tuple(
            _manifest_path(manifest) for manifest in fixture.archived
        ),
        devices=("cuda:999",),
        plan_only=True,
    )
    _, _, snapshot = _selection(
        fixture,
        tmp_path / "selection",
        lower_elos=(-10.0, -5.0),
    )
    with pytest.raises(WarmStartError, match="only be applied to a fork"):
        prepare_champion_warm_start(
            fixture.root,
            fixture.profile,
            apply=True,
            selection_snapshot=snapshot,
        )

    assert planned["status"] == "planned"
    assert _tree_bytes(fixture.root) == before


def test_snapshot_tampering_is_rejected(tmp_path: Path) -> None:
    fixture = _run_fixture(tmp_path)
    _, _, snapshot = _selection(
        fixture,
        tmp_path / "selection",
        lower_elos=(5.0, 10.0),
    )
    os.chmod(snapshot, 0o644)
    payload = json.loads(snapshot.read_text(encoding="utf-8"))
    payload["selected_identity"] = fixture.champion.model_identity
    snapshot.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ManifestSelectionError, match="selected_identity"):
        verify_selection_snapshot(snapshot)


@pytest.mark.parametrize(
    ("field", "message"),
    (
        ("opening_seed", "opening"),
        ("forced_opening", "opening"),
        ("opening_action", "opening"),
        ("candidate_search", "candidate search"),
        ("baseline_search", "baseline search"),
    ),
)
def test_result_opening_and_search_tampering_is_rejected(
    tmp_path: Path,
    field: str,
    message: str,
) -> None:
    fixture = _run_fixture(tmp_path)
    plan, _ = _plan(fixture, tmp_path / "selection")
    candidate = plan.candidates[0]
    result = _result(
        plan,
        0,
        tmp_path / "selection" / "result.json",
        lower_elo=0,
    )
    payload = json.loads(result.read_text(encoding="utf-8"))
    if field == "candidate_search":
        payload["search"]["simulations"] += 1
    elif field == "baseline_search":
        payload["baseline_metadata"]["search_budget"]["max_considered"] += 1
    else:
        pair = payload["pairs"][0]
        if field == "forced_opening":
            pair[field] = not pair[field]
        elif pair[field] is None:
            pair[field] = 0
        else:
            pair[field] += 1
    atomic_json(result, payload)

    with pytest.raises(ManifestSelectionError, match=message):
        load_persisted_selection_result(plan, candidate, result)


def test_plan_only_reuses_plan_and_terminal_results_without_cuda(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _run_fixture(tmp_path)
    output = tmp_path / "restart"
    candidates = tuple(_manifest_path(item) for item in fixture.archived)
    first = evaluate_archived_manifests(
        source_run_root=fixture.root,
        profile=fixture.profile,
        output_directory=output,
        candidate_manifest_paths=candidates,
        plan_only=True,
    )
    second = evaluate_archived_manifests(
        source_run_root=fixture.root,
        profile=fixture.profile,
        output_directory=output,
        candidate_manifest_paths=candidates,
        plan_only=True,
    )
    assert first["plan_digest"] == second["plan_digest"]
    assert second["plan_reused"] is True
    with pytest.raises(
        ManifestSelectionError,
        match="frozen selection plan differs",
    ):
        evaluate_archived_manifests(
            source_run_root=fixture.root,
            profile=fixture.profile,
            output_directory=output,
            candidate_manifest_paths=candidates,
            shortlist_size=1,
            plan_only=True,
        )

    plan = plan_archived_manifest_evaluation(
        source_run_root=fixture.root,
        profile=fixture.profile,
        candidate_manifest_paths=candidates,
    )
    results = output / "results"
    results.mkdir()
    for index, candidate in enumerate(plan.candidates):
        _result(
            plan,
            index,
            results / f"{candidate.model_identity}.json",
            lower_elo=0,
        )
    monkeypatch.setattr(
        archived_selection_script,
        "load_star_native",
        lambda **_options: pytest.fail("terminal result reuse loaded native code"),
    )
    completed = evaluate_archived_manifests(
        source_run_root=fixture.root,
        profile=fixture.profile,
        output_directory=output,
        candidate_manifest_paths=candidates,
        devices=("cuda:999",),
    )
    repeated = evaluate_archived_manifests(
        source_run_root=fixture.root,
        profile=fixture.profile,
        output_directory=output,
        candidate_manifest_paths=candidates,
        devices=("cuda:999",),
    )
    assert completed["status"] == "verified"
    assert completed["plan_reused"] is True
    assert repeated["snapshot_reused"] is True


def test_partial_wave_is_validated_and_resumed_from_next_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _run_fixture(tmp_path)
    output = tmp_path / "resume"
    candidates = tuple(_manifest_path(item) for item in fixture.archived)
    evaluate_archived_manifests(
        source_run_root=fixture.root,
        profile=fixture.profile,
        output_directory=output,
        candidate_manifest_paths=candidates,
        plan_only=True,
    )
    plan = plan_archived_manifest_evaluation(
        source_run_root=fixture.root,
        profile=fixture.profile,
        candidate_manifest_paths=candidates,
    )
    results = output / "results"
    results.mkdir()
    partial_candidate = plan.candidates[0]
    _result(
        plan,
        0,
        results / f"{partial_candidate.model_identity}.json",
        lower_elo=0,
        completed_pairs=plan.contract.initial_pairs,
    )
    _result(
        plan,
        1,
        results / f"{plan.candidates[1].model_identity}.json",
        lower_elo=0,
    )
    resumed_starts: list[int] = []

    def fake_lane(**options: object) -> dict[str, Path]:
        lane_candidates = options["candidates"]
        assert lane_candidates == (partial_candidate,)
        persisted = load_persisted_selection_result(
            plan,
            partial_candidate,
            results / f"{partial_candidate.model_identity}.json",
        )
        resumed_starts.append(len(persisted.pairs))
        path = _result(
            plan,
            0,
            results / f"{partial_candidate.model_identity}.json",
            lower_elo=0,
        )
        return {partial_candidate.model_identity: path}

    monkeypatch.setattr(
        archived_selection_script,
        "load_star_native",
        lambda **_options: object(),
    )
    monkeypatch.setattr(
        archived_selection_script,
        "_evaluate_device_lane",
        fake_lane,
    )
    report = evaluate_archived_manifests(
        source_run_root=fixture.root,
        profile=fixture.profile,
        output_directory=output,
        candidate_manifest_paths=candidates,
        devices=("cpu",),
    )
    assert report["status"] == "verified"
    assert resumed_starts == [plan.contract.initial_pairs]


def test_device_aliases_are_canonicalized_and_duplicates_rejected() -> None:
    aliases = {
        "auto": "cuda",
        "cuda": "cuda",
        "cuda:0": "cuda:0",
        "cuda:1": "cuda:1",
    }

    def resolver(device: str) -> str:
        return aliases[device]

    assert canonicalize_device_lanes(
        ("cuda", "cuda:1"),
        resolver=resolver,
    ) == ("cuda:0", "cuda:1")
    with pytest.raises(ManifestSelectionError, match="duplicate lane cuda:0"):
        canonicalize_device_lanes(
            ("auto", "cuda", "cuda:0"),
            resolver=resolver,
        )
    with pytest.raises(ManifestSelectionError, match="same physical lane"):
        static_device_assignments((), ("cuda:0", "cuda:0"))


def test_selection_falls_back_to_source_champion_without_proof(
    tmp_path: Path,
) -> None:
    fixture = _run_fixture(tmp_path)
    _, evidence, snapshot = _selection(
        fixture,
        tmp_path / "selection",
        lower_elos=(-4.0, 0.0),
    )

    verified = verify_selection_snapshot(snapshot)
    assert evidence.fallback_used is True
    assert verified.selected_manifest.model_identity == fixture.champion.model_identity
    assert "no_shortlisted_candidate" in str(evidence.fallback_reason)


def test_selection_ranks_and_chooses_highest_proven_lower_bound(
    tmp_path: Path,
) -> None:
    fixture = _run_fixture(tmp_path)
    plan, evidence, snapshot = _selection(
        fixture,
        tmp_path / "selection",
        lower_elos=(4.0, 12.0),
    )

    verified = verify_selection_snapshot(snapshot)
    assert evidence.fallback_used is False
    assert evidence.ranking == (
        plan.candidates[1].model_identity,
        plan.candidates[0].model_identity,
    )
    assert (
        verified.selected_manifest.model_identity == plan.candidates[1].model_identity
    )


def test_fork_anchors_only_child_to_verified_archived_manifest(
    tmp_path: Path,
) -> None:
    fixture = _run_fixture(tmp_path)
    plan, _, snapshot = _selection(
        fixture,
        tmp_path / "selection",
        lower_elos=(3.0, 9.0),
    )
    parent_before = _tree_bytes(fixture.root)

    child, metadata = _fork(fixture, tmp_path, snapshot, prefix="anchored")

    parent_champion = load_model_manifest(fixture.root / "learner" / "champion.json")
    child_champion = load_model_manifest(child / "learner" / "champion.json")
    assert _tree_bytes(fixture.root) == parent_before
    assert parent_champion.model_identity == fixture.champion.model_identity
    assert child_champion.model_identity == plan.candidates[1].model_identity
    anchor = metadata["anchor"]
    assert isinstance(anchor, dict)
    assert anchor["model_identity"] == child_champion.model_identity
    selection = metadata["source_manifest_selection"]
    assert isinstance(selection, dict)
    assert selection["plan_digest"] == plan.plan_digest
    cutover = json.loads(
        (child / "learner" / "selection-cutover.json").read_text(encoding="utf-8")
    )
    assert cutover["status"] == "pending"
    with pytest.raises(RuntimeError, match="pending a verified warm-start"):
        require_active_selection_cutover(child / "learner")


def test_ring10_only_fork_preserves_objective_metadata(tmp_path: Path) -> None:
    fixture = _run_fixture(tmp_path)
    (fixture.root / "arena").mkdir()
    (fixture.root / "arena" / "legacy-generalist.json").write_text(
        "{}\n",
        encoding="utf-8",
    )
    _, _, snapshot = _selection(
        fixture,
        tmp_path / "selection",
        lower_elos=(3.0, 9.0),
    )

    child, metadata = _fork(
        fixture,
        tmp_path,
        snapshot,
        prefix="ring10",
        treatment="ring10-only",
    )

    profile = load_config(child / "profile-elo-ablation.yaml")
    assert profile.orchestration.training_objective == "ring10_only"
    assert profile.arena.rings == (10,)
    assert metadata["training_objective"] == "ring10_only"
    assert metadata["promotion_objective"] == "ring_10_only"
    assert metadata["per_ring_guarantees"] is False
    assert list((child / "arena").iterdir()) == []
    assert (child / "ablation-parent" / "arena" / "legacy-generalist.json").is_file()


def test_warm_start_uses_archived_non_champion_ema_with_fresh_state(
    tmp_path: Path,
) -> None:
    fixture = _run_fixture(tmp_path)
    plan, _, snapshot = _selection(
        fixture,
        tmp_path / "selection",
        lower_elos=(2.0, 8.0),
    )
    child, _ = _fork(fixture, tmp_path, snapshot, prefix="warm")
    selected_path = child / Path(plan.candidates[1].manifest.path).relative_to(
        fixture.root
    )
    active_paths = tuple(
        path
        for path in (
            child / "learner" / "champion.json",
            child / "learner" / "candidate.json",
            child / "learner" / "recovery.json",
            child / "learner" / "resume-cutover.json",
            child / "learner" / "cadence.json",
            child / "learner" / "utd-segment.json",
        )
        if path.is_file()
    )
    active_before = {path: path.read_bytes() for path in active_paths}

    prepared_report = prepare_champion_warm_start(
        child,
        child / "profile-elo-ablation.yaml",
        prepare_only=True,
        selection_snapshot=child / "selection-snapshot.json",
        source_manifest=selected_path,
    )
    prepared_marker = prepared_report["warm_start"]
    assert isinstance(prepared_marker, dict)
    assert prepared_marker["status"] == "prepared"
    assert prepared_marker["source_model_identity"] == plan.candidates[1].model_identity
    assert {path: path.read_bytes() for path in active_paths} == active_before
    assert (
        json.loads(
            (child / "learner" / "selection-cutover.json").read_text(encoding="utf-8")
        )["status"]
        == "pending"
    )
    assert (
        json.loads(
            (child / "learner" / "cutover-staging.json").read_text(encoding="utf-8")
        )["status"]
        == "pending"
    )

    report = prepare_champion_warm_start(
        child,
        child / "profile-elo-ablation.yaml",
        apply=True,
        selection_snapshot=child / "selection-snapshot.json",
        source_manifest=selected_path,
    )
    marker = report["warm_start"]
    assert isinstance(marker, dict)
    assert marker["source_model_identity"] == plan.candidates[1].model_identity
    assert marker["source_model_identity"] != fixture.champion.model_identity
    prepared = torch.load(
        child / "learner" / str(marker["checkpoint"]),
        map_location="cpu",
        weights_only=True,
    )
    source = torch.load(
        load_model_manifest(selected_path).checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    for name, expected in source["ema"]["shadow"].items():
        torch.testing.assert_close(prepared["model"][name], expected)
        torch.testing.assert_close(prepared["ema"]["shadow"][name], expected)
    assert prepared["optimizer"]["state"] == {}
    assert prepared["ema"]["num_updates"] == 0
    child_config = load_config(child / "profile-elo-ablation.yaml")
    assert prepared["ema"]["decay"] == child_config.train.resolved_ema_decay(
        len(child_config.orchestration.learner_gpus)
    )
    assert prepared["step"] == plan.candidates[1].model_step
    cutover = json.loads(
        (child / "learner" / "selection-cutover.json").read_text(encoding="utf-8")
    )
    assert cutover["status"] == "active"
    require_active_selection_cutover(child / "learner")
    require_launch_ready(child / "learner")


def test_warm_start_rejects_incompatible_run_and_generation(tmp_path: Path) -> None:
    fixture = _run_fixture(tmp_path, name="primary")
    other = _run_fixture(tmp_path, name="other")

    with pytest.raises(WarmStartError, match="another run or generation"):
        prepare_champion_warm_start(
            fixture.root,
            fixture.profile,
            source_manifest=_manifest_path(other.archived[0]),
        )
