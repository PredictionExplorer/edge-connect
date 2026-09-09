from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts import validate_cuda_graph_runtime as probe
from startrain.config import load_config
from startrain.inference import GraphInferenceAdapter, InferenceConfig


def production_config():
    return load_config(
        Path(__file__).parents[1] / "configs/h100-8gpu-largest-board-priority.yaml"
    )


def test_probe_changes_only_explicit_inference_controls():
    source = production_config()
    changed = probe.probe_config(source)
    before = asdict(source.orchestration.model_refresh.inference)
    after = asdict(changed.orchestration.model_refresh.inference)
    allowed = {
        "cache_max_entries",
        "cache_max_bytes",
        "deduplicate",
        "cuda_graphs",
        "cuda_graph_max_entries",
        "cuda_graph_max_bytes",
    }
    assert {k for k in before if before[k] != after[k]} <= allowed
    assert source.model == changed.model and source.train == changed.train
    assert source.game == changed.game and source.selfplay == changed.selfplay
    assert changed.orchestration.model_refresh.inference.cuda_graph_max_entries == 16
    assert (
        changed.orchestration.model_refresh.inference.cuda_graph_max_bytes
        == 12 * 1024**3
    )
    assert not changed.orchestration.model_refresh.inference.deduplicate


def test_production_contract_rejects_wrong_size_precision_and_compiler(monkeypatch):
    from startrain import model

    config = production_config()
    monkeypatch.setattr(model, "model_parameter_count", lambda _: 17402775)
    probe.validate_config(config)
    for altered in (
        replace(config, train=replace(config.train, precision="fp32")),
        replace(config, train=replace(config.train, compile=False)),
        replace(
            config,
            orchestration=replace(
                config.orchestration,
                model_refresh=replace(
                    config.orchestration.model_refresh, inference_compile_dynamic=False
                ),
            ),
        ),
    ):
        with pytest.raises(ValueError):
            probe.validate_config(altered)
    monkeypatch.setattr(model, "model_parameter_count", lambda _: 100)
    with pytest.raises(ValueError, match="17,402,775"):
        probe.validate_config(config)


def test_probe_requires_memory_headroom_without_changing_fraction():
    gib = 1024**3
    assert probe.require_memory_headroom(70 * gib, 80 * gib) == 16 * gib
    with pytest.raises(RuntimeError, match="free GPU memory"):
        probe.require_memory_headroom(20 * gib, 80 * gib)
    with pytest.raises(RuntimeError, match="20%"):
        probe.require_memory_headroom(32 * gib, 32 * gib)


def test_output_cannot_touch_production_or_existing_artifacts(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    config = SimpleNamespace(
        orchestration=SimpleNamespace(directories=SimpleNamespace(root=str(run)))
    )
    with pytest.raises(ValueError, match="outside"):
        probe.validate_output(
            run / "report.json", config, run / "profile", run / "manifest"
        )
    output = tmp_path / "report.json"
    output.write_text("existing")
    with pytest.raises(ValueError, match="new artifact"):
        probe.validate_output(output, config, run / "profile", run / "manifest")


def test_regular_reference_keeps_exact_graph_padding_policy(monkeypatch):
    from startrain import inference

    cleared = []

    class Adapter:
        def __init__(self, model, **kwargs):
            self.model = model
            self.__dict__.update(kwargs)
            self._graphs = SimpleNamespace(clear=lambda: cleared.append(True))

    source = SimpleNamespace(
        model=object(),
        device=torch.device("cuda:7"),
        config=InferenceConfig(cuda_graphs=True),
        homogeneous_relational_bias=True,
        model_identity="pinned",
        model_version="pinned",
        model_step=123,
    )
    monkeypatch.setattr(inference, "GraphInferenceAdapter", Adapter)
    regular = probe.regular_adapter(source)
    assert (
        regular._graphs is None
        and regular.config is source.config
        and cleared == [True]
    )
    assert [
        GraphInferenceAdapter._inference_batch_rows(regular, n) for n in probe.SHAPES
    ] == list(probe.SHAPES)
    assert GraphInferenceAdapter._inference_batch_rows(regular, 80) == 96
    assert GraphInferenceAdapter._inference_batch_rows(regular, 160) == 192


def response():
    return SimpleNamespace(
        tokens=[1], policy_offsets=[0, 2], policy_logits=[0.5, -0.125], values=[0.25]
    )


@pytest.mark.parametrize(
    "field", ["tokens", "policy_offsets", "policy_logits", "values"]
)
def test_prediction_comparison_is_exact_and_rejects_route_or_value_changes(field):
    expected = response()
    actual = deepcopy(expected)
    assert probe.compare_predictions(expected, actual) == {
        "policy_logits": 0.0,
        "values": 0.0,
    }
    getattr(actual, field)[0] += 1
    with pytest.raises(ValueError):
        probe.compare_predictions(expected, actual)


def test_nonfinite_predictions_and_graph_fallbacks_fail_closed():
    actual = response()
    actual.values = [float("nan")]
    with pytest.raises(FloatingPointError):
        probe.compare_predictions(response(), actual)
    for field in (
        "graph_fallbacks",
        "graph_validation_failures",
        "graph_negative_entries",
    ):
        with pytest.raises(RuntimeError, match=field):
            probe.check_health(SimpleNamespace(efficiency_snapshot=lambda: {field: 1}))


def test_live_entries_must_have_distinct_exclusive_streams():
    def entry(rows, stream):
        return SimpleNamespace(
            args=(torch.zeros(rows, 1),),
            retained_bytes=123,
            stream_lease=SimpleNamespace(stream=SimpleNamespace(cuda_stream=stream)),
        )

    adapter = SimpleNamespace(
        _graphs=SimpleNamespace(
            _entries={
                ("first", (), ()): entry(64, 10),
                ("second", (), ()): entry(32, 11),
            }
        )
    )
    assert [record["physical_rows"] for record in probe.entry_records(adapter)] == [
        64,
        32,
    ]
    adapter._graphs._entries[("second", (), ())] = entry(32, 10)
    with pytest.raises(RuntimeError, match="share a capture stream"):
        probe.entry_records(adapter)


def test_entry_inventory_preserves_original_signatures_and_runtime_memory_accounting(
    monkeypatch,
):
    input_tensor, output = torch.zeros(64, 1), torch.ones(64, 1)
    original_args = (("tensor", (64, 1), (2, 1), "torch.float32", "cuda:7"),)
    original_kwargs = (
        ("mask", ("tensor", (64, 1), (1, 1), "torch.float32", "cuda:7")),
    )
    entry = SimpleNamespace(
        args=(input_tensor,),
        kwargs={"mask": input_tensor},
        output=output,
        graph=SimpleNamespace(pool=lambda: (9, 2)),
        device=torch.device("cuda:7"),
        retained_bytes=5000,
        stream_lease=SimpleNamespace(stream=SimpleNamespace(cuda_stream=10)),
    )
    snapshot = [
        {"device": 7, "segment_pool_id": (9, 2), "total_size": 4096, "address": 0}
    ]
    for tensor in (input_tensor, output):
        storage = tensor.untyped_storage()
        snapshot.append(
            {
                "device": 7,
                "segment_pool_id": (0, 0),
                "address": storage.data_ptr(),
                "total_size": storage.nbytes(),
                "blocks": [{"address": storage.data_ptr(), "size": storage.nbytes()}],
            }
        )
    monkeypatch.setattr(torch.cuda, "memory_snapshot", lambda **kwargs: snapshot)
    adapter = SimpleNamespace(
        _graphs=SimpleNamespace(
            _entries={("model", original_args, original_kwargs): entry}
        )
    )
    (record,) = probe.entry_records(adapter, include_memory=True)
    assert record["original_argument_signatures"] == original_args
    assert record["original_keyword_signatures"] == original_kwargs
    assert record["graph_pool_id"] == (9, 2)
    assert record["private_pool_bytes"] == 4096
    assert (
        record["external_static_block_bytes"] == 512
    )  # The aliased mask is charged once.
    assert record["current_accounted_bytes"] == 4608
    assert record["retained_bytes"] == 5000 and record["retained_bytes_delta"] == -392


def test_shape_order_accepts_only_a_permutation_of_existing_buckets():
    order = (1, 128, 96, 192, 256, 64, 32, 16, 8, 4, 2)
    assert probe.parse_shape_order(",".join(map(str, order))) == order
    for value in ("1,2", ",".join(map(str, (*probe.SHAPES[:-1], 1))), "not-an-integer"):
        with pytest.raises(probe.argparse.ArgumentTypeError):
            probe.parse_shape_order(value)


def test_native_identity_hashes_loaded_binary_not_just_python_wrapper(tmp_path):
    from startrain.checkpoint import sha256_file

    wrapper = tmp_path / "__init__.py"
    wrapper.write_bytes(b"from .star_native import *")
    binary = tmp_path / "star_native.abi3.so"
    binary.write_bytes(b"compiled implementation")
    module = SimpleNamespace(__name__="star_native", __file__=str(wrapper))
    modules = {
        "star_native": module,
        "star_native.star_native": SimpleNamespace(__file__=str(binary)),
        "unrelated": SimpleNamespace(__file__=str(tmp_path / "unrelated.so")),
    }
    identity = probe.native_artifacts(module, modules)
    assert identity["native_path"] == str(binary) and identity[
        "native_sha256"
    ] == sha256_file(binary)
    assert identity["native_wrapper_path"] == str(wrapper)
    assert identity["native_wrapper_sha256"] != identity["native_sha256"]


@pytest.mark.parametrize(
    "order,expected",
    [
        (None, ["graph", "regular"]),
        ("graph-first", ["graph", "regular"]),
        ("reference-first", ["regular", "graph"]),
    ],
)
def test_checked_request_controls_actual_initialization_order(
    monkeypatch, order, expected
):
    calls = []

    class Adapter:
        def __init__(self, name):
            self.name = name
            self.rows = 0
            self.captures = 0

        def _inference_batch_rows(self, rows):
            return rows

        def metrics_snapshot(self):
            return SimpleNamespace(neural_rows=self.rows, graph_captures=self.captures)

        def evaluate(self, request):
            calls.append(self.name)
            self.rows += request
            self.captures += self.name == "graph"
            return response()

        def efficiency_snapshot(self):
            return {}

    monkeypatch.setattr(probe, "make_requests", lambda *args, rows, **kwargs: rows)
    options = {} if order is None else {"capture_initialization_order": order}
    assert probe.checked_request(
        None,
        None,
        Adapter("graph"),
        Adapter("regular"),
        rows=64,
        version=0,
        variant="double",
        expected_captures=1,
        **options,
    ) == {"policy_logits": 0.0, "values": 0.0}
    assert calls == expected


@pytest.mark.parametrize("order", [None, "reference-first"])
def test_cli_forwards_initialization_order_to_child_probe(monkeypatch, order):
    observed = []
    monkeypatch.setattr(probe, "run_probe", lambda args: observed.append(args))
    command = [
        "--child",
        "--config",
        "profile.yaml",
        "--manifest",
        "model.json",
        "--device",
        "cuda:7",
        "--output",
        "report.json",
    ]
    if order is not None:
        command += [
            "--capture-initialization-order",
            order,
            "--shape-order",
            ",".join(map(str, reversed(probe.SHAPES))),
        ]
    assert probe.main(command) == 0
    assert observed[0].capture_initialization_order == (order or "graph-first")
    assert observed[0].shape_order == (
        tuple(reversed(probe.SHAPES)) if order else probe.SHAPES
    )


def test_source_identity_uses_imported_runtime_when_probe_is_outside_release(
    monkeypatch, tmp_path
):
    import startrain
    from startrain.checkpoint import sha256_file

    runtime = tmp_path / "release" / "startrain"
    runtime.mkdir(parents=True)
    for name in (
        "__init__.py",
        "actor.py",
        "model.py",
        "inference.py",
        "inference_graphs.py",
        "inference_cache.py",
        "native.py",
        "checkpoint.py",
    ):
        (runtime / name).write_text(name)
    copied_probe = tmp_path / "rollout" / "probe.py"
    copied_probe.parent.mkdir()
    copied_probe.write_text("updated validation tool")
    monkeypatch.setattr(startrain, "__file__", str(runtime / "__init__.py"))
    monkeypatch.setattr(probe, "__file__", str(copied_probe))
    monkeypatch.setattr(probe, "native_artifacts", lambda native: {})
    native = SimpleNamespace(
        native_rules_hash=lambda: 7, native_feature_schema_version=lambda: 4
    )
    identity = probe._source_identity(native)
    assert identity["startrain_source_root"] == str(runtime)
    sources = identity["source_files"]
    assert sources[str(copied_probe)] == sha256_file(copied_probe)
    assert sources[str(runtime / "inference.py")] == sha256_file(
        runtime / "inference.py"
    )
    assert sources[str(runtime / "inference_graphs.py")] == sha256_file(
        runtime / "inference_graphs.py"
    )


@pytest.mark.native
@pytest.mark.parametrize("variant", probe.VARIANTS)
def test_changed_inputs_cover_all_six_native_variants(variant):
    from startrain.native import positions_from_native

    native = pytest.importorskip("star_native")
    config = production_config()
    first = probe.make_requests(
        native, config, rows=3, version=0, variant_label=variant
    )
    second = probe.make_requests(
        native, config, rows=3, version=1, variant_label=variant
    )
    assert len(first) == len(second) == 3
    assert not torch.equal(
        positions_from_native(first.states)[0].stones,
        positions_from_native(second.states)[0].stones,
    )
