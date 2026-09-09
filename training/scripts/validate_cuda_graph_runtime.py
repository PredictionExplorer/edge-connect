#!/usr/bin/env python3
"""Bounded production-model graph correctness and memory probe.

Requires --config PROFILE --manifest MODEL_POINTER_OR_MANIFEST --device cuda:N
--output NEW_JSON. It never opens replay or changes production controls. The
selected GPU may be shared with production: results are correctness/memory
evidence, never timing or speed evidence. A fresh child process owns an isolated
compiler cache, a 20% CUDA allocator limit, and a bounded lifetime.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import gc
from importlib.machinery import EXTENSION_SUFFIXES
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Literal, cast

SHAPES = (1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256)
COLD_SHAPES = (1, 2, 4, 8, 16, 32)
VARIANTS = (
    "double",
    "classic",
    "pie-double",
    "pie-classic",
    "handicap-9-double",
    "handicap-9-classic",
)
GRAPH_BYTES = 12 * 1024**3
MEMORY_FRACTION = 0.20
FORMAT = "startrain.cuda-graph-runtime-validation"


def _control():
    from scripts import benchmark_actor_throughput

    return benchmark_actor_throughput


def validate_config(config) -> None:
    from startrain.model import model_parameter_count

    if model_parameter_count(config.model) != 17_402_775:
        raise ValueError("probe requires the production 17,402,775-parameter model")
    refresh = config.orchestration.model_refresh
    if config.train.precision != "bf16" or config.train.compile is not True:
        raise ValueError("probe requires production BF16 and compiled inference")
    if (
        refresh.inference_compile_dynamic is not True
        or refresh.inference_compile_mode != "default"
    ):
        raise ValueError(
            "probe requires production dynamic/default inference compilation"
        )


def probe_config(config):
    inference = replace(
        config.orchestration.model_refresh.inference,
        cache_max_entries=0,
        cache_max_bytes=0,
        deduplicate=False,
        cuda_graphs=True,
        cuda_graph_max_entries=16,
        cuda_graph_max_bytes=GRAPH_BYTES,
    )
    return replace(
        config,
        orchestration=replace(
            config.orchestration,
            model_refresh=replace(
                config.orchestration.model_refresh, inference=inference
            ),
        ),
    )


def validate_output(
    output: Path, config, config_path: Path, manifest_path: Path
) -> None:
    resolved = output.resolve()
    run_root = Path(config.orchestration.directories.root).expanduser().resolve()
    if resolved == run_root or run_root in resolved.parents:
        raise ValueError("probe output must be outside the production run root")
    if resolved in (config_path.resolve(), manifest_path.resolve()) or output.exists():
        raise ValueError("probe output must be a new artifact")
    for component in (output, *output.parents):
        if component.is_symlink():
            raise ValueError("probe output path contains a symlink")


def require_memory_headroom(free_bytes: int, total_bytes: int) -> int:
    limit = int(total_bytes * MEMORY_FRACTION)
    if free_bytes < 2 * limit + 2 * 1024**3:
        raise RuntimeError(
            "insufficient free GPU memory for a bounded shared-GPU probe"
        )
    if limit < GRAPH_BYTES + 2 * 1024**3:
        raise RuntimeError("20% GPU allocation limit cannot fit the production probe")
    return limit


def make_requests(native, config, *, rows: int, version: int, variant_label: str):
    from startrain.selfplay import GameVariant

    variant = GameVariant.parse(variant_label)
    states = native.StateBatch(
        10, rows, mode=variant.mode, handicap=variant.handicap, pie=variant.pie
    )
    nodes = int(states.node_count)
    depth = 1 if version % 2 == 0 else 3
    for ply in range(depth):
        states.apply_many(
            list(range(rows)),
            [(row + version * 13 + ply * 31) % nodes for row in range(rows)],
        )
    pda = (
        config.selfplay.variants.pda_for_handicap(variant.handicap)
        if variant.handicap > 1
        else 0
    )
    search = native.SearchBatch(
        states,
        simulations=1,
        max_considered=2,
        deterministic_seed=17 + version,
        pda_by_seat=[(-pda, pda)] * rows,
    )
    return search.root_requests()


def regular_adapter(graph_adapter):
    """Use the graph row planner, but execute regular production inference."""
    from startrain.inference import GraphInferenceAdapter

    adapter = GraphInferenceAdapter(
        graph_adapter.model,
        device=graph_adapter.device,
        config=graph_adapter.config,
        homogeneous_relational_bias=graph_adapter.homogeneous_relational_bias,
        model_identity=graph_adapter.model_identity,
        model_version=graph_adapter.model_version,
        model_step=graph_adapter.model_step,
    )
    assert adapter._graphs is not None
    # The empty helper owns no streams or CUDA graphs. Keeping config.cuda_graphs
    # true retains the SAME 96/192 padding policy for the ordinary reference.
    adapter._graphs.clear()
    adapter._graphs = None
    return adapter


def check_health(adapter) -> dict[str, int | float]:
    snapshot = adapter.efficiency_snapshot()
    for name in (
        "graph_fallbacks",
        "graph_validation_failures",
        "graph_negative_entries",
    ):
        if snapshot.get(name, 0):
            raise RuntimeError(
                f"graph correctness probe observed {name}: {snapshot[name]}"
            )
    return snapshot


def compare_predictions(expected, actual) -> dict[str, float]:
    import torch

    if (
        expected.tokens != actual.tokens
        or expected.policy_offsets != actual.policy_offsets
    ):
        raise ValueError("graph changed request-token or policy routing")
    differences = {}
    for name in ("policy_logits", "values"):
        reference = torch.tensor(getattr(expected, name), dtype=torch.float32)
        graph = torch.tensor(getattr(actual, name), dtype=torch.float32)
        if reference.shape != graph.shape or not bool(
            torch.isfinite(reference).all() and torch.isfinite(graph).all()
        ):
            raise FloatingPointError(f"invalid or nonfinite {name} output")
        differences[name] = (
            float((reference - graph).abs().max()) if graph.numel() else 0.0
        )
        if not torch.equal(reference.view(torch.uint8), graph.view(torch.uint8)):
            raise ValueError(
                f"graph {name} differs from same-shape regular inference; max_abs={differences[name]}"
            )
    return differences


def entry_records(adapter) -> list[dict[str, int]]:
    graphs = adapter._graphs
    if graphs is None:
        raise RuntimeError("probe graph backend is absent")
    records = []
    for entry in graphs._entries.values():
        lease = entry.stream_lease
        if lease is None:
            raise RuntimeError("live graph does not own an exclusive capture stream")
        records.append(
            {
                "physical_rows": int(entry.args[0].shape[0]),
                "retained_bytes": entry.retained_bytes,
                "capture_stream": int(lease.stream.cuda_stream),
                "entry_identity": id(entry),
            }
        )
    if len({record["capture_stream"] for record in records}) != len(records):
        raise RuntimeError("simultaneously live graphs share a capture stream")
    return records


def checked_request(
    native,
    config,
    graph,
    regular,
    *,
    rows: int,
    version: int,
    variant: str,
    expected_captures: int | None = None,
    capture_initialization_order: Literal[
        "graph-first", "reference-first"
    ] = "graph-first",
):
    if capture_initialization_order not in ("graph-first", "reference-first"):
        raise ValueError("invalid capture initialization order")
    expected_rows = graph._inference_batch_rows(rows)
    if regular._inference_batch_rows(rows) != expected_rows:
        raise RuntimeError("reference and graph use different physical batch shapes")
    request = make_requests(
        native, config, rows=rows, version=version, variant_label=variant
    )
    before_graph = graph.metrics_snapshot()
    before_reference = regular.metrics_snapshot()
    if capture_initialization_order == "graph-first":
        actual = graph.evaluate(request)
        expected = regular.evaluate(request)
    else:
        expected = regular.evaluate(request)
        actual = graph.evaluate(request)
    differences = compare_predictions(expected, actual)
    if (
        graph.metrics_snapshot().neural_rows - before_graph.neural_rows != expected_rows
        or regular.metrics_snapshot().neural_rows - before_reference.neural_rows
        != expected_rows
    ):
        raise RuntimeError("probe inference rows differ from the physical batch plan")
    if (
        expected_captures is not None
        and graph.metrics_snapshot().graph_captures - before_graph.graph_captures
        != expected_captures
    ):
        raise RuntimeError("unexpected graph capture count")
    check_health(graph)
    return differences


def native_artifacts(native, modules=None) -> dict[str, str]:
    """Identify the actual loaded extension even when the import is a package."""
    from startrain.checkpoint import sha256_file

    modules = sys.modules if modules is None else modules
    prefix = native.__name__.split(".")[0]
    wrapper = Path(native.__file__).resolve()
    binaries = set()
    for name, module in tuple(modules.items()):
        source = getattr(module, "__file__", None)
        if name != prefix and not name.startswith(prefix + "."):
            continue
        if source and any(
            str(source).endswith(suffix) for suffix in EXTENSION_SUFFIXES
        ):
            binaries.add(Path(source).resolve())
    if any(str(wrapper).endswith(suffix) for suffix in EXTENSION_SUFFIXES):
        binaries.add(wrapper)
    if len(binaries) != 1:
        raise RuntimeError("cannot identify exactly one loaded native extension")
    binary = next(iter(binaries))
    return {
        "native_path": str(binary),
        "native_sha256": sha256_file(binary),
        "native_wrapper_path": str(wrapper),
        "native_wrapper_sha256": sha256_file(wrapper),
    }


def _source_identity(native) -> dict[str, object]:
    import torch
    import startrain
    from startrain.checkpoint import sha256_file

    root = Path(startrain.__file__).resolve().parent
    names = (
        "actor.py",
        "model.py",
        "inference.py",
        "inference_graphs.py",
        "inference_cache.py",
        "native.py",
        "checkpoint.py",
    )
    paths = [Path(__file__).resolve(), *(root / name for name in names)]
    return {
        "startrain_source_root": str(root),
        "source_files": {str(path): sha256_file(path) for path in paths},
        "python": sys.version,
        "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        **native_artifacts(native),
        "native_rules_hash": int(native.native_rules_hash()),
        "native_feature_schema_version": int(native.native_feature_schema_version()),
    }


def run_probe(args) -> dict[str, object]:
    import torch
    from startrain.actor import ManifestModelProvider
    from startrain.checkpoint import load_model_manifest, sha256_file
    from startrain.config import load_config
    from startrain.inference import GraphInferenceAdapter
    from startrain.native import load_star_native
    from startrain.runtime import RunIdentity, atomic_json

    source_config = load_config(args.config)
    validate_config(source_config)
    validate_output(args.output, source_config, args.config, args.manifest)
    if sha256_file(args.config) != args.config_sha256:
        raise ValueError("profile changed after the parent pinned it")
    manifest = load_model_manifest(args.manifest)
    if manifest.manifest_sha256 != args.manifest_sha256:
        raise ValueError("immutable model manifest changed after parent pinning")
    device = torch.device(args.device)
    if device.type != "cuda" or device.index is None or not torch.cuda.is_available():
        raise ValueError("probe requires an explicit CUDA device")
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(MEMORY_FRACTION, device)
    free, total = torch.cuda.mem_get_info(device)
    limit = require_memory_headroom(free, total)
    native = load_star_native(required=True)
    assert native is not None
    config = probe_config(source_config)
    # CPU budget comes from the matching production actor, without changing its
    # model/precision/compiler/transfer settings or touching production affinity.
    assignment = next(
        (
            g
            for g in config.orchestration.gpus
            if g.gpu_id == device.index and g.role == "actor"
        ),
        None,
    )
    if assignment is not None:
        torch.set_num_threads(assignment.blas_threads or assignment.cpu_threads)
    identity = RunIdentity(
        args.output.parent / "unused-probe-run.json",
        manifest.run_id,
        manifest.generation_family,
        0,
    )
    provider = ManifestModelProvider(
        config,
        args.manifest,
        device=str(device),
        run_identity=identity,
        expected_role=cast(Literal["champion", "candidate", "direct"], manifest.role),
    )
    graph = provider.refresh()
    assert isinstance(graph, GraphInferenceAdapter) and graph._graphs is not None
    regular = regular_adapter(graph)
    properties = torch.cuda.get_device_properties(device)
    report: dict[str, Any] = {
        "format": FORMAT,
        "schema_version": 1,
        "status": "running",
        "scope": "shared-GPU correctness and retained memory only; no timing or speed claim",
        "source": _source_identity(native),
        "config_sha256": args.config_sha256,
        "production_config": source_config.as_dict(),
        "probe_inference": asdict(config.orchestration.model_refresh.inference),
        "capture_initialization_order": args.capture_initialization_order,
        "model": {
            "identity": manifest.model_identity,
            "step": manifest.model_step,
            "manifest_sha256": manifest.manifest_sha256,
            "checkpoint_sha256": manifest.checkpoint_sha256,
            "checkpoint_bytes": manifest.checkpoint_bytes,
            "parameters": 17402775,
        },
        "device": str(device),
        "gpu_name": properties.name,
        "gpu_uuid": str(properties.uuid),
        "allocator_fraction": MEMORY_FRACTION,
        "allocator_limit_bytes": limit,
        "free_bytes_before": free,
        "total_bytes": total,
        "blas_threads": torch.get_num_threads(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "shapes": [],
        "stress": {},
    }
    atomic_json(args.output, report)
    try:
        for rows in SHAPES:
            for version, variant in enumerate(VARIANTS):
                checked_request(
                    native,
                    config,
                    graph,
                    regular,
                    rows=rows,
                    version=version,
                    variant=variant,
                    expected_captures=1 if version == 0 else 0,
                    capture_initialization_order=args.capture_initialization_order,
                )
            entries = entry_records(graph)
            entry = next(
                record for record in entries if record["physical_rows"] == rows
            )
            report["shapes"].append(
                entry
                | {
                    "changed_input_cases": len(VARIANTS),
                    "allocated_bytes": torch.cuda.memory_allocated(device),
                    "reserved_bytes": torch.cuda.memory_reserved(device),
                }
            )
            report["inventory_counters"] = check_health(graph)
            atomic_json(args.output, report)
        if graph.metrics_snapshot().graph_captures != len(SHAPES):
            raise RuntimeError(
                "inventory did not retain exactly one graph per physical shape"
            )
        graph._graphs.clear()
        graph._graphs.max_entries = 2
        checked_request(
            native,
            config,
            graph,
            regular,
            rows=64,
            version=50,
            variant="double",
            expected_captures=1,
            capture_initialization_order=args.capture_initialization_order,
        )
        hot = entry_records(graph)[0]
        before = graph.metrics_snapshot()
        for index in range(args.stress_captures):
            variant = VARIANTS[index % len(VARIANTS)]
            checked_request(
                native,
                config,
                graph,
                regular,
                rows=64,
                version=100 + index,
                variant=variant,
                expected_captures=0,
                capture_initialization_order=args.capture_initialization_order,
            )
            cold_rows = COLD_SHAPES[index % len(COLD_SHAPES)]
            checked_request(
                native,
                config,
                graph,
                regular,
                rows=cold_rows,
                version=200 + index,
                variant=variant,
                expected_captures=1,
                capture_initialization_order=args.capture_initialization_order,
            )
            entries = entry_records(graph)
            current_hot = next(
                record for record in entries if record["physical_rows"] == 64
            )
            if (
                current_hot["entry_identity"] != hot["entry_identity"]
                or current_hot["capture_stream"] != hot["capture_stream"]
            ):
                raise RuntimeError("hot graph was replaced during cold-stream churn")
            if (index + 1) % 8 == 0:
                scratch = [
                    torch.empty(size * 1024**2, dtype=torch.uint8, device=device).fill_(
                        index % 251
                    )
                    for size in (64, 128)
                ]
                torch.cuda.synchronize(device)
                del scratch
                gc.collect()
                torch.cuda.empty_cache()
                checked_request(
                    native,
                    config,
                    graph,
                    regular,
                    rows=64,
                    version=500 + index,
                    variant=variant,
                    expected_captures=0,
                    capture_initialization_order=args.capture_initialization_order,
                )
                report["stress"] = {
                    "completed_cold_captures": index + 1,
                    "hot_graph": hot,
                    "live_entries": entry_records(graph),
                    "counters": check_health(graph),
                }
                atomic_json(args.output, report)
        delta = graph.metrics_snapshot().delta(before)
        if (
            delta.graph_captures != args.stress_captures
            or delta.graph_evictions < args.stress_captures - 1
        ):
            raise RuntimeError(
                "stress did not exercise the required capture/eviction count"
            )
        report["stress"]["delta"] = asdict(delta)
        report["final_counters"] = check_health(graph)
        report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        report["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        if (
            sha256_file(args.config) != args.config_sha256
            or _source_identity(native) != report["source"]
        ):
            raise RuntimeError("probe source/config changed during validation")
        report["status"] = "passed"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        atomic_json(args.output, report)
        raise
    finally:
        cleanup_errors = []
        for adapter in (graph, regular):
            try:
                adapter.close()
            except Exception as error:
                cleanup_errors.append(f"{type(error).__name__}: {error}")
        if cleanup_errors:
            report["status"] = "failed"
            report["cleanup_errors"] = cleanup_errors
        atomic_json(args.output, report)
    if report["status"] != "passed":
        raise RuntimeError("probe cleanup failed")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stress-captures", type=int, default=96)
    parser.add_argument("--timeout-seconds", type=float, default=1200)
    parser.add_argument(
        "--capture-initialization-order",
        choices=("graph-first", "reference-first"),
        default="graph-first",
        help="graph-first matches actor startup; reference-first reproduces the original probe",
    )
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--manifest-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--config-sha256", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not 96 <= args.stress_captures <= 256 or args.stress_captures % 8:
        parser.error("stress captures must be96..256 and a multiple of8")
    if not 1 <= args.timeout_seconds <= 1800:
        parser.error("timeout must be1..1800 seconds")
    if args.child:
        run_probe(args)
        return 0
    from startrain.checkpoint import load_model_manifest, sha256_file
    from startrain.config import load_config
    from startrain.training import isolated_compile_cache

    config = load_config(args.config)
    validate_config(config)
    validate_output(args.output, config, args.config, args.manifest)
    manifest = load_model_manifest(args.manifest)
    immutable = manifest.artifact_manifest or manifest.path
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cache = args.output.parent / (args.output.stem + "-compiler-cache")
    if cache.exists():
        raise ValueError("probe compiler cache must be new")
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--config",
        str(args.config.resolve()),
        "--manifest",
        str(immutable.resolve()),
        "--manifest-sha256",
        manifest.manifest_sha256,
        "--config-sha256",
        sha256_file(args.config),
        "--device",
        args.device,
        "--output",
        str(args.output.resolve()),
        "--stress-captures",
        str(args.stress_captures),
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--capture-initialization-order",
        args.capture_initialization_order,
    ]
    control = _control()
    with control._controller_signals(), isolated_compile_cache(cache) as provenance:
        try:
            child = control._run_owned(
                command,
                env=dict(os.environ, **provenance.environment),
                timeout=args.timeout_seconds,
            )
        except (subprocess.TimeoutExpired, control.BenchmarkInterrupted) as error:
            for stream in ("stdout", "stderr"):
                value = getattr(error, stream, "") or ""
                if isinstance(value, bytes):
                    value = value.decode(errors="replace")
                args.output.with_suffix(f".{stream}.log").write_text(value)
            raise
    args.output.with_suffix(".stdout.log").write_text(child.stdout)
    args.output.with_suffix(".stderr.log").write_text(child.stderr)
    if child.returncode:
        return child.returncode
    result = json.loads(args.output.read_text())
    if (
        result.get("status") != "passed"
        or len(result.get("shapes", [])) != len(SHAPES)
        or result.get("capture_initialization_order")
        != args.capture_initialization_order
    ):
        raise RuntimeError("child did not complete graph validation")
    print(
        json.dumps(
            {
                "status": "passed",
                "output": str(args.output.resolve()),
                "stress_captures": args.stress_captures,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
