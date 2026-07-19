"""Fail-closed NVIDIA GPU health inspection for training hosts.

Checks are fail-closed for hardware that reports a feature in a bad state,
and tolerant of hardware that does not expose a feature at all (consumer
GPUs report ``N/A`` for ECC/MIG/row-remap fields, and some drivers omit the
nodes entirely). The production H100 gate is expressed through
``require_gpu_model`` instead of being hardcoded.
"""

from __future__ import annotations

import math
import re
import subprocess
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from typing import Protocol

GPU_HEALTH_SCHEMA_VERSION = 1
_INTEGER = re.compile(r"^-?\d+")
_ABSENT = ("", "n/a", "none", "not active", "unknown error")


class CompletedProcessProtocol(Protocol):
    returncode: int
    stdout: str
    stderr: str


CommandRunner = Callable[..., CompletedProcessProtocol]


@dataclass(frozen=True, slots=True)
class GPUHealth:
    index: int
    uuid: str
    serial: str
    pci_bus_id: str
    product_name: str
    mig_mode: str
    ecc_mode: str
    recovery_action: str
    volatile_sram_uncorrectable_parity: int
    volatile_sram_uncorrectable_secded: int
    volatile_dram_uncorrectable: int
    aggregate_sram_uncorrectable_parity: int
    aggregate_sram_uncorrectable_secded: int
    aggregate_dram_uncorrectable: int
    sram_threshold_exceeded: bool
    channel_repair_pending: bool
    tpc_repair_pending: bool
    remapped_rows_pending: bool
    remapped_rows_failure: bool
    reasons: tuple[str, ...]

    @property
    def healthy(self) -> bool:
        return not self.reasons

    def as_dict(self) -> dict[str, object]:
        return {**asdict(self), "healthy": self.healthy}


def _text(node: ET.Element, path: str, *, required: bool = True) -> str:
    child = node.find(path)
    value = "" if child is None or child.text is None else child.text.strip()
    if required and not value:
        raise ValueError(f"nvidia-smi XML is missing gpu/{path}")
    return value


def _optional_text(node: ET.Element, path: str, *, default: str) -> str:
    value = _text(node, path, required=False)
    return value if value else default


def _integer(node: ET.Element, path: str) -> int:
    """Parse a counter; missing or N/A nodes mean the feature is absent."""

    value = _text(node, path, required=False)
    if value.casefold() in _ABSENT:
        return 0
    match = _INTEGER.match(value)
    if match is None:
        raise ValueError(f"nvidia-smi XML gpu/{path} is not an integer: {value!r}")
    result = int(match.group())
    if result < 0:
        raise ValueError(f"nvidia-smi XML gpu/{path} is negative")
    return result


def _yes(value: str, *, path: str) -> bool:
    normalized = value.casefold()
    if normalized in ("yes", "pending"):
        return True
    if normalized in ("no", *_ABSENT):
        return False
    raise ValueError(f"nvidia-smi XML gpu/{path} has unknown state {value!r}")


def parse_nvidia_smi_xml(
    payload: str,
    *,
    expected_indices: Iterable[int] | None = None,
    require_gpu_model: str | None = "H100",
    fail_on_aggregate_uncorrectable: bool = True,
) -> dict[str, object]:
    """Parse ``nvidia-smi -q -x`` and return a deterministic health report."""

    try:
        root = ET.fromstring(payload)
    except ET.ParseError as exc:
        raise ValueError(f"cannot parse nvidia-smi XML: {exc}") from exc
    if root.tag != "nvidia_smi_log":
        raise ValueError("nvidia-smi XML root is not nvidia_smi_log")

    driver_version = _text(root, "driver_version")
    records: list[GPUHealth] = []
    for fallback, gpu in enumerate(root.findall("gpu")):
        # XML order is the nvidia-smi logical index. ``minor_number`` is a
        # device-node identifier and is not guaranteed to match that index.
        index = fallback
        product_name = _text(gpu, "product_name")
        mig_mode = _optional_text(gpu, "mig_mode/current_mig", default="N/A")
        ecc_mode = _optional_text(gpu, "ecc_mode/current_ecc", default="N/A")
        recovery_action = _optional_text(gpu, "gpu_recovery_action", default="None")
        volatile_parity = _integer(gpu, "ecc_errors/volatile/sram_uncorrectable_parity")
        volatile_secded = _integer(gpu, "ecc_errors/volatile/sram_uncorrectable_secded")
        volatile_dram = _integer(gpu, "ecc_errors/volatile/dram_uncorrectable")
        aggregate_parity = _integer(
            gpu, "ecc_errors/aggregate/sram_uncorrectable_parity"
        )
        aggregate_secded = _integer(
            gpu, "ecc_errors/aggregate/sram_uncorrectable_secded"
        )
        aggregate_dram = _integer(gpu, "ecc_errors/aggregate/dram_uncorrectable")
        threshold = _yes(
            _optional_text(
                gpu,
                "ecc_errors/aggregate/sram_threshold_exceeded",
                default="No",
            ),
            path="ecc_errors/aggregate/sram_threshold_exceeded",
        )
        channel_repair = _yes(
            _optional_text(gpu, "ecc_errors/channel_repair_pending", default="No"),
            path="ecc_errors/channel_repair_pending",
        )
        tpc_repair = _yes(
            _optional_text(gpu, "ecc_errors/tpc_repair_pending", default="No"),
            path="ecc_errors/tpc_repair_pending",
        )
        remap_pending = _yes(
            _optional_text(gpu, "remapped_rows/remapped_row_pending", default="No"),
            path="remapped_rows/remapped_row_pending",
        )
        remap_failure = _yes(
            _optional_text(gpu, "remapped_rows/remapped_row_failure", default="No"),
            path="remapped_rows/remapped_row_failure",
        )

        reasons = []
        if require_gpu_model and require_gpu_model not in product_name:
            reasons.append("unexpected_gpu_model")
        # "N/A" means the GPU does not support the feature; only an actively
        # bad state (MIG on, ECC deliberately disabled) fails the gate.
        if mig_mode.casefold() not in ("disabled", "n/a"):
            reasons.append("mig_enabled")
        if ecc_mode.casefold() not in ("enabled", "n/a"):
            reasons.append("ecc_disabled")
        if recovery_action.casefold() not in ("none", "n/a"):
            reasons.append("gpu_recovery_action")
        if volatile_parity or volatile_secded or volatile_dram:
            reasons.append("volatile_uncorrectable_ecc")
        if fail_on_aggregate_uncorrectable and (
            aggregate_parity or aggregate_secded or aggregate_dram
        ):
            reasons.append("aggregate_uncorrectable_ecc")
        if threshold:
            reasons.append("sram_threshold_exceeded")
        if channel_repair:
            reasons.append("channel_repair_pending")
        if tpc_repair:
            reasons.append("tpc_repair_pending")
        if remap_pending:
            reasons.append("row_remap_pending")
        if remap_failure:
            reasons.append("row_remap_failure")

        records.append(
            GPUHealth(
                index=index,
                uuid=_text(gpu, "uuid"),
                serial=_optional_text(gpu, "serial", default="N/A"),
                pci_bus_id=_text(gpu, "pci/pci_bus_id"),
                product_name=product_name,
                mig_mode=mig_mode,
                ecc_mode=ecc_mode,
                recovery_action=recovery_action,
                volatile_sram_uncorrectable_parity=volatile_parity,
                volatile_sram_uncorrectable_secded=volatile_secded,
                volatile_dram_uncorrectable=volatile_dram,
                aggregate_sram_uncorrectable_parity=aggregate_parity,
                aggregate_sram_uncorrectable_secded=aggregate_secded,
                aggregate_dram_uncorrectable=aggregate_dram,
                sram_threshold_exceeded=threshold,
                channel_repair_pending=channel_repair,
                tpc_repair_pending=tpc_repair,
                remapped_rows_pending=remap_pending,
                remapped_rows_failure=remap_failure,
                reasons=tuple(dict.fromkeys(reasons)),
            )
        )

    if not records:
        raise ValueError("nvidia-smi XML contains no GPUs")
    indices = [record.index for record in records]
    if len(indices) != len(set(indices)):
        raise ValueError("nvidia-smi XML contains duplicate GPU indices")
    expected = None if expected_indices is None else sorted(set(expected_indices))
    missing = [] if expected is None else sorted(set(expected) - set(indices))
    unexpected = [] if expected is None else sorted(set(indices) - set(expected))
    selected = (
        records
        if expected is None
        else [item for item in records if item.index in expected]
    )
    # Extra unassigned GPUs are permitted (for example, a four-GPU profile on
    # an eight-GPU host). Only configured indices participate in this gate.
    healthy = not missing and all(item.healthy for item in selected)
    return {
        "schema_version": GPU_HEALTH_SCHEMA_VERSION,
        "driver_version": driver_version,
        "healthy": healthy,
        "expected_indices": expected,
        "missing_indices": missing,
        "unexpected_indices": unexpected,
        "gpus": [item.as_dict() for item in records],
    }


def query_gpu_health(
    *,
    expected_indices: Iterable[int] | None = None,
    require_gpu_model: str | None = "H100",
    fail_on_aggregate_uncorrectable: bool = True,
    runner: CommandRunner = subprocess.run,
    timeout: float = 30.0,
) -> dict[str, object]:
    """Run ``nvidia-smi`` and parse its complete XML health payload."""

    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("GPU health timeout must be finite and positive")
    try:
        completed = runner(
            ["nvidia-smi", "-q", "-x"],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"nvidia-smi health query failed: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or f"exit status {completed.returncode}"
        raise RuntimeError(f"nvidia-smi health query failed: {detail}")
    return parse_nvidia_smi_xml(
        completed.stdout,
        expected_indices=expected_indices,
        require_gpu_model=require_gpu_model,
        fail_on_aggregate_uncorrectable=fail_on_aggregate_uncorrectable,
    )


def unhealthy_reasons(report: dict[str, object]) -> tuple[str, ...]:
    reasons = []
    missing = report.get("missing_indices")
    for index in missing if isinstance(missing, list) else []:
        reasons.append(f"GPU {index}: missing")
    expected_value = report.get("expected_indices")
    expected = set(expected_value) if isinstance(expected_value, list) else None
    rows = report.get("gpus")
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        index = row.get("index")
        if expected is not None and index not in expected:
            continue
        row_reasons = row.get("reasons")
        for reason in row_reasons if isinstance(row_reasons, (list, tuple)) else []:
            reasons.append(f"GPU {index}: {reason}")
    return tuple(reasons)
