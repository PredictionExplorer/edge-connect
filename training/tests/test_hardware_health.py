from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import hardware_health_preflight
from startrain.hardware_health import parse_nvidia_smi_xml, query_gpu_health


def _gpu_xml(
    index: int,
    *,
    aggregate_parity: int = 0,
    volatile_parity: int = 0,
    threshold: str = "No",
) -> str:
    return f"""
    <gpu id="00000000:{index:02x}:00.0">
      <product_name>NVIDIA H100 80GB HBM3</product_name>
      <uuid>GPU-{index}</uuid>
      <serial>serial-{index}</serial>
      <minor_number>{index + 4}</minor_number>
      <gpu_recovery_action>None</gpu_recovery_action>
      <pci><pci_bus_id>00000000:{index:02x}:00.0</pci_bus_id></pci>
      <mig_mode><current_mig>Disabled</current_mig></mig_mode>
      <ecc_mode><current_ecc>Enabled</current_ecc></ecc_mode>
      <ecc_errors>
        <volatile>
          <sram_uncorrectable_parity>{volatile_parity}</sram_uncorrectable_parity>
          <sram_uncorrectable_secded>0</sram_uncorrectable_secded>
          <dram_uncorrectable>0</dram_uncorrectable>
        </volatile>
        <aggregate>
          <sram_uncorrectable_parity>{aggregate_parity}</sram_uncorrectable_parity>
          <sram_uncorrectable_secded>0</sram_uncorrectable_secded>
          <dram_uncorrectable>0</dram_uncorrectable>
          <sram_threshold_exceeded>{threshold}</sram_threshold_exceeded>
        </aggregate>
        <channel_repair_pending>No</channel_repair_pending>
        <tpc_repair_pending>No</tpc_repair_pending>
      </ecc_errors>
      <remapped_rows>
        <remapped_row_pending>No</remapped_row_pending>
        <remapped_row_failure>No</remapped_row_failure>
      </remapped_rows>
    </gpu>
    """


def _xml(*gpus: str) -> str:
    return (
        "<nvidia_smi_log><driver_version>580.105.08</driver_version>"
        + "".join(gpus)
        + "</nvidia_smi_log>"
    )


def test_health_parser_uses_logical_order_not_minor_number() -> None:
    report = parse_nvidia_smi_xml(
        _xml(_gpu_xml(0), _gpu_xml(1)),
        expected_indices=(0, 1),
    )

    assert report["healthy"] is True
    assert [row["index"] for row in report["gpus"]] == [0, 1]


def test_aggregate_sram_threshold_fails_with_zero_volatile_errors() -> None:
    report = parse_nvidia_smi_xml(
        _xml(_gpu_xml(0, aggregate_parity=65_535, threshold="Yes")),
        expected_indices=(0,),
    )

    assert report["healthy"] is False
    row = report["gpus"][0]
    assert row["volatile_sram_uncorrectable_parity"] == 0
    assert row["aggregate_sram_uncorrectable_parity"] == 65_535
    assert row["reasons"] == (
        "aggregate_uncorrectable_ecc",
        "sram_threshold_exceeded",
    )


def test_missing_configured_gpu_fails_closed() -> None:
    report = parse_nvidia_smi_xml(
        _xml(_gpu_xml(0)),
        expected_indices=(0, 1),
    )

    assert report["healthy"] is False
    assert report["missing_indices"] == [1]


def test_query_failure_is_explicit() -> None:
    def runner(*_args, **_kwargs):
        return SimpleNamespace(returncode=9, stdout="", stderr="driver unavailable")

    with pytest.raises(RuntimeError, match="driver unavailable"):
        query_gpu_health(expected_indices=(0,), runner=runner)


def test_preflight_writes_failure_report(tmp_path: Path, monkeypatch) -> None:
    profile = Path(__file__).parents[1] / "configs" / "small.yaml"
    output = tmp_path / "health.json"
    monkeypatch.setattr(
        hardware_health_preflight,
        "query_gpu_health",
        lambda **_kwargs: parse_nvidia_smi_xml(
            _xml(_gpu_xml(0, aggregate_parity=4, threshold="Yes")),
            expected_indices=(0,),
        ),
    )

    assert (
        hardware_health_preflight.main(
            ["--config", str(profile), "--output", str(output)]
        )
        == 2
    )
    assert json.loads(output.read_text(encoding="utf-8"))["healthy"] is False
