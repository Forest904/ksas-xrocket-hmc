from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from ksas_xrocket.audit import (
    EXPECTED_CHANNELS,
    DatasetAuditError,
    audit_dataset,
    parse_movement_filename,
)


def test_parse_movement_filename_maps_labels_participants_and_arms() -> None:
    parts = parse_movement_filename("5-20-i.csv")

    assert parts.movement_label_id == 5
    assert parts.movement_label == "rear elbow block"
    assert parts.participant_number == 20
    assert parts.participant_id == "P20"
    assert parts.arm_code == "i"
    assert parts.arm == "left"
    assert parts.sample_id == "ksas_m5_p20_i"


def test_parse_movement_filename_rejects_unknown_label() -> None:
    with pytest.raises(DatasetAuditError, match="Unexpected movement label"):
        parse_movement_filename("6-1-d.csv")


def test_audit_dataset_writes_manifest_summary_and_ranges(tmp_path: Path) -> None:
    raw_dir = make_synthetic_ksas_dataset(tmp_path)
    manifest_path = tmp_path / "manifests" / "samples.csv"
    audit_dir = tmp_path / "audit"
    provenance_path = tmp_path / "manifests" / "ksas_provenance.json"

    result = audit_dataset(
        raw_dir=raw_dir,
        manifest_path=manifest_path,
        audit_dir=audit_dir,
        provenance_path=provenance_path,
        project_root=tmp_path,
    )

    manifest_rows = read_csv_rows(result.manifest_path)
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    range_rows = read_csv_rows(result.numeric_ranges_path)
    provenance = json.loads(result.provenance_path.read_text(encoding="utf-8"))

    assert len(manifest_rows) == 240
    assert manifest_rows[0]["sample_id"] == "ksas_m0_p01_d"
    assert manifest_rows[0]["participant_id"] == "P01"
    assert manifest_rows[0]["arm"] == "right"
    assert manifest_rows[0]["sampling_rate_hz"] == ""
    assert manifest_rows[0]["processed_length"] == manifest_rows[0]["original_length"]
    assert json.loads(manifest_rows[0]["channel_names"]) == list(EXPECTED_CHANNELS)
    assert summary["validation_status"] == "passed"
    assert summary["csv_file_count"] == 240
    assert summary["missing_values_total"] == 0
    assert summary["duplicate_rows_total"] == 0
    assert len(range_rows) == 240 * len(EXPECTED_CHANNELS)
    assert provenance["source"] == "Course-provided local dataset"
    assert {"lic" + "ense", "source" + "_url"}.isdisjoint(provenance)


def test_audit_dataset_fails_on_missing_class_coverage(tmp_path: Path) -> None:
    raw_dir = make_synthetic_ksas_dataset(tmp_path, omit=("5", "20", "i"))

    with pytest.raises(DatasetAuditError, match="Expected 240 CSV files"):
        audit_dataset(
            raw_dir=raw_dir,
            manifest_path=tmp_path / "samples.csv",
            audit_dir=tmp_path / "audit",
            provenance_path=tmp_path / "provenance.json",
            project_root=tmp_path,
        )


def make_synthetic_ksas_dataset(
    tmp_path: Path,
    omit: tuple[str, str, str] | None = None,
) -> Path:
    raw_dir = tmp_path / "KSAS-Dataset"
    movements_dir = raw_dir / "movements"
    movements_dir.mkdir(parents=True)
    raw_dir.joinpath("README.md").write_text("# KSAS-Dataset\n", encoding="utf-8")

    for label_id in range(6):
        for participant in range(1, 21):
            for arm_code in ("d", "i"):
                key = (str(label_id), str(participant), arm_code)
                if key == omit:
                    continue
                filename = f"{label_id}-{participant}-{arm_code}.csv"
                write_synthetic_csv(
                    movements_dir / filename,
                    base=label_id * 1000 + participant * 10 + (0 if arm_code == "d" else 1),
                )

    return raw_dir


def write_synthetic_csv(path: Path, base: int) -> None:
    rows = []
    for row_index in range(2):
        rows.append(
            [
                f"{base + row_index + column_index / 100:.2f}"
                for column_index, _channel in enumerate(EXPECTED_CHANNELS)
            ]
        )

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(EXPECTED_CHANNELS)
        writer.writerows(rows)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
