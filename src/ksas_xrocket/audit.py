"""KSAS raw-data audit utilities."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from statistics import median
from typing import Any

EXPECTED_FILE_COUNT = 240
EXPECTED_LABELS = {
    0: "no movement",
    1: "upward block",
    2: "hammering inward block",
    3: "extended outward block",
    4: "outward downward block",
    5: "rear elbow block",
}
EXPECTED_PARTICIPANTS = tuple(range(1, 21))
EXPECTED_ARMS = {"i": "left", "d": "right"}
EXPECTED_CHANNELS = (
    "accelerometer_x",
    "accelerometer_y",
    "accelerometer_z",
    "gravity_x",
    "gravity_y",
    "gravity_z",
    "gyros_x",
    "gyros_y",
    "gyros_z",
    "lin_accel_x",
    "lin_accel_y",
    "lin_accel_z",
    "game_rot_vec_x",
    "game_rot_vec_y",
    "game_rot_vec_z",
    "magn_field_x",
    "magn_field_y",
    "magn_field_z",
)
SENSOR_UNITS = {
    "accelerometer": "m/s^2",
    "gravity": "m/s^2",
    "gyros": "rad/s",
    "lin_accel": "m/s^2",
    "game_rot_vec": "unitless",
    "magn_field": "microtesla",
}

_FILENAME_RE = re.compile(r"^(?P<label>\d+)-(?P<participant>\d+)-(?P<arm>[id])\.csv$")
CsvRowValue = str | int | float


class DatasetAuditError(ValueError):
    """Raised when the KSAS raw dataset fails an audit invariant."""


@dataclass(frozen=True)
class FilenameParts:
    """Parsed KSAS movement filename fields."""

    movement_label_id: int
    participant_number: int
    arm_code: str

    @property
    def participant_id(self) -> str:
        """Return the pseudonymous participant identifier used in manifests."""
        return f"P{self.participant_number:02d}"

    @property
    def movement_label(self) -> str:
        """Return the movement label text."""
        return EXPECTED_LABELS[self.movement_label_id]

    @property
    def arm(self) -> str:
        """Return the human-readable arm label."""
        return EXPECTED_ARMS[self.arm_code]

    @property
    def sample_id(self) -> str:
        """Return a stable sample identifier."""
        return f"ksas_m{self.movement_label_id}_p{self.participant_number:02d}_{self.arm_code}"


@dataclass(frozen=True)
class CsvStats:
    """Audit statistics for one CSV file."""

    row_count: int
    missing_values: int
    duplicate_rows: int
    non_numeric_columns: tuple[str, ...]
    wrong_width_rows: int
    numeric_ranges: dict[str, dict[str, float]]


@dataclass(frozen=True)
class AuditResult:
    """Paths and summary returned by a completed audit."""

    manifest_path: Path
    summary_path: Path
    numeric_ranges_path: Path
    provenance_path: Path
    summary: dict[str, Any]


def parse_movement_filename(filename: str) -> FilenameParts:
    """Parse a KSAS movement filename."""
    match = _FILENAME_RE.fullmatch(filename)
    if match is None:
        raise DatasetAuditError(f"Invalid KSAS movement filename: {filename}")

    label_id = int(match.group("label"))
    participant_number = int(match.group("participant"))
    arm_code = match.group("arm")

    if label_id not in EXPECTED_LABELS:
        raise DatasetAuditError(f"Unexpected movement label {label_id} in {filename}")
    if participant_number not in EXPECTED_PARTICIPANTS:
        raise DatasetAuditError(f"Unexpected participant {participant_number} in {filename}")
    if arm_code not in EXPECTED_ARMS:
        raise DatasetAuditError(f"Unexpected arm code {arm_code!r} in {filename}")

    return FilenameParts(
        movement_label_id=label_id,
        participant_number=participant_number,
        arm_code=arm_code,
    )


def audit_dataset(
    raw_dir: Path,
    manifest_path: Path,
    audit_dir: Path,
    provenance_path: Path,
    project_root: Path | None = None,
) -> AuditResult:
    """Audit the raw KSAS dataset and write manifest/provenance artifacts."""
    project_root = Path.cwd() if project_root is None else project_root
    movements_dir = raw_dir / "movements"
    if not movements_dir.is_dir():
        raise DatasetAuditError(f"Expected movements directory not found: {movements_dir}")

    csv_paths = sorted(movements_dir.glob("*.csv"), key=lambda path: path.name)
    manifest_rows: list[dict[str, CsvRowValue]] = []
    numeric_range_rows: list[dict[str, CsvRowValue]] = []
    errors: list[str] = []
    checksums: dict[str, list[str]] = defaultdict(list)
    observed_keys: list[tuple[int, int, str]] = []
    class_counts: Counter[int] = Counter()
    arm_counts: Counter[str] = Counter()
    participant_counts: Counter[str] = Counter()
    row_counts: list[int] = []
    missing_total = 0
    duplicate_rows_total = 0
    non_numeric_files: list[str] = []
    wrong_width_files: list[str] = []

    if len(csv_paths) != EXPECTED_FILE_COUNT:
        errors.append(f"Expected {EXPECTED_FILE_COUNT} CSV files, found {len(csv_paths)}")

    for csv_path in csv_paths:
        try:
            parts = parse_movement_filename(csv_path.name)
        except DatasetAuditError as exc:
            errors.append(str(exc))
            continue

        checksum = sha256_file(csv_path)
        checksums[checksum].append(csv_path.name)
        stats = read_csv_stats(csv_path)
        observed_keys.append((parts.movement_label_id, parts.participant_number, parts.arm_code))
        class_counts[parts.movement_label_id] += 1
        arm_counts[parts.arm_code] += 1
        participant_counts[parts.participant_id] += 1
        row_counts.append(stats.row_count)
        missing_total += stats.missing_values
        duplicate_rows_total += stats.duplicate_rows

        if stats.non_numeric_columns:
            non_numeric_files.append(csv_path.name)
            errors.append(f"{csv_path.name} has non-numeric columns: {stats.non_numeric_columns}")
        if stats.wrong_width_rows:
            wrong_width_files.append(csv_path.name)
            errors.append(f"{csv_path.name} has {stats.wrong_width_rows} row(s) with wrong width")
        if stats.missing_values:
            errors.append(f"{csv_path.name} has {stats.missing_values} missing value(s)")
        if stats.duplicate_rows:
            errors.append(f"{csv_path.name} has {stats.duplicate_rows} duplicate row(s)")

        quality_flag = (
            "warning"
            if (
                stats.non_numeric_columns
                or stats.wrong_width_rows
                or stats.missing_values
                or stats.duplicate_rows
            )
            else "valid"
        )
        manifest_rows.append(
            build_manifest_row(
                csv_path=csv_path,
                parts=parts,
                checksum=checksum,
                row_count=stats.row_count,
                quality_flag=quality_flag,
                project_root=project_root,
            )
        )

        for channel, ranges in stats.numeric_ranges.items():
            numeric_range_rows.append(
                {
                    "sample_id": parts.sample_id,
                    "filename": csv_path.name,
                    "channel": channel,
                    "min": ranges["min"],
                    "max": ranges["max"],
                    "mean": ranges["mean"],
                    "original_length": stats.row_count,
                }
            )

    duplicate_checksum_groups = [names for names in checksums.values() if len(names) > 1]
    for names in duplicate_checksum_groups:
        errors.append(f"Exact duplicate file contents found: {', '.join(sorted(names))}")

    expected_keys = {
        (label_id, participant, arm_code)
        for label_id in EXPECTED_LABELS
        for participant in EXPECTED_PARTICIPANTS
        for arm_code in EXPECTED_ARMS
    }
    observed_key_counts = Counter(observed_keys)
    missing_keys = sorted(expected_keys.difference(observed_key_counts))
    duplicate_keys = sorted(key for key, count in observed_key_counts.items() if count > 1)
    unexpected_keys = sorted(set(observed_key_counts).difference(expected_keys))

    if missing_keys:
        errors.append(f"Missing participant-arm-label combinations: {format_keys(missing_keys)}")
    if duplicate_keys:
        errors.append(
            f"Duplicate participant-arm-label combinations: {format_keys(duplicate_keys)}"
        )
    if unexpected_keys:
        errors.append(
            f"Unexpected participant-arm-label combinations: {format_keys(unexpected_keys)}"
        )

    audit_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    numeric_ranges_path = audit_dir / "ksas_numeric_ranges.csv"
    summary_path = audit_dir / "ksas_audit_summary.json"

    write_dict_rows(manifest_path, manifest_rows)
    write_dict_rows(numeric_ranges_path, numeric_range_rows)

    summary = build_summary(
        raw_dir=raw_dir,
        csv_count=len(csv_paths),
        class_counts=class_counts,
        arm_counts=arm_counts,
        participant_counts=participant_counts,
        row_counts=row_counts,
        missing_total=missing_total,
        duplicate_rows_total=duplicate_rows_total,
        duplicate_checksum_groups=duplicate_checksum_groups,
        non_numeric_files=non_numeric_files,
        wrong_width_files=wrong_width_files,
        missing_keys=missing_keys,
        duplicate_keys=duplicate_keys,
        errors=errors,
        manifest_path=manifest_path,
        numeric_ranges_path=numeric_ranges_path,
    )
    write_json(summary_path, summary)
    provenance = build_provenance(raw_dir=raw_dir, audit_date=date.today().isoformat())
    write_json(provenance_path, provenance)

    if errors:
        raise DatasetAuditError("KSAS audit failed:\n- " + "\n- ".join(errors))

    return AuditResult(
        manifest_path=manifest_path,
        summary_path=summary_path,
        numeric_ranges_path=numeric_ranges_path,
        provenance_path=provenance_path,
        summary=summary,
    )


def read_csv_stats(csv_path: Path) -> CsvStats:
    """Read one KSAS CSV and compute validation statistics."""
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            raise DatasetAuditError(f"{csv_path.name} is empty") from None
        rows = list(reader)

    if tuple(header) != EXPECTED_CHANNELS:
        raise DatasetAuditError(
            f"{csv_path.name} has unexpected columns: {header}; expected {list(EXPECTED_CHANNELS)}"
        )

    column_values: dict[str, list[float]] = {channel: [] for channel in EXPECTED_CHANNELS}
    non_numeric_columns: set[str] = set()
    missing_values = 0
    wrong_width_rows = 0
    row_tuples: list[tuple[str, ...]] = []

    for row in rows:
        row_tuples.append(tuple(row))
        if len(row) != len(EXPECTED_CHANNELS):
            wrong_width_rows += 1
            continue
        for channel, value in zip(EXPECTED_CHANNELS, row, strict=True):
            stripped = value.strip()
            if stripped == "":
                missing_values += 1
                continue
            try:
                column_values[channel].append(float(stripped))
            except ValueError:
                non_numeric_columns.add(channel)

    duplicate_rows = len(row_tuples) - len(set(row_tuples))
    numeric_ranges = {
        channel: {
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
        }
        for channel, values in column_values.items()
        if values
    }
    missing_numeric_columns = set(EXPECTED_CHANNELS).difference(numeric_ranges)
    non_numeric_columns.update(missing_numeric_columns)

    return CsvStats(
        row_count=len(rows),
        missing_values=missing_values,
        duplicate_rows=duplicate_rows,
        non_numeric_columns=tuple(sorted(non_numeric_columns)),
        wrong_width_rows=wrong_width_rows,
        numeric_ranges=numeric_ranges,
    )


def build_manifest_row(
    csv_path: Path,
    parts: FilenameParts,
    checksum: str,
    row_count: int,
    quality_flag: str,
    project_root: Path,
) -> dict[str, CsvRowValue]:
    """Build one row for the canonical sample manifest."""
    return {
        "sample_id": parts.sample_id,
        "filename": csv_path.name,
        "source_path": relative_path(csv_path, project_root),
        "checksum_sha256": checksum,
        "movement_label_id": parts.movement_label_id,
        "movement_label": parts.movement_label,
        "participant_id": parts.participant_id,
        "arm_code": parts.arm_code,
        "arm": parts.arm,
        "recording_id": csv_path.stem,
        "split_group": parts.participant_id,
        "sensor_location": "smartphone",
        "device_model": "XiaoMi Mi A2",
        "device_orientation": "",
        "sampling_rate_hz": "",
        "original_length": row_count,
        "processed_length": row_count,
        "channel_names": json.dumps(list(EXPECTED_CHANNELS)),
        "quality_flag": quality_flag,
    }


def build_summary(
    raw_dir: Path,
    csv_count: int,
    class_counts: Counter[int],
    arm_counts: Counter[str],
    participant_counts: Counter[str],
    row_counts: list[int],
    missing_total: int,
    duplicate_rows_total: int,
    duplicate_checksum_groups: list[list[str]],
    non_numeric_files: list[str],
    wrong_width_files: list[str],
    missing_keys: list[tuple[int, int, str]],
    duplicate_keys: list[tuple[int, int, str]],
    errors: list[str],
    manifest_path: Path,
    numeric_ranges_path: Path,
) -> dict[str, Any]:
    """Build the JSON audit summary."""
    row_count_summary: dict[str, int | float | None] = {
        "min": min(row_counts) if row_counts else None,
        "max": max(row_counts) if row_counts else None,
    }
    if row_counts:
        row_count_summary["median"] = median(row_counts)

    return {
        "audit_timestamp_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "validation_status": "failed" if errors else "passed",
        "raw_dir": raw_dir.as_posix(),
        "expected_file_count": EXPECTED_FILE_COUNT,
        "csv_file_count": csv_count,
        "expected_labels": {str(key): value for key, value in EXPECTED_LABELS.items()},
        "class_counts": {str(key): class_counts[key] for key in sorted(class_counts)},
        "participants": {
            "expected_count": len(EXPECTED_PARTICIPANTS),
            "observed_count": len(participant_counts),
            "ids": sorted(participant_counts),
        },
        "arms": {
            code: {"label": EXPECTED_ARMS[code], "count": arm_counts[code]}
            for code in sorted(EXPECTED_ARMS)
        },
        "participant_arm_groups": len(EXPECTED_PARTICIPANTS) * len(EXPECTED_ARMS),
        "missing_participant_arm_label_combinations": format_keys(missing_keys),
        "duplicate_participant_arm_label_combinations": format_keys(duplicate_keys),
        "channels": list(EXPECTED_CHANNELS),
        "sensor_units": SENSOR_UNITS,
        "sequence_length_samples": row_count_summary,
        "missing_values_total": missing_total,
        "duplicate_rows_total": duplicate_rows_total,
        "duplicate_file_checksum_groups": duplicate_checksum_groups,
        "non_numeric_files": sorted(non_numeric_files),
        "wrong_width_files": sorted(wrong_width_files),
        "manifest_path": manifest_path.as_posix(),
        "numeric_ranges_path": numeric_ranges_path.as_posix(),
        "errors": errors,
    }


def build_provenance(raw_dir: Path, audit_date: str) -> dict[str, Any]:
    """Build KSAS dataset provenance metadata."""
    readme_path = raw_dir / "README.md"
    return {
        "dataset_name": "KSAS-Dataset",
        "source": "Course-provided local dataset",
        "local_acquisition_date": audit_date,
        "audit_date": audit_date,
        "local_readme_path": readme_path.as_posix(),
        "local_readme_sha256": sha256_file(readme_path) if readme_path.is_file() else None,
        "raw_data_git_policy": "Raw KSAS CSV files remain local-only and ignored by Git.",
        "known_unresolved_facts": [
            "Sampling frequency is not stated in the dataset README or CSV files.",
            "CSV files do not include timestamps.",
            "Phone placement and device orientation protocol are not fully specified.",
        ],
    }


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_dict_rows(path: Path, rows: list[dict[str, CsvRowValue]]) -> None:
    """Write a list of dictionaries as CSV."""
    if not rows:
        path.write_text("", encoding="utf-8", newline="")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, content: dict[str, Any]) -> None:
    """Write indented JSON with stable key order."""
    path.write_text(json.dumps(content, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def relative_path(path: Path, project_root: Path) -> str:
    """Return a portable relative path when possible."""
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def format_keys(keys: list[tuple[int, int, str]]) -> list[str]:
    """Format label-participant-arm coverage keys for JSON and errors."""
    return [
        f"label={label_id},participant=P{participant:02d},arm={arm_code}"
        for label_id, participant, arm_code in keys
    ]
