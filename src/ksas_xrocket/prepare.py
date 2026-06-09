"""M2 preprocessing for KSAS manifest-backed tensors."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ksas_xrocket.audit import EXPECTED_CHANNELS, EXPECTED_LABELS, sha256_file, write_json

REQUIRED_MANIFEST_COLUMNS = (
    "sample_id",
    "source_path",
    "checksum_sha256",
    "movement_label_id",
    "movement_label",
    "participant_id",
    "arm_code",
    "arm",
    "split_group",
    "original_length",
    "channel_names",
    "quality_flag",
)
DEFAULT_TARGET_LENGTH = 56


class PreparationError(ValueError):
    """Raised when manifest-backed tensor preparation fails."""


@dataclass(frozen=True)
class PreparedData:
    """In-memory representation of the M2 tensor contract."""

    x: np.ndarray
    y: np.ndarray
    valid_mask: np.ndarray
    sample_ids: np.ndarray
    participant_ids: np.ndarray
    arm_codes: np.ndarray
    original_lengths: np.ndarray
    manifest_rows: list[dict[str, str]]
    raw_checksums: dict[str, str]


@dataclass(frozen=True)
class PreparationResult:
    """Paths returned by a completed M2 preparation run."""

    output_dir: Path
    tensor_path: Path
    metadata_path: Path
    contract_path: Path
    provenance_path: Path


def prepare_ksas_tensors(
    manifest_path: Path,
    output_dir: Path,
    target_length: int = DEFAULT_TARGET_LENGTH,
    project_root: Path | None = None,
    resolved_config: dict[str, Any] | None = None,
) -> PreparationResult:
    """Build padded model-ready tensors from the canonical sample manifest."""
    project_root = Path.cwd() if project_root is None else project_root
    prepared = build_prepared_data(
        manifest_path=manifest_path,
        target_length=target_length,
        project_root=project_root,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    tensor_path = output_dir / "tensors.npz"
    metadata_path = output_dir / "metadata.csv"
    contract_path = output_dir / "tensor_contract.json"
    provenance_path = output_dir / "provenance.json"
    resolved_config_path = output_dir / "resolved_config.json"

    np.savez_compressed(
        tensor_path,
        X=prepared.x,
        y=prepared.y,
        valid_mask=prepared.valid_mask,
        sample_id=prepared.sample_ids,
        participant_id=prepared.participant_ids,
        arm_code=prepared.arm_codes,
        original_length=prepared.original_lengths,
    )
    write_metadata_csv(metadata_path, prepared.manifest_rows)
    write_json(contract_path, build_tensor_contract(prepared, target_length))
    write_json(
        resolved_config_path,
        resolved_config
        or {
            "manifest": manifest_path.as_posix(),
            "output_dir": output_dir.as_posix(),
            "target_length": target_length,
            "length_strategy": "right_pad_with_zeros",
            "normalization": "none",
            "smoothing": "none",
            "derived_channels": "none",
            "target": "movement_label_id",
            "classes": list(EXPECTED_LABELS),
        },
    )
    write_json(
        provenance_path,
        {
            "manifest_path": manifest_path.as_posix(),
            "manifest_sha256": sha256_file(manifest_path),
            "tensor_path": tensor_path.as_posix(),
            "metadata_path": metadata_path.as_posix(),
            "target_length": target_length,
            "length_strategy": "right_pad_with_zeros",
            "normalization": "none",
            "smoothing": "none",
            "derived_channels": "none",
            "primary_target": "six_class_movement_label_id",
            "raw_file_count": len(prepared.raw_checksums),
            "raw_inputs_sha256": digest_mapping(prepared.raw_checksums),
            "resolved_config_path": resolved_config_path.as_posix(),
        },
    )

    return PreparationResult(
        output_dir=output_dir,
        tensor_path=tensor_path,
        metadata_path=metadata_path,
        contract_path=contract_path,
        provenance_path=provenance_path,
    )


def build_prepared_data(
    manifest_path: Path,
    target_length: int = DEFAULT_TARGET_LENGTH,
    project_root: Path | None = None,
) -> PreparedData:
    """Load CSVs listed in the manifest and return padded tensors."""
    project_root = Path.cwd() if project_root is None else project_root
    rows = read_manifest(manifest_path)
    if not rows:
        raise PreparationError(f"Manifest contains no rows: {manifest_path}")

    n_samples = len(rows)
    n_channels = len(EXPECTED_CHANNELS)
    x = np.zeros((n_samples, n_channels, target_length), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.int64)
    valid_mask = np.zeros((n_samples, target_length), dtype=np.bool_)
    sample_ids: list[str] = []
    participant_ids: list[str] = []
    arm_codes: list[str] = []
    original_lengths: list[int] = []
    output_rows: list[dict[str, str]] = []
    raw_checksums: dict[str, str] = {}

    for index, row in enumerate(rows):
        validate_manifest_row(row, manifest_path)
        csv_path = resolve_manifest_source_path(row["source_path"], project_root)
        if not csv_path.is_file():
            raise PreparationError(f"Source CSV not found: {csv_path}")
        actual_checksum = sha256_file(csv_path)
        expected_checksum = row["checksum_sha256"]
        if actual_checksum != expected_checksum:
            raise PreparationError(
                f"{row['sample_id']} checksum mismatch for {csv_path}: "
                f"manifest={expected_checksum}, actual={actual_checksum}"
            )
        raw_checksums[row["source_path"]] = actual_checksum
        sequence = load_sequence_csv(csv_path)
        original_length = int(row["original_length"])
        if sequence.shape[0] != original_length:
            raise PreparationError(
                f"{row['sample_id']} original_length={original_length} but CSV has "
                f"{sequence.shape[0]} rows"
            )
        if original_length > target_length:
            raise PreparationError(
                f"{row['sample_id']} length {original_length} exceeds target_length "
                f"{target_length}; truncation is not enabled"
            )

        label_id = int(row["movement_label_id"])
        if label_id not in EXPECTED_LABELS:
            raise PreparationError(f"{row['sample_id']} has unsupported label {label_id}")

        x[index, :, :original_length] = sequence.T
        y[index] = label_id
        valid_mask[index, :original_length] = True
        sample_ids.append(row["sample_id"])
        participant_ids.append(row["participant_id"])
        arm_codes.append(row["arm_code"])
        original_lengths.append(original_length)
        output_rows.append(
            row
            | {
                "sample_index": str(index),
                "processed_length": str(target_length),
                "length_strategy": "right_pad_with_zeros",
                "tensor_axis_order": "sample,channel,time",
            }
        )

    return PreparedData(
        x=x,
        y=y,
        valid_mask=valid_mask,
        sample_ids=np.asarray(sample_ids),
        participant_ids=np.asarray(participant_ids),
        arm_codes=np.asarray(arm_codes),
        original_lengths=np.asarray(original_lengths, dtype=np.int64),
        manifest_rows=output_rows,
        raw_checksums=raw_checksums,
    )


def read_manifest(manifest_path: Path) -> list[dict[str, str]]:
    """Read and validate the sample manifest schema."""
    if not manifest_path.is_file():
        raise PreparationError(f"Manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise PreparationError(f"Manifest has no header: {manifest_path}")
        missing = sorted(set(REQUIRED_MANIFEST_COLUMNS).difference(reader.fieldnames))
        if missing:
            raise PreparationError(
                f"Manifest {manifest_path} is missing required columns: {', '.join(missing)}"
            )
        return list(reader)


def validate_manifest_row(row: dict[str, str], manifest_path: Path) -> None:
    """Validate one manifest row before loading source data."""
    if row["quality_flag"] != "valid":
        raise PreparationError(f"{row['sample_id']} has quality_flag={row['quality_flag']!r}")
    try:
        channels = json.loads(row["channel_names"])
    except json.JSONDecodeError as exc:
        raise PreparationError(
            f"{row['sample_id']} has invalid channel_names JSON in {manifest_path}"
        ) from exc
    if channels != list(EXPECTED_CHANNELS):
        raise PreparationError(
            f"{row['sample_id']} channel order does not match audited EXPECTED_CHANNELS"
        )
    if row["movement_label"] != EXPECTED_LABELS.get(int(row["movement_label_id"]), ""):
        raise PreparationError(f"{row['sample_id']} label id/name mapping is inconsistent")
    if row["split_group"] != row["participant_id"]:
        raise PreparationError(f"{row['sample_id']} split_group must equal participant_id")


def resolve_manifest_source_path(source_path: str, project_root: Path) -> Path:
    """Resolve a manifest source path relative to the project root."""
    path = Path(source_path)
    if path.is_absolute():
        return path
    return project_root / path


def load_sequence_csv(csv_path: Path) -> np.ndarray:
    """Load one KSAS sequence with the audited channel order."""
    if not csv_path.is_file():
        raise PreparationError(f"Source CSV not found: {csv_path}")
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            raise PreparationError(f"Source CSV is empty: {csv_path}") from None
        if tuple(header) != EXPECTED_CHANNELS:
            raise PreparationError(f"{csv_path} has unexpected channel header")
        rows: list[list[float]] = []
        for row_number, row in enumerate(reader, start=2):
            if len(row) != len(EXPECTED_CHANNELS):
                raise PreparationError(
                    f"{csv_path} row {row_number} has {len(row)} values; "
                    f"expected {len(EXPECTED_CHANNELS)}"
                )
            parsed_row: list[float] = []
            for channel, value in zip(EXPECTED_CHANNELS, row, strict=True):
                stripped = value.strip()
                if stripped == "":
                    raise PreparationError(
                        f"{csv_path} row {row_number} channel {channel} is empty"
                    )
                try:
                    parsed_row.append(float(stripped))
                except ValueError as exc:
                    raise PreparationError(
                        f"{csv_path} row {row_number} channel {channel} has "
                        f"non-numeric value {value!r}"
                    ) from exc
            rows.append(parsed_row)

    if not rows:
        raise PreparationError(f"Source CSV has no data rows: {csv_path}")
    sequence = np.asarray(rows, dtype=np.float32)
    if sequence.ndim != 2 or sequence.shape[1] != len(EXPECTED_CHANNELS):
        raise PreparationError(f"{csv_path} has invalid sequence shape {sequence.shape}")
    if np.isnan(sequence).any():
        raise PreparationError(f"{csv_path} contains NaN after numeric parsing")
    return sequence


def digest_mapping(values: dict[str, str]) -> str:
    """Return a deterministic digest for a path-to-checksum mapping."""
    digest = hashlib.sha256()
    for key in sorted(values):
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(values[key].encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def write_metadata_csv(path: Path, rows: list[dict[str, str]]) -> None:
    """Write processed sample metadata aligned to tensor sample indices."""
    if not rows:
        path.write_text("", encoding="utf-8", newline="")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_tensor_contract(prepared: PreparedData, target_length: int) -> dict[str, Any]:
    """Return a JSON-serializable description of the M2 tensor contract."""
    return {
        "axis_order": ["sample", "channel", "time"],
        "shape": list(prepared.x.shape),
        "dtype": str(prepared.x.dtype),
        "target_length": target_length,
        "padding": {
            "side": "right",
            "value": 0.0,
            "valid_mask_shape": list(prepared.valid_mask.shape),
        },
        "channels": list(EXPECTED_CHANNELS),
        "sample_order": "data/manifests/samples.csv row order",
        "label_field": "movement_label_id",
        "classes": {str(key): value for key, value in EXPECTED_LABELS.items()},
        "group_field": "participant_id",
        "metadata_fields": [
            "sample_id",
            "participant_id",
            "arm_code",
            "original_length",
        ],
    }
