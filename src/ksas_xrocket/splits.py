"""Participant-grouped split generation and diagnostics."""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from ksas_xrocket.audit import EXPECTED_LABELS, write_json
from ksas_xrocket.prepare import read_manifest


class SplitError(ValueError):
    """Raised when grouped split generation fails."""


@dataclass(frozen=True)
class SplitResult:
    """Paths returned by grouped split generation."""

    split_manifest_path: Path
    diagnostics_path: Path
    summary_path: Path


def generate_grouped_splits(
    manifest_path: Path,
    output_dir: Path,
    n_splits: int = 5,
    random_state: int = 42,
) -> SplitResult:
    """Generate participant-safe stratified grouped fold manifests."""
    rows = read_manifest(manifest_path)
    sample_ids = np.asarray([row["sample_id"] for row in rows])
    labels = np.asarray([int(row["movement_label_id"]) for row in rows], dtype=np.int64)
    groups = np.asarray([row["split_group"] for row in rows])
    arms = np.asarray([row["arm_code"] for row in rows])

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    split_rows: list[dict[str, str | int]] = []
    diagnostics: list[dict[str, Any]] = []

    for fold_index, (train_indices, test_indices) in enumerate(
        splitter.split(np.zeros(len(labels)), labels, groups)
    ):
        train_groups = set(groups[train_indices])
        test_groups = set(groups[test_indices])
        overlap = sorted(train_groups.intersection(test_groups))
        if overlap:
            raise SplitError(f"Fold {fold_index} has participant leakage: {', '.join(overlap)}")
        absent_train = absent_classes(labels[train_indices])
        absent_test = absent_classes(labels[test_indices])
        if absent_train or absent_test:
            raise SplitError(
                f"Fold {fold_index} has absent classes: train={absent_train}, test={absent_test}"
            )

        for split_name, indices in (("train", train_indices), ("test", test_indices)):
            for sample_index in indices:
                split_rows.append(
                    {
                        "fold": fold_index,
                        "split": split_name,
                        "sample_index": int(sample_index),
                        "sample_id": sample_ids[sample_index],
                        "movement_label_id": int(labels[sample_index]),
                        "participant_id": groups[sample_index],
                        "arm_code": arms[sample_index],
                    }
                )

        diagnostics.append(
            build_fold_diagnostics(
                fold_index=fold_index,
                train_indices=train_indices,
                test_indices=test_indices,
                labels=labels,
                groups=groups,
                arms=arms,
            )
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    split_manifest_path = output_dir / "m2_grouped_folds.csv"
    diagnostics_path = output_dir / "m2_grouped_fold_diagnostics.csv"
    summary_path = output_dir / "m2_grouped_split_summary.json"
    write_split_manifest(split_manifest_path, split_rows)
    write_diagnostics_csv(diagnostics_path, diagnostics)
    write_json(
        summary_path,
        {
            "splitter": "StratifiedGroupKFold",
            "n_splits": n_splits,
            "random_state": random_state,
            "group_field": "split_group",
            "target_field": "movement_label_id",
            "participant_overlap_assertion": "passed",
            "folds": diagnostics,
        },
    )

    return SplitResult(
        split_manifest_path=split_manifest_path,
        diagnostics_path=diagnostics_path,
        summary_path=summary_path,
    )


def absent_classes(labels: np.ndarray) -> list[int]:
    """Return expected class IDs absent from a label vector."""
    present = set(int(label) for label in labels)
    return [label_id for label_id in EXPECTED_LABELS if label_id not in present]


def build_fold_diagnostics(
    fold_index: int,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    arms: np.ndarray,
) -> dict[str, Any]:
    """Build per-fold grouped split diagnostics."""
    train_groups = sorted(set(groups[train_indices]))
    test_groups = sorted(set(groups[test_indices]))
    participant_overlap = sorted(set(train_groups).intersection(test_groups))
    return {
        "fold": fold_index,
        "train_sample_count": int(len(train_indices)),
        "test_sample_count": int(len(test_indices)),
        "train_participants": train_groups,
        "test_participants": test_groups,
        "participant_overlap_count": len(participant_overlap),
        "participant_overlap": participant_overlap,
        "train_class_counts": count_as_string_dict(labels[train_indices]),
        "test_class_counts": count_as_string_dict(labels[test_indices]),
        "train_arm_counts": count_as_string_dict(arms[train_indices]),
        "test_arm_counts": count_as_string_dict(arms[test_indices]),
        "absent_train_classes": absent_classes(labels[train_indices]),
        "absent_test_classes": absent_classes(labels[test_indices]),
    }


def count_as_string_dict(values: np.ndarray) -> dict[str, int]:
    """Count NumPy values and return JSON/CSV-friendly keys."""
    counts = Counter(str(value) for value in values)
    return {key: counts[key] for key in sorted(counts)}


def write_split_manifest(path: Path, rows: list[dict[str, str | int]]) -> None:
    """Write split membership as CSV."""
    if not rows:
        raise SplitError("No split rows were generated")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_diagnostics_csv(path: Path, diagnostics: list[dict[str, Any]]) -> None:
    """Write fold diagnostics with complex values JSON-encoded."""
    fieldnames = [
        "fold",
        "train_sample_count",
        "test_sample_count",
        "train_participants",
        "test_participants",
        "participant_overlap_count",
        "participant_overlap",
        "train_class_counts",
        "test_class_counts",
        "train_arm_counts",
        "test_arm_counts",
        "absent_train_classes",
        "absent_test_classes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in diagnostics:
            writer.writerow(
                {
                    key: json.dumps(value) if isinstance(value, (dict, list)) else value
                    for key, value in row.items()
                }
            )
