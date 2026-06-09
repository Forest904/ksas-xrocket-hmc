"""M2 statistical baseline training on fixed participant-grouped folds."""

from __future__ import annotations

import csv
import platform
import subprocess
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ksas_xrocket.audit import EXPECTED_LABELS, sha256_file, write_json

BASELINE_MODELS = ("majority", "statistical_logistic_regression", "statistical_random_forest")
STAT_NAMES = ("mean", "std", "min", "max", "median", "q25", "q75")
REQUIRED_SPLIT_COLUMNS = (
    "fold",
    "split",
    "sample_index",
    "sample_id",
    "movement_label_id",
    "participant_id",
    "arm_code",
)


class BaselineError(ValueError):
    """Raised when M2 baseline training cannot proceed."""


@dataclass(frozen=True)
class BaselineResult:
    """Paths returned by baseline training."""

    output_dir: Path
    metrics_path: Path
    aggregate_metrics_path: Path
    predictions_path: Path
    confusion_matrices_path: Path
    provenance_path: Path


def run_baselines(
    processed_dir: Path,
    splits_dir: Path,
    output_dir: Path,
    random_state: int = 42,
    label_ids: tuple[int, ...] = tuple(EXPECTED_LABELS),
    resolved_config: dict[str, Any] | None = None,
) -> BaselineResult:
    """Train baseline models on the saved M2 grouped folds."""
    arrays = load_processed_arrays(processed_dir / "tensors.npz")
    split_rows = read_split_manifest(splits_dir / "m2_grouped_folds.csv")
    validate_split_manifest_against_arrays(split_rows, arrays)
    folds = sorted({int(row["fold"]) for row in split_rows})
    features = extract_statistical_features(arrays["X"], arrays["valid_mask"])

    metrics_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []

    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    for model_name in BASELINE_MODELS:
        fold_metric_rows: list[dict[str, Any]] = []
        for fold in folds:
            train_indices, test_indices = indices_for_fold(split_rows, fold)
            assert_no_group_overlap(arrays["participant_id"], train_indices, test_indices, fold)
            estimator = build_estimator(model_name, random_state)
            estimator.fit(features[train_indices], arrays["y"][train_indices])
            validate_estimator_classes(estimator, label_ids, model_name, fold)
            y_pred = estimator.predict(features[test_indices])
            probabilities = predict_probabilities(estimator, features[test_indices])
            estimator_classes = estimator_class_labels(estimator)

            model_path = models_dir / f"{model_name}_fold_{fold}.joblib"
            joblib.dump(estimator, model_path)

            fold_metrics: dict[str, Any] = compute_metrics(
                arrays["y"][test_indices],
                y_pred,
                label_ids=label_ids,
            )
            fold_metrics.update(
                {
                    "model": model_name,
                    "fold": fold,
                    "train_sample_count": int(len(train_indices)),
                    "test_sample_count": int(len(test_indices)),
                    "model_path": model_path.as_posix(),
                }
            )
            metrics_rows.append(fold_metrics)
            fold_metric_rows.append(fold_metrics)

            prediction_rows.extend(
                build_prediction_rows(
                    model_name=model_name,
                    fold=fold,
                    sample_indices=test_indices,
                    arrays=arrays,
                    y_pred=y_pred,
                    probabilities=probabilities,
                    probability_classes=estimator_classes,
                    label_ids=label_ids,
                )
            )
            confusion_rows.extend(
                build_confusion_rows(
                    model_name=model_name,
                    fold=fold,
                    y_true=arrays["y"][test_indices],
                    y_pred=y_pred,
                    label_ids=label_ids,
                )
            )

        aggregate_rows.append(aggregate_metrics(model_name, fold_metric_rows))

    metrics_path = output_dir / "fold_metrics.csv"
    aggregate_metrics_path = output_dir / "aggregate_metrics.csv"
    predictions_path = output_dir / "predictions.csv"
    confusion_matrices_path = output_dir / "confusion_matrices.csv"
    provenance_path = output_dir / "provenance.json"
    resolved_config_path = output_dir / "resolved_config.json"

    write_dicts_csv(metrics_path, metrics_rows)
    write_dicts_csv(aggregate_metrics_path, aggregate_rows)
    write_dicts_csv(predictions_path, prediction_rows)
    write_dicts_csv(confusion_matrices_path, confusion_rows)
    write_json(
        resolved_config_path,
        resolved_config
        or {
            "processed_dir": processed_dir.as_posix(),
            "splits_dir": splits_dir.as_posix(),
            "output_dir": output_dir.as_posix(),
            "models": list(BASELINE_MODELS),
            "feature_statistics": list(STAT_NAMES),
            "random_state": random_state,
            "classes": list(label_ids),
        },
    )
    write_json(
        provenance_path,
        build_baseline_provenance(
            processed_dir=processed_dir,
            splits_dir=splits_dir,
            output_dir=output_dir,
            random_state=random_state,
            label_ids=label_ids,
        ),
    )

    return BaselineResult(
        output_dir=output_dir,
        metrics_path=metrics_path,
        aggregate_metrics_path=aggregate_metrics_path,
        predictions_path=predictions_path,
        confusion_matrices_path=confusion_matrices_path,
        provenance_path=provenance_path,
    )


def load_processed_arrays(tensor_path: Path) -> dict[str, np.ndarray]:
    """Load M2 tensors and metadata arrays from disk."""
    if not tensor_path.is_file():
        raise BaselineError(f"Processed tensor file not found: {tensor_path}")
    with np.load(tensor_path, allow_pickle=False) as data:
        required = {
            "X",
            "y",
            "valid_mask",
            "sample_id",
            "participant_id",
            "arm_code",
            "original_length",
        }
        missing = sorted(required.difference(data.files))
        if missing:
            raise BaselineError(f"{tensor_path} is missing arrays: {', '.join(missing)}")
        return {name: data[name] for name in required}


def read_split_manifest(path: Path) -> list[dict[str, str]]:
    """Read grouped fold membership."""
    if not path.is_file():
        raise BaselineError(f"Split manifest not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise BaselineError(f"Split manifest has no header: {path}")
        missing = sorted(set(REQUIRED_SPLIT_COLUMNS).difference(reader.fieldnames))
        if missing:
            raise BaselineError(
                f"Split manifest {path} is missing required columns: {', '.join(missing)}"
            )
        rows = list(reader)
    if not rows:
        raise BaselineError(f"Split manifest contains no rows: {path}")
    return rows


def validate_split_manifest_against_arrays(
    split_rows: list[dict[str, str]],
    arrays: dict[str, np.ndarray],
) -> None:
    """Validate saved split rows against tensor metadata before training."""
    sample_count = len(arrays["y"])
    folds = sorted({int(row["fold"]) for row in split_rows})
    for fold in folds:
        seen: dict[int, str] = {}
        for row in split_rows:
            if int(row["fold"]) != fold:
                continue
            split_name = row["split"]
            if split_name not in {"train", "test"}:
                raise BaselineError(
                    f"Fold {fold} sample {row['sample_index']} has invalid split {split_name!r}"
                )
            sample_index = int(row["sample_index"])
            if sample_index < 0 or sample_index >= sample_count:
                raise BaselineError(
                    f"Fold {fold} sample_index {sample_index} is outside tensor range "
                    f"0..{sample_count - 1}"
                )
            if sample_index in seen:
                raise BaselineError(
                    f"Fold {fold} sample_index {sample_index} appears in both "
                    f"{seen[sample_index]!r} and {split_name!r}"
                )
            seen[sample_index] = split_name
            validate_split_row_matches_arrays(row, arrays, sample_index, fold)
        missing_indices = sorted(set(range(sample_count)).difference(seen))
        if missing_indices:
            preview = ", ".join(str(index) for index in missing_indices[:10])
            raise BaselineError(
                f"Fold {fold} does not assign every sample; missing sample_index {preview}"
            )


def validate_split_row_matches_arrays(
    row: dict[str, str],
    arrays: dict[str, np.ndarray],
    sample_index: int,
    fold: int,
) -> None:
    """Validate one split row against tensor metadata arrays."""
    expected = {
        "sample_id": str(arrays["sample_id"][sample_index]),
        "movement_label_id": str(int(arrays["y"][sample_index])),
        "participant_id": str(arrays["participant_id"][sample_index]),
        "arm_code": str(arrays["arm_code"][sample_index]),
    }
    for field, expected_value in expected.items():
        if row[field] != expected_value:
            raise BaselineError(
                f"Fold {fold} sample_index {sample_index} {field} mismatch: "
                f"split={row[field]!r}, tensor={expected_value!r}"
            )


def extract_statistical_features(x: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Extract per-channel statistics while ignoring padded timesteps."""
    feature_rows: list[list[float]] = []
    for sample_index in range(x.shape[0]):
        valid = valid_mask[sample_index]
        if not valid.any():
            raise BaselineError(f"Sample {sample_index} has no valid timesteps")
        sample_features: list[float] = []
        for channel_index in range(x.shape[1]):
            values = x[sample_index, channel_index, valid]
            sample_features.extend(
                [
                    float(np.mean(values)),
                    float(np.std(values)),
                    float(np.min(values)),
                    float(np.max(values)),
                    float(np.median(values)),
                    float(np.quantile(values, 0.25)),
                    float(np.quantile(values, 0.75)),
                ]
            )
        feature_rows.append(sample_features)
    return np.asarray(feature_rows, dtype=np.float32)


def indices_for_fold(split_rows: list[dict[str, str]], fold: int) -> tuple[np.ndarray, np.ndarray]:
    """Return train and test sample indices for one fold."""
    train = [
        int(row["sample_index"])
        for row in split_rows
        if int(row["fold"]) == fold and row["split"] == "train"
    ]
    test = [
        int(row["sample_index"])
        for row in split_rows
        if int(row["fold"]) == fold and row["split"] == "test"
    ]
    if not train or not test:
        raise BaselineError(f"Fold {fold} is missing train or test samples")
    return np.asarray(train, dtype=np.int64), np.asarray(test, dtype=np.int64)


def assert_no_group_overlap(
    participant_ids: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    fold: int,
) -> None:
    """Fail if a participant appears in both train and test."""
    overlap = set(participant_ids[train_indices]).intersection(participant_ids[test_indices])
    if overlap:
        raise BaselineError(f"Fold {fold} participant overlap: {', '.join(sorted(overlap))}")


def build_estimator(model_name: str, random_state: int) -> Any:
    """Create one baseline estimator."""
    if model_name == "majority":
        return DummyClassifier(strategy="most_frequent")
    if model_name == "statistical_logistic_regression":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(max_iter=2000, random_state=random_state),
                ),
            ]
        )
    if model_name == "statistical_random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            random_state=random_state,
            class_weight="balanced",
        )
    raise BaselineError(f"Unknown baseline model: {model_name}")


def validate_estimator_classes(
    estimator: Any,
    label_ids: tuple[int, ...],
    model_name: str,
    fold: int,
) -> None:
    """Fail if an estimator was not trained with all configured classes."""
    classes = estimator_class_labels(estimator)
    missing = sorted(set(label_ids).difference(classes))
    if missing:
        raise BaselineError(
            f"{model_name} fold {fold} estimator is missing configured classes: {missing}"
        )


def estimator_class_labels(estimator: Any) -> tuple[int, ...]:
    """Return estimator class labels as integers."""
    classes = getattr(estimator, "classes_", None)
    if classes is None and hasattr(estimator, "named_steps"):
        classifier = estimator.named_steps.get("classifier")
        classes = getattr(classifier, "classes_", None)
    if classes is None:
        raise BaselineError("Estimator does not expose classes_ after fitting")
    return tuple(int(label) for label in classes)


def predict_probabilities(estimator: Any, features: np.ndarray) -> np.ndarray | None:
    """Return class probabilities when the estimator supports them."""
    if hasattr(estimator, "predict_proba"):
        return cast(np.ndarray, estimator.predict_proba(features))
    return None


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_ids: tuple[int, ...] = tuple(EXPECTED_LABELS),
) -> dict[str, float]:
    """Compute required fold metrics."""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(label_ids),
        zero_division=0,
    )
    metrics: dict[str, float] = {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }
    for position, label_id in enumerate(label_ids):
        metrics[f"class_{label_id}_precision"] = float(precision[position])
        metrics[f"class_{label_id}_recall"] = float(recall[position])
        metrics[f"class_{label_id}_f1"] = float(f1[position])
        metrics[f"class_{label_id}_support"] = float(support[position])
    return metrics


def build_prediction_rows(
    model_name: str,
    fold: int,
    sample_indices: np.ndarray,
    arrays: dict[str, np.ndarray],
    y_pred: np.ndarray,
    probabilities: np.ndarray | None,
    probability_classes: tuple[int, ...],
    label_ids: tuple[int, ...] = tuple(EXPECTED_LABELS),
) -> list[dict[str, Any]]:
    """Build one prediction row per test sample."""
    rows: list[dict[str, Any]] = []
    for local_index, sample_index in enumerate(sample_indices):
        row: dict[str, Any] = {
            "model": model_name,
            "fold": fold,
            "sample_index": int(sample_index),
            "sample_id": str(arrays["sample_id"][sample_index]),
            "participant_id": str(arrays["participant_id"][sample_index]),
            "arm_code": str(arrays["arm_code"][sample_index]),
            "y_true": int(arrays["y"][sample_index]),
            "y_pred": int(y_pred[local_index]),
        }
        if probabilities is not None:
            probability_positions = {
                label_id: position for position, label_id in enumerate(probability_classes)
            }
            for label_id in label_ids:
                if label_id not in probability_positions:
                    raise BaselineError(
                        f"Probability output is missing configured class {label_id}"
                    )
                row[f"prob_class_{label_id}"] = float(
                    probabilities[local_index, probability_positions[label_id]]
                )
        rows.append(row)
    return rows


def build_confusion_rows(
    model_name: str,
    fold: int,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_ids: tuple[int, ...] = tuple(EXPECTED_LABELS),
) -> list[dict[str, Any]]:
    """Build long-form confusion matrix rows."""
    matrix = confusion_matrix(y_true, y_pred, labels=list(label_ids))
    rows: list[dict[str, Any]] = []
    for true_position, true_label in enumerate(label_ids):
        for pred_position, pred_label in enumerate(label_ids):
            rows.append(
                {
                    "model": model_name,
                    "fold": fold,
                    "true_label": true_label,
                    "predicted_label": pred_label,
                    "count": int(matrix[true_position, pred_position]),
                }
            )
    return rows


def aggregate_metrics(model_name: str, fold_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate fold metrics with mean and standard deviation."""
    if not fold_rows:
        raise BaselineError(f"No fold metrics to aggregate for {model_name}")
    aggregate: dict[str, Any] = {"model": model_name, "fold_count": len(fold_rows)}
    metric_names = [
        key
        for key, value in fold_rows[0].items()
        if isinstance(value, float) and not key.endswith("_support")
    ]
    for metric_name in metric_names:
        values = np.asarray([row[metric_name] for row in fold_rows], dtype=np.float64)
        aggregate[f"{metric_name}_mean"] = float(np.mean(values))
        aggregate[f"{metric_name}_std"] = float(np.std(values, ddof=0))
    return aggregate


def write_dicts_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write heterogeneous dictionaries to CSV with unioned fieldnames."""
    if not rows:
        path.write_text("", encoding="utf-8", newline="")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_baseline_provenance(
    processed_dir: Path,
    splits_dir: Path,
    output_dir: Path,
    random_state: int,
    label_ids: tuple[int, ...],
) -> dict[str, Any]:
    """Return baseline provenance metadata."""
    return {
        "processed_dir": processed_dir.as_posix(),
        "splits_dir": splits_dir.as_posix(),
        "output_dir": output_dir.as_posix(),
        "random_state": random_state,
        "classes": list(label_ids),
        "tensor_sha256": sha256_file(processed_dir / "tensors.npz"),
        "split_manifest_sha256": sha256_file(splits_dir / "m2_grouped_folds.csv"),
        "git_commit": git_commit(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                package: package_version(package) for package in ("numpy", "scikit-learn", "joblib")
            },
        },
    }


def git_commit() -> str | None:
    """Return the current Git commit SHA when available."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def package_version(package: str) -> str | None:
    """Return an installed package version."""
    try:
        return version(package)
    except PackageNotFoundError:
        return None
