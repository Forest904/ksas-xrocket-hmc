"""M3 XROCKET grouped-fold experiment and traceability artifacts."""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ksas_xrocket.audit import EXPECTED_CHANNELS, EXPECTED_LABELS, sha256_file, write_json
from ksas_xrocket.baselines import (
    aggregate_metrics,
    assert_no_group_overlap,
    build_confusion_rows,
    build_prediction_rows,
    compute_metrics,
    estimator_class_labels,
    git_commit,
    indices_for_fold,
    load_processed_arrays,
    package_version,
    predict_probabilities,
    read_split_manifest,
    validate_estimator_classes,
    validate_split_manifest_against_arrays,
    write_dicts_csv,
)
from ksas_xrocket.xrocket_adapter import XRocketAdapter

PRIMARY_MODEL = "xrocket_random_forest"
SENSITIVITY_MODEL = "xrocket_logistic_regression"


class XRocketExperimentError(ValueError):
    """Raised when an M3 experiment cannot be completed safely."""


@dataclass(frozen=True)
class XRocketExperimentResult:
    """Paths returned by the M3 experiment."""

    output_dir: Path
    metrics_path: Path
    aggregate_metrics_path: Path
    predictions_path: Path
    confusion_matrices_path: Path
    runtime_path: Path
    provenance_path: Path


def run_xrocket_experiment(
    *,
    processed_dir: Path,
    splits_dir: Path,
    output_dir: Path,
    folds: tuple[int, ...] | None = None,
    random_state: int = 42,
    label_ids: tuple[int, ...] = tuple(EXPECTED_LABELS),
    batch_size: int = 16,
    feature_cap: int = 10_000,
    kernel_length: int = 9,
    max_dilations: int = 32,
    combination_order: int = 1,
    combination_method: str = "additive",
    nominal_sampling_rate_hz: float = 50.0,
    random_forest_estimators: int = 500,
    logistic_max_iter: int = 2000,
    overwrite: bool = False,
    resolved_config: dict[str, Any] | None = None,
) -> XRocketExperimentResult:
    """Run leak-safe XROCKET training on saved participant-grouped folds."""
    started = time.perf_counter()
    arrays = load_processed_arrays(processed_dir / "tensors.npz")
    split_path = splits_dir / "m2_grouped_folds.csv"
    split_rows = read_split_manifest(split_path)
    validate_split_manifest_against_arrays(split_rows, arrays)
    available_folds = tuple(sorted({int(row["fold"]) for row in split_rows}))
    selected_folds = folds or available_folds
    unknown_folds = sorted(set(selected_folds).difference(available_folds))
    if unknown_folds:
        raise XRocketExperimentError(f"Unknown folds requested: {unknown_folds}")
    if not selected_folds:
        raise XRocketExperimentError("At least one fold must be selected")
    if arrays["X"].shape[1] != len(EXPECTED_CHANNELS):
        raise XRocketExperimentError(
            f"Expected {len(EXPECTED_CHANNELS)} channels, received {arrays['X'].shape[1]}"
        )
    _prepare_output_dir(output_dir, overwrite=overwrite)

    metrics_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    actual_dilations: tuple[int, ...] | None = None
    feature_dims: dict[str, int] | None = None

    for fold in selected_folds:
        fold_started = time.perf_counter()
        fold_dir = output_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True)
        train_indices, test_indices = indices_for_fold(split_rows, fold)
        assert_no_group_overlap(arrays["participant_id"], train_indices, test_indices, fold)

        adapter = XRocketAdapter(
            in_channels=arrays["X"].shape[1],
            max_kernel_span=arrays["X"].shape[2],
            channel_names=EXPECTED_CHANNELS,
            combination_order=combination_order,
            combination_method=combination_method,
            feature_cap=feature_cap,
            kernel_length=kernel_length,
            max_dilations=max_dilations,
            nominal_sampling_rate_hz=nominal_sampling_rate_hz,
        )
        fit_started = time.perf_counter()
        adapter.fit(arrays["X"][train_indices])
        fit_seconds = time.perf_counter() - fit_started

        transform_started = time.perf_counter()
        features = adapter.transform(arrays["X"], batch_size=batch_size)
        transform_seconds = time.perf_counter() - transform_started
        actual_dilations = adapter.dilations
        feature_dims = adapter.feature_dims

        metadata = pd.DataFrame(adapter.feature_metadata())
        metadata_path = fold_dir / "feature_metadata.parquet"
        metadata.to_parquet(metadata_path, index=False)
        reloaded_metadata = pd.read_parquet(metadata_path)
        pd.testing.assert_frame_equal(metadata, reloaded_metadata, check_dtype=True)

        features_path = fold_dir / "features.npz"
        np.savez_compressed(
            features_path,
            features=features.astype(np.float32, copy=False),
            train_indices=train_indices,
            test_indices=test_indices,
        )
        adapter_path = fold_dir / "xrocket_adapter.joblib"
        adapter.save(adapter_path)

        models = _build_models(
            random_state=random_state,
            random_forest_estimators=random_forest_estimators,
            logistic_max_iter=logistic_max_iter,
        )
        padded_predictions: dict[str, np.ndarray] = {}
        for model_name, estimator in models.items():
            classifier_started = time.perf_counter()
            estimator.fit(features[train_indices], arrays["y"][train_indices])
            classifier_seconds = time.perf_counter() - classifier_started
            validate_estimator_classes(estimator, label_ids, model_name, fold)
            y_pred = estimator.predict(features[test_indices])
            padded_predictions[model_name] = y_pred
            probabilities = predict_probabilities(estimator, features[test_indices])
            classes = estimator_class_labels(estimator)
            model_path = fold_dir / f"{model_name}.joblib"
            joblib.dump(estimator, model_path)

            metrics: dict[str, Any] = dict(
                compute_metrics(arrays["y"][test_indices], y_pred, label_ids=label_ids)
            )
            metrics.update(
                {
                    "model": model_name,
                    "fold": fold,
                    "train_sample_count": len(train_indices),
                    "test_sample_count": len(test_indices),
                    "feature_count": features.shape[1],
                    "model_path": model_path.as_posix(),
                }
            )
            metrics_rows.append(metrics)
            prediction_rows.extend(
                build_prediction_rows(
                    model_name=model_name,
                    fold=fold,
                    sample_indices=test_indices,
                    arrays=arrays,
                    y_pred=y_pred,
                    probabilities=probabilities,
                    probability_classes=classes,
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
            runtime_rows.append(
                {
                    "fold": fold,
                    "stage": f"classifier_fit:{model_name}",
                    "seconds": classifier_seconds,
                }
            )

        primary = models[PRIMARY_MODEL]
        importances = np.asarray(primary.feature_importances_, dtype=np.float64)
        if len(importances) != len(metadata):
            raise XRocketExperimentError("Random-forest importance does not align to metadata")
        importance = metadata.copy()
        importance["importance"] = importances
        importance.to_parquet(fold_dir / "feature_importance.parquet", index=False)

        cropped_features = _transform_cropped(
            adapter,
            arrays["X"],
            arrays["original_length"],
            batch_size=batch_size,
        )
        _write_padding_diagnostics(
            fold_dir=fold_dir,
            fold=fold,
            arrays=arrays,
            train_indices=train_indices,
            test_indices=test_indices,
            padded_features=features,
            cropped_features=cropped_features,
            metadata=metadata,
            models=models,
            padded_predictions=padded_predictions,
        )
        _verify_reload(
            adapter_path=adapter_path,
            model_paths={name: fold_dir / f"{name}.joblib" for name in models},
            x=arrays["X"][test_indices[: min(4, len(test_indices))]],
            expected_features=features[test_indices[: min(4, len(test_indices))]],
            expected_predictions={
                name: predictions[: min(4, len(test_indices))]
                for name, predictions in padded_predictions.items()
            },
            batch_size=batch_size,
        )

        runtime_rows.extend(
            [
                {"fold": fold, "stage": "xrocket_fit", "seconds": fit_seconds},
                {"fold": fold, "stage": "xrocket_transform_all", "seconds": transform_seconds},
                {
                    "fold": fold,
                    "stage": "fold_total",
                    "seconds": time.perf_counter() - fold_started,
                },
            ]
        )

    aggregate_rows = [
        aggregate_metrics(
            model_name,
            [row for row in metrics_rows if row["model"] == model_name],
        )
        for model_name in (PRIMARY_MODEL, SENSITIVITY_MODEL)
    ]
    metrics_path = output_dir / "fold_metrics.csv"
    aggregate_metrics_path = output_dir / "aggregate_metrics.csv"
    predictions_path = output_dir / "predictions.csv"
    confusion_matrices_path = output_dir / "confusion_matrices.csv"
    runtime_path = output_dir / "runtime.csv"
    provenance_path = output_dir / "provenance.json"
    write_dicts_csv(metrics_path, metrics_rows)
    write_dicts_csv(aggregate_metrics_path, aggregate_rows)
    write_dicts_csv(predictions_path, prediction_rows)
    write_dicts_csv(confusion_matrices_path, confusion_rows)
    runtime_rows.append(
        {"fold": "all", "stage": "experiment_total", "seconds": time.perf_counter() - started}
    )
    write_dicts_csv(runtime_path, runtime_rows)
    write_json(
        output_dir / "resolved_config.json",
        resolved_config
        or _default_resolved_config(
            processed_dir=processed_dir,
            splits_dir=splits_dir,
            output_dir=output_dir,
            folds=selected_folds,
            random_state=random_state,
            batch_size=batch_size,
            feature_cap=feature_cap,
            kernel_length=kernel_length,
            max_dilations=max_dilations,
            combination_order=combination_order,
            combination_method=combination_method,
            nominal_sampling_rate_hz=nominal_sampling_rate_hz,
            random_forest_estimators=random_forest_estimators,
            logistic_max_iter=logistic_max_iter,
        ),
    )
    write_json(
        provenance_path,
        {
            "processed_dir": processed_dir.as_posix(),
            "splits_dir": splits_dir.as_posix(),
            "output_dir": output_dir.as_posix(),
            "tensor_sha256": sha256_file(processed_dir / "tensors.npz"),
            "split_manifest_sha256": sha256_file(split_path),
            "git_commit": git_commit(),
            "random_state": random_state,
            "folds": list(selected_folds),
            "actual_dilations": list(actual_dilations or ()),
            "feature_dims": feature_dims or {},
            "xrocket_upstream": {
                "repository": "https://github.com/dida-do/xrocket.git",
                "revision": "1511e810c59d0c42f6431ef2f1f9fa57c71e9b2f",
                "license_status": (
                    "Course-authorized academic use; upstream has no public license file"
                ),
            },
            "packages": {
                package: package_version(package)
                for package in (
                    "numpy",
                    "pandas",
                    "pyarrow",
                    "scikit-learn",
                    "torch",
                    "xrocket",
                    "joblib",
                )
            },
        },
    )
    return XRocketExperimentResult(
        output_dir=output_dir,
        metrics_path=metrics_path,
        aggregate_metrics_path=aggregate_metrics_path,
        predictions_path=predictions_path,
        confusion_matrices_path=confusion_matrices_path,
        runtime_path=runtime_path,
        provenance_path=provenance_path,
    )


def _prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise XRocketExperimentError(
                f"Output directory is not empty: {output_dir}; pass --overwrite to replace it"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _build_models(
    *,
    random_state: int,
    random_forest_estimators: int,
    logistic_max_iter: int,
) -> dict[str, Any]:
    if random_forest_estimators <= 0:
        raise XRocketExperimentError("random_forest_estimators must be positive")
    if logistic_max_iter <= 0:
        raise XRocketExperimentError("logistic_max_iter must be positive")
    return {
        PRIMARY_MODEL: RandomForestClassifier(
            n_estimators=random_forest_estimators,
            class_weight="balanced",
            max_features="sqrt",
            random_state=random_state,
            n_jobs=-1,
        ),
        SENSITIVITY_MODEL: Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=logistic_max_iter,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def _transform_cropped(
    adapter: XRocketAdapter,
    x: np.ndarray,
    original_lengths: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    cropped = np.empty((len(x), adapter.num_features), dtype=np.float32)
    for length_value in np.unique(original_lengths):
        length = int(length_value)
        indices = np.flatnonzero(original_lengths == length)
        cropped[indices] = adapter.transform(
            x[indices, :, :length],
            batch_size=batch_size,
        )
    return cropped


def _write_padding_diagnostics(
    *,
    fold_dir: Path,
    fold: int,
    arrays: dict[str, np.ndarray],
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    padded_features: np.ndarray,
    cropped_features: np.ndarray,
    metadata: pd.DataFrame,
    models: dict[str, Any],
    padded_predictions: dict[str, np.ndarray],
) -> None:
    split_by_index = np.full(len(arrays["y"]), "train", dtype="<U5")
    split_by_index[test_indices] = "test"
    selected = np.concatenate([train_indices, test_indices])
    detail_rows: list[dict[str, Any]] = []
    for dilation, group in metadata.groupby("dilation", sort=False):
        feature_indices = group["feature_index"].to_numpy(dtype=np.int64)
        delta = np.abs(padded_features[:, feature_indices] - cropped_features[:, feature_indices])
        receptive_field = int(group["effective_receptive_field_samples"].iloc[0])
        for sample_index in selected:
            original_length = int(arrays["original_length"][sample_index])
            detail_rows.append(
                {
                    "fold": fold,
                    "split": split_by_index[sample_index],
                    "sample_index": int(sample_index),
                    "sample_id": str(arrays["sample_id"][sample_index]),
                    "original_length": original_length,
                    "padding_fraction": 1.0 - original_length / arrays["X"].shape[2],
                    "dilation": int(dilation),
                    "effective_receptive_field_samples": receptive_field,
                    "receptive_field_exceeds_original_length": (receptive_field > original_length),
                    "mean_absolute_feature_delta": float(np.mean(delta[sample_index])),
                    "max_absolute_feature_delta": float(np.max(delta[sample_index])),
                }
            )
    write_dicts_csv(fold_dir / "padding_feature_diagnostics.csv", detail_rows)
    detail = pd.DataFrame(detail_rows)
    summary_rows: list[dict[str, Any]] = []
    for (split_name, dilation), group in detail.groupby(
        ["split", "dilation"],
        sort=False,
    ):
        padding = group["padding_fraction"].to_numpy(dtype=np.float64)
        delta = group["mean_absolute_feature_delta"].to_numpy(dtype=np.float64)
        correlation = (
            float(np.corrcoef(padding, delta)[0, 1])
            if len(group) > 1 and np.std(padding) > 0 and np.std(delta) > 0
            else float("nan")
        )
        summary_rows.append(
            {
                "fold": fold,
                "split": split_name,
                "dilation": int(dilation),
                "sample_count": len(group),
                "effective_receptive_field_samples": int(
                    group["effective_receptive_field_samples"].iloc[0]
                ),
                "mean_padding_fraction": float(np.mean(padding)),
                "mean_absolute_feature_delta": float(np.mean(delta)),
                "median_absolute_feature_delta": float(np.median(delta)),
                "max_absolute_feature_delta": float(np.max(delta)),
                "padding_delta_correlation": correlation,
            }
        )
    write_dicts_csv(fold_dir / "padding_feature_summary.csv", summary_rows)

    threshold_rows: list[dict[str, Any]] = []
    for dilation, group in metadata.groupby("dilation", sort=False):
        values = group["threshold"].to_numpy(dtype=np.float64)
        threshold_rows.append(
            {
                "fold": fold,
                "dilation": int(dilation),
                "threshold_count": len(values),
                "zero_threshold_fraction": float(np.mean(values == 0.0)),
                "near_zero_threshold_fraction": float(np.mean(np.abs(values) <= 1e-8)),
                "threshold_min": float(np.min(values)),
                "threshold_median": float(np.median(values)),
                "threshold_max": float(np.max(values)),
            }
        )
    write_dicts_csv(fold_dir / "padding_threshold_diagnostics.csv", threshold_rows)

    prediction_rows: list[dict[str, Any]] = []
    for model_name, estimator in models.items():
        cropped_predictions = estimator.predict(cropped_features[test_indices])
        padded = padded_predictions[model_name]
        prediction_rows.append(
            {
                "fold": fold,
                "model": model_name,
                "test_sample_count": len(test_indices),
                "prediction_change_count": int(np.sum(cropped_predictions != padded)),
                "prediction_change_fraction": float(np.mean(cropped_predictions != padded)),
            }
        )
    write_dicts_csv(fold_dir / "padding_prediction_diagnostics.csv", prediction_rows)


def _verify_reload(
    *,
    adapter_path: Path,
    model_paths: dict[str, Path],
    x: np.ndarray,
    expected_features: np.ndarray,
    expected_predictions: dict[str, np.ndarray],
    batch_size: int,
) -> None:
    loaded_adapter = XRocketAdapter.load(adapter_path)
    reloaded_features = loaded_adapter.transform(x, batch_size=batch_size)
    if not np.array_equal(reloaded_features, expected_features):
        raise XRocketExperimentError("Reloaded adapter changed transformed features")
    for model_name, path in model_paths.items():
        estimator = joblib.load(path)
        predictions = estimator.predict(reloaded_features)
        if not np.array_equal(predictions, expected_predictions[model_name]):
            raise XRocketExperimentError(f"Reloaded {model_name} changed predictions")


def _default_resolved_config(
    *,
    processed_dir: Path,
    splits_dir: Path,
    output_dir: Path,
    folds: tuple[int, ...],
    random_state: int,
    batch_size: int,
    feature_cap: int,
    kernel_length: int,
    max_dilations: int,
    combination_order: int,
    combination_method: str,
    nominal_sampling_rate_hz: float,
    random_forest_estimators: int,
    logistic_max_iter: int,
) -> dict[str, Any]:
    return {
        "processed_dir": processed_dir.as_posix(),
        "splits_dir": splits_dir.as_posix(),
        "output_dir": output_dir.as_posix(),
        "folds": list(folds),
        "classes": list(EXPECTED_LABELS),
        "random_state": random_state,
        "batch_size": batch_size,
        "xrocket": {
            "feature_cap": feature_cap,
            "kernel_length": kernel_length,
            "max_dilations": max_dilations,
            "combination_order": combination_order,
            "combination_method": combination_method,
            "nominal_sampling_rate_hz": nominal_sampling_rate_hz,
        },
        "classifiers": {
            PRIMARY_MODEL: {"n_estimators": random_forest_estimators},
            SENSITIVITY_MODEL: {"max_iter": logistic_max_iter},
        },
    }
