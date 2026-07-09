"""M7 robustness, confound, and negative-control analyses."""

from __future__ import annotations

import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ksas_xrocket.audit import EXPECTED_LABELS, sha256_file, write_json
from ksas_xrocket.baselines import (
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
from ksas_xrocket.task_1_1_explain import parse_channel_name
from ksas_xrocket.task_1_2_explain import BIN_RULE_DESCRIPTION, temporal_scale_bin
from ksas_xrocket.xrocket_experiment import PRIMARY_MODEL, SENSITIVITY_MODEL

DEFAULT_SEEDS = (7, 13, 29, 42, 101)
DEFAULT_WARNING_THRESHOLDS = {
    "label_shuffle_mean_macro_f1": 0.25,
    "label_shuffle_max_macro_f1": 0.40,
    "metadata_mean_macro_f1": 0.30,
    "rank_correlation_min": 0.75,
    "class_recall_min": 0.80,
}
EXPLANATION_GROUP_LEVELS = (
    "sensor_family",
    "axis",
    "channel",
    "dilation",
    "temporal_scale_bin",
)


class RobustnessError(ValueError):
    """Raised when M7 robustness analysis cannot complete safely."""


@dataclass(frozen=True)
class RobustnessResult:
    """Paths returned by the M7 workflow."""

    controls_output_dir: Path
    stability_output_dir: Path
    controls_summary_path: Path
    stability_summary_path: Path
    control_flags_path: Path


def shuffled_training_labels(y: np.ndarray, train_indices: np.ndarray, *, seed: int) -> np.ndarray:
    """Return a deterministic permutation of training labels only."""
    rng = np.random.default_rng(seed)
    return rng.permutation(y[train_indices])


def build_metadata_feature_frame(arrays: dict[str, np.ndarray]) -> pd.DataFrame:
    """Build length and arm metadata features from processed tensor metadata."""
    original_length = arrays["original_length"].astype(float)
    target_length = float(arrays["X"].shape[2])
    return pd.DataFrame(
        {
            "original_length": original_length,
            "padding_fraction": 1.0 - original_length / target_length,
            "arm_code": arrays["arm_code"].astype(str),
        }
    )


def summarize_ranked_values(frame: pd.DataFrame, *, value_column: str) -> pd.DataFrame:
    """Summarize grouped values with mean, variation, and rank stability."""
    required = {"group_level", "group_value", value_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RobustnessError(f"Rank summary frame is missing columns: {', '.join(missing)}")
    index_columns = [column for column in ("seed", "fold", "arm_code") if column in frame.columns]
    if not index_columns:
        raise RobustnessError("Rank summary frame must include seed/fold or arm_code columns")
    ranked = frame.copy()
    ranked["rank"] = ranked.groupby(index_columns + ["group_level"])[value_column].rank(
        method="average",
        ascending=False,
    )
    rows: list[dict[str, Any]] = []
    for (group_level, group_value), group in ranked.groupby(["group_level", "group_value"]):
        values = group[value_column].to_numpy(dtype=np.float64)
        ranks = group["rank"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "group_level": group_level,
                "group_value": group_value,
                "case_count": int(len(group)),
                f"{value_column}_mean": float(np.mean(values)),
                f"{value_column}_std": float(np.std(values, ddof=0)),
                f"{value_column}_min": float(np.min(values)),
                f"{value_column}_max": float(np.max(values)),
                "mean_rank": float(np.mean(ranks)),
                "rank_std": float(np.std(ranks, ddof=0)),
            }
        )
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    return summary.sort_values(["group_level", f"{value_column}_mean"], ascending=[True, False])


def run_robustness_analysis(
    *,
    processed_dir: Path,
    splits_dir: Path,
    xrocket_dir: Path,
    controls_output_dir: Path,
    stability_output_dir: Path,
    label_ids: tuple[int, ...] = tuple(EXPECTED_LABELS),
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    primary_model: str = PRIMARY_MODEL,
    random_forest_estimators: int = 500,
    logistic_max_iter: int = 2000,
    warning_thresholds: dict[str, float] | None = None,
    overwrite: bool = False,
    resolved_config: dict[str, Any] | None = None,
) -> RobustnessResult:
    """Run M7 controls and stability analysis from saved grouped-fold artifacts."""
    if not seeds:
        raise RobustnessError("At least one seed is required")
    thresholds = {**DEFAULT_WARNING_THRESHOLDS, **(warning_thresholds or {})}
    arrays = load_processed_arrays(processed_dir / "tensors.npz")
    split_path = splits_dir / "m2_grouped_folds.csv"
    split_rows = read_split_manifest(split_path)
    validate_split_manifest_against_arrays(split_rows, arrays)
    folds = sorted({int(row["fold"]) for row in split_rows})
    if not folds:
        raise RobustnessError("No folds found in split manifest")
    _validate_xrocket_artifacts(xrocket_dir, folds)
    _prepare_output_dir(controls_output_dir, overwrite=overwrite)
    _prepare_output_dir(stability_output_dir, overwrite=overwrite)

    leakage_checks = _build_leakage_checks(split_rows, arrays, folds, label_ids)
    write_dicts_csv(controls_output_dir / "leakage_checks.csv", leakage_checks)

    label_shuffle = _run_label_shuffle_controls(
        xrocket_dir=xrocket_dir,
        folds=folds,
        arrays=arrays,
        split_rows=split_rows,
        label_ids=label_ids,
        seeds=seeds,
        random_forest_estimators=random_forest_estimators,
        logistic_max_iter=logistic_max_iter,
    )
    label_shuffle["metrics"].to_csv(controls_output_dir / "label_shuffle_metrics.csv", index=False)
    label_shuffle["aggregate"].to_csv(
        controls_output_dir / "label_shuffle_aggregate_metrics.csv",
        index=False,
    )
    label_shuffle["predictions"].to_csv(
        controls_output_dir / "label_shuffle_predictions.csv",
        index=False,
    )

    metadata = _run_metadata_controls(
        arrays=arrays,
        split_rows=split_rows,
        folds=folds,
        label_ids=label_ids,
        seeds=seeds,
        random_forest_estimators=random_forest_estimators,
        logistic_max_iter=logistic_max_iter,
    )
    metadata["metrics"].to_csv(controls_output_dir / "metadata_baseline_metrics.csv", index=False)
    metadata["aggregate"].to_csv(
        controls_output_dir / "metadata_baseline_aggregate_metrics.csv",
        index=False,
    )
    metadata["predictions"].to_csv(
        controls_output_dir / "metadata_baseline_predictions.csv",
        index=False,
    )

    confounds = _build_confound_summary(arrays, label_ids)
    confounds.to_csv(controls_output_dir / "confound_summary.csv", index=False)

    stability = _run_seed_stability(
        xrocket_dir=xrocket_dir,
        folds=folds,
        arrays=arrays,
        split_rows=split_rows,
        label_ids=label_ids,
        seeds=seeds,
        random_forest_estimators=random_forest_estimators,
        logistic_max_iter=logistic_max_iter,
    )
    stability["metrics"].to_csv(stability_output_dir / "seed_fold_metrics.csv", index=False)
    stability["metric_summary"].to_csv(
        stability_output_dir / "metric_stability_summary.csv",
        index=False,
    )
    stability["importance"].to_parquet(
        stability_output_dir / "explanation_seed_importance.parquet",
        index=False,
    )
    stability["importance_summary"].to_csv(
        stability_output_dir / "explanation_stability_summary.csv",
        index=False,
    )
    stability["rank_correlations"].to_csv(
        stability_output_dir / "rank_correlations.csv",
        index=False,
    )
    stability["topk_overlap"].to_csv(stability_output_dir / "topk_overlap.csv", index=False)

    arm_rankings = _run_arm_axis_rankings(
        xrocket_dir=xrocket_dir,
        folds=folds,
        arrays=arrays,
        label_ids=label_ids,
        seed=seeds[0],
        random_forest_estimators=random_forest_estimators,
    )
    arm_rankings["rankings"].to_csv(stability_output_dir / "arm_axis_rankings.csv", index=False)
    arm_rankings["comparison"].to_csv(stability_output_dir / "arm_rank_comparison.csv", index=False)

    per_class_errors = _build_per_class_error_summary(
        xrocket_dir=xrocket_dir,
        primary_model=primary_model,
        label_ids=label_ids,
    )
    per_class_errors.to_csv(stability_output_dir / "per_class_error_summary.csv", index=False)
    error_review_path = stability_output_dir / "confusion_error_review.md"
    error_review_path.write_text(
        _build_confusion_error_review(per_class_errors),
        encoding="utf-8",
    )

    flags = _build_control_flags(
        label_shuffle=label_shuffle["aggregate"],
        metadata=metadata["aggregate"],
        stability=stability,
        per_class_errors=per_class_errors,
        thresholds=thresholds,
    )
    control_flags_path = controls_output_dir / "control_flags.csv"
    write_dicts_csv(control_flags_path, flags)

    controls_summary_path = controls_output_dir / "m7_controls_summary.md"
    controls_summary_path.write_text(
        _build_controls_summary(
            label_shuffle=label_shuffle["aggregate"],
            metadata=metadata["aggregate"],
            confounds=confounds,
            flags=pd.DataFrame(flags),
        ),
        encoding="utf-8",
    )
    stability_summary_path = stability_output_dir / "m7_stability_summary.md"
    stability_summary_path.write_text(
        _build_stability_summary(
            metric_summary=stability["metric_summary"],
            importance_summary=stability["importance_summary"],
            arm_comparison=arm_rankings["comparison"],
            per_class_errors=per_class_errors,
            flags=pd.DataFrame(flags),
        ),
        encoding="utf-8",
    )

    resolved = resolved_config or _default_resolved_config(
        processed_dir=processed_dir,
        splits_dir=splits_dir,
        xrocket_dir=xrocket_dir,
        controls_output_dir=controls_output_dir,
        stability_output_dir=stability_output_dir,
        label_ids=label_ids,
        seeds=seeds,
        primary_model=primary_model,
        random_forest_estimators=random_forest_estimators,
        logistic_max_iter=logistic_max_iter,
        warning_thresholds=thresholds,
    )
    write_json(controls_output_dir / "resolved_config.json", resolved)
    write_json(stability_output_dir / "resolved_config.json", resolved)
    provenance = _build_provenance(
        processed_dir=processed_dir,
        split_path=split_path,
        xrocket_dir=xrocket_dir,
        controls_output_dir=controls_output_dir,
        stability_output_dir=stability_output_dir,
        folds=folds,
        seeds=seeds,
    )
    write_json(controls_output_dir / "provenance.json", provenance)
    write_json(stability_output_dir / "provenance.json", provenance)
    return RobustnessResult(
        controls_output_dir=controls_output_dir,
        stability_output_dir=stability_output_dir,
        controls_summary_path=controls_summary_path,
        stability_summary_path=stability_summary_path,
        control_flags_path=control_flags_path,
    )


def _prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise RobustnessError(
                f"Output directory is not empty: {output_dir}; pass --overwrite to replace it"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _validate_xrocket_artifacts(xrocket_dir: Path, folds: list[int]) -> None:
    if not xrocket_dir.is_dir():
        raise RobustnessError(f"M3 XROCKET directory not found: {xrocket_dir}")
    for fold in folds:
        fold_dir = xrocket_dir / f"fold_{fold}"
        missing = [
            path.name
            for path in (
                fold_dir / "features.npz",
                fold_dir / "feature_metadata.parquet",
                fold_dir / f"{PRIMARY_MODEL}.joblib",
                fold_dir / f"{SENSITIVITY_MODEL}.joblib",
            )
            if not path.is_file()
        ]
        if missing:
            raise RobustnessError(
                f"Fold {fold} is missing required M3 artifacts: {', '.join(missing)}"
            )


def _run_label_shuffle_controls(
    *,
    xrocket_dir: Path,
    folds: list[int],
    arrays: dict[str, np.ndarray],
    split_rows: list[dict[str, str]],
    label_ids: tuple[int, ...],
    seeds: tuple[int, ...],
    random_forest_estimators: int,
    logistic_max_iter: int,
) -> dict[str, pd.DataFrame]:
    metrics_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for seed in seeds:
        for fold in folds:
            fold_dir = xrocket_dir / f"fold_{fold}"
            features, train_indices, test_indices = _load_fold_features(fold_dir)
            assert_no_group_overlap(arrays["participant_id"], train_indices, test_indices, fold)
            y_shuffled = shuffled_training_labels(arrays["y"], train_indices, seed=seed + fold)
            models = _build_xrocket_feature_models(
                random_state=seed,
                random_forest_estimators=random_forest_estimators,
                logistic_max_iter=logistic_max_iter,
            )
            for model_name, estimator in models.items():
                control_name = f"{model_name}_label_shuffle"
                estimator.fit(features[train_indices], y_shuffled)
                validate_estimator_classes(estimator, label_ids, control_name, fold)
                y_pred = estimator.predict(features[test_indices])
                probabilities = predict_probabilities(estimator, features[test_indices])
                classes = estimator_class_labels(estimator)
                row: dict[str, Any] = compute_metrics(
                    arrays["y"][test_indices],
                    y_pred,
                    label_ids=label_ids,
                )
                row.update(
                    {
                        "control": "label_shuffle",
                        "model": control_name,
                        "seed": seed,
                        "fold": fold,
                        "train_sample_count": int(len(train_indices)),
                        "test_sample_count": int(len(test_indices)),
                    }
                )
                metrics_rows.append(row)
                rows = build_prediction_rows(
                    model_name=control_name,
                    fold=fold,
                    sample_indices=test_indices,
                    arrays=arrays,
                    y_pred=y_pred,
                    probabilities=probabilities,
                    probability_classes=classes,
                    label_ids=label_ids,
                )
                for prediction in rows:
                    prediction["seed"] = seed
                    prediction["control"] = "label_shuffle"
                prediction_rows.extend(rows)
    metrics = pd.DataFrame(metrics_rows)
    return {
        "metrics": metrics,
        "aggregate": _aggregate_metrics_frame(metrics, ["control", "model"]),
        "predictions": pd.DataFrame(prediction_rows),
    }


def _run_metadata_controls(
    *,
    arrays: dict[str, np.ndarray],
    split_rows: list[dict[str, str]],
    folds: list[int],
    label_ids: tuple[int, ...],
    seeds: tuple[int, ...],
    random_forest_estimators: int,
    logistic_max_iter: int,
) -> dict[str, pd.DataFrame]:
    features = build_metadata_feature_frame(arrays)
    metrics_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for seed in seeds:
        for fold in folds:
            train_indices, test_indices = indices_for_fold(split_rows, fold)
            assert_no_group_overlap(arrays["participant_id"], train_indices, test_indices, fold)
            models = _build_metadata_models(
                random_state=seed,
                random_forest_estimators=random_forest_estimators,
                logistic_max_iter=logistic_max_iter,
            )
            for model_name, estimator in models.items():
                x_train = _metadata_columns_for_model(features, model_name).iloc[train_indices]
                x_test = _metadata_columns_for_model(features, model_name).iloc[test_indices]
                estimator.fit(x_train, arrays["y"][train_indices])
                validate_estimator_classes(estimator, label_ids, model_name, fold)
                y_pred = estimator.predict(x_test)
                probabilities = predict_probabilities(estimator, x_test)
                classes = estimator_class_labels(estimator)
                row: dict[str, Any] = compute_metrics(
                    arrays["y"][test_indices],
                    y_pred,
                    label_ids=label_ids,
                )
                row.update(
                    {
                        "control": "metadata_baseline",
                        "model": model_name,
                        "seed": seed,
                        "fold": fold,
                        "train_sample_count": int(len(train_indices)),
                        "test_sample_count": int(len(test_indices)),
                    }
                )
                metrics_rows.append(row)
                rows = build_prediction_rows(
                    model_name=model_name,
                    fold=fold,
                    sample_indices=test_indices,
                    arrays=arrays,
                    y_pred=y_pred,
                    probabilities=probabilities,
                    probability_classes=classes,
                    label_ids=label_ids,
                )
                for prediction in rows:
                    prediction["seed"] = seed
                    prediction["control"] = "metadata_baseline"
                prediction_rows.extend(rows)
    metrics = pd.DataFrame(metrics_rows)
    return {
        "metrics": metrics,
        "aggregate": _aggregate_metrics_frame(metrics, ["control", "model"]),
        "predictions": pd.DataFrame(prediction_rows),
    }


def _run_seed_stability(
    *,
    xrocket_dir: Path,
    folds: list[int],
    arrays: dict[str, np.ndarray],
    split_rows: list[dict[str, str]],
    label_ids: tuple[int, ...],
    seeds: tuple[int, ...],
    random_forest_estimators: int,
    logistic_max_iter: int,
) -> dict[str, pd.DataFrame]:
    metrics_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    importance_rows: list[pd.DataFrame] = []
    for seed in seeds:
        for fold in folds:
            fold_dir = xrocket_dir / f"fold_{fold}"
            features, train_indices, test_indices = _load_fold_features(fold_dir)
            assert_no_group_overlap(arrays["participant_id"], train_indices, test_indices, fold)
            models = _build_xrocket_feature_models(
                random_state=seed,
                random_forest_estimators=random_forest_estimators,
                logistic_max_iter=logistic_max_iter,
            )
            for model_name, estimator in models.items():
                estimator.fit(features[train_indices], arrays["y"][train_indices])
                validate_estimator_classes(estimator, label_ids, model_name, fold)
                y_pred = estimator.predict(features[test_indices])
                row: dict[str, Any] = compute_metrics(
                    arrays["y"][test_indices],
                    y_pred,
                    label_ids=label_ids,
                )
                row.update(
                    {
                        "model": model_name,
                        "seed": seed,
                        "fold": fold,
                        "train_sample_count": int(len(train_indices)),
                        "test_sample_count": int(len(test_indices)),
                    }
                )
                metrics_rows.append(row)
                confusion_rows.extend(
                    build_confusion_rows(
                        model_name=f"{model_name}_seed_{seed}",
                        fold=fold,
                        y_true=arrays["y"][test_indices],
                        y_pred=y_pred,
                        label_ids=label_ids,
                    )
                )
                if model_name == PRIMARY_MODEL:
                    importance_rows.append(
                        _aggregate_seed_importance(
                            fold_dir=fold_dir,
                            estimator=estimator,
                            seed=seed,
                            fold=fold,
                        )
                    )
    metrics = pd.DataFrame(metrics_rows)
    importance = pd.concat(importance_rows, ignore_index=True)
    rank_stability = _rank_correlations(importance, top_k=3)
    return {
        "metrics": metrics,
        "metric_summary": _aggregate_metrics_frame(metrics, ["model"]),
        "confusion": pd.DataFrame(confusion_rows),
        "importance": importance,
        "importance_summary": summarize_ranked_values(importance, value_column="importance"),
        "rank_correlations": rank_stability["rank_correlations"],
        "topk_overlap": rank_stability["topk_overlap"],
    }


def _aggregate_seed_importance(
    *,
    fold_dir: Path,
    estimator: Any,
    seed: int,
    fold: int,
) -> pd.DataFrame:
    metadata = _metadata_with_groups(fold_dir)
    importances = np.asarray(estimator.feature_importances_, dtype=np.float64)
    if len(importances) != len(metadata):
        raise RobustnessError(f"Fold {fold} seed {seed} RF importance does not align to metadata")
    total = float(np.sum(importances))
    if total <= 0.0:
        raise RobustnessError(f"Fold {fold} seed {seed} RF importance has zero total")
    metadata = metadata.copy()
    metadata["importance"] = importances / total
    frames: list[pd.DataFrame] = []
    for group_level in EXPLANATION_GROUP_LEVELS:
        grouped = (
            metadata.groupby(group_level, as_index=False)["importance"]
            .sum()
            .rename(columns={group_level: "group_value"})
        )
        grouped["group_value"] = grouped["group_value"].astype(str)
        grouped["seed"] = seed
        grouped["fold"] = fold
        grouped["group_level"] = group_level
        frames.append(grouped[["seed", "fold", "group_level", "group_value", "importance"]])
    return pd.concat(frames, ignore_index=True)


def _run_arm_axis_rankings(
    *,
    xrocket_dir: Path,
    folds: list[int],
    arrays: dict[str, np.ndarray],
    label_ids: tuple[int, ...],
    seed: int,
    random_forest_estimators: int,
) -> dict[str, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    for fold in folds:
        fold_dir = xrocket_dir / f"fold_{fold}"
        features, train_indices, test_indices = _load_fold_features(fold_dir)
        metadata = _metadata_with_groups(fold_dir)
        for arm_code in sorted(set(arrays["arm_code"].astype(str))):
            train_arm = train_indices[arrays["arm_code"][train_indices].astype(str) == arm_code]
            test_arm = test_indices[arrays["arm_code"][test_indices].astype(str) == arm_code]
            if not _has_full_class_coverage(arrays["y"], train_arm, label_ids) or not (
                _has_full_class_coverage(arrays["y"], test_arm, label_ids)
            ):
                continue
            estimator = _build_rf(
                random_state=seed + fold,
                random_forest_estimators=random_forest_estimators,
            )
            estimator.fit(features[train_arm], arrays["y"][train_arm])
            importances = np.asarray(estimator.feature_importances_, dtype=np.float64)
            if np.sum(importances) <= 0.0:
                continue
            temp = metadata[["axis"]].copy()
            temp["importance"] = importances / float(np.sum(importances))
            grouped = (
                temp.groupby("axis", as_index=False)["importance"]
                .sum()
                .sort_values("importance", ascending=False)
            )
            grouped["rank"] = grouped["importance"].rank(method="average", ascending=False)
            grouped["fold"] = fold
            grouped["arm_code"] = arm_code
            grouped["train_sample_count"] = int(len(train_arm))
            grouped["test_sample_count"] = int(len(test_arm))
            rows.append(grouped)
    rankings = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if rankings.empty:
        comparison = pd.DataFrame(
            [
                {
                    "status": "skipped",
                    "reason": "class coverage did not support arm-stratified axis ranking",
                }
            ]
        )
    else:
        comparison_rows: list[dict[str, Any]] = []
        for arm_code, group in rankings.groupby("arm_code", sort=True):
            summary = summarize_ranked_values(
                group.rename(columns={"axis": "group_value"}).assign(group_level="axis"),
                value_column="importance",
            )
            leader = summary.sort_values("importance_mean", ascending=False).iloc[0]
            comparison_rows.append(
                {
                    "status": "completed",
                    "arm_code": arm_code,
                    "top_axis": str(leader["group_value"]),
                    "top_axis_importance_mean": float(leader["importance_mean"]),
                    "axis_rank_std_max": float(summary["rank_std"].max()),
                }
            )
        completed = pd.DataFrame(comparison_rows)
        top_axes = sorted(set(completed["top_axis"]))
        completed["arm_top_axis_conflict"] = len(top_axes) > 1
        comparison = completed
    return {"rankings": rankings, "comparison": comparison}


def _build_leakage_checks(
    split_rows: list[dict[str, str]],
    arrays: dict[str, np.ndarray],
    folds: list[int],
    label_ids: tuple[int, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fold in folds:
        train_indices, test_indices = indices_for_fold(split_rows, fold)
        train_groups = set(arrays["participant_id"][train_indices].astype(str))
        test_groups = set(arrays["participant_id"][test_indices].astype(str))
        overlap = sorted(train_groups.intersection(test_groups))
        train_classes = set(int(value) for value in arrays["y"][train_indices])
        test_classes = set(int(value) for value in arrays["y"][test_indices])
        rows.append(
            {
                "fold": fold,
                "train_sample_count": int(len(train_indices)),
                "test_sample_count": int(len(test_indices)),
                "train_participant_count": len(train_groups),
                "test_participant_count": len(test_groups),
                "participant_overlap_count": len(overlap),
                "participant_overlap": ";".join(overlap),
                "train_has_all_classes": set(label_ids).issubset(train_classes),
                "test_has_all_classes": set(label_ids).issubset(test_classes),
                "status": (
                    "pass"
                    if not overlap
                    and set(label_ids).issubset(train_classes)
                    and set(label_ids).issubset(test_classes)
                    else "fail"
                ),
            }
        )
    return rows


def _build_confound_summary(
    arrays: dict[str, np.ndarray],
    label_ids: tuple[int, ...],
) -> pd.DataFrame:
    feature_frame = build_metadata_feature_frame(arrays)
    feature_frame["label"] = arrays["y"]
    rows: list[dict[str, Any]] = []
    for label_id in label_ids:
        group = feature_frame.loc[feature_frame["label"] == label_id]
        rows.append(
            {
                "level": "label",
                "value": label_id,
                "sample_count": int(len(group)),
                "original_length_mean": float(group["original_length"].mean()),
                "original_length_std": float(group["original_length"].std(ddof=0)),
                "original_length_min": float(group["original_length"].min()),
                "original_length_max": float(group["original_length"].max()),
                "padding_fraction_mean": float(group["padding_fraction"].mean()),
                "arm_d_count": int((group["arm_code"] == "d").sum()),
                "arm_i_count": int((group["arm_code"] == "i").sum()),
            }
        )
    for arm_code, group in feature_frame.groupby("arm_code", sort=True):
        rows.append(
            {
                "level": "arm_code",
                "value": arm_code,
                "sample_count": int(len(group)),
                "original_length_mean": float(group["original_length"].mean()),
                "original_length_std": float(group["original_length"].std(ddof=0)),
                "original_length_min": float(group["original_length"].min()),
                "original_length_max": float(group["original_length"].max()),
                "padding_fraction_mean": float(group["padding_fraction"].mean()),
                "arm_d_count": int((group["arm_code"] == "d").sum()),
                "arm_i_count": int((group["arm_code"] == "i").sum()),
            }
        )
    return pd.DataFrame(rows)


def _build_per_class_error_summary(
    *,
    xrocket_dir: Path,
    primary_model: str,
    label_ids: tuple[int, ...],
) -> pd.DataFrame:
    path = xrocket_dir / "confusion_matrices.csv"
    if not path.is_file():
        raise RobustnessError(f"Missing confusion matrix artifact: {path}")
    matrix = pd.read_csv(path)
    matrix = matrix.loc[matrix["model"] == primary_model].copy()
    if matrix.empty:
        raise RobustnessError(f"No confusion rows found for primary model {primary_model!r}")
    rows: list[dict[str, Any]] = []
    for label_id in label_ids:
        label_rows = matrix.loc[matrix["true_label"] == label_id]
        support = float(label_rows["count"].sum())
        correct = float(label_rows.loc[label_rows["predicted_label"] == label_id, "count"].sum())
        predicted_as = float(matrix.loc[matrix["predicted_label"] == label_id, "count"].sum())
        recall = correct / support if support else 0.0
        precision = correct / predicted_as if predicted_as else 0.0
        confusions = label_rows.loc[label_rows["predicted_label"] != label_id].copy()
        confusions = confusions.sort_values("count", ascending=False)
        top_confusion = ""
        top_confusion_count = 0
        if not confusions.empty and int(confusions.iloc[0]["count"]) > 0:
            top_confusion = str(int(confusions.iloc[0]["predicted_label"]))
            top_confusion_count = int(confusions.iloc[0]["count"])
        rows.append(
            {
                "model": primary_model,
                "class_label": label_id,
                "support": int(support),
                "correct_count": int(correct),
                "recall": recall,
                "precision": precision,
                "top_confused_as": top_confusion,
                "top_confused_count": top_confusion_count,
            }
        )
    return pd.DataFrame(rows)


def _build_control_flags(
    *,
    label_shuffle: pd.DataFrame,
    metadata: pd.DataFrame,
    stability: dict[str, pd.DataFrame],
    per_class_errors: pd.DataFrame,
    thresholds: dict[str, float],
) -> list[dict[str, Any]]:
    flags: list[dict[str, Any]] = []
    shuffle_mean = float(label_shuffle["macro_f1_mean"].max()) if not label_shuffle.empty else 0.0
    shuffle_max = float(label_shuffle["macro_f1_max"].max()) if not label_shuffle.empty else 0.0
    metadata_mean = float(metadata["macro_f1_mean"].max()) if not metadata.empty else 0.0
    rank_corr = stability["rank_correlations"]
    corr_min = (
        float(rank_corr["spearman_rank_correlation"].min()) if not rank_corr.empty else float("nan")
    )
    top_changed = _top_group_changed(stability["importance"])
    weak_classes = per_class_errors.loc[per_class_errors["recall"] < thresholds["class_recall_min"]]
    flags.append(
        _flag_row(
            "label_shuffle_mean_macro_f1",
            shuffle_mean > thresholds["label_shuffle_mean_macro_f1"],
            shuffle_mean,
            thresholds["label_shuffle_mean_macro_f1"],
            "Label-shuffle mean macro F1 exceeds threshold",
        )
    )
    flags.append(
        _flag_row(
            "label_shuffle_max_macro_f1",
            shuffle_max > thresholds["label_shuffle_max_macro_f1"],
            shuffle_max,
            thresholds["label_shuffle_max_macro_f1"],
            "Any label-shuffle fold/seed macro F1 exceeds threshold",
        )
    )
    flags.append(
        _flag_row(
            "metadata_mean_macro_f1",
            metadata_mean > thresholds["metadata_mean_macro_f1"],
            metadata_mean,
            thresholds["metadata_mean_macro_f1"],
            "Metadata-only control exceeds threshold",
        )
    )
    flags.append(
        _flag_row(
            "rank_correlation_min",
            bool(not math.isnan(corr_min) and corr_min < thresholds["rank_correlation_min"]),
            corr_min,
            thresholds["rank_correlation_min"],
            "Seed/fold explanation rank correlation below threshold",
        )
    )
    for group_level, changed in top_changed.items():
        flags.append(
            {
                "flag": f"{group_level}_top_rank_changed",
                "triggered": bool(changed),
                "observed": str(changed),
                "threshold": "top group should be stable for strong claims",
                "message": f"Top {group_level} changed across seed/fold cases",
            }
        )
    flags.append(
        {
            "flag": "movement_specific_weakness",
            "triggered": not weak_classes.empty,
            "observed": ";".join(
                f"class_{int(row['class_label'])}_recall_{float(row['recall']):.3f}"
                for row in weak_classes.to_dict("records")
            ),
            "threshold": thresholds["class_recall_min"],
            "message": "At least one movement class has low recall",
        }
    )
    return flags


def _flag_row(
    flag: str,
    triggered: bool,
    observed: float,
    threshold: float,
    message: str,
) -> dict[str, Any]:
    return {
        "flag": flag,
        "triggered": triggered,
        "observed": observed,
        "threshold": threshold,
        "message": message,
    }


def _top_group_changed(importance: pd.DataFrame) -> dict[str, bool]:
    changes: dict[str, bool] = {}
    for group_level, group in importance.groupby("group_level", sort=False):
        leaders: set[str] = set()
        for (_seed, _fold), case in group.groupby(["seed", "fold"], sort=False):
            leader = case.sort_values("importance", ascending=False).iloc[0]
            leaders.add(str(leader["group_value"]))
        changes[str(group_level)] = len(leaders) > 1
    return changes


def _aggregate_metrics_frame(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    if frame.empty:
        return frame
    metric_columns = [
        column
        for column in frame.columns
        if pd.api.types.is_numeric_dtype(frame[column])
        and column not in {"fold", "seed", "train_sample_count", "test_sample_count"}
        and not column.endswith("_support")
    ]
    rows: list[dict[str, Any]] = []
    for key, group in frame.groupby(group_columns, sort=False):
        key_values = key if isinstance(key, tuple) else (key,)
        row = {column: key_values[index] for index, column in enumerate(group_columns)}
        row["case_count"] = int(len(group))
        for metric in metric_columns:
            values = group[metric].to_numpy(dtype=np.float64)
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values, ddof=0))
            row[f"{metric}_min"] = float(np.min(values))
            row[f"{metric}_max"] = float(np.max(values))
        rows.append(row)
    return pd.DataFrame(rows)


def _rank_correlations(frame: pd.DataFrame, *, top_k: int) -> dict[str, pd.DataFrame]:
    correlation_rows: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []
    working = frame.copy()
    working["case_id"] = working["seed"].astype(str) + "_fold_" + working["fold"].astype(str)
    for group_level, group in working.groupby("group_level", sort=False):
        pivot = group.pivot(index="case_id", columns="group_value", values="importance").fillna(0.0)
        case_ids = [str(value) for value in pivot.index]
        for left_pos, left_case in enumerate(case_ids):
            for right_case in case_ids[left_pos + 1 :]:
                left = pivot.loc[left_case]
                right = pivot.loc[right_case]
                correlation = float(left.corr(right, method="spearman"))
                if math.isnan(correlation):
                    correlation = 0.0
                left_top = set(left.sort_values(ascending=False).head(top_k).index)
                right_top = set(right.sort_values(ascending=False).head(top_k).index)
                union = left_top.union(right_top)
                correlation_rows.append(
                    {
                        "group_level": group_level,
                        "left_case": left_case,
                        "right_case": right_case,
                        "spearman_rank_correlation": correlation,
                    }
                )
                overlap_rows.append(
                    {
                        "group_level": group_level,
                        "left_case": left_case,
                        "right_case": right_case,
                        "top_k": top_k,
                        "jaccard_overlap": (
                            float(len(left_top.intersection(right_top)) / len(union))
                            if union
                            else 0.0
                        ),
                    }
                )
    return {
        "rank_correlations": pd.DataFrame(
            correlation_rows,
            columns=[
                "group_level",
                "left_case",
                "right_case",
                "spearman_rank_correlation",
            ],
        ),
        "topk_overlap": pd.DataFrame(
            overlap_rows,
            columns=["group_level", "left_case", "right_case", "top_k", "jaccard_overlap"],
        ),
    }


def _metadata_with_groups(fold_dir: Path) -> pd.DataFrame:
    path = fold_dir / "feature_metadata.parquet"
    if not path.is_file():
        raise RobustnessError(f"Missing feature metadata artifact: {path}")
    metadata = pd.read_parquet(path).copy()
    parsed = [parse_channel_name(str(channel)) for channel in metadata["channel_name"]]
    metadata["sensor_family"] = [row["sensor_family"] for row in parsed]
    metadata["axis"] = [row["axis"] for row in parsed]
    metadata["channel"] = [row["channel"] for row in parsed]
    metadata["temporal_scale_bin"] = [
        temporal_scale_bin(int(span), target_length=56)
        for span in metadata["effective_receptive_field_samples"]
    ]
    return metadata


def _load_fold_features(fold_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = fold_dir / "features.npz"
    if not path.is_file():
        raise RobustnessError(f"Missing transformed features artifact: {path}")
    with np.load(path, allow_pickle=False) as data:
        return (
            data["features"].astype(np.float32, copy=False),
            data["train_indices"].astype(np.int64, copy=False),
            data["test_indices"].astype(np.int64, copy=False),
        )


def _build_xrocket_feature_models(
    *,
    random_state: int,
    random_forest_estimators: int,
    logistic_max_iter: int,
) -> dict[str, Any]:
    return {
        PRIMARY_MODEL: _build_rf(
            random_state=random_state,
            random_forest_estimators=random_forest_estimators,
        ),
        SENSITIVITY_MODEL: Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(max_iter=logistic_max_iter, random_state=random_state),
                ),
            ]
        ),
    }


def _build_metadata_models(
    *,
    random_state: int,
    random_forest_estimators: int,
    logistic_max_iter: int,
) -> dict[str, Any]:
    metadata_preprocessor = ColumnTransformer(
        [
            ("numeric", StandardScaler(), ["original_length", "padding_fraction"]),
            ("arm", OneHotEncoder(handle_unknown="ignore"), ["arm_code"]),
        ]
    )
    return {
        "length_only_logistic": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(max_iter=logistic_max_iter, random_state=random_state),
                ),
            ]
        ),
        "metadata_logistic": Pipeline(
            [
                ("preprocessor", metadata_preprocessor),
                (
                    "classifier",
                    LogisticRegression(max_iter=logistic_max_iter, random_state=random_state),
                ),
            ]
        ),
        "metadata_random_forest": Pipeline(
            [
                ("preprocessor", metadata_preprocessor),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=random_forest_estimators,
                        class_weight="balanced",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def _metadata_columns_for_model(frame: pd.DataFrame, model_name: str) -> pd.DataFrame:
    if model_name == "length_only_logistic":
        return frame[["original_length", "padding_fraction"]]
    return frame[["original_length", "padding_fraction", "arm_code"]]


def _build_rf(*, random_state: int, random_forest_estimators: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=random_forest_estimators,
        class_weight="balanced",
        max_features="sqrt",
        random_state=random_state,
        n_jobs=-1,
    )


def _has_full_class_coverage(
    y: np.ndarray,
    indices: np.ndarray,
    label_ids: tuple[int, ...],
) -> bool:
    if len(indices) == 0:
        return False
    return set(label_ids).issubset(set(int(value) for value in y[indices]))


def _build_controls_summary(
    *,
    label_shuffle: pd.DataFrame,
    metadata: pd.DataFrame,
    confounds: pd.DataFrame,
    flags: pd.DataFrame,
) -> str:
    shuffle = _best_row(label_shuffle, "macro_f1_mean")
    meta = _best_row(metadata, "macro_f1_mean")
    triggered = flags.loc[flags["triggered"].astype(str).isin(["True", "true", "1"])]
    length_rows = confounds.loc[confounds["level"] == "label"]
    length_range = (
        float(length_rows["original_length_mean"].max() - length_rows["original_length_mean"].min())
        if not length_rows.empty
        else 0.0
    )
    return f"""# M7 Controls Summary

## Negative Controls

The strongest label-shuffle control was `{shuffle.get("model", "n/a")}` with mean
macro F1 `{float(shuffle.get("macro_f1_mean", 0.0)):.4f}` and max fold/seed macro
F1 `{float(shuffle.get("macro_f1_max", 0.0)):.4f}`.

The strongest metadata-only control was `{meta.get("model", "n/a")}` with mean
macro F1 `{float(meta.get("macro_f1_mean", 0.0)):.4f}`.

## Confound Check

Mean original sequence length differs by up to `{length_range:.2f}` samples across
movement labels. Arm coverage is balanced in the audited dataset, and metadata
controls remain the formal check for whether length or arm can explain labels.

## Triggered Flags

{_markdown_table(triggered, ["flag", "observed", "threshold", "message"])}
"""


def _build_stability_summary(
    *,
    metric_summary: pd.DataFrame,
    importance_summary: pd.DataFrame,
    arm_comparison: pd.DataFrame,
    per_class_errors: pd.DataFrame,
    flags: pd.DataFrame,
) -> str:
    primary = metric_summary.loc[metric_summary["model"] == PRIMARY_MODEL]
    primary_row = primary.iloc[0].to_dict() if not primary.empty else {}
    top_axis = _top_summary_value(importance_summary, "axis")
    top_sensor = _top_summary_value(importance_summary, "sensor_family")
    top_dilation = _top_summary_value(importance_summary, "dilation")
    weak = per_class_errors.loc[
        per_class_errors["recall"] < DEFAULT_WARNING_THRESHOLDS["class_recall_min"]
    ]
    weak_columns = ["class_label", "recall", "precision", "top_confused_as", "top_confused_count"]
    triggered = flags.loc[flags["triggered"].astype(str).isin(["True", "true", "1"])]
    return f"""# M7 Stability Summary

## Metric Stability

Retraining classifiers on saved XROCKET features across seeds produced primary
mean macro F1 `{float(primary_row.get("macro_f1_mean", 0.0)):.4f}` with standard
deviation `{float(primary_row.get("macro_f1_std", 0.0)):.4f}` across seed/fold
cases.

## Explanation Stability

Top seed-stability groups: sensor family `{top_sensor}`, axis `{top_axis}`, and
dilation `{top_dilation}`. These summarize random-forest feature importance after
refitting only the classifier layer on the saved XROCKET representation.

## Arm-Stratified Axis Check

{_markdown_table(arm_comparison, list(arm_comparison.columns))}

## Per-Class Weaknesses

{_markdown_table(weak, weak_columns)}

## Triggered Flags

{_markdown_table(triggered, ["flag", "observed", "threshold", "message"])}
"""


def _build_confusion_error_review(per_class_errors: pd.DataFrame) -> str:
    columns = [
        "class_label",
        "support",
        "recall",
        "precision",
        "top_confused_as",
        "top_confused_count",
    ]
    return f"""# Confusion And Per-Class Error Review

Aggregated primary-model per-class errors:

{_markdown_table(per_class_errors, columns)}

Classes with lower recall should be treated as movement-specific weaknesses in
the final report rather than averaged away by the overall macro score.
"""


def _top_summary_value(summary: pd.DataFrame, group_level: str) -> str:
    selected = summary.loc[summary["group_level"] == group_level]
    if selected.empty:
        return "n/a"
    return str(selected.sort_values("importance_mean", ascending=False).iloc[0]["group_value"])


def _best_row(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    if frame.empty:
        return {}
    row: dict[str, Any] = frame.sort_values(column, ascending=False).iloc[0].to_dict()
    return row


def _default_resolved_config(
    *,
    processed_dir: Path,
    splits_dir: Path,
    xrocket_dir: Path,
    controls_output_dir: Path,
    stability_output_dir: Path,
    label_ids: tuple[int, ...],
    seeds: tuple[int, ...],
    primary_model: str,
    random_forest_estimators: int,
    logistic_max_iter: int,
    warning_thresholds: dict[str, float],
) -> dict[str, Any]:
    return {
        "processed_dir": processed_dir.as_posix(),
        "splits_dir": splits_dir.as_posix(),
        "xrocket_dir": xrocket_dir.as_posix(),
        "controls_output_dir": controls_output_dir.as_posix(),
        "stability_output_dir": stability_output_dir.as_posix(),
        "classes": list(label_ids),
        "seeds": list(seeds),
        "primary_model": primary_model,
        "random_forest": {"n_estimators": random_forest_estimators},
        "logistic_regression": {"max_iter": logistic_max_iter},
        "warning_thresholds": warning_thresholds,
        "temporal_bin_rule": BIN_RULE_DESCRIPTION,
        "encoder_reruns": "not performed in M7 report-safe workflow",
    }


def _build_provenance(
    *,
    processed_dir: Path,
    split_path: Path,
    xrocket_dir: Path,
    controls_output_dir: Path,
    stability_output_dir: Path,
    folds: list[int],
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    xrocket_provenance = xrocket_dir / "provenance.json"
    return {
        "processed_dir": processed_dir.as_posix(),
        "split_manifest": split_path.as_posix(),
        "xrocket_dir": xrocket_dir.as_posix(),
        "controls_output_dir": controls_output_dir.as_posix(),
        "stability_output_dir": stability_output_dir.as_posix(),
        "tensor_sha256": sha256_file(processed_dir / "tensors.npz"),
        "split_manifest_sha256": sha256_file(split_path),
        "xrocket_provenance_sha256": (
            sha256_file(xrocket_provenance) if xrocket_provenance.is_file() else None
        ),
        "git_commit": git_commit(),
        "folds": folds,
        "seeds": list(seeds),
        "packages": {
            package: package_version(package)
            for package in ("numpy", "pandas", "pyarrow", "scikit-learn", "joblib")
        },
    }


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_No rows available._"
    selected = frame[columns].copy()
    for column in selected.columns:
        if pd.api.types.is_float_dtype(selected[column]):
            selected[column] = selected[column].map(lambda value: f"{float(value):.4f}")
    text_rows = [[str(value) for value in row] for row in selected.to_numpy().tolist()]
    widths = [
        max(len(str(column)), *(len(row[index]) for row in text_rows))
        for index, column in enumerate(selected.columns)
    ]
    header = (
        "| "
        + " | ".join(
            str(column).ljust(widths[index]) for index, column in enumerate(selected.columns)
        )
        + " |"
    )
    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    body = [
        "| " + " | ".join(row[index].ljust(widths[index]) for index in range(len(widths))) + " |"
        for row in text_rows
    ]
    return "\n".join([header, separator, *body])
