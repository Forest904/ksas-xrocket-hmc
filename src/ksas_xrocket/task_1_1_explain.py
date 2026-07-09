"""Task 1.1 sensor-axis contribution analysis."""

from __future__ import annotations

import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ksas_xrocket.audit import EXPECTED_CHANNELS, EXPECTED_LABELS, sha256_file, write_json
from ksas_xrocket.baselines import (
    compute_metrics,
    git_commit,
    load_processed_arrays,
    package_version,
)
from ksas_xrocket.xrocket_experiment import PRIMARY_MODEL

SENSOR_FAMILIES = ("accelerometer", "gravity", "gyros", "lin_accel", "game_rot_vec", "magn_field")
AXES = ("x", "y", "z")
NATIVE_GROUP_LEVELS = ("sensor_family", "axis", "channel", "family_axis", "channel_combination")
VALIDATION_GROUP_LEVELS = ("sensor_family", "channel")


class Task11ExplanationError(ValueError):
    """Raised when Task 1.1 explanations cannot be generated safely."""


@dataclass(frozen=True)
class Task11ExplanationResult:
    """Paths returned by the Task 1.1 explanation workflow."""

    output_dir: Path
    native_importance_path: Path
    ablation_path: Path
    permutation_path: Path
    method_agreement_path: Path
    answer_path: Path
    provenance_path: Path


def parse_channel_name(channel_name: str) -> dict[str, str]:
    """Parse one audited KSAS channel into sensor family, axis, and channel IDs."""
    try:
        sensor_family, axis = channel_name.rsplit("_", 1)
    except ValueError as exc:
        raise Task11ExplanationError(f"Cannot parse channel name: {channel_name!r}") from exc
    if sensor_family not in SENSOR_FAMILIES:
        raise Task11ExplanationError(f"Unknown sensor family {sensor_family!r} in {channel_name!r}")
    if axis not in AXES:
        raise Task11ExplanationError(f"Unknown axis {axis!r} in {channel_name!r}")
    return {
        "sensor_family": sensor_family,
        "axis": axis,
        "channel": channel_name,
        "family_axis": f"{sensor_family}_{axis}",
        "channel_combination": channel_name,
    }


def normalize_importance(values: pd.Series, *, fold: int) -> pd.Series:
    """Normalize a non-negative importance vector to sum to one."""
    if values.isna().any():
        raise Task11ExplanationError(f"Fold {fold} native importance contains missing values")
    array = values.to_numpy(dtype=np.float64)
    if not np.isfinite(array).all():
        raise Task11ExplanationError(f"Fold {fold} native importance contains non-finite values")
    if np.any(array < 0.0):
        raise Task11ExplanationError(f"Fold {fold} native importance contains negative values")
    total = float(np.sum(array))
    if total <= 0.0:
        raise Task11ExplanationError(f"Fold {fold} native importance has zero total")
    return pd.Series(array / total, index=values.index)


def aggregate_fold_importance(
    frame: pd.DataFrame,
    *,
    group_level: str,
    value_column: str = "normalized_importance",
) -> pd.DataFrame:
    """Aggregate normalized feature importance to one grouping level per fold."""
    required = {"fold", group_level, value_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise Task11ExplanationError(f"Importance frame is missing columns: {', '.join(missing)}")
    grouped = (
        frame.groupby(["fold", group_level], sort=False, as_index=False)[value_column]
        .sum()
        .rename(columns={group_level: "group_value", value_column: "importance"})
    )
    grouped.insert(1, "group_level", group_level)
    return grouped


def summarize_fold_values(frame: pd.DataFrame, *, value_column: str) -> pd.DataFrame:
    """Summarize fold-level values with uncertainty and rank variation."""
    required = {"fold", "group_level", "group_value", value_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise Task11ExplanationError(f"Summary frame is missing columns: {', '.join(missing)}")
    ranked = frame.copy()
    ranked["rank"] = ranked.groupby(["fold", "group_level"])[value_column].rank(
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
                "fold_count": int(len(group)),
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


def run_task_1_1_explanation(
    *,
    processed_dir: Path,
    xrocket_dir: Path,
    output_dir: Path,
    label_ids: tuple[int, ...] = tuple(EXPECTED_LABELS),
    random_state: int = 42,
    random_forest_estimators: int = 500,
    permutation_repeats: int = 5,
    top_k: int = 3,
    overwrite: bool = False,
    resolved_config: dict[str, Any] | None = None,
) -> Task11ExplanationResult:
    """Run Task 1.1 analysis from saved M3 XROCKET artifacts."""
    _prepare_output_dir(output_dir, overwrite=overwrite)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    arrays = load_processed_arrays(processed_dir / "tensors.npz")
    folds = _discover_folds(xrocket_dir)
    if not folds:
        raise Task11ExplanationError(f"No fold directories found in {xrocket_dir}")

    native = _load_native_importance(xrocket_dir=xrocket_dir, folds=folds)
    native_path = output_dir / "fold_native_importance.parquet"
    native.to_parquet(native_path, index=False)

    native_fold = pd.concat(
        [aggregate_fold_importance(native, group_level=level) for level in NATIVE_GROUP_LEVELS],
        ignore_index=True,
    )
    native_fold.to_csv(output_dir / "native_fold_group_importance.csv", index=False)
    native_summary = summarize_fold_values(native_fold, value_column="importance")
    _write_importance_summaries(output_dir, native_summary)

    stability = _compute_stability(native_fold, top_k=top_k)
    stability["rank_correlations"].to_csv(
        output_dir / "stability_rank_correlations.csv",
        index=False,
    )
    stability["topk_overlap"].to_csv(output_dir / "stability_topk_overlap.csv", index=False)

    class_specific = _compute_class_specific_importance(
        xrocket_dir=xrocket_dir,
        folds=folds,
        arrays=arrays,
        label_ids=label_ids,
        random_state=random_state,
        random_forest_estimators=random_forest_estimators,
    )
    class_specific["channel"].to_csv(
        output_dir / "class_specific_channel_importance.csv",
        index=False,
    )
    class_specific["sensor_family"].to_csv(
        output_dir / "class_specific_sensor_family_importance.csv",
        index=False,
    )

    ablation = _run_ablation(
        xrocket_dir=xrocket_dir,
        folds=folds,
        arrays=arrays,
        label_ids=label_ids,
        random_state=random_state,
        random_forest_estimators=random_forest_estimators,
    )
    ablation_path = output_dir / "ablation_metrics.csv"
    ablation.to_csv(ablation_path, index=False)

    permutation = _run_permutation(
        xrocket_dir=xrocket_dir,
        folds=folds,
        arrays=arrays,
        label_ids=label_ids,
        random_state=random_state,
        permutation_repeats=permutation_repeats,
    )
    permutation_path = output_dir / "permutation_importance.csv"
    permutation.to_csv(permutation_path, index=False)

    method_agreement = _build_method_agreement(native_fold, ablation, permutation)
    method_agreement_path = output_dir / "method_agreement.csv"
    method_agreement.to_csv(method_agreement_path, index=False)

    _write_figures(
        figures_dir=figures_dir,
        native_fold=native_fold,
        native_summary=native_summary,
        stability=stability,
        ablation=ablation,
        class_specific_channel=class_specific["channel"],
        method_agreement=method_agreement,
    )
    _write_figure_captions(figures_dir)

    answer_path = output_dir / "task_1_1_answer.md"
    answer_path.write_text(
        _build_task_answer(
            native_summary=native_summary,
            ablation=ablation,
            permutation=permutation,
            method_agreement=method_agreement,
            class_specific_channel=class_specific["channel"],
        ),
        encoding="utf-8",
    )

    write_json(
        output_dir / "resolved_config.json",
        resolved_config
        or {
            "task": "task_1_1",
            "processed_dir": processed_dir.as_posix(),
            "xrocket_dir": xrocket_dir.as_posix(),
            "output_dir": output_dir.as_posix(),
            "classes": list(label_ids),
            "random_state": random_state,
            "random_forest_estimators": random_forest_estimators,
            "permutation_repeats": permutation_repeats,
            "top_k": top_k,
            "validation_group_levels": list(VALIDATION_GROUP_LEVELS),
        },
    )
    provenance_path = output_dir / "provenance.json"
    write_json(
        provenance_path,
        {
            "processed_dir": processed_dir.as_posix(),
            "xrocket_dir": xrocket_dir.as_posix(),
            "output_dir": output_dir.as_posix(),
            "tensor_sha256": sha256_file(processed_dir / "tensors.npz"),
            "m3_provenance_sha256": sha256_file(xrocket_dir / "provenance.json"),
            "git_commit": git_commit(),
            "folds": folds,
            "packages": {
                package: package_version(package)
                for package in ("numpy", "pandas", "matplotlib", "scikit-learn", "joblib")
            },
        },
    )

    return Task11ExplanationResult(
        output_dir=output_dir,
        native_importance_path=native_path,
        ablation_path=ablation_path,
        permutation_path=permutation_path,
        method_agreement_path=method_agreement_path,
        answer_path=answer_path,
        provenance_path=provenance_path,
    )


def _prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise Task11ExplanationError(
                f"Output directory is not empty: {output_dir}; pass --overwrite to replace it"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _discover_folds(xrocket_dir: Path) -> list[int]:
    if not xrocket_dir.is_dir():
        raise Task11ExplanationError(f"M3 XROCKET directory not found: {xrocket_dir}")
    folds: list[int] = []
    for path in xrocket_dir.iterdir():
        if path.is_dir() and path.name.startswith("fold_"):
            try:
                folds.append(int(path.name.removeprefix("fold_")))
            except ValueError:
                continue
    return sorted(folds)


def _load_native_importance(*, xrocket_dir: Path, folds: list[int]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for fold in folds:
        path = xrocket_dir / f"fold_{fold}" / "feature_importance.parquet"
        if not path.is_file():
            raise Task11ExplanationError(f"Missing feature importance artifact: {path}")
        frame = pd.read_parquet(path)
        if "channel_name" not in frame or "importance" not in frame:
            raise Task11ExplanationError(f"Importance artifact has unexpected schema: {path}")
        if set(frame["channel_name"].dropna()) != set(EXPECTED_CHANNELS):
            raise Task11ExplanationError(f"Fold {fold} does not contain the expected channels")
        frame = frame.copy()
        parsed = [parse_channel_name(str(channel)) for channel in frame["channel_name"]]
        for column in ("sensor_family", "axis", "channel", "family_axis", "channel_combination"):
            frame[column] = [row[column] for row in parsed]
        frame["fold"] = fold
        frame["native_importance"] = frame["importance"].astype(float)
        frame["normalized_importance"] = normalize_importance(frame["native_importance"], fold=fold)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _write_importance_summaries(output_dir: Path, summary: pd.DataFrame) -> None:
    for level in NATIVE_GROUP_LEVELS:
        path = output_dir / f"{level}_importance_summary.csv"
        summary.loc[summary["group_level"] == level].to_csv(path, index=False)


def _compute_stability(native_fold: pd.DataFrame, *, top_k: int) -> dict[str, pd.DataFrame]:
    correlation_rows: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []
    for group_level, group in native_fold.groupby("group_level", sort=False):
        pivot = group.pivot(index="fold", columns="group_value", values="importance").fillna(0.0)
        folds = [int(value) for value in pivot.index]
        for left_pos, left_fold in enumerate(folds):
            for right_fold in folds[left_pos + 1 :]:
                left = pivot.loc[left_fold]
                right = pivot.loc[right_fold]
                correlation = float(left.corr(right, method="spearman"))
                if math.isnan(correlation):
                    correlation = 0.0
                correlation_rows.append(
                    {
                        "group_level": group_level,
                        "left_fold": left_fold,
                        "right_fold": int(right_fold),
                        "spearman_rank_correlation": correlation,
                    }
                )
                left_top = set(left.sort_values(ascending=False).head(top_k).index)
                right_top = set(right.sort_values(ascending=False).head(top_k).index)
                union = left_top.union(right_top)
                overlap_rows.append(
                    {
                        "group_level": group_level,
                        "left_fold": left_fold,
                        "right_fold": int(right_fold),
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
                "left_fold",
                "right_fold",
                "spearman_rank_correlation",
            ],
        ),
        "topk_overlap": pd.DataFrame(
            overlap_rows,
            columns=["group_level", "left_fold", "right_fold", "top_k", "jaccard_overlap"],
        ),
    }


def _compute_class_specific_importance(
    *,
    xrocket_dir: Path,
    folds: list[int],
    arrays: dict[str, np.ndarray],
    label_ids: tuple[int, ...],
    random_state: int,
    random_forest_estimators: int,
) -> dict[str, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for fold in folds:
        fold_dir = xrocket_dir / f"fold_{fold}"
        metadata = _metadata_with_groups(fold_dir)
        features, train_indices, _test_indices = _load_fold_features(fold_dir)
        for label_id in label_ids:
            y_binary = (arrays["y"][train_indices] == label_id).astype(np.int64)
            estimator = _build_rf(
                random_state=random_state + fold * 100 + label_id,
                random_forest_estimators=random_forest_estimators,
            )
            estimator.fit(features[train_indices], y_binary)
            importances = np.asarray(estimator.feature_importances_, dtype=np.float64)
            normalized = normalize_importance(pd.Series(importances), fold=fold).to_numpy()
            temp = metadata[["sensor_family", "channel"]].copy()
            temp["importance"] = normalized
            for group_level in ("channel", "sensor_family"):
                grouped = temp.groupby(group_level, as_index=False)["importance"].sum()
                for item in grouped.to_dict("records"):
                    rows.append(
                        {
                            "fold": fold,
                            "class_label": label_id,
                            "group_level": group_level,
                            "group_value": item[group_level],
                            "importance": float(item["importance"]),
                        }
                    )
    fold_frame = pd.DataFrame(rows)
    class_summary_rows: list[dict[str, Any]] = []
    for (class_label, group_level, group_value), group in fold_frame.groupby(
        ["class_label", "group_level", "group_value"],
        sort=False,
    ):
        values = group["importance"].to_numpy(dtype=np.float64)
        class_summary_rows.append(
            {
                "class_label": int(class_label),
                "group_level": group_level,
                "group_value": group_value,
                "fold_count": int(len(group)),
                "importance_mean": float(np.mean(values)),
                "importance_std": float(np.std(values, ddof=0)),
                "importance_min": float(np.min(values)),
                "importance_max": float(np.max(values)),
            }
        )
    class_summary = pd.DataFrame(class_summary_rows).sort_values(
        ["class_label", "group_level", "importance_mean"],
        ascending=[True, True, False],
    )
    return {
        "channel": class_summary.loc[class_summary["group_level"] == "channel"].copy(),
        "sensor_family": class_summary.loc[class_summary["group_level"] == "sensor_family"].copy(),
    }


def _run_ablation(
    *,
    xrocket_dir: Path,
    folds: list[int],
    arrays: dict[str, np.ndarray],
    label_ids: tuple[int, ...],
    random_state: int,
    random_forest_estimators: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold in folds:
        fold_dir = xrocket_dir / f"fold_{fold}"
        metadata = _metadata_with_groups(fold_dir)
        features, train_indices, test_indices = _load_fold_features(fold_dir)
        full_model = joblib.load(fold_dir / f"{PRIMARY_MODEL}.joblib")
        full_pred = full_model.predict(features[test_indices])
        full_metrics = compute_metrics(arrays["y"][test_indices], full_pred, label_ids=label_ids)
        for group_level in VALIDATION_GROUP_LEVELS:
            for group_value in sorted(metadata[group_level].unique()):
                keep = metadata[group_level].to_numpy() != group_value
                if not np.any(keep):
                    raise Task11ExplanationError(
                        f"Fold {fold} ablation for {group_value!r} would remove all features"
                    )
                estimator = _build_rf(
                    random_state=random_state + fold * 1000 + len(rows),
                    random_forest_estimators=random_forest_estimators,
                )
                estimator.fit(features[train_indices][:, keep], arrays["y"][train_indices])
                y_pred = estimator.predict(features[test_indices][:, keep])
                metrics = compute_metrics(arrays["y"][test_indices], y_pred, label_ids=label_ids)
                rows.append(
                    _validation_row(
                        method="ablation",
                        fold=fold,
                        group_level=group_level,
                        group_value=str(group_value),
                        baseline_metrics=full_metrics,
                        perturbed_metrics=metrics,
                    )
                )
    frame = pd.DataFrame(rows)
    return _add_validation_normalization(frame)


def _run_permutation(
    *,
    xrocket_dir: Path,
    folds: list[int],
    arrays: dict[str, np.ndarray],
    label_ids: tuple[int, ...],
    random_state: int,
    permutation_repeats: int,
) -> pd.DataFrame:
    if permutation_repeats <= 0:
        raise Task11ExplanationError("permutation_repeats must be positive")
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(random_state)
    for fold in folds:
        fold_dir = xrocket_dir / f"fold_{fold}"
        metadata = _metadata_with_groups(fold_dir)
        features, _train_indices, test_indices = _load_fold_features(fold_dir)
        full_model = joblib.load(fold_dir / f"{PRIMARY_MODEL}.joblib")
        full_pred = full_model.predict(features[test_indices])
        full_metrics = compute_metrics(arrays["y"][test_indices], full_pred, label_ids=label_ids)
        test_features = features[test_indices]
        for group_level in VALIDATION_GROUP_LEVELS:
            for group_value in sorted(metadata[group_level].unique()):
                columns = np.flatnonzero(metadata[group_level].to_numpy() == group_value)
                for repeat in range(permutation_repeats):
                    permuted = test_features.copy()
                    order = rng.permutation(len(test_indices))
                    permuted[:, columns] = permuted[order][:, columns]
                    y_pred = full_model.predict(permuted)
                    metrics = compute_metrics(
                        arrays["y"][test_indices],
                        y_pred,
                        label_ids=label_ids,
                    )
                    row = _validation_row(
                        method="permutation",
                        fold=fold,
                        group_level=group_level,
                        group_value=str(group_value),
                        baseline_metrics=full_metrics,
                        perturbed_metrics=metrics,
                    )
                    row["repeat"] = repeat
                    rows.append(row)
    frame = pd.DataFrame(rows)
    return _add_validation_normalization(frame)


def _validation_row(
    *,
    method: str,
    fold: int,
    group_level: str,
    group_value: str,
    baseline_metrics: dict[str, float],
    perturbed_metrics: dict[str, float],
) -> dict[str, Any]:
    return {
        "method": method,
        "fold": fold,
        "group_level": group_level,
        "group_value": group_value,
        "baseline_macro_f1": baseline_metrics["macro_f1"],
        "metric_macro_f1": perturbed_metrics["macro_f1"],
        "macro_f1_drop": baseline_metrics["macro_f1"] - perturbed_metrics["macro_f1"],
        "baseline_balanced_accuracy": baseline_metrics["balanced_accuracy"],
        "metric_balanced_accuracy": perturbed_metrics["balanced_accuracy"],
        "balanced_accuracy_drop": (
            baseline_metrics["balanced_accuracy"] - perturbed_metrics["balanced_accuracy"]
        ),
    }


def _add_validation_normalization(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["positive_macro_f1_drop"] = frame["macro_f1_drop"].clip(lower=0.0)
    frame["normalized_positive_drop"] = 0.0
    group_columns = ["method", "fold", "group_level"]
    for _key, group in frame.groupby(group_columns, sort=False):
        total = float(group["positive_macro_f1_drop"].sum())
        if total > 0.0:
            frame.loc[group.index, "normalized_positive_drop"] = (
                group["positive_macro_f1_drop"] / total
            )
    return frame


def _build_method_agreement(
    native_fold: pd.DataFrame,
    ablation: pd.DataFrame,
    permutation: pd.DataFrame,
) -> pd.DataFrame:
    native = native_fold.loc[native_fold["group_level"].isin(VALIDATION_GROUP_LEVELS)].copy()
    native_summary = summarize_fold_values(native, value_column="importance")
    native_summary = native_summary.rename(columns={"importance_mean": "native_importance_mean"})

    validation_rows: list[pd.DataFrame] = []
    for method_name, frame in (("ablation", ablation), ("permutation", permutation)):
        collapsed = _collapse_validation_for_summary(frame)
        summary = summarize_fold_values(collapsed, value_column="normalized_positive_drop")
        summary = summary.rename(
            columns={"normalized_positive_drop_mean": f"{method_name}_normalized_drop_mean"}
        )
        validation_rows.append(
            summary[
                [
                    "group_level",
                    "group_value",
                    f"{method_name}_normalized_drop_mean",
                    "mean_rank",
                ]
            ].rename(columns={"mean_rank": f"{method_name}_mean_rank"})
        )
    agreement = native_summary[
        ["group_level", "group_value", "native_importance_mean", "mean_rank"]
    ].rename(columns={"mean_rank": "native_mean_rank"})
    for frame in validation_rows:
        agreement = agreement.merge(frame, on=["group_level", "group_value"], how="left")
    for method_name in ("ablation", "permutation"):
        agreement[f"native_vs_{method_name}_rank_delta"] = (
            agreement["native_mean_rank"] - agreement[f"{method_name}_mean_rank"]
        ).abs()
    agreement["conflict_flag"] = (agreement["native_vs_ablation_rank_delta"] >= 3.0) | (
        agreement["native_vs_permutation_rank_delta"] >= 3.0
    )
    return agreement.sort_values(
        ["group_level", "native_importance_mean"],
        ascending=[True, False],
    )


def _metadata_with_groups(fold_dir: Path) -> pd.DataFrame:
    path = fold_dir / "feature_metadata.parquet"
    if not path.is_file():
        raise Task11ExplanationError(f"Missing feature metadata artifact: {path}")
    metadata = pd.read_parquet(path).copy()
    parsed = [parse_channel_name(str(channel)) for channel in metadata["channel_name"]]
    for column in ("sensor_family", "axis", "channel", "family_axis", "channel_combination"):
        metadata[column] = [row[column] for row in parsed]
    return metadata


def _load_fold_features(fold_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = fold_dir / "features.npz"
    if not path.is_file():
        raise Task11ExplanationError(f"Missing transformed features artifact: {path}")
    with np.load(path, allow_pickle=False) as data:
        return (
            data["features"].astype(np.float32, copy=False),
            data["train_indices"].astype(np.int64, copy=False),
            data["test_indices"].astype(np.int64, copy=False),
        )


def _build_rf(*, random_state: int, random_forest_estimators: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=random_forest_estimators,
        class_weight="balanced",
        max_features="sqrt",
        random_state=random_state,
        n_jobs=-1,
    )


def _write_figures(
    *,
    figures_dir: Path,
    native_fold: pd.DataFrame,
    native_summary: pd.DataFrame,
    stability: dict[str, pd.DataFrame],
    ablation: pd.DataFrame,
    class_specific_channel: pd.DataFrame,
    method_agreement: pd.DataFrame,
) -> None:
    family = _summary_for(native_summary, "sensor_family")
    channel = _summary_for(native_summary, "channel")
    family_axis = _summary_for(native_summary, "family_axis")
    _bar_with_error(
        family,
        value_column="importance_mean",
        error_column="importance_std",
        title="Sensor-family contribution to movement classification",
        xlabel="Normalized native importance",
        path_stem=figures_dir / "sensor_family_contribution",
    )
    _channel_heatmap(family_axis, figures_dir / "axis_channel_contribution_heatmap")
    _bar_with_error(
        channel,
        value_column="importance_mean",
        error_column="importance_std",
        title="Ranked sensor-axis channel contribution",
        xlabel="Normalized native importance",
        path_stem=figures_dir / "ranked_channel_importance",
        horizontal=True,
    )
    _stability_figure(
        native_fold,
        stability["rank_correlations"],
        figures_dir / "fold_stability",
    )
    ablation_family = (
        ablation.loc[ablation["group_level"] == "sensor_family"]
        .groupby("group_value", as_index=False)["macro_f1_drop"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(
            columns={
                "group_value": "group_value",
                "mean": "macro_f1_drop_mean",
                "std": "macro_f1_drop_std",
            }
        )
    )
    _bar_with_error(
        ablation_family,
        value_column="macro_f1_drop_mean",
        error_column="macro_f1_drop_std",
        title="Sensor-family ablation impact",
        xlabel="Macro-F1 drop after feature-group removal",
        path_stem=figures_dir / "ablation_impact",
    )
    _class_specific_heatmap(
        class_specific_channel,
        figures_dir / "class_specific_channel_profiles",
    )
    _method_agreement_figure(method_agreement, figures_dir / "method_agreement")


def _summary_for(native_summary: pd.DataFrame, group_level: str) -> pd.DataFrame:
    return native_summary.loc[native_summary["group_level"] == group_level].copy()


def _bar_with_error(
    frame: pd.DataFrame,
    *,
    value_column: str,
    error_column: str,
    title: str,
    xlabel: str,
    path_stem: Path,
    horizontal: bool = False,
) -> None:
    plot = frame.sort_values(value_column, ascending=horizontal).copy()
    labels = plot["group_value"].astype(str).to_numpy()
    values = plot[value_column].to_numpy(dtype=np.float64)
    errors = plot[error_column].fillna(0.0).to_numpy(dtype=np.float64)
    height = max(4.0, 0.35 * len(plot) + 1.5)
    fig, ax = plt.subplots(figsize=(8.5, height if horizontal else 4.8), constrained_layout=True)
    if horizontal:
        positions = np.arange(len(labels))
        ax.barh(positions, values, xerr=errors, color="#4C78A8")
        ax.set_yticks(positions, labels)
        ax.set_xlabel(xlabel)
    else:
        positions = np.arange(len(labels))
        ax.bar(positions, values, yerr=errors, color="#4C78A8", capsize=4)
        ax.set_xticks(positions, labels, rotation=30, ha="right")
        ax.set_ylabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="x" if horizontal else "y", alpha=0.25)
    _save_figure(fig, path_stem)


def _channel_heatmap(frame: pd.DataFrame, path_stem: Path) -> None:
    temp = frame.copy()
    parsed = [parse_channel_name(str(value)) for value in temp["group_value"]]
    temp["sensor_family"] = [row["sensor_family"] for row in parsed]
    temp["axis"] = [row["axis"] for row in parsed]
    pivot = temp.pivot(index="sensor_family", columns="axis", values="importance_mean").reindex(
        index=list(SENSOR_FAMILIES),
        columns=list(AXES),
    )
    fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    image = ax.imshow(pivot.to_numpy(dtype=np.float64), cmap="viridis")
    ax.set_xticks(np.arange(len(AXES)), AXES)
    ax.set_yticks(np.arange(len(SENSOR_FAMILIES)), SENSOR_FAMILIES)
    ax.set_title("Sensor-family by device-axis contribution")
    ax.set_xlabel("Device-frame axis")
    ax.set_ylabel("Sensor family")
    fig.colorbar(image, ax=ax, label="Mean normalized native importance")
    _save_figure(fig, path_stem)


def _stability_figure(
    native_fold: pd.DataFrame,
    rank_correlations: pd.DataFrame,
    path_stem: Path,
) -> None:
    channel = native_fold.loc[native_fold["group_level"] == "channel"]
    pivot = channel.pivot(index="fold", columns="group_value", values="importance").fillna(0.0)
    matrix = pivot.T.corr(method="spearman").to_numpy(dtype=np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), constrained_layout=True)
    image = axes[0].imshow(matrix, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    axes[0].set_title("Channel-rank stability by fold")
    axes[0].set_xlabel("Fold")
    axes[0].set_ylabel("Fold")
    axes[0].set_xticks(np.arange(len(pivot.index)), [str(value) for value in pivot.index])
    axes[0].set_yticks(np.arange(len(pivot.index)), [str(value) for value in pivot.index])
    fig.colorbar(image, ax=axes[0], label="Spearman correlation")
    channel_corr = rank_correlations.loc[rank_correlations["group_level"] == "channel"]
    if channel_corr.empty:
        axes[1].text(0.5, 0.5, "Only one fold available", ha="center", va="center")
        axes[1].set_xlim(0.0, 1.0)
    else:
        axes[1].hist(channel_corr["spearman_rank_correlation"], bins=8, color="#F58518")
    axes[1].set_title("Pairwise channel-rank correlations")
    axes[1].set_xlabel("Spearman correlation")
    axes[1].set_ylabel("Fold-pair count")
    _save_figure(fig, path_stem)


def _class_specific_heatmap(frame: pd.DataFrame, path_stem: Path) -> None:
    top_channels = (
        frame.groupby("group_value")["importance_mean"]
        .mean()
        .sort_values(ascending=False)
        .head(12)
        .index
    )
    selected = frame.loc[frame["group_value"].isin(top_channels)]
    pivot = selected.pivot(
        index="class_label",
        columns="group_value",
        values="importance_mean",
    ).fillna(0.0)
    fig, ax = plt.subplots(figsize=(11.0, 4.8), constrained_layout=True)
    image = ax.imshow(pivot.to_numpy(dtype=np.float64), cmap="magma")
    ax.set_xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)), [str(value) for value in pivot.index])
    ax.set_title("One-vs-rest class-specific channel profiles")
    ax.set_xlabel("Sensor-axis channel")
    ax.set_ylabel("Movement class")
    fig.colorbar(image, ax=ax, label="Mean normalized one-vs-rest importance")
    _save_figure(fig, path_stem)


def _method_agreement_figure(frame: pd.DataFrame, path_stem: Path) -> None:
    channel = frame.loc[frame["group_level"] == "channel"]
    fig, ax = plt.subplots(figsize=(6.5, 5.2), constrained_layout=True)
    ax.scatter(
        channel["native_importance_mean"],
        channel["ablation_normalized_drop_mean"],
        c=channel["permutation_normalized_drop_mean"],
        cmap="viridis",
        s=60,
    )
    ax.set_title("Native importance and validation agreement")
    ax.set_xlabel("Mean normalized native importance")
    ax.set_ylabel("Mean normalized ablation drop")
    ax.grid(alpha=0.25)
    _save_figure(fig, path_stem)


def _save_figure(fig: Any, path_stem: Path) -> None:
    fig.savefig(path_stem.with_suffix(".png"), dpi=200)
    fig.savefig(path_stem.with_suffix(".pdf"))
    plt.close(fig)


def _write_figure_captions(figures_dir: Path) -> None:
    captions = """# Task 1.1 Figure Captions

- `sensor_family_contribution`: Mean normalized random-forest native importance by
  sensor family with fold standard deviation.
- `axis_channel_contribution_heatmap`: Mean normalized native importance by sensor
  family and Android device-frame axis.
- `ranked_channel_importance`: Ranked sensor-axis channels with fold standard
  deviation.
- `fold_stability`: Fold-to-fold channel-rank stability using Spearman correlation.
- `ablation_impact`: Macro-F1 drop after removing saved XROCKET feature groups for
  each sensor family.
- `class_specific_channel_profiles`: One-vs-rest random-forest channel profiles by
  movement class.
- `method_agreement`: Agreement between native importance and validation drops for
  channels.
"""
    (figures_dir / "figure_captions.md").write_text(captions, encoding="utf-8")


def _build_task_answer(
    *,
    native_summary: pd.DataFrame,
    ablation: pd.DataFrame,
    permutation: pd.DataFrame,
    method_agreement: pd.DataFrame,
    class_specific_channel: pd.DataFrame,
) -> str:
    family = _summary_for(native_summary, "sensor_family").head(3)
    channel = _summary_for(native_summary, "channel").head(6)
    axis = _summary_for(native_summary, "axis").head(3)
    ablation_family = _validation_summary(ablation, "sensor_family").head(3)
    permutation_family = _validation_summary(permutation, "sensor_family").head(3)
    conflicts = method_agreement.loc[method_agreement["conflict_flag"]]
    class_top = (
        class_specific_channel.sort_values(
            ["class_label", "importance_mean"],
            ascending=[True, False],
        )
        .groupby("class_label")
        .head(2)
    )
    validation_columns = [
        "group_value",
        "macro_f1_drop_mean",
        "positive_macro_f1_drop_mean",
    ]
    return f"""# Task 1.1 Answer: Sensor-Axis Contribution

## Direct Answer

The M4 evidence indicates that the most useful signals are the top-ranked
sensor families and channels below. These values are normalized within each
participant-held-out fold before aggregation, so they describe relative model
use in the saved padded XROCKET representation rather than physical effect
sizes.

## Native Importance Evidence

Top sensor families:

{_markdown_table(family, ["group_value", "importance_mean", "importance_std", "mean_rank"])}

Top device-frame axes:

{_markdown_table(axis, ["group_value", "importance_mean", "importance_std", "mean_rank"])}

Top sensor-axis channels:

{_markdown_table(channel, ["group_value", "importance_mean", "importance_std", "mean_rank"])}

## Validation Evidence

Largest sensor-family macro-F1 drops under feature-group ablation:

{_markdown_table(ablation_family, validation_columns)}

Largest sensor-family macro-F1 drops under grouped test-set permutation:

{_markdown_table(permutation_family, validation_columns)}

## Class-Specific Profiles

Top one-vs-rest channels per class:

{_markdown_table(class_top, ["class_label", "group_value", "importance_mean", "importance_std"])}

## Biomechanical Interpretation

The observed rankings support a device-frame sensor contribution claim, not a
direct causal biomechanics claim. Channels with high native importance and
validation support are plausible indicators of forearm acceleration, angular
velocity, orientation-related gravity/rotation structure, or magnetic-field
variation during the Kenpo blocks. Because the phone is mounted on the forearm
in an Android device coordinate frame and both arms are included, anatomical
meaning must be treated as a protocol-based inference and checked against
arm-stratified diagnostics before any strong left/right or flexion/rotation
claim is made.

## Agreement And Caveats

Native, ablation, and permutation evidence should be read together. Conflict
flags were raised for {len(conflicts)} channel or sensor-family rows, so claims
should emphasize groups that are consistently high across native importance and
at least one validation method. The analysis remains specific to the padded M3
XROCKET representation, and feature-group ablation validates transformed
feature reliance rather than proving raw-sensor necessity.
"""


def _validation_summary(frame: pd.DataFrame, group_level: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    selected = _collapse_validation_for_summary(frame)
    selected = selected.loc[selected["group_level"] == group_level]
    for group_value, group in selected.groupby("group_value", sort=False):
        values = group["macro_f1_drop"].to_numpy(dtype=np.float64)
        positive = group["positive_macro_f1_drop"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "group_value": group_value,
                "macro_f1_drop_mean": float(np.mean(values)),
                "macro_f1_drop_std": float(np.std(values, ddof=0)),
                "positive_macro_f1_drop_mean": float(np.mean(positive)),
            }
        )
    return pd.DataFrame(rows).sort_values("positive_macro_f1_drop_mean", ascending=False)


def _collapse_validation_for_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse repeated validation rows to one value per fold and group."""
    value_columns = [
        "macro_f1_drop",
        "balanced_accuracy_drop",
        "positive_macro_f1_drop",
        "normalized_positive_drop",
    ]
    return (
        frame.groupby(["method", "fold", "group_level", "group_value"], as_index=False)[
            value_columns
        ]
        .mean()
        .copy()
    )


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
