"""Task 1.2 dilation and temporal-scale contribution analysis."""

from __future__ import annotations

import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ksas_xrocket.audit import EXPECTED_LABELS, sha256_file, write_json
from ksas_xrocket.baselines import git_commit, load_processed_arrays, package_version
from ksas_xrocket.task_1_1_explain import normalize_importance

TEMPORAL_GROUP_LEVELS = ("dilation", "temporal_scale_bin")
SCALE_BIN_ORDER = ("short", "intermediate", "long")
BIN_RULE_DESCRIPTION = (
    "short: span/target_length <= 1/3; intermediate: 1/3 < span/target_length <= 2/3; "
    "long: span/target_length > 2/3"
)


class Task12ExplanationError(ValueError):
    """Raised when Task 1.2 explanations cannot be generated safely."""


@dataclass(frozen=True)
class Task12ExplanationResult:
    """Paths returned by the Task 1.2 explanation workflow."""

    output_dir: Path
    feature_importance_path: Path
    important_features_path: Path
    dilation_summary_path: Path
    temporal_scale_summary_path: Path
    answer_path: Path
    provenance_path: Path


def effective_span_samples(dilation: int, kernel_length: int) -> int:
    """Return the effective temporal span for a dilated XROCKET kernel."""
    if dilation <= 0:
        raise Task12ExplanationError("dilation must be positive")
    if kernel_length <= 0:
        raise Task12ExplanationError("kernel_length must be positive")
    return 1 + dilation * (kernel_length - 1)


def nominal_seconds(span_samples: int, sampling_rate_hz: float) -> float:
    """Convert a sample span to approximate seconds using a nominal rate."""
    if span_samples <= 0:
        raise Task12ExplanationError("span_samples must be positive")
    if sampling_rate_hz <= 0.0:
        raise Task12ExplanationError("sampling_rate_hz must be positive")
    return span_samples / sampling_rate_hz


def temporal_scale_bin(span_samples: int, *, target_length: int = 56) -> str:
    """Assign a temporal-scale bin using the predeclared thirds-of-window rule."""
    if span_samples <= 0:
        raise Task12ExplanationError("span_samples must be positive")
    if target_length <= 0:
        raise Task12ExplanationError("target_length must be positive")
    relative_span = span_samples / target_length
    if relative_span <= 1.0 / 3.0:
        return "short"
    if relative_span <= 2.0 / 3.0:
        return "intermediate"
    return "long"


def classify_temporal_evidence(scale_summary: pd.DataFrame) -> str:
    """Classify the global temporal-scale profile from mean normalized importance."""
    required = {"group_value", "importance_mean"}
    missing = sorted(required.difference(scale_summary.columns))
    if missing:
        raise Task12ExplanationError(f"Scale summary is missing columns: {', '.join(missing)}")
    values = {
        str(row["group_value"]): float(row["importance_mean"])
        for row in scale_summary.to_dict("records")
    }
    if not values:
        raise Task12ExplanationError("Scale summary is empty")
    short = values.get("short", 0.0)
    long = values.get("long", 0.0)
    next_after_short = max((value for key, value in values.items() if key != "short"), default=0.0)
    next_after_long = max((value for key, value in values.items() if key != "long"), default=0.0)
    if short >= 0.50 and short >= 1.5 * next_after_short:
        return "short-scale"
    if long >= 0.50 and long >= 1.5 * next_after_long:
        return "long-scale"
    leader = max(values.items(), key=lambda item: item[1])[0]
    return f"mixed temporal-scale evidence; largest contribution: {leader}"


def aggregate_temporal_importance(
    frame: pd.DataFrame,
    *,
    group_level: str,
    value_column: str = "normalized_importance",
) -> pd.DataFrame:
    """Aggregate normalized temporal importance to one grouping level per fold."""
    required = {"fold", group_level, value_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise Task12ExplanationError(f"Importance frame is missing columns: {', '.join(missing)}")
    grouped = (
        frame.groupby(["fold", group_level], sort=False, as_index=False)[value_column]
        .sum()
        .rename(columns={group_level: "group_value", value_column: "importance"})
    )
    grouped.insert(1, "group_level", group_level)
    return grouped


def summarize_temporal_importance(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize fold-level temporal importance with rank variation."""
    required = {"fold", "group_level", "group_value", "importance"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise Task12ExplanationError(f"Summary frame is missing columns: {', '.join(missing)}")
    ranked = frame.copy()
    ranked["rank"] = ranked.groupby(["fold", "group_level"])["importance"].rank(
        method="average",
        ascending=False,
    )
    rows: list[dict[str, Any]] = []
    for (group_level, group_value), group in ranked.groupby(["group_level", "group_value"]):
        values = group["importance"].to_numpy(dtype=np.float64)
        ranks = group["rank"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "group_level": group_level,
                "group_value": group_value,
                "fold_count": int(len(group)),
                "importance_mean": float(np.mean(values)),
                "importance_std": float(np.std(values, ddof=0)),
                "importance_min": float(np.min(values)),
                "importance_max": float(np.max(values)),
                "mean_rank": float(np.mean(ranks)),
                "rank_std": float(np.std(ranks, ddof=0)),
            }
        )
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    return summary.sort_values(["group_level", "importance_mean"], ascending=[True, False])


def run_task_1_2_explanation(
    *,
    processed_dir: Path,
    xrocket_dir: Path,
    output_dir: Path,
    label_ids: tuple[int, ...] = tuple(EXPECTED_LABELS),
    random_state: int = 42,
    random_forest_estimators: int = 500,
    top_k: int = 2,
    target_length: int = 56,
    nominal_sampling_rate_hz: float = 50.0,
    overwrite: bool = False,
    resolved_config: dict[str, Any] | None = None,
) -> Task12ExplanationResult:
    """Run Task 1.2 analysis from saved M3 XROCKET artifacts."""
    _prepare_output_dir(output_dir, overwrite=overwrite)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    arrays = load_processed_arrays(processed_dir / "tensors.npz")
    folds = _discover_folds(xrocket_dir)
    if not folds:
        raise Task12ExplanationError(f"No fold directories found in {xrocket_dir}")

    native = _load_temporal_feature_importance(
        xrocket_dir=xrocket_dir,
        folds=folds,
        target_length=target_length,
        nominal_sampling_rate_hz=nominal_sampling_rate_hz,
    )
    feature_importance_path = output_dir / "fold_temporal_feature_importance.parquet"
    native.to_parquet(feature_importance_path, index=False)

    important = native.loc[native["native_importance"] > 0.0].copy()
    important_features_path = output_dir / "important_temporal_features.parquet"
    important.to_parquet(important_features_path, index=False)

    span_mapping = _build_span_mapping(
        native,
        target_length=target_length,
        nominal_sampling_rate_hz=nominal_sampling_rate_hz,
    )
    span_mapping.to_csv(output_dir / "temporal_span_mapping.csv", index=False)

    dilation_fold = aggregate_temporal_importance(native, group_level="dilation")
    scale_fold = aggregate_temporal_importance(native, group_level="temporal_scale_bin")
    dilation_fold.to_csv(output_dir / "dilation_fold_importance.csv", index=False)
    scale_fold.to_csv(output_dir / "temporal_scale_fold_importance.csv", index=False)

    temporal_fold = pd.concat([dilation_fold, scale_fold], ignore_index=True)
    temporal_summary = summarize_temporal_importance(temporal_fold)
    dilation_summary = temporal_summary.loc[temporal_summary["group_level"] == "dilation"].copy()
    scale_summary = temporal_summary.loc[
        temporal_summary["group_level"] == "temporal_scale_bin"
    ].copy()
    dilation_summary_path = output_dir / "dilation_importance_summary.csv"
    temporal_scale_summary_path = output_dir / "temporal_scale_importance_summary.csv"
    dilation_summary.to_csv(dilation_summary_path, index=False)
    scale_summary.to_csv(temporal_scale_summary_path, index=False)

    stability = _compute_stability(temporal_fold, top_k=top_k)
    stability["rank_correlations"].to_csv(
        output_dir / "stability_rank_correlations.csv",
        index=False,
    )
    stability["topk_overlap"].to_csv(output_dir / "stability_topk_overlap.csv", index=False)

    class_specific = _compute_class_specific_temporal_importance(
        xrocket_dir=xrocket_dir,
        folds=folds,
        arrays=arrays,
        label_ids=label_ids,
        random_state=random_state,
        random_forest_estimators=random_forest_estimators,
        target_length=target_length,
        nominal_sampling_rate_hz=nominal_sampling_rate_hz,
    )
    class_specific["dilation"].to_csv(
        output_dir / "class_specific_dilation_importance.csv",
        index=False,
    )
    class_specific["temporal_scale_bin"].to_csv(
        output_dir / "class_specific_temporal_scale_importance.csv",
        index=False,
    )

    padding = _summarize_padding_temporal_diagnostics(xrocket_dir=xrocket_dir, folds=folds)
    padding.to_csv(output_dir / "padding_temporal_diagnostics.csv", index=False)

    _write_figures(
        figures_dir=figures_dir,
        span_mapping=span_mapping,
        dilation_summary=dilation_summary,
        scale_summary=scale_summary,
        class_specific_scale=class_specific["temporal_scale_bin"],
        temporal_fold=temporal_fold,
        stability=stability,
    )
    _write_figure_captions(figures_dir)

    classification = classify_temporal_evidence(scale_summary)
    answer_path = output_dir / "task_1_2_answer.md"
    answer_path.write_text(
        _build_task_answer(
            span_mapping=span_mapping,
            dilation_summary=dilation_summary,
            scale_summary=scale_summary,
            class_specific_scale=class_specific["temporal_scale_bin"],
            padding=padding,
            classification=classification,
        ),
        encoding="utf-8",
    )

    write_json(
        output_dir / "resolved_config.json",
        resolved_config
        or {
            "task": "task_1_2",
            "processed_dir": processed_dir.as_posix(),
            "xrocket_dir": xrocket_dir.as_posix(),
            "output_dir": output_dir.as_posix(),
            "classes": list(label_ids),
            "random_state": random_state,
            "random_forest_estimators": random_forest_estimators,
            "top_k": top_k,
            "target_length": target_length,
            "nominal_sampling_rate_hz": nominal_sampling_rate_hz,
            "temporal_bin_rule": BIN_RULE_DESCRIPTION,
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
            "temporal_bin_rule": BIN_RULE_DESCRIPTION,
            "packages": {
                package: package_version(package)
                for package in ("numpy", "pandas", "matplotlib", "scikit-learn")
            },
        },
    )

    return Task12ExplanationResult(
        output_dir=output_dir,
        feature_importance_path=feature_importance_path,
        important_features_path=important_features_path,
        dilation_summary_path=dilation_summary_path,
        temporal_scale_summary_path=temporal_scale_summary_path,
        answer_path=answer_path,
        provenance_path=provenance_path,
    )


def _prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise Task12ExplanationError(
                f"Output directory is not empty: {output_dir}; pass --overwrite to replace it"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _discover_folds(xrocket_dir: Path) -> list[int]:
    if not xrocket_dir.is_dir():
        raise Task12ExplanationError(f"M3 XROCKET directory not found: {xrocket_dir}")
    folds: list[int] = []
    for path in xrocket_dir.iterdir():
        if path.is_dir() and path.name.startswith("fold_"):
            try:
                folds.append(int(path.name.removeprefix("fold_")))
            except ValueError:
                continue
    return sorted(folds)


def _load_temporal_feature_importance(
    *,
    xrocket_dir: Path,
    folds: list[int],
    target_length: int,
    nominal_sampling_rate_hz: float,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    required_columns = {
        "feature_index",
        "kernel_id",
        "dilation",
        "kernel_length",
        "effective_receptive_field_samples",
        "effective_receptive_field_seconds_nominal",
        "threshold",
        "channel_name",
        "importance",
    }
    for fold in folds:
        path = xrocket_dir / f"fold_{fold}" / "feature_importance.parquet"
        if not path.is_file():
            raise Task12ExplanationError(f"Missing feature importance artifact: {path}")
        frame = pd.read_parquet(path).copy()
        missing = sorted(required_columns.difference(frame.columns))
        if missing:
            raise Task12ExplanationError(
                f"Importance artifact has unexpected schema {path}: missing {', '.join(missing)}"
            )
        _validate_temporal_metadata(
            frame,
            fold=fold,
            nominal_sampling_rate_hz=nominal_sampling_rate_hz,
        )
        frame["fold"] = fold
        frame["native_importance"] = frame["importance"].astype(float)
        frame["normalized_importance"] = normalize_importance(
            frame["native_importance"],
            fold=fold,
        )
        frame["temporal_scale_bin"] = [
            temporal_scale_bin(int(span), target_length=target_length)
            for span in frame["effective_receptive_field_samples"]
        ]
        frame["temporal_bin_rule"] = BIN_RULE_DESCRIPTION
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _validate_temporal_metadata(
    frame: pd.DataFrame,
    *,
    fold: int,
    nominal_sampling_rate_hz: float,
) -> None:
    unique_spans = frame[
        ["dilation", "kernel_length", "effective_receptive_field_samples"]
    ].drop_duplicates()
    for row in unique_spans.to_dict("records"):
        expected = effective_span_samples(int(row["dilation"]), int(row["kernel_length"]))
        actual = int(row["effective_receptive_field_samples"])
        if actual != expected:
            raise Task12ExplanationError(
                f"Fold {fold} dilation {row['dilation']} has span {actual}, expected {expected}"
            )
    expected_seconds = frame["effective_receptive_field_samples"].astype(float).to_numpy()
    expected_seconds = expected_seconds / nominal_sampling_rate_hz
    actual_seconds = frame["effective_receptive_field_seconds_nominal"].astype(float).to_numpy()
    if not np.allclose(actual_seconds, expected_seconds, rtol=1e-8, atol=1e-10):
        raise Task12ExplanationError(
            f"Fold {fold} nominal seconds do not match the configured sampling rate"
        )


def _build_span_mapping(
    native: pd.DataFrame,
    *,
    target_length: int,
    nominal_sampling_rate_hz: float,
) -> pd.DataFrame:
    mapping = native[
        [
            "dilation",
            "kernel_length",
            "effective_receptive_field_samples",
            "effective_receptive_field_seconds_nominal",
            "temporal_scale_bin",
        ]
    ].drop_duplicates()
    mapping = mapping.sort_values("dilation").copy()
    mapping["relative_span_of_padded_window"] = (
        mapping["effective_receptive_field_samples"] / target_length
    )
    mapping["target_length_samples"] = target_length
    mapping["nominal_sampling_rate_hz"] = nominal_sampling_rate_hz
    mapping["temporal_bin_rule"] = BIN_RULE_DESCRIPTION
    return mapping


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


def _compute_class_specific_temporal_importance(
    *,
    xrocket_dir: Path,
    folds: list[int],
    arrays: dict[str, np.ndarray],
    label_ids: tuple[int, ...],
    random_state: int,
    random_forest_estimators: int,
    target_length: int,
    nominal_sampling_rate_hz: float,
) -> dict[str, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for fold in folds:
        fold_dir = xrocket_dir / f"fold_{fold}"
        metadata = _load_temporal_metadata(
            fold_dir,
            fold=fold,
            target_length=target_length,
            nominal_sampling_rate_hz=nominal_sampling_rate_hz,
        )
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
            temp = metadata[
                [
                    "dilation",
                    "temporal_scale_bin",
                    "effective_receptive_field_samples",
                    "effective_receptive_field_seconds_nominal",
                ]
            ].copy()
            temp["importance"] = normalized
            for group_level in TEMPORAL_GROUP_LEVELS:
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
    summary_rows: list[dict[str, Any]] = []
    for (class_label, group_level, group_value), group in fold_frame.groupby(
        ["class_label", "group_level", "group_value"],
        sort=False,
    ):
        values = group["importance"].to_numpy(dtype=np.float64)
        summary_rows.append(
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
    class_summary = pd.DataFrame(summary_rows).sort_values(
        ["class_label", "group_level", "importance_mean"],
        ascending=[True, True, False],
    )
    return {
        "dilation": class_summary.loc[class_summary["group_level"] == "dilation"].copy(),
        "temporal_scale_bin": class_summary.loc[
            class_summary["group_level"] == "temporal_scale_bin"
        ].copy(),
    }


def _load_temporal_metadata(
    fold_dir: Path,
    *,
    fold: int,
    target_length: int,
    nominal_sampling_rate_hz: float,
) -> pd.DataFrame:
    path = fold_dir / "feature_metadata.parquet"
    if not path.is_file():
        raise Task12ExplanationError(f"Missing feature metadata artifact: {path}")
    metadata = pd.read_parquet(path).copy()
    _validate_temporal_metadata(
        metadata,
        fold=fold,
        nominal_sampling_rate_hz=nominal_sampling_rate_hz,
    )
    metadata["temporal_scale_bin"] = [
        temporal_scale_bin(int(span), target_length=target_length)
        for span in metadata["effective_receptive_field_samples"]
    ]
    return metadata


def _load_fold_features(fold_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = fold_dir / "features.npz"
    if not path.is_file():
        raise Task12ExplanationError(f"Missing transformed features artifact: {path}")
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


def _summarize_padding_temporal_diagnostics(*, xrocket_dir: Path, folds: list[int]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for fold in folds:
        fold_dir = xrocket_dir / f"fold_{fold}"
        feature_path = fold_dir / "padding_feature_summary.csv"
        threshold_path = fold_dir / "padding_threshold_diagnostics.csv"
        if not feature_path.is_file() or not threshold_path.is_file():
            continue
        feature = pd.read_csv(feature_path)
        thresholds = pd.read_csv(threshold_path)
        merged = feature.merge(thresholds, on=["fold", "dilation"], how="left")
        rows.append(merged)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _write_figures(
    *,
    figures_dir: Path,
    span_mapping: pd.DataFrame,
    dilation_summary: pd.DataFrame,
    scale_summary: pd.DataFrame,
    class_specific_scale: pd.DataFrame,
    temporal_fold: pd.DataFrame,
    stability: dict[str, pd.DataFrame],
) -> None:
    _dilation_importance_figure(dilation_summary, span_mapping, figures_dir / "dilation_importance")
    _scale_contribution_figure(scale_summary, figures_dir / "temporal_scale_contribution")
    _class_specific_scale_figure(
        class_specific_scale,
        figures_dir / "class_specific_scale_profiles",
    )
    _fold_stability_figure(
        temporal_fold,
        stability["rank_correlations"],
        figures_dir / "fold_stability",
    )


def _dilation_importance_figure(
    dilation_summary: pd.DataFrame,
    span_mapping: pd.DataFrame,
    path_stem: Path,
) -> None:
    plot = dilation_summary.sort_values("group_value").copy()
    spans = span_mapping.set_index("dilation")["effective_receptive_field_samples"].to_dict()
    labels = [f"d={int(value)}\n{int(spans[int(value)])} samples" for value in plot["group_value"]]
    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    ax.bar(
        np.arange(len(plot)),
        plot["importance_mean"].to_numpy(dtype=np.float64),
        yerr=plot["importance_std"].fillna(0.0).to_numpy(dtype=np.float64),
        color="#4C78A8",
        capsize=4,
    )
    ax.set_xticks(np.arange(len(plot)), labels)
    ax.set_title("Dilation contribution to movement classification")
    ax.set_ylabel("Mean normalized native importance")
    ax.set_xlabel("XROCKET dilation and effective receptive field")
    ax.grid(axis="y", alpha=0.25)
    _save_figure(fig, path_stem)


def _scale_contribution_figure(scale_summary: pd.DataFrame, path_stem: Path) -> None:
    plot = _ordered_scale_frame(scale_summary)
    fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    ax.bar(
        np.arange(len(plot)),
        plot["importance_mean"].to_numpy(dtype=np.float64),
        yerr=plot["importance_std"].fillna(0.0).to_numpy(dtype=np.float64),
        color="#54A24B",
        capsize=4,
    )
    ax.set_xticks(np.arange(len(plot)), plot["group_value"].astype(str).to_numpy())
    ax.set_title("Temporal-scale contribution")
    ax.set_ylabel("Mean normalized native importance")
    ax.set_xlabel("Predeclared temporal-scale bin")
    ax.grid(axis="y", alpha=0.25)
    _save_figure(fig, path_stem)


def _class_specific_scale_figure(frame: pd.DataFrame, path_stem: Path) -> None:
    temp = frame.copy()
    temp["group_value"] = pd.Categorical(
        temp["group_value"],
        categories=list(SCALE_BIN_ORDER),
        ordered=True,
    )
    pivot = (
        temp.pivot(
            index="class_label",
            columns="group_value",
            values="importance_mean",
        )
        .reindex(columns=list(SCALE_BIN_ORDER))
        .fillna(0.0)
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    image = ax.imshow(pivot.to_numpy(dtype=np.float64), cmap="magma")
    ax.set_xticks(np.arange(len(pivot.columns)), pivot.columns.astype(str))
    ax.set_yticks(np.arange(len(pivot.index)), [str(value) for value in pivot.index])
    ax.set_title("Class-specific temporal-scale profiles")
    ax.set_xlabel("Temporal-scale bin")
    ax.set_ylabel("Movement class")
    fig.colorbar(image, ax=ax, label="Mean normalized one-vs-rest importance")
    _save_figure(fig, path_stem)


def _fold_stability_figure(
    temporal_fold: pd.DataFrame,
    rank_correlations: pd.DataFrame,
    path_stem: Path,
) -> None:
    dilation = temporal_fold.loc[temporal_fold["group_level"] == "dilation"]
    pivot = dilation.pivot(index="fold", columns="group_value", values="importance").fillna(0.0)
    matrix = pivot.T.corr(method="spearman").to_numpy(dtype=np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), constrained_layout=True)
    image = axes[0].imshow(matrix, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    axes[0].set_title("Dilation-rank stability by fold")
    axes[0].set_xlabel("Fold")
    axes[0].set_ylabel("Fold")
    axes[0].set_xticks(np.arange(len(pivot.index)), [str(value) for value in pivot.index])
    axes[0].set_yticks(np.arange(len(pivot.index)), [str(value) for value in pivot.index])
    fig.colorbar(image, ax=axes[0], label="Spearman correlation")
    dilation_corr = rank_correlations.loc[rank_correlations["group_level"] == "dilation"]
    if dilation_corr.empty:
        axes[1].text(0.5, 0.5, "Only one fold available", ha="center", va="center")
        axes[1].set_xlim(0.0, 1.0)
    else:
        axes[1].hist(dilation_corr["spearman_rank_correlation"], bins=8, color="#F58518")
    axes[1].set_title("Pairwise dilation-rank correlations")
    axes[1].set_xlabel("Spearman correlation")
    axes[1].set_ylabel("Fold-pair count")
    _save_figure(fig, path_stem)


def _ordered_scale_frame(frame: pd.DataFrame) -> pd.DataFrame:
    temp = frame.copy()
    temp["group_value"] = pd.Categorical(
        temp["group_value"],
        categories=list(SCALE_BIN_ORDER),
        ordered=True,
    )
    return temp.sort_values("group_value")


def _save_figure(fig: Any, path_stem: Path) -> None:
    fig.savefig(path_stem.with_suffix(".png"), dpi=200)
    fig.savefig(path_stem.with_suffix(".pdf"))
    plt.close(fig)


def _write_figure_captions(figures_dir: Path) -> None:
    captions = """# Task 1.2 Figure Captions

- `dilation_importance`: Mean normalized random-forest native importance by
  XROCKET dilation with effective receptive-field spans in samples.
- `temporal_scale_contribution`: Mean normalized native importance by the
  predeclared short, intermediate, and long temporal-scale bins.
- `class_specific_scale_profiles`: One-vs-rest random-forest temporal-scale
  profiles by movement class.
- `fold_stability`: Fold-to-fold dilation-rank stability using Spearman
  correlation.
"""
    (figures_dir / "figure_captions.md").write_text(captions, encoding="utf-8")


def _build_task_answer(
    *,
    span_mapping: pd.DataFrame,
    dilation_summary: pd.DataFrame,
    scale_summary: pd.DataFrame,
    class_specific_scale: pd.DataFrame,
    padding: pd.DataFrame,
    classification: str,
) -> str:
    class_profiles = _class_profile_summary(class_specific_scale)
    padding_summary = _padding_summary_for_answer(padding)
    dilation_table = _markdown_table(
        dilation_summary,
        ["group_value", "importance_mean", "importance_std", "mean_rank"],
    )
    scale_table = _markdown_table(
        scale_summary,
        ["group_value", "importance_mean", "importance_std", "mean_rank"],
    )
    class_profile_table = _markdown_table(
        class_profiles,
        ["class_label", "classification", "largest_scale", "largest_importance"],
    )
    return f"""# Task 1.2 Answer: Temporal-Scale Analysis

## Direct Answer

The Task 1.2 evidence classifies the saved padded XROCKET representation as
**{classification}**. Importance is normalized within each participant-held-out
fold before aggregation. Dilation is interpreted as an effective temporal span,
not as a Fourier frequency.

## Temporal Span Mapping

{
        _markdown_table(
            span_mapping,
            [
                "dilation",
                "kernel_length",
                "effective_receptive_field_samples",
                "effective_receptive_field_seconds_nominal",
                "relative_span_of_padded_window",
                "temporal_scale_bin",
            ],
        )
    }

The seconds column is approximate at nominal 50 Hz. The KSAS Android app
requested this rate, but realized sensor timing and jitter were not retained in
the CSV exports.

## Native Importance Evidence

Dilation ranking:

{dilation_table}

Temporal-scale ranking:

{scale_table}

## Class-Specific Profiles

Secondary one-vs-rest temporal-scale profiles:

{class_profile_table}

## Padding Caveat

{padding_summary}

These temporal claims are traceable to saved XROCKET metadata:
`dilation`, `kernel_length`, and `effective_receptive_field_samples`. They
should still be read as evidence about the padded M3 feature representation,
not as timing-verified claims about the original Android sensor stream.
"""


def _class_profile_summary(class_specific_scale: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for class_label, group in class_specific_scale.groupby("class_label", sort=True):
        classification = classify_temporal_evidence(
            group.rename(columns={"importance_mean": "importance_mean"})
        )
        leader = group.sort_values("importance_mean", ascending=False).iloc[0]
        rows.append(
            {
                "class_label": int(class_label),
                "classification": classification,
                "largest_scale": str(leader["group_value"]),
                "largest_importance": float(leader["importance_mean"]),
            }
        )
    return pd.DataFrame(rows)


def _padding_summary_for_answer(padding: pd.DataFrame) -> str:
    if padding.empty:
        return "No padding diagnostics were available for the temporal-scale answer."
    test = padding.loc[padding["split"] == "test"].copy()
    if test.empty:
        test = padding.copy()
    delta_mean = float(test["mean_absolute_feature_delta"].mean())
    zero_threshold = float(test["zero_threshold_fraction"].mean())
    return (
        "Existing M3 padding diagnostics are retained as a guardrail: across the "
        f"available temporal diagnostic rows, the mean absolute feature delta was "
        f"{delta_mean:.4f}, and the mean zero-threshold fraction was {zero_threshold:.4f}. "
        "This supports reporting temporal-scale evidence with a padded-representation caveat."
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
