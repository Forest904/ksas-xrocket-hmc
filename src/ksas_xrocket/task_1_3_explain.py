"""Task 1.3 discriminative-pattern interpretation analysis."""

from __future__ import annotations

import ast
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from ksas_xrocket.audit import (
    EXPECTED_CHANNELS,
    EXPECTED_LABELS,
    SENSOR_UNITS,
    sha256_file,
    write_json,
)
from ksas_xrocket.baselines import git_commit, load_processed_arrays, package_version
from ksas_xrocket.task_1_1_explain import normalize_importance, parse_channel_name
from ksas_xrocket.xrocket_experiment import PRIMARY_MODEL


class Task13ExplanationError(ValueError):
    """Raised when Task 1.3 explanations cannot be generated safely."""


@dataclass(frozen=True)
class Task13ExplanationResult:
    """Paths returned by the Task 1.3 explanation workflow."""

    output_dir: Path
    selected_patterns_path: Path
    pattern_cases_path: Path
    response_traces_path: Path
    feature_distributions_path: Path
    answer_path: Path
    provenance_path: Path


@dataclass(frozen=True)
class ResponseLocalization:
    """Localized representative interval for one PPV feature response."""

    response_start_index: int
    response_end_index: int
    response_max_index: int
    processed_start_index: int
    processed_end_index: int
    original_start_index: int
    original_end_index: int
    touches_edge_padding: bool
    touches_right_padding: bool
    edge_padding_fraction: float
    right_padding_fraction: float
    above_threshold_count: int
    localized_above_threshold_count: int
    max_margin: float


def run_task_1_3_explanation(
    *,
    processed_dir: Path,
    xrocket_dir: Path,
    output_dir: Path,
    label_ids: tuple[int, ...] = tuple(EXPECTED_LABELS),
    stable_candidate_count: int = 12,
    correct_case_count: int = 3,
    include_failure_case: bool = True,
    target_length: int = 56,
    nominal_sampling_rate_hz: float = 50.0,
    primary_model: str = PRIMARY_MODEL,
    overwrite: bool = False,
    resolved_config: dict[str, Any] | None = None,
) -> Task13ExplanationResult:
    """Run Task 1.3 pattern interpretation from saved M3 XROCKET artifacts."""
    if stable_candidate_count <= 0:
        raise Task13ExplanationError("stable_candidate_count must be positive")
    if correct_case_count <= 0:
        raise Task13ExplanationError("correct_case_count must be positive")
    if target_length <= 0:
        raise Task13ExplanationError("target_length must be positive")
    if nominal_sampling_rate_hz <= 0.0:
        raise Task13ExplanationError("nominal_sampling_rate_hz must be positive")

    _prepare_output_dir(output_dir, overwrite=overwrite)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    arrays = load_processed_arrays(processed_dir / "tensors.npz")
    metadata = _load_processed_metadata(processed_dir)
    folds = _discover_folds(xrocket_dir)
    if not folds:
        raise Task13ExplanationError(f"No fold directories found in {xrocket_dir}")

    predictions = _load_primary_predictions(xrocket_dir, primary_model=primary_model)
    selected = select_stable_patterns(
        xrocket_dir=xrocket_dir,
        folds=folds,
        stable_candidate_count=stable_candidate_count,
    )
    distributions = build_feature_distributions(
        xrocket_dir=xrocket_dir,
        folds=folds,
        selected=selected,
        predictions=predictions,
        label_ids=label_ids,
    )
    selected = add_pattern_class_summaries(selected, distributions)
    selected_patterns_path = output_dir / "selected_patterns.csv"
    selected.to_csv(selected_patterns_path, index=False)

    case_specs = select_pattern_cases(
        selected=selected,
        distributions=distributions,
        predictions=predictions,
        correct_case_count=correct_case_count,
        include_failure_case=include_failure_case,
    )
    case_rows, trace_rows = materialize_pattern_cases(
        xrocket_dir=xrocket_dir,
        arrays=arrays,
        metadata=metadata,
        selected=selected,
        case_specs=case_specs,
        target_length=target_length,
        nominal_sampling_rate_hz=nominal_sampling_rate_hz,
    )
    cases = pd.DataFrame(case_rows)
    traces = pd.DataFrame(trace_rows)
    pattern_cases_path = output_dir / "pattern_cases.csv"
    response_traces_path = output_dir / "pattern_response_traces.parquet"
    feature_distributions_path = output_dir / "pattern_feature_distributions.csv"
    cases.to_csv(pattern_cases_path, index=False)
    traces.to_parquet(response_traces_path, index=False)
    distributions.to_csv(feature_distributions_path, index=False)

    _write_figures(
        figures_dir=figures_dir,
        arrays=arrays,
        cases=cases,
        traces=traces,
        distributions=distributions,
        selected=selected,
        target_length=target_length,
    )
    _write_figure_captions(figures_dir)

    answer_path = output_dir / "task_1_3_answer.md"
    answer_path.write_text(
        _build_task_answer(selected=selected, cases=cases),
        encoding="utf-8",
    )

    write_json(
        output_dir / "resolved_config.json",
        resolved_config
        or {
            "task": "task_1_3",
            "processed_dir": processed_dir.as_posix(),
            "xrocket_dir": xrocket_dir.as_posix(),
            "output_dir": output_dir.as_posix(),
            "classes": list(label_ids),
            "patterns": {
                "stable_candidate_count": stable_candidate_count,
                "correct_case_count": correct_case_count,
                "include_failure_case": include_failure_case,
                "target_length": target_length,
                "nominal_sampling_rate_hz": nominal_sampling_rate_hz,
                "primary_model": primary_model,
            },
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
            "localization": (
                "PPV features are localized to a representative strongest "
                "above-threshold response interval; PPV does not imply a unique instant."
            ),
            "packages": {
                package: package_version(package)
                for package in ("numpy", "pandas", "matplotlib", "scikit-learn", "torch")
            },
        },
    )
    return Task13ExplanationResult(
        output_dir=output_dir,
        selected_patterns_path=selected_patterns_path,
        pattern_cases_path=pattern_cases_path,
        response_traces_path=response_traces_path,
        feature_distributions_path=feature_distributions_path,
        answer_path=answer_path,
        provenance_path=provenance_path,
    )


def select_stable_patterns(
    *,
    xrocket_dir: Path,
    folds: list[int],
    stable_candidate_count: int,
) -> pd.DataFrame:
    """Select stable, high-importance features across participant-held-out folds."""
    frames: list[pd.DataFrame] = []
    signature_columns = _signature_columns()
    for fold in folds:
        path = xrocket_dir / f"fold_{fold}" / "feature_importance.parquet"
        if not path.is_file():
            raise Task13ExplanationError(f"Missing feature importance artifact: {path}")
        frame = pd.read_parquet(path).copy()
        missing = sorted(set(signature_columns + ["importance", "threshold"]).difference(frame))
        if missing:
            raise Task13ExplanationError(
                f"Feature importance artifact is missing columns: {', '.join(missing)}"
            )
        frame["fold"] = fold
        frame["native_importance"] = frame["importance"].astype(float)
        frame["normalized_importance"] = normalize_importance(
            frame["native_importance"],
            fold=fold,
        )
        frames.append(frame)
    all_features = pd.concat(frames, ignore_index=True)
    grouped = all_features.groupby(signature_columns, dropna=False, sort=False)
    rows: list[dict[str, Any]] = []
    for key, group in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        values = group["normalized_importance"].to_numpy(dtype=np.float64)
        native = group["native_importance"].to_numpy(dtype=np.float64)
        thresholds = group["threshold"].to_numpy(dtype=np.float64)
        row = {column: key[index] for index, column in enumerate(signature_columns)}
        row.update(
            {
                "fold_count": int(group["fold"].nunique()),
                "nonzero_fold_count": int(np.sum(native > 0.0)),
                "mean_normalized_importance": float(np.mean(values)),
                "std_normalized_importance": float(np.std(values, ddof=0)),
                "min_normalized_importance": float(np.min(values)),
                "max_normalized_importance": float(np.max(values)),
                "threshold_mean": float(np.mean(thresholds)),
                "threshold_std": float(np.std(thresholds, ddof=0)),
                "threshold_min": float(np.min(thresholds)),
                "threshold_max": float(np.max(thresholds)),
            }
        )
        rows.append(row)
    summary = pd.DataFrame(rows)
    stable = summary.loc[
        (summary["fold_count"] == len(folds)) & (summary["nonzero_fold_count"] == len(folds))
    ].copy()
    if stable.empty:
        raise Task13ExplanationError("No stable nonzero-importance features were found")
    stable = stable.sort_values(
        ["mean_normalized_importance", "std_normalized_importance", "feature_index"],
        ascending=[False, True, True],
    ).head(stable_candidate_count)
    stable.insert(0, "pattern_rank", np.arange(1, len(stable) + 1))
    return stable.reset_index(drop=True)


def build_feature_distributions(
    *,
    xrocket_dir: Path,
    folds: list[int],
    selected: pd.DataFrame,
    predictions: pd.DataFrame,
    label_ids: tuple[int, ...],
) -> pd.DataFrame:
    """Build held-out transformed-feature distributions for selected patterns."""
    selected_indices = selected["feature_index"].to_numpy(dtype=np.int64)
    selected_by_index = selected.set_index("feature_index")
    rows: list[dict[str, Any]] = []
    for fold in folds:
        features, _train_indices, test_indices = _load_fold_features(xrocket_dir / f"fold_{fold}")
        fold_predictions = predictions.loc[predictions["fold"] == fold].set_index("sample_index")
        for sample_index in test_indices:
            if int(sample_index) not in fold_predictions.index:
                raise Task13ExplanationError(
                    f"Missing prediction row for fold {fold} sample {int(sample_index)}"
                )
            prediction = fold_predictions.loc[int(sample_index)]
            correct = int(prediction["y_true"]) == int(prediction["y_pred"])
            for feature_index in selected_indices:
                pattern = selected_by_index.loc[int(feature_index)]
                rows.append(
                    {
                        "pattern_rank": int(pattern["pattern_rank"]),
                        "feature_index": int(feature_index),
                        "fold": fold,
                        "sample_index": int(sample_index),
                        "sample_id": str(prediction["sample_id"]),
                        "participant_id": str(prediction["participant_id"]),
                        "arm_code": str(prediction["arm_code"]),
                        "y_true": int(prediction["y_true"]),
                        "y_pred": int(prediction["y_pred"]),
                        "correct": bool(correct),
                        "feature_value": float(features[int(sample_index), int(feature_index)]),
                    }
                )
    frame = pd.DataFrame(rows)
    if not set(frame["y_true"]).issubset(set(label_ids)):
        raise Task13ExplanationError("Feature distributions contain labels outside configuration")
    return frame


def add_pattern_class_summaries(
    selected: pd.DataFrame,
    distributions: pd.DataFrame,
) -> pd.DataFrame:
    """Add associated class and separation diagnostics to selected patterns."""
    rows: list[dict[str, Any]] = []
    for feature_index, group in distributions.groupby("feature_index", sort=False):
        correct = group.loc[group["correct"]].copy()
        if correct.empty:
            correct = group.copy()
        medians = correct.groupby("y_true")["feature_value"].median().sort_values(ascending=False)
        associated_class = int(medians.index[0])
        values = group["feature_value"].to_numpy(dtype=np.float64)
        target = (group["y_true"].to_numpy(dtype=np.int64) == associated_class).astype(np.int64)
        auc = _safe_auc(target, values)
        rows.append(
            {
                "feature_index": int(feature_index),
                "associated_class": associated_class,
                "associated_class_label": EXPECTED_LABELS.get(
                    associated_class,
                    str(associated_class),
                ),
                "class_separation_auc": auc,
                "associated_class_median_value": float(medians.iloc[0]),
                "global_value_q75": float(np.quantile(values, 0.75)),
                "global_value_q90": float(np.quantile(values, 0.90)),
            }
        )
    additions = pd.DataFrame(rows)
    merged = selected.merge(additions, on="feature_index", how="left", validate="one_to_one")
    return merged.sort_values("pattern_rank").reset_index(drop=True)


def select_pattern_cases(
    *,
    selected: pd.DataFrame,
    distributions: pd.DataFrame,
    predictions: pd.DataFrame,
    correct_case_count: int,
    include_failure_case: bool,
) -> list[dict[str, Any]]:
    """Select representative correct and failure/ambiguous pattern cases."""
    cases: list[dict[str, Any]] = []
    used_samples: set[int] = set()
    for pattern in selected.to_dict("records"):
        if len([case for case in cases if case["case_type"] == "correct"]) >= correct_case_count:
            break
        feature_rows = distributions.loc[distributions["feature_index"] == pattern["feature_index"]]
        associated = int(pattern["associated_class"])
        class_rows = feature_rows.loc[
            (feature_rows["correct"])
            & (feature_rows["y_true"] == associated)
            & (~feature_rows["sample_index"].isin(used_samples))
        ].copy()
        if class_rows.empty:
            continue
        q75 = float(class_rows["feature_value"].quantile(0.75))
        q90 = float(class_rows["feature_value"].quantile(0.90))
        high = class_rows.loc[class_rows["feature_value"] >= q75].copy()
        if high.empty:
            high = class_rows.sort_values("feature_value", ascending=False).head(1).copy()
        high["distance_to_q90"] = (high["feature_value"] - q90).abs()
        chosen = high.sort_values(
            ["distance_to_q90", "feature_value", "sample_index"],
            ascending=[True, False, True],
        ).iloc[0]
        sample_index = int(chosen["sample_index"])
        used_samples.add(sample_index)
        cases.append(
            {
                "case_id": f"pattern_case_{len(cases) + 1:02d}",
                "case_type": "correct",
                "pattern_rank": int(pattern["pattern_rank"]),
                "feature_index": int(pattern["feature_index"]),
                "fold": int(chosen["fold"]),
                "sample_index": sample_index,
                "selection_note": "correct high activation closest to associated-class q90",
            }
        )
    if len([case for case in cases if case["case_type"] == "correct"]) < correct_case_count:
        raise Task13ExplanationError("Could not select the requested number of correct cases")

    if include_failure_case:
        failure = _select_failure_case(selected, distributions, predictions)
        if failure is not None:
            cases.append(failure)
    return cases


def reconstruct_feature_response(
    *,
    adapter: Any,
    x: np.ndarray,
    metadata_row: pd.Series,
    saved_feature_value: float,
) -> tuple[np.ndarray, float]:
    """Reconstruct one feature response and verify the saved PPV value."""
    block = adapter.encoder.blocks[int(metadata_row["dilation_index"])]
    tensor = torch.from_numpy(x.astype(np.float32, copy=False))
    with torch.inference_mode():
        convolved = block.conv(tensor)
        mixed = block.mix(convolved)
    response = (
        mixed[
            0,
            int(metadata_row["pattern_index"]),
            int(metadata_row["channel_combination_index"]),
            :,
        ]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64, copy=False)
    )
    threshold = float(metadata_row["threshold"])
    ppv = float(np.mean(response > threshold))
    if not math.isclose(ppv, saved_feature_value, rel_tol=1e-6, abs_tol=1e-6):
        raise Task13ExplanationError(
            f"Reconstructed PPV {ppv:.8f} does not match saved value {saved_feature_value:.8f}"
        )
    return response, ppv


def localize_response_interval(
    *,
    response: np.ndarray,
    threshold: float,
    dilation: int,
    kernel_length: int,
    padding_per_side: int,
    original_length: int,
    target_length: int,
) -> ResponseLocalization:
    """Localize the contiguous above-threshold segment containing max response."""
    if response.ndim != 1:
        raise Task13ExplanationError("response must be one-dimensional")
    margins = response - threshold
    above = margins > 0.0
    max_index = int(np.argmax(margins))
    if above[max_index]:
        start = max_index
        while start > 0 and bool(above[start - 1]):
            start -= 1
        end = max_index
        while end + 1 < len(above) and bool(above[end + 1]):
            end += 1
    else:
        start = max_index
        end = max_index
    footprints = _footprints_for_response_interval(
        start=start,
        end=end,
        dilation=dilation,
        kernel_length=kernel_length,
        padding_per_side=padding_per_side,
    )
    processed_valid = footprints[(footprints >= 0) & (footprints < target_length)]
    original_valid = footprints[(footprints >= 0) & (footprints < original_length)]
    processed_start = int(np.min(processed_valid)) if len(processed_valid) else 0
    processed_end = int(np.max(processed_valid)) if len(processed_valid) else 0
    original_start = int(np.min(original_valid)) if len(original_valid) else -1
    original_end = int(np.max(original_valid)) if len(original_valid) else -1
    edge_padding = (footprints < 0) | (footprints >= target_length)
    right_padding = footprints >= original_length
    return ResponseLocalization(
        response_start_index=start,
        response_end_index=end,
        response_max_index=max_index,
        processed_start_index=processed_start,
        processed_end_index=processed_end,
        original_start_index=original_start,
        original_end_index=original_end,
        touches_edge_padding=bool(np.any(edge_padding)),
        touches_right_padding=bool(np.any(right_padding)),
        edge_padding_fraction=float(np.mean(edge_padding)),
        right_padding_fraction=float(np.mean(right_padding)),
        above_threshold_count=int(np.sum(above)),
        localized_above_threshold_count=int(np.sum(above[start : end + 1])),
        max_margin=float(margins[max_index]),
    )


def materialize_pattern_cases(
    *,
    xrocket_dir: Path,
    arrays: dict[str, np.ndarray],
    metadata: pd.DataFrame,
    selected: pd.DataFrame,
    case_specs: list[dict[str, Any]],
    target_length: int,
    nominal_sampling_rate_hz: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Reconstruct responses and produce case and trace rows."""
    selected_by_feature = selected.set_index("feature_index")
    case_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    for spec in case_specs:
        fold = int(spec["fold"])
        feature_index = int(spec["feature_index"])
        sample_index = int(spec["sample_index"])
        fold_dir = xrocket_dir / f"fold_{fold}"
        fold_metadata = pd.read_parquet(fold_dir / "feature_metadata.parquet").set_index(
            "feature_index"
        )
        if feature_index not in fold_metadata.index:
            raise Task13ExplanationError(f"Feature {feature_index} missing in fold {fold}")
        metadata_row = fold_metadata.loc[feature_index]
        features, _train_indices, _test_indices = _load_fold_features(fold_dir)
        adapter = joblib.load(fold_dir / "xrocket_adapter.joblib")
        response, ppv = reconstruct_feature_response(
            adapter=adapter,
            x=arrays["X"][sample_index : sample_index + 1],
            metadata_row=metadata_row,
            saved_feature_value=float(features[sample_index, feature_index]),
        )
        localization = localize_response_interval(
            response=response,
            threshold=float(metadata_row["threshold"]),
            dilation=int(metadata_row["dilation"]),
            kernel_length=int(metadata_row["kernel_length"]),
            padding_per_side=int(metadata_row["padding_per_side"]),
            original_length=int(arrays["original_length"][sample_index]),
            target_length=target_length,
        )
        pattern = selected_by_feature.loc[feature_index]
        sample_meta = metadata.set_index("sample_index").loc[sample_index]
        prediction = _prediction_for_sample(xrocket_dir, fold=fold, sample_index=sample_index)
        meaningfulness = classify_meaningfulness(
            auc=float(pattern["class_separation_auc"]),
            localization=localization,
        )
        interpretation = interpret_pattern(
            channel_name=str(metadata_row["channel_name"]),
            dilation=int(metadata_row["dilation"]),
            response_span=int(metadata_row["effective_receptive_field_samples"]),
            meaningfulness=meaningfulness,
            touches_right_padding=localization.touches_right_padding,
        )
        case_row = {
            **spec,
            "sample_id": str(arrays["sample_id"][sample_index]),
            "participant_id": str(arrays["participant_id"][sample_index]),
            "participant_pseudonym": str(arrays["participant_id"][sample_index]),
            "arm_code": str(arrays["arm_code"][sample_index]),
            "arm": str(sample_meta.get("arm", arrays["arm_code"][sample_index])),
            "movement_label": EXPECTED_LABELS.get(int(arrays["y"][sample_index]), "unknown"),
            "y_true": int(prediction["y_true"]),
            "y_pred": int(prediction["y_pred"]),
            "prediction_correct": int(prediction["y_true"]) == int(prediction["y_pred"]),
            "prediction_margin": float(prediction["probability_margin"]),
            "feature_value_ppv": ppv,
            "threshold": float(metadata_row["threshold"]),
            "kernel_id": str(metadata_row["kernel_id"]),
            "pattern_index": int(metadata_row["pattern_index"]),
            "pattern_weights": str(metadata_row["pattern_weights"]),
            "channel_name": str(metadata_row["channel_name"]),
            "channel_names": str(metadata_row["channel_names"]),
            "dilation": int(metadata_row["dilation"]),
            "kernel_length": int(metadata_row["kernel_length"]),
            "effective_receptive_field_samples": int(
                metadata_row["effective_receptive_field_samples"]
            ),
            "effective_receptive_field_seconds_nominal": float(
                metadata_row["effective_receptive_field_seconds_nominal"]
            ),
            "response_start_index": localization.response_start_index,
            "response_end_index": localization.response_end_index,
            "response_max_index": localization.response_max_index,
            "processed_start_index": localization.processed_start_index,
            "processed_end_index": localization.processed_end_index,
            "original_start_index": localization.original_start_index,
            "original_end_index": localization.original_end_index,
            "original_start_seconds_nominal": (
                localization.original_start_index / nominal_sampling_rate_hz
                if localization.original_start_index >= 0
                else float("nan")
            ),
            "original_end_seconds_nominal": (
                localization.original_end_index / nominal_sampling_rate_hz
                if localization.original_end_index >= 0
                else float("nan")
            ),
            "touches_edge_padding": localization.touches_edge_padding,
            "touches_right_padding": localization.touches_right_padding,
            "edge_padding_fraction": localization.edge_padding_fraction,
            "right_padding_fraction": localization.right_padding_fraction,
            "above_threshold_count": localization.above_threshold_count,
            "localized_above_threshold_count": localization.localized_above_threshold_count,
            "max_margin": localization.max_margin,
            "associated_class": int(pattern["associated_class"]),
            "associated_class_label": str(pattern["associated_class_label"]),
            "class_separation_auc": float(pattern["class_separation_auc"]),
            "human_meaningfulness": meaningfulness,
            "interpretation": interpretation,
            "localization_caveat": (
                "Representative strongest PPV response interval; the PPV feature pools "
                "over all response positions and does not identify a unique causal instant."
            ),
        }
        case_rows.append(case_row)
        margins = response - float(metadata_row["threshold"])
        above = margins > 0.0
        for response_index, value in enumerate(response):
            trace_rows.append(
                {
                    "case_id": str(spec["case_id"]),
                    "case_type": str(spec["case_type"]),
                    "fold": fold,
                    "sample_index": sample_index,
                    "feature_index": feature_index,
                    "response_index": response_index,
                    "response_value": float(value),
                    "threshold": float(metadata_row["threshold"]),
                    "margin": float(margins[response_index]),
                    "above_threshold": bool(above[response_index]),
                    "in_localized_interval": bool(
                        localization.response_start_index
                        <= response_index
                        <= localization.response_end_index
                    ),
                }
            )
    return case_rows, trace_rows


def classify_meaningfulness(*, auc: float, localization: ResponseLocalization) -> str:
    """Classify cautious human meaningfulness for one pattern case."""
    if localization.above_threshold_count == 0 or localization.right_padding_fraction >= 0.75:
        return "not meaningful"
    if (
        auc >= 0.80
        and not localization.touches_right_padding
        and not localization.touches_edge_padding
    ):
        return "clear"
    if auc >= 0.65 and localization.right_padding_fraction <= 0.25:
        return "plausible"
    return "ambiguous"


def interpret_pattern(
    *,
    channel_name: str,
    dilation: int,
    response_span: int,
    meaningfulness: str,
    touches_right_padding: bool,
) -> str:
    """Create cautious sensor-coordinate interpretation text."""
    parsed = parse_channel_name(channel_name)
    family = parsed["sensor_family"]
    axis = parsed["axis"]
    family_text = {
        "accelerometer": "device-frame acceleration",
        "gravity": "orientation-related gravity structure",
        "gyros": "device-frame angular velocity",
        "lin_accel": "gravity-removed linear acceleration",
        "game_rot_vec": "game-rotation-vector orientation structure",
        "magn_field": "magnetic-field variation",
    }.get(family, "sensor-coordinate signal variation")
    scale = "long-span" if dilation >= 5 else "intermediate-span" if dilation >= 3 else "short-span"
    caveat = " The interval overlaps padded samples, so the meaning is representation-dependent."
    if not touches_right_padding:
        caveat = ""
    if meaningfulness == "not meaningful":
        return (
            f"This {scale} feature on {channel_name} did not provide a reliable human "
            f"interpretation beyond model-specific signal variation.{caveat}"
        )
    return (
        f"This {scale} PPV pattern captures {family_text} on the device {axis}-axis "
        f"over about {response_span} samples. It may reflect broad movement shape, "
        f"transition timing, or sustained orientation/velocity structure, but only in "
        f"sensor coordinates.{caveat}"
    )


def _select_failure_case(
    selected: pd.DataFrame,
    distributions: pd.DataFrame,
    predictions: pd.DataFrame,
) -> dict[str, Any] | None:
    errors = predictions.loc[predictions["y_true"] != predictions["y_pred"]].copy()
    if errors.empty:
        return None
    grouped = (
        errors.groupby(["y_true", "y_pred"], as_index=False)
        .agg(count=("sample_index", "size"), mean_margin=("probability_margin", "mean"))
        .sort_values(
            ["count", "mean_margin", "y_true", "y_pred"],
            ascending=[False, True, True, True],
        )
    )
    pair = grouped.iloc[0]
    candidates = errors.loc[
        (errors["y_true"] == int(pair["y_true"])) & (errors["y_pred"] == int(pair["y_pred"]))
    ].copy()
    chosen = candidates.sort_values(["probability_margin", "sample_index"]).iloc[0]
    sample_index = int(chosen["sample_index"])
    sample_distribution = distributions.loc[distributions["sample_index"] == sample_index].copy()
    q75_by_feature = distributions.groupby("feature_index")["feature_value"].quantile(0.75)
    sample_distribution["is_high"] = [
        float(row["feature_value"]) >= float(q75_by_feature.loc[int(row["feature_index"])])
        for row in sample_distribution.to_dict("records")
    ]
    high = sample_distribution.loc[sample_distribution["is_high"]]
    if high.empty:
        selected_feature = (
            sample_distribution.sort_values(
                ["feature_value", "pattern_rank"],
                ascending=[False, True],
            )
            .iloc[0]
            .to_dict()
        )
        note = "failure case linked to highest available selected-feature value"
    else:
        selected_feature = high.sort_values("pattern_rank").iloc[0].to_dict()
        note = "failure case linked to highest-ranked high-activation selected feature"
    return {
        "case_id": "pattern_case_failure_or_ambiguous",
        "case_type": "failure_or_ambiguous",
        "pattern_rank": int(selected_feature["pattern_rank"]),
        "feature_index": int(selected_feature["feature_index"]),
        "fold": int(chosen["fold"]),
        "sample_index": sample_index,
        "selection_note": note,
    }


def _prediction_for_sample(xrocket_dir: Path, *, fold: int, sample_index: int) -> pd.Series:
    predictions = _load_primary_predictions(xrocket_dir, primary_model=PRIMARY_MODEL)
    selected = predictions.loc[
        (predictions["fold"] == fold) & (predictions["sample_index"] == sample_index)
    ]
    if selected.empty:
        raise Task13ExplanationError(f"Missing prediction for fold {fold} sample {sample_index}")
    return selected.iloc[0]


def _load_primary_predictions(xrocket_dir: Path, *, primary_model: str) -> pd.DataFrame:
    path = xrocket_dir / "predictions.csv"
    if not path.is_file():
        raise Task13ExplanationError(f"Missing predictions artifact: {path}")
    predictions = pd.read_csv(path)
    required = {
        "model",
        "fold",
        "sample_index",
        "sample_id",
        "participant_id",
        "arm_code",
        "y_true",
        "y_pred",
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise Task13ExplanationError(
            f"Predictions artifact is missing columns: {', '.join(missing)}"
        )
    frame = predictions.loc[predictions["model"] == primary_model].copy()
    if frame.empty:
        raise Task13ExplanationError(f"No predictions found for model {primary_model!r}")
    probability_columns = [column for column in frame.columns if column.startswith("prob_class_")]
    if not probability_columns:
        raise Task13ExplanationError("Predictions artifact has no probability columns")
    probabilities = frame[probability_columns].to_numpy(dtype=np.float64)
    sorted_probabilities = np.sort(probabilities, axis=1)
    frame["probability_margin"] = sorted_probabilities[:, -1] - sorted_probabilities[:, -2]
    frame["correct"] = frame["y_true"].astype(int) == frame["y_pred"].astype(int)
    return frame


def _load_processed_metadata(processed_dir: Path) -> pd.DataFrame:
    path = processed_dir / "metadata.csv"
    if not path.is_file():
        raise Task13ExplanationError(f"Missing processed metadata artifact: {path}")
    metadata = pd.read_csv(path)
    if "sample_index" not in metadata:
        raise Task13ExplanationError("Processed metadata is missing sample_index")
    return metadata


def _prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise Task13ExplanationError(
                f"Output directory is not empty: {output_dir}; pass --overwrite to replace it"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _discover_folds(xrocket_dir: Path) -> list[int]:
    if not xrocket_dir.is_dir():
        raise Task13ExplanationError(f"M3 XROCKET directory not found: {xrocket_dir}")
    folds: list[int] = []
    for path in xrocket_dir.iterdir():
        if path.is_dir() and path.name.startswith("fold_"):
            try:
                folds.append(int(path.name.removeprefix("fold_")))
            except ValueError:
                continue
    return sorted(folds)


def _load_fold_features(fold_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = fold_dir / "features.npz"
    if not path.is_file():
        raise Task13ExplanationError(f"Missing transformed features artifact: {path}")
    with np.load(path, allow_pickle=False) as data:
        return (
            data["features"].astype(np.float32, copy=False),
            data["train_indices"].astype(np.int64, copy=False),
            data["test_indices"].astype(np.int64, copy=False),
        )


def _signature_columns() -> list[str]:
    return [
        "feature_index",
        "kernel_id",
        "pattern_index",
        "pattern_weights",
        "dilation_index",
        "dilation",
        "kernel_length",
        "padding_per_side",
        "channel_combination_index",
        "channel_indices",
        "channel_names",
        "channel_count",
        "channel_index",
        "channel_name",
        "combination_order",
        "combination_method",
        "threshold_index",
        "feature_type",
        "effective_receptive_field_samples",
        "effective_receptive_field_seconds_nominal",
        "relative_span",
    ]


def _safe_auc(target: np.ndarray, values: np.ndarray) -> float:
    if len(np.unique(target)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(target, values))
    except ValueError:
        return float("nan")


def _footprints_for_response_interval(
    *,
    start: int,
    end: int,
    dilation: int,
    kernel_length: int,
    padding_per_side: int,
) -> np.ndarray:
    offsets = np.arange(kernel_length, dtype=np.int64) * int(dilation)
    rows = [
        int(response_index) - int(padding_per_side) + offsets
        for response_index in range(start, end + 1)
    ]
    return np.concatenate(rows).astype(np.int64, copy=False)


def _write_figures(
    *,
    figures_dir: Path,
    arrays: dict[str, np.ndarray],
    cases: pd.DataFrame,
    traces: pd.DataFrame,
    distributions: pd.DataFrame,
    selected: pd.DataFrame,
    target_length: int,
) -> None:
    for index, case in enumerate(cases.to_dict("records"), start=1):
        stem = (
            "pattern_case_failure_or_ambiguous"
            if case["case_type"] == "failure_or_ambiguous"
            else f"pattern_case_{index:02d}"
        )
        _case_figure(
            arrays=arrays,
            case=case,
            trace=traces.loc[traces["case_id"] == case["case_id"]],
            path_stem=figures_dir / stem,
            target_length=target_length,
        )
    _distribution_figure(distributions, selected, figures_dir / "pattern_feature_distributions")
    _summary_table_figure(cases, figures_dir / "pattern_summary_table")


def _case_figure(
    *,
    arrays: dict[str, np.ndarray],
    case: dict[str, Any],
    trace: pd.DataFrame,
    path_stem: Path,
    target_length: int,
) -> None:
    sample_index = int(case["sample_index"])
    channel_names = _same_family_channels(str(case["channel_name"]))
    channel_indices = [EXPECTED_CHANNELS.index(channel) for channel in channel_names]
    x_values = np.arange(target_length)
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 8.0), constrained_layout=True)
    for channel, channel_index in zip(channel_names, channel_indices, strict=True):
        axes[0].plot(x_values, arrays["X"][sample_index, channel_index], label=channel)
    axes[0].axvspan(
        int(case["processed_start_index"]),
        int(case["processed_end_index"]),
        color="#F58518",
        alpha=0.20,
        label="localized footprint",
    )
    axes[0].axvline(int(arrays["original_length"][sample_index]) - 0.5, color="#666666", ls="--")
    parsed = parse_channel_name(str(case["channel_name"]))
    unit = SENSOR_UNITS.get(parsed["sensor_family"], "sensor units")
    axes[0].set_title(
        f"{case['case_id']}: {case['movement_label']} | arm {case['arm']} | "
        f"{case['participant_pseudonym']}"
    )
    axes[0].set_ylabel(f"Signal ({unit})")
    axes[0].legend(loc="best", fontsize=8)
    axes[0].grid(alpha=0.25)

    axes[1].plot(trace["response_index"], trace["response_value"], color="#4C78A8")
    axes[1].axhline(float(case["threshold"]), color="#E45756", ls="--", label="threshold")
    axes[1].axvspan(
        int(case["response_start_index"]),
        int(case["response_end_index"]),
        color="#F58518",
        alpha=0.20,
        label="representative interval",
    )
    axes[1].set_ylabel("Kernel response")
    axes[1].set_xlabel("Processed sample index")
    axes[1].legend(loc="best", fontsize=8)
    axes[1].grid(alpha=0.25)

    axes[2].axis("off")
    text = (
        f"Feature {case['feature_index']} | {case['kernel_id']} | "
        f"dilation {case['dilation']} | span {case['effective_receptive_field_samples']} samples\n"
        f"PPV={case['feature_value_ppv']:.3f}, AUC={case['class_separation_auc']:.3f}, "
        f"meaningfulness={case['human_meaningfulness']}\n"
        f"True={case['y_true']} Pred={case['y_pred']} Margin={case['prediction_margin']:.3f}\n"
        f"{case['interpretation']}"
    )
    axes[2].text(0.01, 0.92, text, va="top", ha="left", wrap=True)
    _save_figure(fig, path_stem)


def _distribution_figure(
    distributions: pd.DataFrame,
    selected: pd.DataFrame,
    path_stem: Path,
) -> None:
    top = selected.sort_values("pattern_rank").head(3)
    fig, axes = plt.subplots(1, len(top), figsize=(4.2 * len(top), 4.8), constrained_layout=True)
    if len(top) == 1:
        axes = [axes]
    for ax, pattern in zip(axes, top.to_dict("records"), strict=True):
        frame = distributions.loc[distributions["feature_index"] == int(pattern["feature_index"])]
        labels = sorted(frame["y_true"].unique())
        values = [
            frame.loc[frame["y_true"] == label, "feature_value"].to_numpy(dtype=np.float64)
            for label in labels
        ]
        ax.boxplot(values, tick_labels=[str(label) for label in labels], showfliers=False)
        ax.set_title(f"Feature {int(pattern['feature_index'])}")
        ax.set_xlabel("Movement class")
        ax.set_ylabel("Held-out PPV")
        ax.grid(axis="y", alpha=0.25)
    _save_figure(fig, path_stem)


def _summary_table_figure(cases: pd.DataFrame, path_stem: Path) -> None:
    columns = [
        "case_id",
        "channel_name",
        "dilation",
        "feature_value_ppv",
        "class_separation_auc",
        "human_meaningfulness",
    ]
    table = cases[columns].copy()
    for column in ("feature_value_ppv", "class_separation_auc"):
        table[column] = table[column].map(lambda value: f"{float(value):.3f}")
    fig, ax = plt.subplots(figsize=(11.0, max(2.8, 0.55 * len(table) + 1.6)))
    ax.axis("off")
    rendered = ax.table(
        cellText=table.to_numpy().tolist(),
        colLabels=table.columns.tolist(),
        loc="center",
        cellLoc="left",
    )
    rendered.auto_set_font_size(False)
    rendered.set_fontsize(8)
    rendered.scale(1.0, 1.4)
    ax.set_title("Task 1.3 Pattern Summary")
    _save_figure(fig, path_stem)


def _same_family_channels(channel_name: str) -> list[str]:
    parsed = parse_channel_name(channel_name)
    family = parsed["sensor_family"]
    return [channel for channel in EXPECTED_CHANNELS if channel.startswith(f"{family}_")]


def _save_figure(fig: Any, path_stem: Path) -> None:
    fig.savefig(path_stem.with_suffix(".png"), dpi=200)
    fig.savefig(path_stem.with_suffix(".pdf"))
    plt.close(fig)


def _write_figure_captions(figures_dir: Path) -> None:
    captions = """# Task 1.3 Figure Captions

- `pattern_case_01` through `pattern_case_03`: Representative high-activation
  correctly classified PPV patterns with same-family signal channels, response
  trace, threshold, localized footprint, prediction metadata, and interpretation.
- `pattern_case_failure_or_ambiguous`: Representative failure or ambiguous case
  from a common random-forest confusion.
- `pattern_feature_distributions`: Held-out selected-feature PPV distributions
  by movement class.
- `pattern_summary_table`: Case-level feature metadata, separation, and human
  meaningfulness labels.
"""
    (figures_dir / "figure_captions.md").write_text(captions, encoding="utf-8")


def _build_task_answer(*, selected: pd.DataFrame, cases: pd.DataFrame) -> str:
    selected_table = _markdown_table(
        selected.head(6),
        [
            "pattern_rank",
            "feature_index",
            "channel_name",
            "dilation",
            "mean_normalized_importance",
            "class_separation_auc",
            "associated_class_label",
        ],
    )
    case_table = _markdown_table(
        cases,
        [
            "case_id",
            "case_type",
            "movement_label",
            "channel_name",
            "feature_value_ppv",
            "human_meaningfulness",
        ],
    )
    clear_count = int(np.sum(cases["human_meaningfulness"] == "clear"))
    plausible_count = int(np.sum(cases["human_meaningfulness"] == "plausible"))
    ambiguous_count = int(np.sum(cases["human_meaningfulness"] == "ambiguous"))
    not_meaningful_count = int(np.sum(cases["human_meaningfulness"] == "not meaningful"))
    return f"""# Task 1.3 Answer: Discriminative-Pattern Interpretation

## Direct Answer

The most discriminative saved XROCKET patterns can be linked to representative
signal intervals, but the localization is approximate. The selected features
are PPV features, so each transformed value is the fraction of response
positions above a fitted threshold. The reported interval is therefore the
strongest representative above-threshold segment, not a unique causal instant.

## Selection Rule

Features were selected by stable random-forest native importance across the
participant-held-out folds: importance was normalized within fold, features had
to be nonzero in every fold, and ties were resolved by higher mean importance,
lower fold variation, then lower feature index.

Top selected patterns:

{selected_table}

## Case Studies

{case_table}

Human-meaningfulness labels across these cases were: clear={clear_count},
plausible={plausible_count}, ambiguous={ambiguous_count}, not meaningful={not_meaningful_count}.

## Interpretation

The documented cases support cautious sensor-coordinate interpretations such as
broad movement shape, transition timing, sustained orientation structure,
device-frame acceleration, angular velocity, or magnetic-field variation. They
do not validate a coaching system, expertise assessment, force estimate, joint
mechanics claim, or learning-gain claim. A future learning or performance
assessment system could use this style of explanation to identify which signal
regions a model used and to guide expert review, but that would require separate
validation with pedagogical or biomechanical ground truth.

## Limitation

The analysis inherits the M3 padded-representation caveat. Several important
features use long spans and may overlap right-padding or convolution edge
padding. Those cases are retained because they clarify model behavior, but their
human meaning is marked as ambiguous or not meaningful when padding dominates.
"""


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


def parse_pattern_weights(value: str) -> list[float]:
    """Parse persisted pattern weights for tests and downstream reporting."""
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, list):
        raise Task13ExplanationError("pattern_weights must decode to a list")
    return [float(item) for item in parsed]
