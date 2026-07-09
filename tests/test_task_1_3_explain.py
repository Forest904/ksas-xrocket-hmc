from __future__ import annotations

import csv
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from ksas_xrocket.audit import EXPECTED_CHANNELS
from ksas_xrocket.cli import main
from ksas_xrocket.task_1_3_explain import (
    Task13ExplanationError,
    build_feature_distributions,
    localize_response_interval,
    reconstruct_feature_response,
    run_task_1_3_explanation,
    select_pattern_cases,
    select_stable_patterns,
)
from ksas_xrocket.xrocket_experiment import run_xrocket_experiment


def test_stable_pattern_selection_uses_documented_tie_breaks(tmp_path: Path) -> None:
    xrocket_dir = tmp_path / "xrocket"
    for fold in (0, 1):
        fold_dir = xrocket_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True)
        frame = pd.DataFrame(
            [
                make_importance_row(0, importance=0.2),
                make_importance_row(1, importance=0.3),
                make_importance_row(2, importance=0.3),
            ]
        )
        if fold == 1:
            frame.loc[frame["feature_index"] == 2, "importance"] = 0.0
        frame.to_parquet(fold_dir / "feature_importance.parquet", index=False)

    selected = select_stable_patterns(
        xrocket_dir=xrocket_dir,
        folds=[0, 1],
        stable_candidate_count=2,
    )

    assert selected["feature_index"].tolist() == [1, 0]
    assert selected["nonzero_fold_count"].tolist() == [2, 2]


def test_reconstruct_feature_response_matches_saved_ppv(tmp_path: Path) -> None:
    processed_dir, splits_dir = make_experiment_fixture(tmp_path)
    xrocket_dir = tmp_path / "xrocket"
    run_xrocket_experiment(
        processed_dir=processed_dir,
        splits_dir=splits_dir,
        output_dir=xrocket_dir,
        folds=(0,),
        label_ids=(0, 1),
        feature_cap=1512,
        max_dilations=1,
        random_forest_estimators=5,
        logistic_max_iter=100,
        batch_size=4,
    )
    fold_dir = xrocket_dir / "fold_0"
    metadata = pd.read_parquet(fold_dir / "feature_metadata.parquet")
    features = np.load(fold_dir / "features.npz", allow_pickle=False)["features"]
    with np.load(processed_dir / "tensors.npz", allow_pickle=False) as data:
        x = data["X"][0:1]
    adapter = joblib.load(fold_dir / "xrocket_adapter.joblib")
    row = metadata.iloc[0]

    _response, ppv = reconstruct_feature_response(
        adapter=adapter,
        x=x,
        metadata_row=row,
        saved_feature_value=float(features[0, int(row["feature_index"])]),
    )

    assert ppv == pytest.approx(float(features[0, int(row["feature_index"])]))
    with pytest.raises(Task13ExplanationError, match="does not match"):
        reconstruct_feature_response(
            adapter=adapter,
            x=x,
            metadata_row=row,
            saved_feature_value=ppv + 0.5,
        )


def test_response_localization_records_padding_flags() -> None:
    response = np.asarray([-1.0, 0.2, 0.4, 0.3, -0.2], dtype=np.float64)

    localized = localize_response_interval(
        response=response,
        threshold=0.0,
        dilation=2,
        kernel_length=3,
        padding_per_side=2,
        original_length=4,
        target_length=5,
    )

    assert localized.response_start_index == 1
    assert localized.response_end_index == 3
    assert localized.response_max_index == 2
    assert localized.touches_right_padding
    assert localized.touches_edge_padding
    assert localized.right_padding_fraction > 0.0
    assert localized.edge_padding_fraction > 0.0


def test_case_selection_includes_common_failure() -> None:
    selected = pd.DataFrame(
        [
            {
                "pattern_rank": 1,
                "feature_index": 10,
                "associated_class": 1,
                "associated_class_label": "upward block",
                "class_separation_auc": 0.8,
            }
        ]
    )
    distributions = pd.DataFrame(
        [
            distribution_row(10, 0, 0, 1, 1, True, 0.9),
            distribution_row(10, 0, 1, 1, 1, True, 0.8),
            distribution_row(10, 0, 2, 2, 3, False, 0.85),
            distribution_row(10, 0, 3, 2, 3, False, 0.7),
        ]
    )
    predictions = pd.DataFrame(
        [
            prediction_row(0, 0, 1, 1, 0.6),
            prediction_row(0, 1, 1, 1, 0.7),
            prediction_row(0, 2, 2, 3, 0.2),
            prediction_row(0, 3, 2, 3, 0.1),
        ]
    )

    cases = select_pattern_cases(
        selected=selected,
        distributions=distributions,
        predictions=predictions,
        correct_case_count=1,
        include_failure_case=True,
    )

    assert [case["case_type"] for case in cases] == ["correct", "failure_or_ambiguous"]
    assert cases[1]["sample_index"] == 3


def test_task_1_3_cli_smoke_writes_expected_outputs(tmp_path: Path) -> None:
    processed_dir, splits_dir = make_experiment_fixture(tmp_path)
    xrocket_dir = tmp_path / "xrocket"
    output_dir = tmp_path / "task_1_3"
    run_xrocket_experiment(
        processed_dir=processed_dir,
        splits_dir=splits_dir,
        output_dir=xrocket_dir,
        folds=(0,),
        label_ids=(0, 1),
        feature_cap=1512,
        max_dilations=1,
        random_forest_estimators=5,
        logistic_max_iter=100,
        batch_size=4,
    )
    write_processed_metadata(processed_dir)
    config_path = tmp_path / "task_1_3.yaml"
    config_path.write_text(
        "\n".join(
            [
                "task: task_1_3",
                f"processed_dir: {processed_dir.as_posix()}",
                f"xrocket_dir: {xrocket_dir.as_posix()}",
                f"output_dir: {output_dir.as_posix()}",
                "classes: [0, 1]",
                "patterns:",
                "  stable_candidate_count: 3",
                "  correct_case_count: 1",
                "  include_failure_case: false",
                "  target_length: 9",
                "  nominal_sampling_rate_hz: 50.0",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(["explain", "--config", str(config_path)])

    assert exit_code == 0
    expected = {
        "resolved_config.json",
        "provenance.json",
        "selected_patterns.csv",
        "pattern_cases.csv",
        "pattern_response_traces.parquet",
        "pattern_feature_distributions.csv",
        "task_1_3_answer.md",
    }
    assert expected.issubset({path.name for path in output_dir.iterdir()})
    figures = output_dir / "figures"
    for stem in ("pattern_case_01", "pattern_feature_distributions", "pattern_summary_table"):
        assert (figures / f"{stem}.png").stat().st_size > 0
        assert (figures / f"{stem}.pdf").stat().st_size > 0
    cases = pd.read_csv(output_dir / "pattern_cases.csv")
    assert len(cases) == 1
    traces = pd.read_parquet(output_dir / "pattern_response_traces.parquet")
    assert set(traces["case_id"]) == set(cases["case_id"])
    resolved = json.loads((output_dir / "resolved_config.json").read_text(encoding="utf-8"))
    assert resolved["task"] == "task_1_3"

    distributions = build_feature_distributions(
        xrocket_dir=xrocket_dir,
        folds=[0],
        selected=pd.read_csv(output_dir / "selected_patterns.csv"),
        predictions=pd.read_csv(xrocket_dir / "predictions.csv").query(
            "model == 'xrocket_random_forest'"
        ),
        label_ids=(0, 1),
    )
    assert not distributions.empty

    with pytest.raises(Task13ExplanationError, match="not empty"):
        run_task_1_3_explanation(
            processed_dir=processed_dir,
            xrocket_dir=xrocket_dir,
            output_dir=output_dir,
            label_ids=(0, 1),
            stable_candidate_count=3,
            correct_case_count=1,
        )


def make_importance_row(feature_index: int, *, importance: float) -> dict[str, object]:
    channel_name = EXPECTED_CHANNELS[feature_index % len(EXPECTED_CHANNELS)]
    return {
        "feature_index": feature_index,
        "kernel_id": f"d00_p{feature_index:03d}",
        "pattern_index": feature_index,
        "pattern_weights": "[2.0, 2.0, 2.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]",
        "dilation_index": 0,
        "dilation": 1,
        "kernel_length": 9,
        "padding_per_side": 4,
        "channel_combination_index": feature_index % len(EXPECTED_CHANNELS),
        "channel_indices": f"[{feature_index % len(EXPECTED_CHANNELS)}]",
        "channel_names": f'["{channel_name}"]',
        "channel_count": 1,
        "channel_index": feature_index % len(EXPECTED_CHANNELS),
        "channel_name": channel_name,
        "combination_order": 1,
        "combination_method": "additive",
        "threshold_index": 0,
        "threshold": float(feature_index),
        "feature_type": "ppv",
        "effective_receptive_field_samples": 9,
        "effective_receptive_field_seconds_nominal": 0.18,
        "relative_span": 1.0,
        "importance": importance,
    }


def distribution_row(
    feature_index: int,
    fold: int,
    sample_index: int,
    y_true: int,
    y_pred: int,
    correct: bool,
    feature_value: float,
) -> dict[str, object]:
    return {
        "pattern_rank": 1,
        "feature_index": feature_index,
        "fold": fold,
        "sample_index": sample_index,
        "sample_id": f"sample_{sample_index}",
        "participant_id": "P01",
        "arm_code": "d",
        "y_true": y_true,
        "y_pred": y_pred,
        "correct": correct,
        "feature_value": feature_value,
    }


def prediction_row(
    fold: int,
    sample_index: int,
    y_true: int,
    y_pred: int,
    margin: float,
) -> dict[str, object]:
    return {
        "model": "xrocket_random_forest",
        "fold": fold,
        "sample_index": sample_index,
        "sample_id": f"sample_{sample_index}",
        "participant_id": "P01",
        "arm_code": "d",
        "y_true": y_true,
        "y_pred": y_pred,
        "probability_margin": margin,
    }


def make_experiment_fixture(tmp_path: Path) -> tuple[Path, Path]:
    processed_dir = tmp_path / "processed"
    splits_dir = tmp_path / "splits"
    processed_dir.mkdir()
    splits_dir.mkdir()
    rng = np.random.default_rng(7)
    labels = np.asarray([0, 1] * 6, dtype=np.int64)
    x = rng.normal(size=(12, len(EXPECTED_CHANNELS), 9)).astype(np.float32)
    x += labels[:, None, None] * 0.75
    original_lengths = np.asarray([6, 7, 8, 9] * 3, dtype=np.int64)
    valid_mask = np.zeros((12, 9), dtype=np.bool_)
    for index, length in enumerate(original_lengths):
        valid_mask[index, :length] = True
        x[index, :, length:] = 0.0
    participant_ids = np.asarray([f"P{index // 2 + 1:02d}" for index in range(12)])
    sample_ids = np.asarray([f"sample_{index:02d}" for index in range(12)])
    arm_codes = np.asarray(["d", "i"] * 6)
    np.savez_compressed(
        processed_dir / "tensors.npz",
        X=x,
        y=labels,
        valid_mask=valid_mask,
        sample_id=sample_ids,
        participant_id=participant_ids,
        arm_code=arm_codes,
        original_length=original_lengths,
    )

    fieldnames = [
        "fold",
        "split",
        "sample_index",
        "sample_id",
        "movement_label_id",
        "participant_id",
        "arm_code",
    ]
    rows: list[dict[str, str | int]] = []
    test_indices = {0, 1, 2, 3, 4, 5}
    for index in range(12):
        rows.append(
            {
                "fold": 0,
                "split": "test" if index in test_indices else "train",
                "sample_index": index,
                "sample_id": sample_ids[index],
                "movement_label_id": int(labels[index]),
                "participant_id": participant_ids[index],
                "arm_code": arm_codes[index],
            }
        )
    with (splits_dir / "m2_grouped_folds.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return processed_dir, splits_dir


def write_processed_metadata(processed_dir: Path) -> None:
    rows = []
    with np.load(processed_dir / "tensors.npz", allow_pickle=False) as data:
        for index, sample_id in enumerate(data["sample_id"]):
            arm_code = str(data["arm_code"][index])
            rows.append(
                {
                    "sample_index": index,
                    "sample_id": str(sample_id),
                    "arm_code": arm_code,
                    "arm": "right" if arm_code == "d" else "left",
                }
            )
    pd.DataFrame(rows).to_csv(processed_dir / "metadata.csv", index=False)
