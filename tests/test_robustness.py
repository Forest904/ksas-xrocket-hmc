from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ksas_xrocket.audit import EXPECTED_CHANNELS
from ksas_xrocket.cli import main
from ksas_xrocket.robustness import (
    RobustnessError,
    _build_control_flags,
    build_metadata_feature_frame,
    run_robustness_analysis,
    shuffled_training_labels,
    summarize_ranked_values,
)
from ksas_xrocket.xrocket_experiment import run_xrocket_experiment


def test_shuffled_training_labels_are_deterministic_and_preserve_counts() -> None:
    y = np.asarray([0, 1, 0, 1, 0, 1, 0, 1])
    train_indices = np.asarray([0, 1, 2, 3, 4, 5])

    shuffled_a = shuffled_training_labels(y, train_indices, seed=7)
    shuffled_b = shuffled_training_labels(y, train_indices, seed=7)
    shuffled_c = shuffled_training_labels(y, train_indices, seed=13)

    np.testing.assert_array_equal(shuffled_a, shuffled_b)
    assert sorted(shuffled_a.tolist()) == sorted(y[train_indices].tolist())
    assert not np.array_equal(shuffled_a, shuffled_c)


def test_metadata_feature_frame_uses_length_padding_and_arm() -> None:
    arrays = {
        "X": np.zeros((3, 2, 10), dtype=np.float32),
        "original_length": np.asarray([10, 8, 5]),
        "arm_code": np.asarray(["d", "i", "d"]),
    }

    frame = build_metadata_feature_frame(arrays)

    assert frame["original_length"].tolist() == [10.0, 8.0, 5.0]
    assert frame["padding_fraction"].tolist() == pytest.approx([0.0, 0.2, 0.5])
    assert frame["arm_code"].tolist() == ["d", "i", "d"]


def test_rank_summary_and_flags_detect_instability_and_weak_classes() -> None:
    importance = pd.DataFrame(
        [
            {"seed": 1, "fold": 0, "group_level": "axis", "group_value": "x", "importance": 0.7},
            {"seed": 1, "fold": 0, "group_level": "axis", "group_value": "y", "importance": 0.3},
            {"seed": 2, "fold": 0, "group_level": "axis", "group_value": "x", "importance": 0.4},
            {"seed": 2, "fold": 0, "group_level": "axis", "group_value": "y", "importance": 0.6},
        ]
    )
    summary = summarize_ranked_values(importance, value_column="importance")
    assert set(summary["group_value"]) == {"x", "y"}

    flags = _build_control_flags(
        label_shuffle=pd.DataFrame(
            [{"model": "shuffle", "macro_f1_mean": 0.10, "macro_f1_max": 0.20}]
        ),
        metadata=pd.DataFrame([{"model": "metadata", "macro_f1_mean": 0.35}]),
        stability={
            "rank_correlations": pd.DataFrame(
                [{"group_level": "axis", "spearman_rank_correlation": 0.5}]
            ),
            "importance": importance,
        },
        per_class_errors=pd.DataFrame([{"class_label": 1, "recall": 0.75, "precision": 0.8}]),
        thresholds={
            "label_shuffle_mean_macro_f1": 0.25,
            "label_shuffle_max_macro_f1": 0.40,
            "metadata_mean_macro_f1": 0.30,
            "rank_correlation_min": 0.75,
            "class_recall_min": 0.80,
        },
    )
    triggered = {row["flag"] for row in flags if row["triggered"]}
    assert "metadata_mean_macro_f1" in triggered
    assert "rank_correlation_min" in triggered
    assert "axis_top_rank_changed" in triggered
    assert "movement_specific_weakness" in triggered


def test_robustness_cli_smoke_writes_expected_outputs(tmp_path: Path) -> None:
    processed_dir, splits_dir = make_experiment_fixture(tmp_path)
    xrocket_dir = tmp_path / "xrocket"
    controls_dir = tmp_path / "controls"
    stability_dir = tmp_path / "stability"
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
    config_path = tmp_path / "m7.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"processed_dir: {processed_dir.as_posix()}",
                f"splits_dir: {splits_dir.as_posix()}",
                f"xrocket_dir: {xrocket_dir.as_posix()}",
                f"controls_output_dir: {controls_dir.as_posix()}",
                f"stability_output_dir: {stability_dir.as_posix()}",
                "classes: [0, 1]",
                "seeds: [7]",
                "primary_model: xrocket_random_forest",
                "random_forest:",
                "  n_estimators: 5",
                "logistic_regression:",
                "  max_iter: 100",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(["robustness", "--config", str(config_path)])

    assert exit_code == 0
    expected_controls = {
        "label_shuffle_metrics.csv",
        "label_shuffle_aggregate_metrics.csv",
        "label_shuffle_predictions.csv",
        "metadata_baseline_metrics.csv",
        "metadata_baseline_aggregate_metrics.csv",
        "metadata_baseline_predictions.csv",
        "confound_summary.csv",
        "leakage_checks.csv",
        "control_flags.csv",
        "resolved_config.json",
        "provenance.json",
        "m7_controls_summary.md",
    }
    expected_stability = {
        "seed_fold_metrics.csv",
        "metric_stability_summary.csv",
        "explanation_seed_importance.parquet",
        "explanation_stability_summary.csv",
        "rank_correlations.csv",
        "topk_overlap.csv",
        "arm_axis_rankings.csv",
        "arm_rank_comparison.csv",
        "per_class_error_summary.csv",
        "confusion_error_review.md",
        "resolved_config.json",
        "provenance.json",
        "m7_stability_summary.md",
    }
    assert expected_controls.issubset({path.name for path in controls_dir.iterdir()})
    assert expected_stability.issubset({path.name for path in stability_dir.iterdir()})
    leakage = pd.read_csv(controls_dir / "leakage_checks.csv")
    assert leakage["status"].tolist() == ["pass"]
    importance = pd.read_parquet(stability_dir / "explanation_seed_importance.parquet")
    assert {"sensor_family", "axis", "channel", "dilation", "temporal_scale_bin"}.issubset(
        set(importance["group_level"])
    )
    resolved = json.loads((controls_dir / "resolved_config.json").read_text(encoding="utf-8"))
    assert resolved["seeds"] == [7]

    with pytest.raises(RobustnessError, match="not empty"):
        run_robustness_analysis(
            processed_dir=processed_dir,
            splits_dir=splits_dir,
            xrocket_dir=xrocket_dir,
            controls_output_dir=controls_dir,
            stability_output_dir=stability_dir,
            label_ids=(0, 1),
            seeds=(7,),
            random_forest_estimators=5,
            logistic_max_iter=100,
        )


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
