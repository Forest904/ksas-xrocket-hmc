from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ksas_xrocket.audit import EXPECTED_CHANNELS
from ksas_xrocket.cli import main
from ksas_xrocket.task_1_2_explain import (
    Task12ExplanationError,
    _validate_temporal_metadata,
    aggregate_temporal_importance,
    classify_temporal_evidence,
    effective_span_samples,
    nominal_seconds,
    run_task_1_2_explanation,
    summarize_temporal_importance,
    temporal_scale_bin,
)
from ksas_xrocket.xrocket_experiment import run_xrocket_experiment


def test_temporal_span_and_bin_rules_are_predeclared() -> None:
    assert effective_span_samples(1, 9) == 9
    assert effective_span_samples(6, 9) == 49
    assert nominal_seconds(25, 50.0) == pytest.approx(0.5)

    assert temporal_scale_bin(9, target_length=56) == "short"
    assert temporal_scale_bin(17, target_length=56) == "short"
    assert temporal_scale_bin(25, target_length=56) == "intermediate"
    assert temporal_scale_bin(33, target_length=56) == "intermediate"
    assert temporal_scale_bin(41, target_length=56) == "long"
    assert temporal_scale_bin(49, target_length=56) == "long"

    with pytest.raises(Task12ExplanationError, match="dilation must be positive"):
        effective_span_samples(0, 9)
    with pytest.raises(Task12ExplanationError, match="sampling_rate_hz must be positive"):
        nominal_seconds(9, 0.0)


def test_classify_temporal_evidence_uses_documented_thresholds() -> None:
    short = pd.DataFrame(
        [
            {"group_value": "short", "importance_mean": 0.62},
            {"group_value": "intermediate", "importance_mean": 0.25},
            {"group_value": "long", "importance_mean": 0.13},
        ]
    )
    long = pd.DataFrame(
        [
            {"group_value": "short", "importance_mean": 0.10},
            {"group_value": "intermediate", "importance_mean": 0.25},
            {"group_value": "long", "importance_mean": 0.65},
        ]
    )
    mixed = pd.DataFrame(
        [
            {"group_value": "short", "importance_mean": 0.40},
            {"group_value": "intermediate", "importance_mean": 0.35},
            {"group_value": "long", "importance_mean": 0.25},
        ]
    )

    assert classify_temporal_evidence(short) == "short-scale"
    assert classify_temporal_evidence(long) == "long-scale"
    assert classify_temporal_evidence(mixed) == (
        "mixed temporal-scale evidence; largest contribution: short"
    )


def test_temporal_aggregation_and_summary_preserve_fold_totals() -> None:
    frame = pd.DataFrame(
        [
            {
                "fold": 0,
                "dilation": 1,
                "temporal_scale_bin": "short",
                "normalized_importance": 0.2,
            },
            {
                "fold": 0,
                "dilation": 2,
                "temporal_scale_bin": "short",
                "normalized_importance": 0.3,
            },
            {
                "fold": 0,
                "dilation": 5,
                "temporal_scale_bin": "long",
                "normalized_importance": 0.5,
            },
            {
                "fold": 1,
                "dilation": 1,
                "temporal_scale_bin": "short",
                "normalized_importance": 0.6,
            },
            {
                "fold": 1,
                "dilation": 5,
                "temporal_scale_bin": "long",
                "normalized_importance": 0.4,
            },
        ]
    )

    scale = aggregate_temporal_importance(frame, group_level="temporal_scale_bin")
    assert scale.groupby("fold")["importance"].sum().to_dict() == {
        0: pytest.approx(1.0),
        1: pytest.approx(1.0),
    }
    summary = summarize_temporal_importance(scale)
    short = summary.loc[summary["group_value"] == "short"].iloc[0]
    assert short["importance_mean"] == pytest.approx(0.55)


def test_temporal_metadata_validation_rejects_inconsistent_span() -> None:
    frame = pd.DataFrame(
        [
            {
                "dilation": 2,
                "kernel_length": 9,
                "effective_receptive_field_samples": 18,
                "effective_receptive_field_seconds_nominal": 0.36,
            }
        ]
    )

    with pytest.raises(Task12ExplanationError, match="expected 17"):
        _validate_temporal_metadata(frame, fold=0, nominal_sampling_rate_hz=50.0)


def test_task_1_2_cli_smoke_writes_expected_outputs(tmp_path: Path) -> None:
    processed_dir, splits_dir = make_experiment_fixture(tmp_path)
    xrocket_dir = tmp_path / "xrocket"
    output_dir = tmp_path / "task_1_2"
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
    config_path = tmp_path / "task_1_2.yaml"
    config_path.write_text(
        "\n".join(
            [
                "task: task_1_2",
                f"processed_dir: {processed_dir.as_posix()}",
                f"xrocket_dir: {xrocket_dir.as_posix()}",
                f"output_dir: {output_dir.as_posix()}",
                "classes: [0, 1]",
                "random_state: 42",
                "temporal:",
                "  target_length: 56",
                "  nominal_sampling_rate_hz: 50.0",
                "validation:",
                "  top_k: 1",
                "models:",
                "  random_forest:",
                "    n_estimators: 5",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(["explain", "--config", str(config_path)])

    assert exit_code == 0
    expected = {
        "resolved_config.json",
        "provenance.json",
        "temporal_span_mapping.csv",
        "fold_temporal_feature_importance.parquet",
        "important_temporal_features.parquet",
        "dilation_fold_importance.csv",
        "dilation_importance_summary.csv",
        "temporal_scale_fold_importance.csv",
        "temporal_scale_importance_summary.csv",
        "class_specific_dilation_importance.csv",
        "class_specific_temporal_scale_importance.csv",
        "stability_rank_correlations.csv",
        "stability_topk_overlap.csv",
        "padding_temporal_diagnostics.csv",
        "task_1_2_answer.md",
    }
    assert expected.issubset({path.name for path in output_dir.iterdir()})
    figures = output_dir / "figures"
    for stem in (
        "dilation_importance",
        "temporal_scale_contribution",
        "class_specific_scale_profiles",
        "fold_stability",
    ):
        assert (figures / f"{stem}.png").stat().st_size > 0
        assert (figures / f"{stem}.pdf").stat().st_size > 0

    native = pd.read_parquet(output_dir / "fold_temporal_feature_importance.parquet")
    assert native.groupby("fold")["normalized_importance"].sum().iloc[0] == pytest.approx(1.0)
    mapping = pd.read_csv(output_dir / "temporal_span_mapping.csv")
    assert mapping["temporal_scale_bin"].tolist() == ["short"]
    resolved = json.loads((output_dir / "resolved_config.json").read_text(encoding="utf-8"))
    assert resolved["task"] == "task_1_2"
    assert resolved["temporal"]["target_length"] == 56

    with pytest.raises(Task12ExplanationError, match="not empty"):
        run_task_1_2_explanation(
            processed_dir=processed_dir,
            xrocket_dir=xrocket_dir,
            output_dir=output_dir,
            label_ids=(0, 1),
            random_forest_estimators=5,
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
