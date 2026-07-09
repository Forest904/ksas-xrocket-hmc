from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ksas_xrocket.audit import EXPECTED_CHANNELS
from ksas_xrocket.cli import main
from ksas_xrocket.task_1_1_explain import (
    Task11ExplanationError,
    aggregate_fold_importance,
    normalize_importance,
    parse_channel_name,
    run_task_1_1_explanation,
)
from ksas_xrocket.xrocket_experiment import run_xrocket_experiment


def test_parse_channel_name_handles_multiword_sensor_families() -> None:
    assert parse_channel_name("lin_accel_z") == {
        "sensor_family": "lin_accel",
        "axis": "z",
        "channel": "lin_accel_z",
        "family_axis": "lin_accel_z",
        "channel_combination": "lin_accel_z",
    }
    assert parse_channel_name("game_rot_vec_x")["sensor_family"] == "game_rot_vec"

    with pytest.raises(Task11ExplanationError, match="Unknown sensor family"):
        parse_channel_name("unknown_x")
    with pytest.raises(Task11ExplanationError, match="Unknown axis"):
        parse_channel_name("gyros_bad")


def test_normalize_importance_rejects_invalid_values() -> None:
    normalized = normalize_importance(pd.Series([1.0, 3.0]), fold=0)

    assert normalized.tolist() == pytest.approx([0.25, 0.75])

    with pytest.raises(Task11ExplanationError, match="negative"):
        normalize_importance(pd.Series([1.0, -1.0]), fold=0)
    with pytest.raises(Task11ExplanationError, match="zero total"):
        normalize_importance(pd.Series([0.0, 0.0]), fold=0)


def test_aggregate_fold_importance_preserves_normalized_total() -> None:
    frame = pd.DataFrame(
        [
            {"fold": 0, "sensor_family": "accelerometer", "normalized_importance": 0.2},
            {"fold": 0, "sensor_family": "gyros", "normalized_importance": 0.8},
            {"fold": 1, "sensor_family": "accelerometer", "normalized_importance": 0.4},
            {"fold": 1, "sensor_family": "gyros", "normalized_importance": 0.6},
        ]
    )

    aggregated = aggregate_fold_importance(frame, group_level="sensor_family")

    totals = aggregated.groupby("fold")["importance"].sum().to_dict()
    assert totals == {0: pytest.approx(1.0), 1: pytest.approx(1.0)}
    assert set(aggregated["group_value"]) == {"accelerometer", "gyros"}


def test_task_1_1_cli_smoke_writes_expected_outputs(tmp_path: Path) -> None:
    processed_dir, splits_dir = make_experiment_fixture(tmp_path)
    xrocket_dir = tmp_path / "xrocket"
    output_dir = tmp_path / "task_1_1"
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
    config_path = tmp_path / "task_1_1.yaml"
    config_path.write_text(
        "\n".join(
            [
                "task: task_1_1",
                f"processed_dir: {processed_dir.as_posix()}",
                f"xrocket_dir: {xrocket_dir.as_posix()}",
                f"output_dir: {output_dir.as_posix()}",
                "classes: [0, 1]",
                "random_state: 42",
                "validation:",
                "  permutation_repeats: 1",
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
        "fold_native_importance.parquet",
        "sensor_family_importance_summary.csv",
        "axis_importance_summary.csv",
        "channel_importance_summary.csv",
        "family_axis_importance_summary.csv",
        "channel_combination_importance_summary.csv",
        "class_specific_channel_importance.csv",
        "class_specific_sensor_family_importance.csv",
        "ablation_metrics.csv",
        "permutation_importance.csv",
        "method_agreement.csv",
        "task_1_1_answer.md",
    }
    assert expected.issubset({path.name for path in output_dir.iterdir()})
    figures = output_dir / "figures"
    for stem in (
        "sensor_family_contribution",
        "axis_channel_contribution_heatmap",
        "ranked_channel_importance",
        "fold_stability",
        "ablation_impact",
        "class_specific_channel_profiles",
        "method_agreement",
    ):
        assert (figures / f"{stem}.png").stat().st_size > 0
        assert (figures / f"{stem}.pdf").stat().st_size > 0

    native = pd.read_parquet(output_dir / "fold_native_importance.parquet")
    assert native.groupby("fold")["normalized_importance"].sum().iloc[0] == pytest.approx(1.0)
    ablation = pd.read_csv(output_dir / "ablation_metrics.csv")
    permutation = pd.read_csv(output_dir / "permutation_importance.csv")
    assert {"sensor_family", "channel"} == set(ablation["group_level"])
    assert {"sensor_family", "channel"} == set(permutation["group_level"])
    resolved = json.loads((output_dir / "resolved_config.json").read_text(encoding="utf-8"))
    assert resolved["task"] == "task_1_1"

    with pytest.raises(Task11ExplanationError, match="not empty"):
        run_task_1_1_explanation(
            processed_dir=processed_dir,
            xrocket_dir=xrocket_dir,
            output_dir=output_dir,
            label_ids=(0, 1),
            random_forest_estimators=5,
            permutation_repeats=1,
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
