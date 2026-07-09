from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ksas_xrocket.audit import EXPECTED_CHANNELS
from ksas_xrocket.xrocket_adapter import XRocketAdapter, XRocketAdapterError
from ksas_xrocket.xrocket_experiment import (
    PRIMARY_MODEL,
    SENSITIVITY_MODEL,
    XRocketExperimentError,
    run_xrocket_experiment,
)


def test_adapter_requires_explicit_fit_and_valid_input() -> None:
    adapter = make_adapter()
    x = make_adapter_data()

    with pytest.raises(XRocketAdapterError, match="explicitly fitted"):
        adapter.transform(x)
    assert not adapter.is_fitted

    with pytest.raises(XRocketAdapterError, match="Expected 2 channels"):
        adapter.fit(x[:, :1, :])
    bad = x.copy()
    bad[0, 0, 0] = np.nan
    with pytest.raises(XRocketAdapterError, match="non-finite"):
        adapter.fit(bad)
    with pytest.raises(XRocketAdapterError, match="kernel_length=9"):
        XRocketAdapter(
            in_channels=2,
            max_kernel_span=9,
            channel_names=("a", "b"),
            feature_cap=168,
            kernel_length=7,
        )


def test_adapter_metadata_batching_and_round_trip(tmp_path: Path) -> None:
    x = make_adapter_data()
    adapter = make_adapter().fit(x[:4])
    one_batch = adapter.transform(x, batch_size=len(x))
    small_batches = adapter.transform(x, batch_size=2)

    np.testing.assert_array_equal(one_batch, small_batches)
    assert one_batch.shape == (6, 168)
    assert one_batch.dtype == np.float32

    metadata = adapter.feature_metadata()
    assert len(metadata) == one_batch.shape[1]
    assert [row["feature_index"] for row in metadata] == list(range(168))
    assert metadata[0]["kernel_id"] == "d00_p000"
    assert metadata[0]["pattern_index"] == 0
    assert metadata[0]["dilation"] == 1
    assert metadata[0]["padding_per_side"] == 4
    assert metadata[0]["kernel_length"] == 9
    assert metadata[0]["feature_type"] == "ppv"
    assert metadata[0]["channel_name"] == "a"
    assert metadata[0]["effective_receptive_field_samples"] == 9
    assert metadata[0]["effective_receptive_field_seconds_nominal"] == pytest.approx(0.18)
    assert metadata[0]["relative_span"] == 1.0
    assert metadata[-1]["channel_name"] == "b"

    metadata_path = tmp_path / "metadata.parquet"
    frame = pd.DataFrame(metadata)
    frame.to_parquet(metadata_path, index=False)
    pd.testing.assert_frame_equal(frame, pd.read_parquet(metadata_path))

    adapter_path = tmp_path / "adapter.joblib"
    adapter.save(adapter_path)
    loaded = XRocketAdapter.load(adapter_path)
    np.testing.assert_array_equal(one_batch, loaded.transform(x, batch_size=3))
    assert loaded.feature_metadata() == metadata


def test_m3_experiment_smoke_writes_reloadable_traceability_artifacts(
    tmp_path: Path,
) -> None:
    processed_dir, splits_dir = make_experiment_fixture(tmp_path)
    output_dir = tmp_path / "results"

    result = run_xrocket_experiment(
        processed_dir=processed_dir,
        splits_dir=splits_dir,
        output_dir=output_dir,
        folds=(0,),
        label_ids=(0, 1),
        feature_cap=1512,
        max_dilations=1,
        random_forest_estimators=5,
        logistic_max_iter=100,
        batch_size=4,
    )

    assert result.metrics_path.is_file()
    assert result.aggregate_metrics_path.is_file()
    assert result.predictions_path.is_file()
    assert result.confusion_matrices_path.is_file()
    assert result.runtime_path.is_file()
    assert result.provenance_path.is_file()
    fold_dir = output_dir / "fold_0"
    expected = {
        "features.npz",
        "feature_metadata.parquet",
        "feature_importance.parquet",
        "xrocket_adapter.joblib",
        f"{PRIMARY_MODEL}.joblib",
        f"{SENSITIVITY_MODEL}.joblib",
        "padding_feature_diagnostics.csv",
        "padding_feature_summary.csv",
        "padding_threshold_diagnostics.csv",
        "padding_prediction_diagnostics.csv",
    }
    assert expected.issubset({path.name for path in fold_dir.iterdir()})

    metadata = pd.read_parquet(fold_dir / "feature_metadata.parquet")
    importance = pd.read_parquet(fold_dir / "feature_importance.parquet")
    with np.load(fold_dir / "features.npz", allow_pickle=False) as saved:
        assert saved["features"].shape == (12, 1512)
        assert saved["features"].dtype == np.float32
    assert len(metadata) == 1512
    assert metadata["feature_index"].tolist() == list(range(1512))
    assert importance["feature_index"].tolist() == metadata["feature_index"].tolist()
    assert importance["channel_indices"].tolist() == metadata["channel_indices"].tolist()
    assert importance["dilation"].tolist() == metadata["dilation"].tolist()
    assert importance["importance"].sum() == pytest.approx(1.0)

    metrics = read_rows(result.metrics_path)
    assert {row["model"] for row in metrics} == {PRIMARY_MODEL, SENSITIVITY_MODEL}
    assert {row["fold"] for row in metrics} == {"0"}
    predictions = read_rows(result.predictions_path)
    assert len(predictions) == 12
    assert {row["model"] for row in predictions} == {PRIMARY_MODEL, SENSITIVITY_MODEL}
    assert len(read_rows(fold_dir / "padding_feature_diagnostics.csv")) == 12
    assert len(read_rows(fold_dir / "padding_feature_summary.csv")) == 2
    assert len(read_rows(fold_dir / "padding_threshold_diagnostics.csv")) == 1
    assert len(read_rows(fold_dir / "padding_prediction_diagnostics.csv")) == 2

    with pytest.raises(XRocketExperimentError, match="not empty"):
        run_xrocket_experiment(
            processed_dir=processed_dir,
            splits_dir=splits_dir,
            output_dir=output_dir,
            folds=(0,),
            label_ids=(0, 1),
            feature_cap=1512,
            max_dilations=1,
            random_forest_estimators=5,
            logistic_max_iter=100,
        )


def make_adapter() -> XRocketAdapter:
    return XRocketAdapter(
        in_channels=2,
        max_kernel_span=9,
        channel_names=("a", "b"),
        feature_cap=168,
        max_dilations=1,
    )


def make_adapter_data() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.normal(size=(6, 2, 9)).astype(np.float32)


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
    test_by_fold = {
        0: {0, 1, 2, 3, 4, 5},
        1: {6, 7, 8, 9, 10, 11},
    }
    for fold, test_indices in test_by_fold.items():
        for index in range(12):
            rows.append(
                {
                    "fold": fold,
                    "split": "test" if index in test_indices else "train",
                    "sample_index": index,
                    "sample_id": sample_ids[index],
                    "movement_label_id": int(labels[index]),
                    "participant_id": participant_ids[index],
                    "arm_code": arm_codes[index],
                }
            )
    with (splits_dir / "m2_grouped_folds.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return processed_dir, splits_dir


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
