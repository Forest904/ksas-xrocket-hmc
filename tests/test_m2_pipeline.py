from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from ksas_xrocket.audit import EXPECTED_CHANNELS, EXPECTED_LABELS, sha256_file
from ksas_xrocket.baselines import (
    BaselineError,
    build_confusion_rows,
    build_prediction_rows,
    compute_metrics,
    run_baselines,
    validate_split_manifest_against_arrays,
)
from ksas_xrocket.cli import main
from ksas_xrocket.prepare import PreparationError, build_prepared_data, prepare_ksas_tensors
from ksas_xrocket.splits import build_fold_diagnostics, generate_grouped_splits


def test_manifest_validation_requires_core_columns(tmp_path: Path) -> None:
    manifest_path = tmp_path / "samples.csv"
    manifest_path.write_text("sample_id\nexample\n", encoding="utf-8")

    with pytest.raises(PreparationError, match="missing required columns"):
        build_prepared_data(manifest_path=manifest_path, target_length=4, project_root=tmp_path)


def test_prepare_preserves_order_channels_padding_and_mask(tmp_path: Path) -> None:
    manifest_path = make_synthetic_manifest(tmp_path, participant_count=3, include_both_arms=False)

    prepared = build_prepared_data(
        manifest_path=manifest_path,
        target_length=4,
        project_root=tmp_path,
    )

    assert prepared.x.shape == (18, len(EXPECTED_CHANNELS), 4)
    assert prepared.x.dtype == np.float32
    assert prepared.y.tolist()[:6] == list(EXPECTED_LABELS)
    assert prepared.sample_ids.tolist()[0] == "ksas_m0_p01_d"
    assert prepared.sample_ids.tolist()[1] == "ksas_m1_p01_d"
    assert prepared.original_lengths.tolist()[0] == 2
    assert prepared.valid_mask[0].tolist() == [True, True, False, False]
    assert prepared.x[0, 0, :].tolist() == [100.0, 101.0, 0.0, 0.0]
    np.testing.assert_allclose(prepared.x[0, 1, :], [100.01, 101.01, 0.0, 0.0])


def test_prepare_fails_on_raw_checksum_mismatch(tmp_path: Path) -> None:
    manifest_path = make_synthetic_manifest(tmp_path, participant_count=1, include_both_arms=False)
    rows = read_csv_rows(manifest_path)
    rows[0]["checksum_sha256"] = "0" * 64
    write_manifest_rows(manifest_path, rows)

    with pytest.raises(PreparationError, match="checksum mismatch"):
        build_prepared_data(manifest_path=manifest_path, target_length=4, project_root=tmp_path)


@pytest.mark.parametrize(
    ("bad_rows", "expected"),
    [
        ([["1.0"]], "has 1 values"),
        ([[*["1.0"] * 17, ""]], "is empty"),
        ([[*["1.0"] * 17, "oops"]], "non-numeric value"),
    ],
)
def test_prepare_reports_malformed_csv_context(
    tmp_path: Path,
    bad_rows: list[list[str]],
    expected: str,
) -> None:
    manifest_path = make_synthetic_manifest(tmp_path, participant_count=1, include_both_arms=False)
    rows = read_csv_rows(manifest_path)
    bad_path = tmp_path / rows[0]["source_path"]
    with bad_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(EXPECTED_CHANNELS)
        writer.writerows(bad_rows)
    rows[0]["checksum_sha256"] = sha256_file(bad_path)
    rows[0]["original_length"] = str(len(bad_rows))
    write_manifest_rows(manifest_path, rows)

    with pytest.raises(PreparationError, match=expected):
        build_prepared_data(manifest_path=manifest_path, target_length=4, project_root=tmp_path)


def test_grouped_splits_have_no_participant_overlap_and_full_class_coverage(
    tmp_path: Path,
) -> None:
    manifest_path = make_synthetic_manifest(tmp_path, participant_count=6, include_both_arms=False)

    result = generate_grouped_splits(
        manifest_path=manifest_path,
        output_dir=tmp_path / "splits",
        n_splits=3,
        random_state=42,
    )

    split_rows = read_csv_rows(result.split_manifest_path)
    diagnostics = read_csv_rows(result.diagnostics_path)

    assert len(diagnostics) == 3
    for fold in range(3):
        train_participants = {
            row["participant_id"]
            for row in split_rows
            if row["fold"] == str(fold) and row["split"] == "train"
        }
        test_participants = {
            row["participant_id"]
            for row in split_rows
            if row["fold"] == str(fold) and row["split"] == "test"
        }
        test_labels = {
            int(row["movement_label_id"])
            for row in split_rows
            if row["fold"] == str(fold) and row["split"] == "test"
        }
        assert train_participants.isdisjoint(test_participants)
        assert test_labels == set(EXPECTED_LABELS)
        assert diagnostics[fold]["participant_overlap_count"] == "0"
        assert json.loads(diagnostics[fold]["absent_test_classes"]) == []


def test_fold_diagnostics_reports_actual_participant_overlap() -> None:
    labels = np.asarray([0, 1, 0, 1])
    groups = np.asarray(["P01", "P02", "P01", "P03"])
    arms = np.asarray(["d", "d", "i", "i"])

    diagnostics = build_fold_diagnostics(
        fold_index=0,
        train_indices=np.asarray([0, 1]),
        test_indices=np.asarray([2, 3]),
        labels=labels,
        groups=groups,
        arms=arms,
    )

    assert diagnostics["participant_overlap_count"] == 1
    assert diagnostics["participant_overlap"] == ["P01"]


def test_split_manifest_validation_catches_stale_sample_metadata(tmp_path: Path) -> None:
    manifest_path = make_synthetic_manifest(tmp_path, participant_count=6, include_both_arms=False)
    processed_dir = tmp_path / "processed"
    splits_dir = tmp_path / "splits"
    prepare_ksas_tensors(
        manifest_path=manifest_path,
        output_dir=processed_dir,
        target_length=4,
        project_root=tmp_path,
    )
    split_result = generate_grouped_splits(
        manifest_path=manifest_path,
        output_dir=splits_dir,
        n_splits=3,
        random_state=42,
    )
    arrays = np.load(processed_dir / "tensors.npz", allow_pickle=False)
    split_rows = read_csv_rows(split_result.split_manifest_path)
    split_rows[0]["sample_id"] = "stale_sample"
    tensor_arrays = {name: arrays[name] for name in arrays.files}

    with pytest.raises(BaselineError, match="sample_id mismatch"):
        validate_split_manifest_against_arrays(split_rows, tensor_arrays)


def test_split_manifest_validation_catches_out_of_range_index(tmp_path: Path) -> None:
    manifest_path = make_synthetic_manifest(tmp_path, participant_count=6, include_both_arms=False)
    processed_dir = tmp_path / "processed"
    splits_dir = tmp_path / "splits"
    prepare_ksas_tensors(
        manifest_path=manifest_path,
        output_dir=processed_dir,
        target_length=4,
        project_root=tmp_path,
    )
    split_result = generate_grouped_splits(
        manifest_path=manifest_path,
        output_dir=splits_dir,
        n_splits=3,
        random_state=42,
    )
    arrays = np.load(processed_dir / "tensors.npz", allow_pickle=False)
    split_rows = read_csv_rows(split_result.split_manifest_path)
    split_rows[0]["sample_index"] = "999"
    tensor_arrays = {name: arrays[name] for name in arrays.files}

    with pytest.raises(BaselineError, match="outside tensor range"):
        validate_split_manifest_against_arrays(split_rows, tensor_arrays)


def test_metrics_and_confusion_support_non_contiguous_labels() -> None:
    y_true = np.asarray([10, 10, 20, 20])
    y_pred = np.asarray([10, 20, 20, 20])

    metrics = compute_metrics(y_true, y_pred, label_ids=(10, 20))
    confusion = build_confusion_rows(
        model_name="example",
        fold=0,
        y_true=y_true,
        y_pred=y_pred,
        label_ids=(10, 20),
    )

    assert metrics["class_10_recall"] == 0.5
    assert metrics["class_20_recall"] == 1.0
    assert {(row["true_label"], row["predicted_label"]): row["count"] for row in confusion} == {
        (10, 10): 1,
        (10, 20): 1,
        (20, 10): 0,
        (20, 20): 2,
    }


def test_prediction_probabilities_map_by_estimator_classes() -> None:
    arrays = {
        "sample_id": np.asarray(["s1"]),
        "participant_id": np.asarray(["P01"]),
        "arm_code": np.asarray(["d"]),
        "y": np.asarray([10]),
    }

    rows = build_prediction_rows(
        model_name="example",
        fold=0,
        sample_indices=np.asarray([0]),
        arrays=arrays,
        y_pred=np.asarray([20]),
        probabilities=np.asarray([[0.25, 0.75]]),
        probability_classes=(20, 10),
        label_ids=(10, 20),
    )

    assert rows[0]["prob_class_10"] == 0.75
    assert rows[0]["prob_class_20"] == 0.25


def test_baseline_smoke_run_writes_metrics_predictions_and_confusion_matrices(
    tmp_path: Path,
) -> None:
    manifest_path = make_synthetic_manifest(tmp_path, participant_count=6, include_both_arms=False)
    processed_dir = tmp_path / "processed"
    splits_dir = tmp_path / "splits"
    output_dir = tmp_path / "baselines"

    prepare_ksas_tensors(
        manifest_path=manifest_path,
        output_dir=processed_dir,
        target_length=4,
        project_root=tmp_path,
    )
    generate_grouped_splits(
        manifest_path=manifest_path,
        output_dir=splits_dir,
        n_splits=3,
        random_state=42,
    )

    result = run_baselines(
        processed_dir=processed_dir,
        splits_dir=splits_dir,
        output_dir=output_dir,
        random_state=42,
    )

    metrics = read_csv_rows(result.metrics_path)
    aggregate = read_csv_rows(result.aggregate_metrics_path)
    predictions = read_csv_rows(result.predictions_path)
    confusion = read_csv_rows(result.confusion_matrices_path)

    assert {row["model"] for row in metrics} == {
        "majority",
        "statistical_logistic_regression",
        "statistical_random_forest",
    }
    assert len(aggregate) == 3
    assert len(predictions) == 3 * 36
    assert len(confusion) == 3 * 3 * 36
    assert "macro_f1" in metrics[0]
    assert "balanced_accuracy" in metrics[0]


def test_prepare_config_loads_and_cli_overrides_target_length(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest_path = make_synthetic_manifest(tmp_path, participant_count=6, include_both_arms=False)
    config_path = tmp_path / "prepare.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"manifest: {manifest_path.as_posix()}",
                f"output_dir: {(tmp_path / 'config_processed').as_posix()}",
                "target_length: 3",
                "splits:",
                f"  output_dir: {(tmp_path / 'config_splits').as_posix()}",
                "  n_splits: 3",
                "  random_state: 42",
            ]
        ),
        encoding="utf-8",
    )
    override_output_dir = tmp_path / "override_processed"

    exit_code = main(
        [
            "prepare",
            "--config",
            str(config_path),
            "--output-dir",
            str(override_output_dir),
            "--target-length",
            "4",
        ]
    )

    assert exit_code == 0
    resolved = json.loads(
        (override_output_dir / "resolved_config.json").read_text(encoding="utf-8")
    )
    contract = json.loads(
        (override_output_dir / "tensor_contract.json").read_text(encoding="utf-8")
    )
    assert resolved["output_dir"] == override_output_dir.as_posix()
    assert resolved["target_length"] == 4
    assert contract["shape"] == [36, 18, 4]


def test_baseline_config_loads_and_cli_overrides_output_dir(tmp_path: Path) -> None:
    manifest_path = make_synthetic_manifest(tmp_path, participant_count=6, include_both_arms=False)
    processed_dir = tmp_path / "processed"
    splits_dir = tmp_path / "splits"
    baseline_config_output_dir = tmp_path / "config_baselines"
    override_output_dir = tmp_path / "override_baselines"
    prepare_ksas_tensors(
        manifest_path=manifest_path,
        output_dir=processed_dir,
        target_length=4,
        project_root=tmp_path,
    )
    generate_grouped_splits(
        manifest_path=manifest_path,
        output_dir=splits_dir,
        n_splits=3,
        random_state=42,
    )
    config_path = tmp_path / "baseline.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"processed_dir: {processed_dir.as_posix()}",
                f"splits_dir: {splits_dir.as_posix()}",
                f"output_dir: {baseline_config_output_dir.as_posix()}",
                "classes: [0, 1, 2, 3, 4, 5]",
                "random_state: 42",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "baseline",
            "--config",
            str(config_path),
            "--output-dir",
            str(override_output_dir),
        ]
    )

    assert exit_code == 0
    resolved = json.loads(
        (override_output_dir / "resolved_config.json").read_text(encoding="utf-8")
    )
    assert resolved["output_dir"] == override_output_dir.as_posix()
    assert not baseline_config_output_dir.exists()


def make_synthetic_manifest(
    tmp_path: Path,
    participant_count: int,
    include_both_arms: bool,
) -> Path:
    raw_dir = tmp_path / "data" / "raw" / "KSAS-Dataset" / "movements"
    raw_dir.mkdir(parents=True)
    manifest_path = tmp_path / "data" / "manifests" / "samples.csv"
    manifest_path.parent.mkdir(parents=True)
    arms = ("d", "i") if include_both_arms else ("d",)
    rows: list[dict[str, str]] = []

    for participant in range(1, participant_count + 1):
        for arm_code in arms:
            for label_id, movement_label in EXPECTED_LABELS.items():
                filename = f"{label_id}-{participant}-{arm_code}.csv"
                sample_id = f"ksas_m{label_id}_p{participant:02d}_{arm_code}"
                csv_path = raw_dir / filename
                write_synthetic_csv(
                    csv_path,
                    base=participant * 100 + label_id * 10 + (0 if arm_code == "d" else 5),
                )
                rows.append(
                    {
                        "sample_id": sample_id,
                        "filename": filename,
                        "source_path": f"data/raw/KSAS-Dataset/movements/{filename}",
                        "checksum_sha256": sha256_file(csv_path),
                        "movement_label_id": str(label_id),
                        "movement_label": movement_label,
                        "participant_id": f"P{participant:02d}",
                        "arm_code": arm_code,
                        "arm": "right" if arm_code == "d" else "left",
                        "recording_id": filename.removesuffix(".csv"),
                        "split_group": f"P{participant:02d}",
                        "sensor_location": "smartphone",
                        "device_model": "synthetic",
                        "device_orientation": "",
                        "sampling_rate_hz": "",
                        "original_length": "2",
                        "processed_length": "2",
                        "channel_names": json.dumps(list(EXPECTED_CHANNELS)),
                        "quality_flag": "valid",
                    }
                )

    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    return manifest_path


def write_synthetic_csv(path: Path, base: int) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(EXPECTED_CHANNELS)
        for row_index in range(2):
            writer.writerow(
                [
                    f"{base + row_index + channel_index / 100:.2f}"
                    for channel_index, _channel in enumerate(EXPECTED_CHANNELS)
                ]
            )


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_manifest_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
