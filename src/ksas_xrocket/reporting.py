"""Report artifact preparation and PDF rendering for M8."""

from __future__ import annotations

import csv
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from ksas_xrocket.audit import EXPECTED_CHANNELS, SENSOR_UNITS


class ReportBuildError(RuntimeError):
    """Raised when report artifacts or PDF rendering fail validation."""


@dataclass(frozen=True)
class ReportArtifact:
    """A tracked source artifact copied into the report workspace."""

    source: Path
    destination_name: str
    description: str


@dataclass(frozen=True)
class ReportArtifactsResult:
    """Paths produced by report artifact preparation."""

    figures_dir: Path
    tables_dir: Path
    manifest_path: Path
    figure_count: int
    table_count: int


@dataclass(frozen=True)
class ReportBuildResult:
    """Paths produced by report PDF rendering."""

    pdf_path: Path
    artifact_manifest_path: Path


REPOSITORY_URL = "https://github.com/Forest904/ksas-xrocket-hmc.git"

FIGURE_ARTIFACTS: tuple[ReportArtifact, ...] = (
    ReportArtifact(
        Path("results/explanations/task_1_1/figures/sensor_family_contribution.pdf"),
        "task_1_1_sensor_family_contribution.pdf",
        "Task 1.1 sensor-family contribution",
    ),
    ReportArtifact(
        Path("results/explanations/task_1_1/figures/axis_channel_contribution_heatmap.pdf"),
        "task_1_1_axis_channel_contribution_heatmap.pdf",
        "Task 1.1 channel and axis heatmap",
    ),
    ReportArtifact(
        Path("results/explanations/task_1_1/figures/class_specific_channel_profiles.pdf"),
        "task_1_1_class_specific_channel_profiles.pdf",
        "Task 1.1 class-specific channel profiles",
    ),
    ReportArtifact(
        Path("results/explanations/task_1_1/figures/ablation_impact.pdf"),
        "task_1_1_ablation_impact.pdf",
        "Task 1.1 feature-group ablation",
    ),
    ReportArtifact(
        Path("results/explanations/task_1_1/figures/fold_stability.pdf"),
        "task_1_1_fold_stability.pdf",
        "Task 1.1 fold stability",
    ),
    ReportArtifact(
        Path("results/explanations/task_1_2/figures/dilation_importance.pdf"),
        "task_1_2_dilation_importance.pdf",
        "Task 1.2 dilation importance",
    ),
    ReportArtifact(
        Path("results/explanations/task_1_2/figures/temporal_scale_contribution.pdf"),
        "task_1_2_temporal_scale_contribution.pdf",
        "Task 1.2 temporal-scale contribution",
    ),
    ReportArtifact(
        Path("results/explanations/task_1_2/figures/class_specific_scale_profiles.pdf"),
        "task_1_2_class_specific_scale_profiles.pdf",
        "Task 1.2 class-specific temporal-scale profiles",
    ),
    ReportArtifact(
        Path("results/explanations/task_1_2/figures/fold_stability.pdf"),
        "task_1_2_fold_stability.pdf",
        "Task 1.2 fold stability",
    ),
    ReportArtifact(
        Path("results/explanations/task_1_3/figures/pattern_case_01.pdf"),
        "task_1_3_pattern_case_01.pdf",
        "Task 1.3 first representative pattern case",
    ),
    ReportArtifact(
        Path("results/explanations/task_1_3/figures/pattern_case_02.pdf"),
        "task_1_3_pattern_case_02.pdf",
        "Task 1.3 second representative pattern case",
    ),
    ReportArtifact(
        Path("results/explanations/task_1_3/figures/pattern_case_03.pdf"),
        "task_1_3_pattern_case_03.pdf",
        "Task 1.3 third representative pattern case",
    ),
    ReportArtifact(
        Path("results/explanations/task_1_3/figures/pattern_case_failure_or_ambiguous.pdf"),
        "task_1_3_pattern_case_failure_or_ambiguous.pdf",
        "Task 1.3 failure or ambiguous pattern case",
    ),
    ReportArtifact(
        Path("results/explanations/task_1_3/figures/pattern_feature_distributions.pdf"),
        "task_1_3_pattern_feature_distributions.pdf",
        "Task 1.3 pattern feature distributions",
    ),
    ReportArtifact(
        Path("results/explanations/task_1_3/figures/pattern_summary_table.pdf"),
        "task_1_3_pattern_summary_table.pdf",
        "Task 1.3 pattern summary table",
    ),
)

COMPOSITE_FIGURE_NAMES = (
    "task_1_1_core_evidence.pdf",
    "task_1_1_validation_checks.pdf",
    "task_1_2_scale_evidence.pdf",
    "task_1_2_class_and_stability.pdf",
    "task_1_3_case_cards_plausible.pdf",
    "task_1_3_case_cards_ambiguous.pdf",
    "task_1_3_distribution_summary.pdf",
)

TABLE_ARTIFACTS: tuple[ReportArtifact, ...] = (
    ReportArtifact(
        Path("results/baselines/m2_raw_padded/aggregate_metrics.csv"),
        "baseline_aggregate_metrics.csv",
        "Baseline aggregate metrics",
    ),
    ReportArtifact(
        Path("results/xrocket/m3_raw_padded/aggregate_metrics.csv"),
        "xrocket_aggregate_metrics.csv",
        "XROCKET aggregate metrics",
    ),
    ReportArtifact(
        Path("results/explanations/task_1_1/task_1_1_answer.md"),
        "task_1_1_answer.md",
        "Task 1.1 generated answer draft",
    ),
    ReportArtifact(
        Path("results/explanations/task_1_2/task_1_2_answer.md"),
        "task_1_2_answer.md",
        "Task 1.2 generated answer draft",
    ),
    ReportArtifact(
        Path("results/explanations/task_1_3/task_1_3_answer.md"),
        "task_1_3_answer.md",
        "Task 1.3 generated answer draft",
    ),
    ReportArtifact(
        Path("results/controls/m7_raw_padded/m7_controls_summary.md"),
        "m7_controls_summary.md",
        "M7 negative-control summary",
    ),
    ReportArtifact(
        Path("results/stability/m7_raw_padded/m7_stability_summary.md"),
        "m7_stability_summary.md",
        "M7 stability summary",
    ),
)


def prepare_report_artifacts(
    figures_dir: Path = Path("reports/figures"),
    tables_dir: Path = Path("reports/tables"),
    manifest_path: Path = Path("reports/report_artifacts_manifest.md"),
) -> ReportArtifactsResult:
    """Copy validated final report figures and compact table sources."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    copied_figures = _copy_artifacts(FIGURE_ARTIFACTS, figures_dir)
    copied_tables = _copy_artifacts(TABLE_ARTIFACTS, tables_dir)
    _write_pattern_summary_figure(
        source=Path("results/explanations/task_1_3/pattern_cases.csv"),
        destination=figures_dir / "task_1_3_pattern_summary_table.pdf",
    )
    composite_figures = _write_composite_figures(figures_dir)

    manifest_lines = [
        "# Report Artifact Manifest",
        "",
        "Generated by `hmc figures` from tracked result artifacts.",
        "",
        "## Source Figures",
        "",
    ]
    manifest_lines.extend(_manifest_rows(copied_figures, figures_dir))
    manifest_lines.extend(["", "## Report Composite Figures", ""])
    manifest_lines.extend(_manifest_rows(composite_figures, figures_dir))
    manifest_lines.extend(["", "## Tables And Source Extracts", ""])
    manifest_lines.extend(_manifest_rows(copied_tables, tables_dir))
    manifest_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

    return ReportArtifactsResult(
        figures_dir=figures_dir,
        tables_dir=tables_dir,
        manifest_path=manifest_path,
        figure_count=len(copied_figures) + len(composite_figures),
        table_count=len(copied_tables),
    )


def build_report_pdf(
    report_source: Path = Path("reports/technical_report.md"),
    bibliography: Path = Path("reports/references.bib"),
    output_pdf: Path = Path("reports/ksas_xrocket_hmc_report.pdf"),
    figures_dir: Path = Path("reports/figures"),
    tables_dir: Path = Path("reports/tables"),
) -> ReportBuildResult:
    """Validate report inputs and render the final PDF with Pandoc."""
    artifact_result = prepare_report_artifacts(figures_dir=figures_dir, tables_dir=tables_dir)
    _validate_report_source(report_source, bibliography, figures_dir)

    pandoc = shutil.which("pandoc")
    if pandoc is None:
        raise ReportBuildError("Pandoc is required to build the PDF but was not found on PATH.")
    pdf_engine = shutil.which("pdflatex")
    if pdf_engine is None:
        raise ReportBuildError("MiKTeX/pdflatex is required to build the PDF but was not found.")

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    command = [
        pandoc,
        str(report_source),
        "--from",
        "markdown+pipe_tables",
        "--citeproc",
        "--bibliography",
        str(bibliography),
        "--resource-path",
        ".;reports;reports/figures",
        "--pdf-engine",
        pdf_engine,
        "-V",
        "geometry:margin=1in",
        "-V",
        "fontsize=10pt",
        "-V",
        "colorlinks=true",
        "-V",
        "linkcolor=blue",
        "-V",
        "urlcolor=blue",
        "-V",
        "fig-pos=H",
        "-H",
        "reports/latex_header.tex",
        "-o",
        str(output_pdf),
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        stdout = completed.stdout.strip()
        detail = stderr or stdout or "Pandoc failed without diagnostic output."
        raise ReportBuildError(f"Pandoc report build failed:\n{detail}")
    if not output_pdf.exists() or output_pdf.stat().st_size == 0:
        raise ReportBuildError(f"Pandoc did not create a non-empty PDF at {output_pdf}.")

    return ReportBuildResult(
        pdf_path=output_pdf,
        artifact_manifest_path=artifact_result.manifest_path,
    )


def _copy_artifacts(artifacts: tuple[ReportArtifact, ...], destination_dir: Path) -> list[Path]:
    copied_paths: list[Path] = []
    missing = [artifact.source for artifact in artifacts if not artifact.source.exists()]
    if missing:
        missing_text = "\n".join(f"- {path}" for path in missing)
        raise ReportBuildError(f"Missing required report artifacts:\n{missing_text}")

    for artifact in artifacts:
        destination = destination_dir / artifact.destination_name
        shutil.copy2(artifact.source, destination)
        copied_paths.append(destination)
    return copied_paths


def _manifest_rows(paths: list[Path], base_dir: Path) -> list[str]:
    rows = ["| File |", "|---|"]
    for path in paths:
        rows.append(f"| `{base_dir.as_posix()}/{path.name}` |")
    return rows


def _write_pattern_summary_figure(source: Path, destination: Path) -> None:
    """Write a compact, report-readable pattern summary figure from saved cases."""
    if not source.exists():
        raise ReportBuildError(f"Pattern case table is missing: {source}")

    rows: list[list[str]] = []
    with source.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows.append(
                [
                    _short_case_id(str(row["case_id"])),
                    str(row["movement_label"]),
                    str(row["channel_name"]),
                    str(row["dilation"]),
                    f"{float(row['feature_value_ppv']):.3f}",
                    f"{float(row['class_separation_auc']):.3f}",
                    str(row["human_meaningfulness"]),
                ]
            )

    headers = ["case", "movement", "channel", "dil.", "PPV", "AUC", "meaning"]
    pyplot = _load_pyplot()
    figure, axis = pyplot.subplots(figsize=(7.4, 2.0))
    axis.axis("off")
    axis.set_title("Task 1.3 pattern summary", pad=8)
    table = axis.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="center",
        loc="center",
        colWidths=[0.12, 0.24, 0.14, 0.08, 0.09, 0.09, 0.14],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.0)
    table.scale(1.0, 1.45)
    figure.tight_layout()
    figure.savefig(destination, bbox_inches="tight")
    pyplot.close(figure)


def _write_composite_figures(figures_dir: Path) -> list[Path]:
    """Create report-only composite figures from final tracked result tables."""
    _validate_composite_inputs()
    pyplot = _load_pyplot()
    destinations = [
        _write_task_1_1_core_evidence(pyplot, figures_dir / "task_1_1_core_evidence.pdf"),
        _write_task_1_1_validation_checks(
            pyplot, figures_dir / "task_1_1_validation_checks.pdf"
        ),
        _write_task_1_2_scale_evidence(pyplot, figures_dir / "task_1_2_scale_evidence.pdf"),
        _write_task_1_2_class_and_stability(
            pyplot, figures_dir / "task_1_2_class_and_stability.pdf"
        ),
        _write_task_1_3_case_cards(
            pyplot,
            figures_dir / "task_1_3_case_cards_plausible.pdf",
            ("pattern_case_01", "pattern_case_02"),
            "Task 1.3 plausible representative patterns",
        ),
        _write_task_1_3_case_cards(
            pyplot,
            figures_dir / "task_1_3_case_cards_ambiguous.pdf",
            ("pattern_case_03", "pattern_case_failure_or_ambiguous"),
            "Task 1.3 ambiguous and failure-pattern checks",
        ),
        _write_task_1_3_distribution_summary(
            pyplot, figures_dir / "task_1_3_distribution_summary.pdf"
        ),
    ]
    return destinations


def _validate_composite_inputs() -> None:
    required = (
        Path("results/explanations/task_1_1/sensor_family_importance_summary.csv"),
        Path("results/explanations/task_1_1/family_axis_importance_summary.csv"),
        Path("results/explanations/task_1_1/ablation_metrics.csv"),
        Path("results/explanations/task_1_1/stability_rank_correlations.csv"),
        Path("results/explanations/task_1_2/dilation_importance_summary.csv"),
        Path("results/explanations/task_1_2/temporal_scale_importance_summary.csv"),
        Path("results/explanations/task_1_2/class_specific_temporal_scale_importance.csv"),
        Path("results/explanations/task_1_2/stability_rank_correlations.csv"),
        Path("results/explanations/task_1_3/pattern_cases.csv"),
        Path("results/explanations/task_1_3/pattern_feature_distributions.csv"),
        Path("results/explanations/task_1_3/pattern_response_traces.parquet"),
        Path("data/processed/ksas_m2_raw_padded/tensors.npz"),
    )
    missing = [path for path in required if not path.exists()]
    if missing:
        missing_text = "\n".join(f"- {path}" for path in missing)
        raise ReportBuildError(f"Missing required composite figure inputs:\n{missing_text}")


def _write_task_1_1_core_evidence(pyplot: Any, destination: Path) -> Path:
    families = pd.read_csv("results/explanations/task_1_1/sensor_family_importance_summary.csv")
    family_axis = pd.read_csv("results/explanations/task_1_1/family_axis_importance_summary.csv")
    family_axis[["sensor_family", "axis"]] = family_axis["group_value"].str.rsplit(
        "_", n=1, expand=True
    )
    pivot = family_axis.pivot(index="sensor_family", columns="axis", values="importance_mean")
    pivot = pivot.reindex(index=_sensor_family_order(), columns=["x", "y", "z"])

    figure, axes = pyplot.subplots(1, 2, figsize=(10.8, 4.2), constrained_layout=True)
    ordered = families.sort_values("importance_mean", ascending=True)
    axes[0].barh(
        ordered["group_value"],
        ordered["importance_mean"],
        xerr=ordered["importance_std"],
        color="#4C78A8",
        capsize=3,
    )
    axes[0].set_title("Sensor families")
    axes[0].set_xlabel("Mean normalized importance")
    axes[0].grid(axis="x", alpha=0.25)

    image = axes[1].imshow(pivot.to_numpy(dtype=np.float64), cmap="viridis")
    axes[1].set_title("Sensor family by axis")
    axes[1].set_xticks(range(3), labels=["x", "y", "z"])
    axes[1].set_yticks(range(len(pivot.index)), labels=list(pivot.index))
    axes[1].set_xlabel("Device-frame axis")
    for row_index, row in enumerate(pivot.to_numpy(dtype=np.float64)):
        for col_index, value in enumerate(row):
            axes[1].text(col_index, row_index, f"{value:.3f}", ha="center", va="center", fontsize=8)
    figure.colorbar(image, ax=axes[1], shrink=0.78, label="Mean importance")
    _save_report_figure(figure, destination)
    return destination


def _write_task_1_1_validation_checks(pyplot: Any, destination: Path) -> Path:
    ablation = pd.read_csv("results/explanations/task_1_1/ablation_metrics.csv")
    ablation = ablation.loc[ablation["group_level"] == "sensor_family"]
    summary = (
        ablation.groupby("group_value", as_index=False)["macro_f1_drop"]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values("mean")
    )
    correlations = pd.read_csv("results/explanations/task_1_1/stability_rank_correlations.csv")
    correlations = correlations.loc[correlations["group_level"] == "channel"]
    matrix = _fold_correlation_matrix(correlations)

    figure, axes = pyplot.subplots(1, 2, figsize=(10.8, 4.2), constrained_layout=True)
    axes[0].barh(summary["group_value"], summary["mean"], xerr=summary["std"], color="#4C78A8")
    axes[0].axvline(0.0, color="#444444", lw=0.8)
    axes[0].set_title("Sensor-family ablation")
    axes[0].set_xlabel("Macro-F1 drop after removal")
    axes[0].grid(axis="x", alpha=0.25)

    image = axes[1].imshow(matrix, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    axes[1].set_title("Channel-rank stability")
    axes[1].set_xlabel("Fold")
    axes[1].set_ylabel("Fold")
    axes[1].set_xticks(range(matrix.shape[0]))
    axes[1].set_yticks(range(matrix.shape[0]))
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            axes[1].text(
                col_index,
                row_index,
                f"{matrix[row_index, col_index]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )
    figure.colorbar(image, ax=axes[1], shrink=0.78, label="Spearman correlation")
    _save_report_figure(figure, destination)
    return destination


def _write_task_1_2_scale_evidence(pyplot: Any, destination: Path) -> Path:
    dilation = pd.read_csv("results/explanations/task_1_2/dilation_importance_summary.csv")
    scale = pd.read_csv("results/explanations/task_1_2/temporal_scale_importance_summary.csv")
    dilation = dilation.sort_values("group_value", key=lambda series: series.astype(int))
    scale["order"] = scale["group_value"].map({"short": 0, "intermediate": 1, "long": 2})
    scale = scale.sort_values("order")

    figure, axes = pyplot.subplots(1, 2, figsize=(10.8, 4.0), constrained_layout=True)
    labels = [f"d={int(value)}" for value in dilation["group_value"]]
    axes[0].bar(
        labels,
        dilation["importance_mean"],
        yerr=dilation["importance_std"],
        color="#4C78A8",
        capsize=3,
    )
    axes[0].set_title("Dilation importance")
    axes[0].set_ylabel("Mean normalized importance")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(
        scale["group_value"],
        scale["importance_mean"],
        yerr=scale["importance_std"],
        color="#54A24B",
        capsize=3,
    )
    axes[1].set_title("Temporal-scale bins")
    axes[1].set_ylabel("Mean normalized importance")
    axes[1].grid(axis="y", alpha=0.25)
    _save_report_figure(figure, destination)
    return destination


def _write_task_1_2_class_and_stability(pyplot: Any, destination: Path) -> Path:
    class_scale = pd.read_csv(
        "results/explanations/task_1_2/class_specific_temporal_scale_importance.csv"
    )
    pivot = class_scale.pivot(
        index="class_label", columns="group_value", values="importance_mean"
    ).reindex(columns=["short", "intermediate", "long"])
    correlations = pd.read_csv("results/explanations/task_1_2/stability_rank_correlations.csv")
    correlations = correlations.loc[correlations["group_level"] == "dilation"]
    matrix = _fold_correlation_matrix(correlations)

    figure, axes = pyplot.subplots(1, 2, figsize=(10.8, 4.2), constrained_layout=True)
    image = axes[0].imshow(pivot.to_numpy(dtype=np.float64), cmap="magma")
    axes[0].set_title("Class-specific scale use")
    axes[0].set_xticks(range(3), labels=["short", "inter.", "long"])
    axes[0].set_yticks(range(len(pivot.index)), labels=[str(value) for value in pivot.index])
    axes[0].set_xlabel("Temporal-scale bin")
    axes[0].set_ylabel("Movement class")
    for row_index, row in enumerate(pivot.to_numpy(dtype=np.float64)):
        for col_index, value in enumerate(row):
            axes[0].text(col_index, row_index, f"{value:.2f}", ha="center", va="center", fontsize=8)
    figure.colorbar(image, ax=axes[0], shrink=0.78, label="One-vs-rest importance")

    stability = axes[1].imshow(matrix, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    axes[1].set_title("Dilation-rank stability")
    axes[1].set_xlabel("Fold")
    axes[1].set_ylabel("Fold")
    axes[1].set_xticks(range(matrix.shape[0]))
    axes[1].set_yticks(range(matrix.shape[0]))
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            axes[1].text(
                col_index,
                row_index,
                f"{matrix[row_index, col_index]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )
    figure.colorbar(stability, ax=axes[1], shrink=0.78, label="Spearman correlation")
    _save_report_figure(figure, destination)
    return destination


def _write_task_1_3_case_cards(
    pyplot: Any, destination: Path, case_ids: tuple[str, str], title: str
) -> Path:
    cases = pd.read_csv("results/explanations/task_1_3/pattern_cases.csv")
    traces = pd.read_parquet("results/explanations/task_1_3/pattern_response_traces.parquet")
    arrays = np.load("data/processed/ksas_m2_raw_padded/tensors.npz", allow_pickle=False)
    x_values = np.arange(arrays["X"].shape[2])

    figure = pyplot.figure(figsize=(7.4, 8.6), constrained_layout=True)
    grid = figure.add_gridspec(4, 2, height_ratios=[2.0, 0.55, 2.0, 0.55])
    figure.suptitle(title, fontsize=12)
    for case_number, case_id in enumerate(case_ids):
        plot_row = case_number * 2
        case_rows = cases.loc[cases["case_id"] == case_id]
        if case_rows.empty:
            raise ReportBuildError(f"Pattern case is missing: {case_id}")
        case = case_rows.iloc[0]
        sample_index = int(case["sample_index"])
        channel_names = _same_family_channels(str(case["channel_name"]))
        channel_indices = [EXPECTED_CHANNELS.index(channel) for channel in channel_names]
        signal_axis = figure.add_subplot(grid[plot_row, 0])
        response_axis = figure.add_subplot(grid[plot_row, 1])
        metrics_axis = figure.add_subplot(grid[plot_row + 1, :])

        for channel, channel_index in zip(channel_names, channel_indices, strict=True):
            signal_axis.plot(x_values, arrays["X"][sample_index, channel_index], label=channel)
        signal_axis.axvspan(
            int(case["processed_start_index"]),
            int(case["processed_end_index"]),
            color="#F58518",
            alpha=0.18,
        )
        signal_axis.axvline(
            int(arrays["original_length"][sample_index]) - 0.5,
            color="#666666",
            ls="--",
        )
        signal_axis.set_title(_case_title(case), fontsize=10)
        signal_axis.set_ylabel(f"Signal ({_unit_for_channel(str(case['channel_name']))})")
        signal_axis.grid(alpha=0.22)
        signal_axis.legend(loc="upper right", fontsize=6)

        trace = traces.loc[traces["case_id"] == case_id]
        response_axis.plot(trace["response_index"], trace["response_value"], color="#4C78A8")
        response_axis.axhline(float(case["threshold"]), color="#E45756", ls="--", lw=1.0)
        response_axis.axvspan(
            int(case["response_start_index"]),
            int(case["response_end_index"]),
            color="#F58518",
            alpha=0.18,
        )
        response_axis.set_ylabel("Kernel response")
        response_axis.set_xlabel("Processed sample index")
        response_axis.grid(alpha=0.22)

        metrics_axis.axis("off")
        metrics_axis.text(
            0.0,
            0.86,
            _case_metrics_text(case),
            va="top",
            ha="left",
            fontsize=8.5,
            linespacing=1.18,
        )

    _save_report_figure(figure, destination)
    return destination


def _write_task_1_3_distribution_summary(pyplot: Any, destination: Path) -> Path:
    distributions = pd.read_csv("results/explanations/task_1_3/pattern_feature_distributions.csv")
    cases = pd.read_csv("results/explanations/task_1_3/pattern_cases.csv")
    features = list(dict.fromkeys(int(value) for value in cases["feature_index"]))[:3]

    figure = pyplot.figure(figsize=(11.2, 6.2), constrained_layout=True)
    grid = figure.add_gridspec(2, 3, height_ratios=[2.1, 1.15])
    for index, feature_index in enumerate(features):
        axis = figure.add_subplot(grid[0, index])
        frame = distributions.loc[distributions["feature_index"] == feature_index]
        labels = sorted(frame["y_true"].unique())
        values = [
            frame.loc[frame["y_true"] == label, "feature_value"].to_numpy(dtype=np.float64)
            for label in labels
        ]
        axis.boxplot(values, tick_labels=[str(label) for label in labels], showfliers=False)
        axis.set_title(f"Feature {feature_index}")
        axis.set_xlabel("Movement class")
        if index == 0:
            axis.set_ylabel("Held-out PPV")
        axis.grid(axis="y", alpha=0.25)

    table_axis = figure.add_subplot(grid[1, :])
    table_axis.axis("off")
    table = cases[
        [
            "case_id",
            "movement_label",
            "channel_name",
            "dilation",
            "feature_value_ppv",
            "class_separation_auc",
            "human_meaningfulness",
        ]
    ].copy()
    table["case_id"] = table["case_id"].map(_short_case_id)
    table["feature_value_ppv"] = table["feature_value_ppv"].map(lambda value: f"{value:.3f}")
    table["class_separation_auc"] = table["class_separation_auc"].map(lambda value: f"{value:.3f}")
    rendered = table_axis.table(
        cellText=table.to_numpy().tolist(),
        colLabels=["case", "movement", "channel", "dil.", "PPV", "AUC", "meaning"],
        loc="center",
        cellLoc="center",
    )
    rendered.auto_set_font_size(False)
    rendered.set_fontsize(8)
    rendered.scale(1.0, 1.28)
    table_axis.set_title("Selected pattern case summary", pad=8)
    _save_report_figure(figure, destination)
    return destination


def _short_case_id(case_id: str) -> str:
    if case_id == "pattern_case_failure_or_ambiguous":
        return "ambiguous"
    return case_id.replace("pattern_case_", "case ")


def _sensor_family_order() -> list[str]:
    return ["accelerometer", "gravity", "gyros", "lin_accel", "game_rot_vec", "magn_field"]


def _fold_correlation_matrix(frame: pd.DataFrame) -> np.ndarray:
    folds = sorted(set(frame["left_fold"]).union(set(frame["right_fold"])))
    if not folds:
        raise ReportBuildError("Cannot build stability matrix from an empty correlation table.")
    index_by_fold = {int(fold): index for index, fold in enumerate(folds)}
    matrix = np.eye(len(folds), dtype=np.float64)
    for row in frame.to_dict("records"):
        left = index_by_fold[int(row["left_fold"])]
        right = index_by_fold[int(row["right_fold"])]
        value = float(row["spearman_rank_correlation"])
        matrix[left, right] = value
        matrix[right, left] = value
    return matrix


def _same_family_channels(channel_name: str) -> list[str]:
    sensor_family = channel_name.rsplit("_", 1)[0]
    return [channel for channel in EXPECTED_CHANNELS if channel.startswith(f"{sensor_family}_")]


def _unit_for_channel(channel_name: str) -> str:
    sensor_family = channel_name.rsplit("_", 1)[0]
    return SENSOR_UNITS.get(sensor_family, "sensor units")


def _case_title(case: pd.Series) -> str:
    label = _short_case_id(str(case["case_id"]))
    movement = str(case["movement_label"])
    arm = str(case["arm"])
    participant = str(case["participant_pseudonym"])
    return f"{label}: {movement} | {arm} | {participant}"


def _case_metrics_text(case: pd.Series) -> str:
    correct = "correct" if bool(case["prediction_correct"]) else "misclassified"
    return (
        f"Channel: {case['channel_name']} | dilation {int(case['dilation'])} | "
        f"span {int(case['effective_receptive_field_samples'])} samples\n"
        f"PPV {float(case['feature_value_ppv']):.3f} | "
        f"AUC {float(case['class_separation_auc']):.3f} | "
        f"meaning: {case['human_meaningfulness']}\n"
        f"True class {int(case['y_true'])}, predicted {int(case['y_pred'])} "
        f"({correct}), margin {float(case['prediction_margin']):.3f}\n"
        "Orange bands mark representative intervals, not unique causes."
    )


def _save_report_figure(figure: Any, destination: Path) -> None:
    figure.savefig(destination, bbox_inches="tight")
    pyplot = _load_pyplot()
    pyplot.close(figure)


def _load_pyplot() -> Any:
    matplotlib = cast(Any, __import__("matplotlib"))
    matplotlib.use("Agg")
    return cast(Any, __import__("matplotlib.pyplot", fromlist=["pyplot"]))


def _validate_report_source(report_source: Path, bibliography: Path, figures_dir: Path) -> None:
    if not report_source.exists():
        raise ReportBuildError(f"Report source is missing: {report_source}")
    if not bibliography.exists():
        raise ReportBuildError(f"Bibliography is missing: {bibliography}")

    text = report_source.read_text(encoding="utf-8")
    required_text = (
        REPOSITORY_URL,
        "Answer to Task 1.1",
        "Answer to Task 1.2",
        "Answer to Task 1.3",
        "Generative-AI disclosure",
        "References",
    )
    missing_text = [item for item in required_text if item not in text]
    if missing_text:
        missing = "\n".join(f"- {item}" for item in missing_text)
        raise ReportBuildError(f"Report source is missing required content:\n{missing}")

    expected_figure_names = [artifact.destination_name for artifact in FIGURE_ARTIFACTS]
    missing_figures = [name for name in expected_figure_names if not (figures_dir / name).exists()]
    if missing_figures:
        missing = "\n".join(f"- {figures_dir / name}" for name in missing_figures)
        raise ReportBuildError(f"Report figure workspace is incomplete:\n{missing}")
