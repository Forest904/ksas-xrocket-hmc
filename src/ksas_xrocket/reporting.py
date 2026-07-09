"""Report artifact preparation and PDF rendering for M8."""

from __future__ import annotations

import csv
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast


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

    manifest_lines = [
        "# Report Artifact Manifest",
        "",
        "Generated by `hmc figures` from tracked result artifacts.",
        "",
        "## Figures",
        "",
    ]
    manifest_lines.extend(_manifest_rows(copied_figures, figures_dir))
    manifest_lines.extend(["", "## Tables And Source Extracts", ""])
    manifest_lines.extend(_manifest_rows(copied_tables, tables_dir))
    manifest_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

    return ReportArtifactsResult(
        figures_dir=figures_dir,
        tables_dir=tables_dir,
        manifest_path=manifest_path,
        figure_count=len(copied_figures),
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


def _short_case_id(case_id: str) -> str:
    if case_id == "pattern_case_failure_or_ambiguous":
        return "ambiguous"
    return case_id.replace("pattern_case_", "case ")


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
