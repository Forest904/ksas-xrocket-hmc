# Reproducibility

## Environment

The canonical Python environment is defined by `pyproject.toml` and `uv.lock`.

```bash
uv sync
```

The project targets Python 3.11. The final report is rendered with Pandoc and
MiKTeX/pdflatex. These are expected system tools and are not installed as Python
dependencies.

XROCKET is installed directly from `dida-do/xrocket` at immutable Git commit
`1511e810c59d0c42f6431ef2f1f9fa57c71e9b2f`. `pyproject.toml` supplies corrected
dependency metadata because upstream version 0.1 declares the obsolete package
name `pytorch`; the imported runtime package is `torch`.

Verify the locked encoder:

```bash
uv sync --locked
uv run python -c "from xrocket.encoder import XRocket; print(XRocket)"
```

Verify the report toolchain:

```bash
pandoc --version
pdflatex --version
pdftoppm -h
```

## Data

Raw KSAS data must remain immutable and untracked. Local raw files belong under `data/raw/KSAS-Dataset/` after the M1 data-placement step.

## Workflow

The final smoke reproduction gate is:

```bash
make reproduce
```

This runs linting, formatting check, type checking, tests, import checks,
XROCKET importability, CLI version, and a prepare smoke run. It is the closest
documented clean-run subset for final submission when the committed artifacts
under `results/` are already present.

The full artifact rebuild sequence is:

```bash
uv run hmc audit
uv run hmc prepare --config configs/preprocessing_m2_raw_padded.yaml
uv run hmc baseline --config configs/baseline_m2_raw_padded.yaml
uv run hmc train --config configs/experiments/m3_xrocket_raw_padded.yaml --overwrite
uv run hmc explain --config configs/explanations/task_1_1_xrocket_raw_padded.yaml --overwrite
uv run hmc explain --config configs/explanations/task_1_2_xrocket_raw_padded.yaml --overwrite
uv run hmc explain --config configs/explanations/task_1_3_xrocket_raw_padded.yaml --overwrite
uv run hmc robustness --config configs/robustness/m7_raw_padded.yaml --overwrite
make figures
make report
```

The M2 preprocessing, baseline, and M3 model workflow is:

```bash
uv run hmc prepare --config configs/preprocessing_m2_raw_padded.yaml
uv run hmc baseline --config configs/baseline_m2_raw_padded.yaml
uv run hmc train --config configs/experiments/m3_xrocket_raw_padded.yaml
uv run hmc explain --config configs/explanations/task_1_1_xrocket_raw_padded.yaml
uv run hmc explain --config configs/explanations/task_1_2_xrocket_raw_padded.yaml
uv run hmc explain --config configs/explanations/task_1_3_xrocket_raw_padded.yaml
```

`hmc prepare` writes padded tensors and participant-grouped split manifests.
`hmc baseline` trains the statistical baselines on those exact folds. `hmc
train` fits a separate XROCKET encoder on each training fold and saves features,
metadata, models, metrics, predictions, confusion matrices, runtime, provenance,
and padding diagnostics. It refuses to replace a non-empty output directory
unless `--overwrite` is passed explicitly.

`hmc explain` implements Tasks 1.1 through 1.3 for the M3 raw-padded XROCKET
run. Task 1.1 reads the saved M3 fold artifacts, computes normalized native
importance, class-specific one-vs-rest profiles, feature-group ablations,
grouped permutation importance, method-agreement tables, figures, and a
report-ready answer under `results/explanations/task_1_1/`. Task 1.2 reads the
same fold artifacts, validates dilation and receptive-field metadata,
aggregates normalized native importance by dilation and temporal-scale bin,
computes class-specific temporal profiles, writes figures, and drafts the
temporal-scale answer under `results/explanations/task_1_2/`. Task 1.3 reads
the same fold artifacts, selects stable discriminative PPV features,
reconstructs representative kernel-response intervals, writes pattern case
figures, and drafts the pattern-interpretation answer under
`results/explanations/task_1_3/`. Explanation commands refuse to replace a
non-empty output directory unless `--overwrite` is passed explicitly.

Each M3 fold directory contains the fitted adapter and classifiers. The runner
reloads them before completing and verifies that features and smoke-sample
predictions are unchanged.

## Final Report

The final report artifacts are built by:

```bash
make figures
make report
```

`hmc figures` validates and copies curated figures and compact source extracts
from `results/` into `reports/figures/` and `reports/tables/`. `hmc report`
then validates required report content and renders
`reports/ksas_xrocket_hmc_report.pdf` with Pandoc and pdflatex.

Render the PDF to page images for visual inspection:

```bash
New-Item -ItemType Directory -Force tmp/pdfs
pdftoppm -png reports/ksas_xrocket_hmc_report.pdf tmp/pdfs/ksas_report
```

Before submission, verify:

```bash
git ls-files data/raw
git status --short --ignored
```

Only `data/raw/.gitkeep` should be tracked from `data/raw/`; local KSAS raw
data should appear only as ignored files.

Repository URL: <https://github.com/Forest904/ksas-xrocket-hmc.git>

Submission identifier: tag `v1.0-submission` on branch `main`.

License/status statement: no new repository license is added for this
submission. Raw KSAS participant data are excluded. XROCKET is pinned to the
course-authorized upstream commit documented above, and the public upstream
repository had no public license file during this project.

## M4 Task 1.1 Outputs

The primary Task 1.1 artifacts are:

- `results/explanations/task_1_1/fold_native_importance.parquet`;
- `results/explanations/task_1_1/*_importance_summary.csv`;
- `results/explanations/task_1_1/class_specific_*_importance.csv`;
- `results/explanations/task_1_1/ablation_metrics.csv`;
- `results/explanations/task_1_1/permutation_importance.csv`;
- `results/explanations/task_1_1/method_agreement.csv`;
- `results/explanations/task_1_1/figures/*.{png,pdf}`;
- `results/explanations/task_1_1/task_1_1_answer.md`;
- `results/explanations/task_1_1/resolved_config.json`;
- `results/explanations/task_1_1/provenance.json`.

## M5 Task 1.2 Outputs

The primary Task 1.2 artifacts are:

- `results/explanations/task_1_2/temporal_span_mapping.csv`;
- `results/explanations/task_1_2/fold_temporal_feature_importance.parquet`;
- `results/explanations/task_1_2/important_temporal_features.parquet`;
- `results/explanations/task_1_2/dilation_*_importance.csv`;
- `results/explanations/task_1_2/temporal_scale_*_importance.csv`;
- `results/explanations/task_1_2/class_specific_*_importance.csv`;
- `results/explanations/task_1_2/stability_*.csv`;
- `results/explanations/task_1_2/padding_temporal_diagnostics.csv`;
- `results/explanations/task_1_2/figures/*.{png,pdf}`;
- `results/explanations/task_1_2/task_1_2_answer.md`;
- `results/explanations/task_1_2/resolved_config.json`;
- `results/explanations/task_1_2/provenance.json`.

## M6 Task 1.3 Outputs

The primary Task 1.3 artifacts are:

- `results/explanations/task_1_3/selected_patterns.csv`;
- `results/explanations/task_1_3/pattern_cases.csv`;
- `results/explanations/task_1_3/pattern_response_traces.parquet`;
- `results/explanations/task_1_3/pattern_feature_distributions.csv`;
- `results/explanations/task_1_3/figures/*.{png,pdf}`;
- `results/explanations/task_1_3/task_1_3_answer.md`;
- `results/explanations/task_1_3/resolved_config.json`;
- `results/explanations/task_1_3/provenance.json`.
