# Explainable Human Motion Computing with KSAS and XROCKET

This repository contains the implementation for **Part I — Human Motion Computing with Inertial Signals**.

The project applies **XROCKET** to the **KSAS smartphone IMU dataset** to identify:

1. which sensor axes contribute most to movement classification;
2. which temporal scales are most informative; and
3. which signal patterns are most discriminative and meaningful from a biomechanical perspective.

The final deliverable is a reproducible technical report supported by code, experiment outputs, figures, and documented model explanations.

> **Deadline:** 20 July 2026  
> **Detailed requirements:** See [`docs/PRD.md`](docs/PRD.md)

---

## Future direction

This repository remains the completed academic XROCKET research artifact; it
does not contain a mobile application, smartwatch integration, data-collection
backend, computer-vision model, or coaching system.

A separate successor project is proposed to collect timestamped phone and Wear
OS inertial data for controlled Kenpo studies, then investigate learning
feedback, multimodal sensing, and other sports in later research stages. See
the [successor project proposal](docs/future-project-proposal.md) for the
planned scope, architecture, research gates, and responsible-use boundaries.

---

## Research questions

- Which accelerometer, gyroscope, and magnetometer axes are most informative?
- Does the model rely on short-duration local patterns, long-duration movement structure, or both?
- Are the explanations stable across participants and validation folds?
- Can the most important patterns be related to plausible movement characteristics?
- Could the results be affected by participant identity, arm, phone orientation, sequence length, or preprocessing?

---

## Dataset

The project uses the **KSAS Dataset**, a course-provided local dataset containing
smartphone inertial data recorded during American Kenpo Karate Blocking Set I.

Raw data are not included by default. Data access, channels, labels, sampling
frequency, and participant metadata must be verified during the audit stage.

Expected layout:

```text
data/
├── raw/
├── interim/
├── processed/
└── manifests/
```

---

## Method

The planned workflow is:

```text
KSAS recordings
→ data audit and preprocessing
→ participant-grouped evaluation
→ XROCKET feature extraction
→ classifier training
→ axis and dilation analysis
→ discriminative pattern interpretation
→ technical report
```

The final evaluation must keep all samples from the same participant in the same split to avoid data leakage.

Main evaluation metrics include:

- macro F1;
- balanced accuracy;
- per-class precision, recall, and F1;
- confusion matrices; and
- fold-level variation.

---

## Technology stack

- Python 3.11
- `uv`
- NumPy, pandas, SciPy, PyArrow
- scikit-learn
- PyTorch
- XROCKET, pinned to the audited `dida-do/xrocket` Git revision
- Matplotlib
- JupyterLab
- Pandoc and MiKTeX/pdflatex for the final PDF report
- pytest, Ruff, mypy, pre-commit
- GitHub Actions

The core project is designed to run on CPU hardware.

---

## Repository structure

```text
.
├── AGENT.md
├── README.md
├── configs/
├── data/
├── docs/
├── notebooks/
├── src/ksas_xrocket/
├── scripts/
├── tests/
├── results/
└── reports/
```

Reusable code belongs in `src/`. Notebooks are used for exploration and result inspection.

---

## Installation

```bash
git clone https://github.com/Forest904/ksas-xrocket-hmc.git
cd ksas-xrocket-hmc
uv sync
```

---

## Usage

The intended workflow is:

```bash
make data
make audit
make baseline
make train
make explain
make figures
make report
```

Example experiment command:

```bash
uv run hmc train --config configs/experiments/m3_xrocket_raw_padded.yaml
```

This command fits XROCKET thresholds on each training fold only, trains the
random-forest primary model and logistic-regression sensitivity model, and
writes traceable fold artifacts under `results/xrocket/m3_raw_padded/`. Reruns
must opt in to replacement with `--overwrite`.

---

## Planned analyses

### Task 1.1 — Sensor-axis contribution

- sensor-family importance;
- axis and channel importance;
- class-specific profiles;
- channel or sensor-family ablation;
- explanation stability;
- biomechanical interpretation.

### Task 1.2 — Temporal scale

- dilation importance;
- effective receptive field;
- conversion to temporal duration;
- local, intermediate, and global scale contribution;
- class-specific temporal-scale profiles.

### Task 1.3 — Discriminative patterns

- important kernel selection;
- mapping features back to signal regions;
- representative correct and incorrect examples;
- human-meaningfulness assessment;
- implications for learning and performance assessment.

---

## Reproducibility

Each experiment should store:

- resolved configuration;
- data and split manifests;
- random seeds;
- environment information;
- Git commit;
- metrics and predictions;
- feature metadata;
- explanation tables;
- figures and logs.

The intended reproduction command is:

```bash
make reproduce
```

`make reproduce` runs the locked quality and smoke gate: Ruff, formatting
check, mypy, pytest, import checks, CLI version, XROCKET import, and a prepare
smoke run. The full artifact rebuild sequence is documented in
[`docs/reproducibility.md`](docs/reproducibility.md). Existing result
directories are protected by overwrite checks, so full reruns require explicit
`--overwrite` flags when replacing committed artifacts.

The final report is built with Pandoc and MiKTeX/pdflatex:

```bash
make figures
make report
```

The final PDF is written to `reports/ksas_xrocket_hmc_report.pdf`.

## Repository and licensing status

Final repository URL:
<https://github.com/Forest904/ksas-xrocket-hmc.git>

Raw KSAS participant CSVs are not committed. The selected XROCKET dependency is
pinned to `dida-do/xrocket` commit
`1511e810c59d0c42f6431ef2f1f9fa57c71e9b2f` for course-authorized academic use;
the public upstream repository had no public license file during this project.
No new repository license is added for this submission.

---

## Testing

```bash
uv run ruff check .
uv run mypy src
uv run pytest
```

Tests focus on preprocessing, participant-safe splitting, XROCKET metadata, axis aggregation, dilation mapping, and pattern localization.

---

## Responsible use

This is an academic motion-analysis project. It is **not** a validated coaching, clinical, diagnostic, or injury-prevention system.

Model importance indicates association, not biomechanical causation. Smartphone axes represent the device coordinate frame and may depend on phone placement, arm, and orientation.

---

## References

- [XROCKET overview](https://dida.do/blog/explainable-time-series-classification-with-x-rocket)

---
