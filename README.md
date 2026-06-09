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
- XROCKET
- Matplotlib
- JupyterLab
- Quarto and LaTeX
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
git clone <repository-url>
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
uv run hmc train   --config configs/experiments/movement_xrocket_rf.yaml
```

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
