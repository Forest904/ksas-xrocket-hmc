# Agent Notes

## Project

Build a reproducible Human Motion Computing Part I project using **XROCKET** on the **KSAS smartphone IMU dataset**. Final deliverable: a technical PDF report due **20 July 2026**.

## Read First

- `docs/Roadmap.md`: execution order and milestones.
- `README.md`: intended repo layout and commands.
- `docs/PRD.md`: full requirements and quality gates.
- `docs/Human_Motion_Computing_Lecture_Summary_and_Part_I_Assignment.md`: assignment wording.

## Core Scope

Answer the three required tasks:

1. Sensor axes: which IMU signals contribute most?
2. Dilations/frequencies: short, long, or mixed temporal scales?
3. Discriminative patterns: what motion patterns are meaningful to humans?

Computer-vision alternatives, apps, dashboards, and real-time feedback are out of scope for the core submission.

## Data

- Local upstream copy: `KSAS-Dataset/` at repo root, currently ignored.
- Later move or copy it to `data/raw/KSAS-Dataset/` after the skeleton exists.
- Keep raw data immutable and untracked unless a later explicit license decision allows redistribution.
- KSAS has 240 movement CSVs, 20 participants, 2 arms, 6 labels, and 18 Android sensor channels.

## Engineering Rules

- Prefer package code in `src/ksas_xrocket/`; notebooks are for exploration only.
- Use participant-grouped splits; final results must have zero participant leakage.
- Fit learned preprocessing only on training folds.
- Preserve XROCKET metadata traceability from feature to channel, dilation, temporal span, and pattern interval.
- Generate report figures/tables from code and saved artifacts.
- Record configs, seeds, manifests, environment, Git commit, metrics, predictions, and explanation tables.

## Quality Priorities

1. Mandatory assignment answers.
2. Leakage-resistant evaluation.
3. Explainability traceability.
4. Reproducible report artifacts.
5. Robustness checks and limitations.
6. Stretch experiments only after the mandatory report evidence is safe.

## Commands

Planned workflow from `README.md`:

```bash
make data
make audit
make baseline
make train
make explain
make figures
make report
make reproduce
```
