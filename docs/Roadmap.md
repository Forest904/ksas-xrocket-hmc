# Roadmap

## KSAS XROCKET Human Motion Computing Project

This roadmap turns `docs/PRD.md`, `README.md`, and the Part I assignment brief into an ordered execution plan from the current repository state to the final technical PDF submission due on **20 July 2026**.

The roadmap is implementation-facing. It defines what to build first, what evidence must exist before moving on, and which artifacts prove that each stage is complete.

## Current Starting State

- The root contains the M0 repository skeleton: `pyproject.toml`, `uv.lock`, `Makefile`, `.github/workflows/ci.yml`, `src/ksas_xrocket/`, `tests/`, `configs/`, `data/`, `notebooks/`, `reports/`, `results/`, and `scripts/`.
- Long-form documentation lives in `docs/`, including `PRD.md`, `Roadmap.md`, `Human_Motion_Computing_Lecture_Summary_and_Part_I_Assignment.md`, M0 placeholder documentation files, and `docs/ai-use.md`.
- `KSAS-Dataset/` is present at the root only as a temporary local copy. It should remain untracked and later be moved or copied into `data/raw/KSAS-Dataset/` after the repository skeleton exists.
- The KSAS dataset contains 240 CSV instances under `KSAS-Dataset/movements`.
- KSAS filenames follow `a-b-c.csv`, where `a` is the movement label, `b` is the participant ID, and `c` is the arm indicator.
- Each CSV contains 18 Android sensor channels: accelerometer, gravity, gyroscope, linear acceleration, game rotation vector, and magnetic field, each with x, y, and z axes.

## Global Rules

- Keep raw data immutable. Do not edit upstream CSVs in place.
- Keep raw participant data untracked.
- Prevent participant leakage in every final evaluation split.
- Fit preprocessing that learns from data only on training folds, then apply it to validation or test folds.
- Preserve traceability from XROCKET features to kernel metadata, channel or channel combination, dilation, temporal span, and representative signal intervals.
- Generate core report figures and tables from committed code and saved experiment artifacts.
- Record run provenance: configuration, random seed, data manifest, split manifest, environment, Git commit, metrics, predictions, and explanation outputs.
- Prioritize mandatory assignment tasks before recommended and stretch work.
- Freeze stretch work after **11 July 2026** unless every mandatory analysis and report section already has usable evidence.
- Treat negative, unstable, or moderate-performance results as valid if the evaluation is leakage-resistant and the limitations are explicit.

## Milestone Overview

| Milestone | Focus | Target Outcome |
|---|---|---|
| M0 | Repository foundation and skeleton | Complete: a runnable project structure exists |
| M1 | KSAS data placement, provenance, audit, data dictionary | Dataset facts are verified before modeling |
| M2 | Preprocessing, manifests, grouped splits, baseline models | Reliable tensors and leakage-safe baselines exist |
| M3 | XROCKET integration and metadata traceability | Primary explainable model runs end to end |
| M4 | Task 1.1 sensor-axis contribution analysis | Sensor and axis evidence answers "where" |
| M5 | Task 1.2 dilation and temporal-scale analysis | Dilation evidence answers "at what scale" |
| M6 | Task 1.3 discriminative-pattern interpretation | Representative patterns are localized and interpreted |
| M7 | Robustness, confounds, and negative controls | Main findings are stress-tested |
| M8 | Report generation, reproduction check, final submission | PDF and repository are submission-ready |

## M0: Repository Foundation and Skeleton

**Status:** Complete as of 2026-06-09.

**Objective:** Create the project skeleton promised by the README without starting model work too early.

**Ordered checklist:**

1. Create the planned directories: `configs`, `data/raw`, `data/interim`, `data/processed`, `data/manifests`, `docs`, `notebooks`, `reports`, `results`, `scripts`, `src/ksas_xrocket`, and `tests`.
2. Add Python project configuration for Python 3.11 and `uv`.
3. Add initial dependencies for data loading, scientific computing, modeling, plotting, testing, formatting, typing, and reporting.
4. Add a minimal package entrypoint under `src/ksas_xrocket`.
5. Add a Makefile or equivalent task runner matching the README commands: `data`, `audit`, `baseline`, `train`, `explain`, `figures`, `report`, and `reproduce`.
6. Add CI smoke checks for formatting, tests, and importability.
7. Add placeholder documentation files required by the PRD: `docs/data-dictionary.md`, `docs/methodology.md`, `docs/biomechanics.md`, `docs/reproducibility.md`, `docs/limitations.md`, and `docs/decision-log.md`.
8. Add an AI-use log location and initial disclosure template.

**Expected outputs:**

- Repository skeleton matching `README.md`.
- Importable Python package.
- Basic test and lint commands.
- Empty but structured documentation placeholders.

**Exit criteria:**

- Complete: `uv sync --locked` succeeds using Python 3.11.
- Complete: `uv run pytest` runs the M0 smoke tests.
- Complete: `uv run ruff check .`, `uv run ruff format --check .`, and `uv run mypy src` succeed.
- Complete: README `make` commands exist and map to documented placeholders or real quality checks.

**Blocking decisions:**

- Resolved: `pyproject.toml` plus committed `uv.lock` is the canonical environment lock.
- Resolved: CI runs data-free lint, format, type, test, importability, version, and placeholder CLI smoke checks.

**Completed outputs:**

- Added `pyproject.toml`, `uv.lock`, `Makefile`, `.pre-commit-config.yaml`, and `.github/workflows/ci.yml`.
- Added an importable package under `src/ksas_xrocket` with the `hmc` console entrypoint.
- Added successful placeholder CLI commands for `prepare`, `audit`, `baseline`, `train`, `explain`, `figures`, `report`, and `reproduce`.
- Added smoke tests in `tests/` for importability, version metadata, and placeholder CLI behavior.
- Added required documentation placeholders plus `docs/ai-use.md`.
- Added tracked skeleton directories while keeping `KSAS-Dataset/` and raw data ignored.
- Verified `make reproduce`, which runs linting, formatting check, typing, tests, importability, CLI version, and placeholder CLI smoke.

## M1: KSAS Data Placement, Provenance, Audit, and Data Dictionary

**Status:** Complete as of 2026-06-09.

**Objective:** Move from "dataset available locally" to a verified, documented, reproducible data source without committing raw participant data.

**Ordered checklist:**

1. Move or copy the ignored root `KSAS-Dataset/` folder into `data/raw/KSAS-Dataset/` after the skeleton exists.
2. Ensure `data/raw/` and the dataset folder remain ignored by Git.
3. Record local dataset provenance: course-provided source, local acquisition date, local README checksum, and audit date.
4. Create a data audit script that scans all CSVs and extracts filenames, movement labels, participant IDs, arm indicators, row counts, columns, missing values, duplicate rows, and numeric ranges.
5. Generate `data/manifests/samples.csv` from the audit rather than by hand.
6. Validate that the expected 240 files are present.
7. Validate movement label mapping: 0 no movement, 1 upward block, 2 hammering inward block, 3 extended outward block, 4 outward downward block, 5 rear elbow block.
8. Validate that each participant and arm combination has the expected class coverage.
9. Write `docs/data-dictionary.md` with file naming, labels, channels, units, arms, participant identifiers, sequence boundaries, exclusions, and known unknowns.
10. Record unresolved facts, such as sampling frequency or orientation details, in `docs/limitations.md` and `docs/decision-log.md`.

**Expected outputs:**

- Raw KSAS data located under `data/raw/KSAS-Dataset/` locally and ignored.
- `data/manifests/samples.csv`.
- Data audit summary under `results/audit/` or `data/manifests/`.
- Completed first version of `docs/data-dictionary.md`.

**Exit criteria:**

- Audit confirms file count, channel schema, labels, participant IDs, and arm indicators.
- No raw dataset CSVs are staged for commit.
- Any missing or corrupted data are documented before preprocessing begins.

**Blocking decisions:**

- Resolved: copy the local dataset folder into `data/raw/KSAS-Dataset/`; delete the root copy manually only after confirming the copied raw path is sufficient.
- Resolved: sampling frequency remains unknown, so M1 reports temporal spans in samples and defers seconds-based interpretation.

**Completed outputs:**

- Copied the local ignored raw dataset into `data/raw/KSAS-Dataset/`.
- Confirmed raw dataset paths remain ignored by Git.
- Added `hmc audit` to generate `data/manifests/samples.csv`, `data/manifests/ksas_provenance.json`, `results/audit/ksas_audit_summary.json`, and `results/audit/ksas_numeric_ranges.csv`.
- Verified 240 CSV files, expected labels, participants, arms, class coverage, 18-channel schema, numeric values, no missing values, no duplicate rows, and no exact duplicate files.
- Completed `docs/data-dictionary.md` and recorded unresolved sampling-frequency and orientation facts in `docs/limitations.md`, `docs/decision-log.md`, and `docs/open-questions.md`.

## M2: Preprocessing, Manifests, Grouped Splits, and Baseline Models

**Objective:** Produce reproducible model-ready tensors and leakage-safe baseline results before integrating XROCKET.

**Ordered checklist:**

1. Implement CSV loading using the manifest as the source of truth.
2. Define the tensor contract: sample order, channel order, sequence axis, labels, participant groups, arm metadata, and storage format.
3. Implement minimal preprocessing first: numeric parsing, ordering preservation, missing-value handling, and sequence-length homogenization if required.
4. Keep smoothing, normalization, derived channels, and aggressive filtering out of the primary pipeline unless justified by a later sensitivity experiment.
5. Implement participant-grouped splits and automated leakage assertions.
6. Add split diagnostics for class coverage per fold.
7. Train simple baselines on identical grouped folds, such as statistical features with logistic regression or random forest.
8. Save baseline metrics, predictions, confusion matrices, and provenance.
9. Add tests for manifest parsing, label mapping, channel ordering, tensor shape, and participant-safe splitting.

**Expected outputs:**

- `data/processed/` tensors or reproducible tensor build outputs.
- `data/manifests/splits/*.csv` or equivalent split artifacts.
- Baseline experiment outputs under `results/baselines/`.
- Tests for data and split invariants.

**Exit criteria:**

- No participant appears in both train and test for any final split.
- Every preprocessing decision is documented in `docs/methodology.md`.
- Baseline metrics are generated on grouped folds.
- A label-shape and class-coverage report exists before XROCKET training.

**Blocking decisions:**

- Choose the sequence-length strategy required for the first model: pad, truncate, resample, or use an implementation that accepts unequal lengths.
- Decide whether no-movement is included in the primary classification target or treated as a separate sensitivity analysis, based on class balance and assignment fit.

## M3: XROCKET Integration and Metadata Traceability

**Objective:** Fit an XROCKET-based model that exposes enough metadata to answer the assignment's explainability questions.

**Ordered checklist:**

1. Select and license-check the XROCKET implementation or adapted reference code.
2. Build an adapter around the implementation rather than scattering implementation-specific calls through the project.
3. Confirm that the adapter exposes kernel ID, channel or channel combination, dilation, padding, kernel length, feature type, and feature index.
4. Fit a small prototype on one grouped fold.
5. Save transformed features, feature metadata, model artifacts, metrics, and predictions.
6. Train the primary classifier, expected to be random forest unless the prototype shows a clear incompatibility.
7. Add a classifier sensitivity run with ridge or logistic regression if time permits.
8. Add tests that verify metadata length matches transformed feature count and that saved metadata reloads correctly.

**Expected outputs:**

- XROCKET adapter in `src/ksas_xrocket`.
- Primary experiment configuration under `configs/experiments/`.
- Primary model outputs under `results/xrocket/`.
- Feature metadata table suitable for channel and dilation aggregation.

**Exit criteria:**

- One complete grouped fold runs end to end.
- Feature metadata can map each model feature back to channel information and dilation.
- Saved model and metadata reload without changing predictions on a small fixture or smoke sample.
- Primary performance and confusion matrix are available.

**Blocking decisions:**

- If XROCKET metadata are inaccessible, modify or wrap the implementation before continuing; do not use an opaque transform for final explanations.
- If runtime is too high, reduce seeds or kernel count before dropping mandatory analyses.

## M4: Task 1.1 Sensor-Axis Contribution Analysis

**Objective:** Answer Task 1.1: which sensor signals contribute most to classify user movement and how findings relate to movement biomechanics.

**Ordered checklist:**

1. Aggregate model-native feature importance by sensor family, channel, axis, and channel combination.
2. Normalize importance within fold before comparing across folds.
3. Produce global importance rankings with uncertainty across folds.
4. Produce class-specific sensor or channel profiles if the classifier and explanation method support them.
5. Run channel or sensor-family ablations on the same grouped folds.
6. Add permutation importance if feasible without threatening the report schedule.
7. Compare native importance, ablation, and permutation results for agreement or conflict.
8. Produce Task 1.1 figures: sensor-family contribution, axis/channel contribution, fold stability, and ablation impact.
9. Draft the biomechanical interpretation with explicit device-frame and arm-orientation caveats.

**Expected outputs:**

- `results/explanations/task_1_1/` tables and figures.
- Report-ready answer to Task 1.1.
- Updated `docs/biomechanics.md` and `docs/limitations.md`.

**Exit criteria:**

- Sensor-axis claims are supported by at least model-native importance and one validation method where feasible.
- Figures include titles, axes, units or normalized scale labels, captions, and fold variation.
- The written interpretation separates observed evidence from biomechanical speculation.

**Blocking decisions:**

- Decide how to allocate importance for multi-channel kernels: combination view, marginal view, or both.
- Decide whether arm-specific analysis is mandatory based on whether left/right rankings appear meaningfully different.

## M5: Task 1.2 Dilations/Frequencies and Temporal-Scale Analysis

**Objective:** Answer Task 1.2: whether classification relies on short-duration, long-duration, or mixed temporal patterns.

**Ordered checklist:**

1. Extract dilation, kernel length, and receptive-field metadata for every important XROCKET feature.
2. Convert dilation to effective temporal span in samples.
3. Convert spans to seconds only if sampling frequency is verified; otherwise report sample spans and mark real-time interpretation as limited.
4. Define short, intermediate, and long temporal-scale bins using a documented rule.
5. Aggregate importance by dilation and temporal-scale bin across folds.
6. Compare temporal-scale profiles globally and per movement class where possible.
7. Produce Task 1.2 figures: dilation importance, temporal-scale contribution, class-specific scale profiles, and fold stability.
8. Draft the report answer explaining how temporal scale influences movement recognition.

**Expected outputs:**

- `results/explanations/task_1_2/` dilation tables and figures.
- Effective temporal-span mapping.
- Report-ready answer to Task 1.2.

**Exit criteria:**

- Every temporal claim is traceable to XROCKET metadata.
- Time-in-seconds claims are used only if sampling frequency is defensible.
- The answer explicitly classifies the evidence as short-scale, long-scale, or mixed.

**Blocking decisions:**

- Decide the binning rule for temporal scales before inspecting final rankings too heavily.
- Decide how to phrase temporal-frequency language if the dataset lacks a verified sampling rate.

## M6: Task 1.3 Discriminative-Pattern Interpretation

**Objective:** Answer Task 1.3 by mapping important transformed features back to representative signal intervals and interpreting their human meaning.

**Ordered checklist:**

1. Select important features using a documented rule, such as top features by stable cross-fold importance.
2. For each selected feature, identify representative samples where activation or transformed value is high and the model prediction is correct.
3. Include at least one incorrect or ambiguous example if it clarifies limitations.
4. Localize the corresponding signal interval for each selected pattern.
5. Plot the relevant channels around each interval with movement label, arm, participant pseudonym, dilation, and feature metadata.
6. Interpret what aspect of movement may be captured, staying within sensor-coordinate evidence.
7. Classify each pattern's human meaningfulness as clear, plausible, ambiguous, or not meaningful.
8. Discuss how explanations could support future learning or performance assessment without claiming a validated coaching system.

**Expected outputs:**

- `results/explanations/task_1_3/` pattern tables and figures.
- Pattern case studies suitable for the PDF report.
- Report-ready answer to Task 1.3.

**Exit criteria:**

- At least three discriminative pattern examples are documented.
- Each pattern links feature metadata to raw or processed signal intervals.
- Human interpretation is cautious and explicitly tied to evidence.

**Blocking decisions:**

- Decide whether pattern localization is based on maximum convolution response, PPV-related intervals, or another implementation-specific method.
- If localization is technically unreliable, report representative high-activation sequences and document the limitation rather than overclaiming interval precision.

## M7: Robustness, Confounds, and Negative Controls

**Objective:** Stress-test the main findings enough to make the final report scientifically defensible.

**Ordered checklist:**

1. Run a label-shuffle negative control on the same grouped evaluation setup.
2. Run a length-only or metadata-only baseline if sequence length or participant metadata appear confounded with labels.
3. Summarize fold-to-fold and seed-to-seed variation for metrics and explanations.
4. Compare axis rankings across arms if class coverage supports it.
5. Review confusion matrices and per-class errors for movement-specific weaknesses.
6. Document implementation problems, failed attempts, unresolved dataset issues, and interpretation limits as they occur.
7. Decide which recommended analyses belong in the final report and which move to future work.

**Expected outputs:**

- Negative-control outputs under `results/controls/`.
- Stability summaries under `results/stability/`.
- Updated `docs/limitations.md`, `docs/methodology.md`, and `docs/decision-log.md`.

**Exit criteria:**

- No evidence of participant leakage or obvious label leakage remains unresolved.
- Negative-control behavior is documented.
- Main findings are qualified if rankings or metrics are unstable.
- Stretch analyses are stopped unless all mandatory report sections are already complete.

**Blocking decisions:**

- If controls suggest leakage or a severe confound, pause report polishing and fix the evaluation design.
- If results are unstable, decide whether to report instability as a core finding or reduce claim strength in the final conclusions.

## M8: Report Generation, Reproduction Check, and Final Submission

**Objective:** Produce the final PDF and repository state required by the assignment.

**Ordered checklist:**

1. Build the report skeleton early with sections for dataset, methods, problems encountered, repository link, Tasks 1.1-1.3, visualizations, limitations, and AI-use disclosure.
2. Insert final generated figures and tables from saved artifacts.
3. Ensure each assignment task has a direct answer before extended discussion.
4. Add the repository link or final repository URL.
5. Add a concise generative-AI disclosure reflecting actual usage.
6. Add references and XROCKET/code attribution notes when relevant.
7. Run the clean reproduction command or the closest documented clean-run subset.
8. Run tests, linting, and report build checks.
9. Open the generated PDF and verify figures, captions, tables, references, and page readability.
10. Tag or otherwise identify the final submission commit.
11. Keep 18-19 July for corrections only.

**Expected outputs:**

- Final technical report PDF under `reports/`.
- Reproducibility instructions in `docs/reproducibility.md` and README.
- Final results and figures under `results/` or `reports/figures/`.
- Passing tests and documented reproduction command.

**Exit criteria:**

- PDF opens and contains every required assignment element.
- All mandatory figures render and are referenced in the text.
- Repository link works.
- AI-use disclosure is present.
- Clean reproduction path has been tested or limitations are documented.
- No secrets or prohibited raw data are committed.

**Blocking decisions:**

- If the report build toolchain fails close to submission, use the simplest reliable PDF generation path and document the change.
- If final reproduction is too slow, run and document a clean smoke reproduction plus archived full-run artifacts.

## Definition of Done

The project is complete only when all of the following are true:

- A readable technical report PDF exists.
- The report includes a link to the code repository or notebook.
- Task 1.1 is directly answered with sensor-axis evidence and biomechanical discussion.
- Task 1.2 is directly answered with dilation or frequency-scale evidence.
- Task 1.3 is directly answered with discriminative pattern interpretation.
- Visualizations computed from the analysis support the findings.
- Dataset description, preprocessing, evaluation, and implementation steps are documented.
- Problems encountered are documented, including unresolved issues.
- Generative-AI tools used for implementation support are disclosed.
- Participant-independent evaluation has no train-test participant overlap.
- Raw KSAS data remain immutable and untracked.
- A clean reproduction path has been tested or its limits are honestly documented.

## Immediate Next Actions

1. Move or copy `KSAS-Dataset/` into `data/raw/KSAS-Dataset/` now that `data/raw/` exists and is ignored.
2. Record local dataset provenance and acquisition date.
3. Run the first dataset audit and generate `data/manifests/samples.csv`.
4. Fill `docs/data-dictionary.md` before writing preprocessing or modeling code.
5. Record unresolved facts, such as sampling frequency or device orientation, in `docs/limitations.md` and `docs/decision-log.md`.
