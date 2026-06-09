---
title: "Product Requirements Document — Explainable Human Motion Computing with KSAS and XROCKET"
document_type: "PRD"
version: "1.0"
status: "Baseline approved for implementation"
owner: "Forest"
academic_context: "Human Motion Computing — Part I: Inertial Signals"
due_date: "2026-07-20"
repository_name: "ksas-xrocket-hmc"
primary_deliverable: "Technical report in PDF format, supported by a reproducible code repository"
last_updated: "2026-06-09"
---

# Product Requirements Document (PRD)

## Explainable Human Motion Computing with KSAS and XROCKET

> [!IMPORTANT]
> This project is exclusively the selected **Part I — Human Motion Computing with Inertial Signals** assignment. Computer-vision alternatives are outside the submission scope.
>
> The project must apply **XROCKET** to the **KSAS smartphone inertial dataset**, or to a justified alternative inertial dataset, and use the resulting explanations to analyze:
>
> 1. which IMU sensor signals or axes contain discriminative information;
> 2. which temporal scales or dilations contain discriminative information; and
> 3. which signal patterns are most discriminative and whether they are meaningful from a human and biomechanical perspective.

---

# 1. Document purpose

This PRD defines the complete product, research, engineering, analytical, reproducibility, documentation, and submission requirements for the Human Motion Computing Part I project.

The “product” is not a consumer application. It is a **reproducible research artifact** consisting of:

- a version-controlled repository;
- a verified and documented inertial-data pipeline;
- leakage-resistant experiments;
- an XROCKET-based explainable time-series model;
- baseline and sensitivity experiments;
- generated metrics, tables, and visualizations;
- a clear biomechanical interpretation;
- a technical report in PDF format;
- a transparent record of generative-AI use; and
- enough documentation for an evaluator or researcher to reproduce the principal findings.

This PRD intentionally defines a broad, grade-maximizing scope. To prevent scope drift, requirements are separated into:

- **Mandatory:** required to satisfy the assignment and produce a defensible submission.
- **Recommended:** not explicitly demanded by the assignment, but strongly improves scientific quality.
- **Stretch:** attempted only after all mandatory and recommended requirements pass their quality gates.

A sophisticated extension must never replace a missing mandatory analysis.

---

# 2. Source of truth and precedence

When requirements conflict, use the following precedence:

1. The original assignment PDF, `Assignment_HMC_Santos2026.pdf`.
2. The selected-scope Markdown source, `docs/Human_Motion_Computing_Lecture_Summary_and_Part_I_Assignment.md`.
3. Clarifications explicitly received from Professor Olga C. Santos.
4. This `docs/PRD.md`.
5. Experiment configuration files.
6. Implementation details and notebook commentary.

The assignment requires a technical PDF report containing:

- a description of the inertial dataset;
- the steps followed to answer the questions;
- implementation problems encountered;
- a link to the produced code repository or notebook;
- responses to Tasks 1.1–1.3;
- visualizations that support the findings; and
- a declaration of generative-AI tools used to support software implementation.

The project due date is **20 July 2026**.

---

# 3. Product identity

## 3.1 Repository name

```text
ksas-xrocket-hmc
```

## 3.2 Repository description

> Explainable multivariate time-series analysis of KSAS smartphone IMU data with XROCKET, focusing on sensor-axis contributions, temporal dilations, discriminative patterns, and biomechanical interpretation.

## 3.3 README title

```text
Explainable Human Motion Computing with KSAS and XROCKET
```

## 3.4 Working report title

```text
Where and When Is the Information?
Explainable Classification of Kenpo Motion from Smartphone IMU Signals with XROCKET
```

The final title may be revised once the exact classification target and strongest findings are known.

---

# 4. High-level vision

Build a scientifically defensible and reproducible Human Motion Computing pipeline that transforms multivariate smartphone IMU sequences into:

1. reliable movement or performance predictions;
2. stable explanations of **where** the discriminative information is located across sensors and axes;
3. stable explanations of **when** the discriminative information appears across temporal scales;
4. concrete examples of discriminative temporal patterns; and
5. cautious, evidence-based biomechanical interpretations that could inform learning or performance assessment.

The ideal result is not merely a classifier with high accuracy. The ideal result is a transparent analysis that answers:

- **What does the model use?**
- **At what temporal scale does it use it?**
- **Does the explanation remain stable across participants and model runs?**
- **Is the explanation biomechanically plausible?**
- **What are the limitations and possible confounds?**
- **How could the identified patterns support future learning or performance-assessment systems?**

---

# 5. Product principles

The implementation and report must follow these principles.

## 5.1 Explainability before decoration

Plots must answer a research question. Visual complexity is not a substitute for analytical value.

## 5.2 Participant-independent generalization

The model must be evaluated on participants not seen during training. Sample-level random splits are prohibited for final results.

## 5.3 Minimal processing before sensitivity analysis

The main pipeline should preserve as much genuine temporal information as possible. Aggressive smoothing, feature engineering, coordinate transformation, or augmentation must be justified and evaluated separately.

## 5.4 Physical meaning must remain traceable

Every reported importance value must be traceable from:

```text
model feature
→ XROCKET kernel metadata
→ channel or channel combination
→ sensor and axis
→ dilation
→ effective temporal span
→ representative signal interval
```

## 5.5 Uncertainty must be visible

Report fold variation, participant variation, and explanation stability. Do not present one seed, one split, or one importance ranking as definitive.

## 5.6 Negative results remain valid results

The project is successful when it produces valid and interpretable evidence, even if predictive performance is moderate or some explanations are unstable.

## 5.7 Human-centered claims must be cautious

The repository is an academic analysis, not a validated coaching, diagnostic, or injury-prevention system. Biomechanical explanations must be framed as evidence-supported interpretations, not clinical or pedagogical facts.

## 5.8 Reproducibility is a deliverable

The report, figures, metrics, and model explanations must be generated from committed code and configuration rather than manually assembled from transient notebook state.

---

# 6. Stakeholders and users

| Stakeholder | Need | Project response |
|---|---|---|
| Student/researcher | Complete the assignment with strong scientific and engineering quality | Clear requirements, reproducible workflows, report-ready outputs |
| Professor/evaluator | Verify that Tasks 1.1–1.3 were answered with evidence | Traceability matrix, figures, metrics, repository link, limitations |
| Future researcher | Re-run or extend the analysis | Locked environment, data manifest, configurations, tests, documentation |
| Movement-domain reader | Understand the meaning of model explanations | Biomechanical interpretation, representative patterns, cautious terminology |
| Software reviewer | Inspect maintainability and correctness | Package-first architecture, typed interfaces, tests, CI |
| Data subject | Expect responsible use of motion data | Pseudonymization, minimization, no unnecessary exposure of raw recordings |

---

# 7. Assignment scope

## 7.1 Mandatory assignment questions

### Task 1.1 — Sensor axes contribution analysis

The system must determine which sensor signals contribute most to classification and relate the findings to movement biomechanics.

### Task 1.2 — Temporal pattern analysis

The system must analyze selected dilation values and determine whether classification depends mainly on:

- short-duration, high-frequency motion patterns;
- long-duration, global movement structures; or
- a combination of both.

The report must explain how temporal scale influences movement recognition.

### Task 1.3 — Explainable Human Motion Computing

The system must interpret the most discriminative patterns and discuss:

- what aspects of movement are captured;
- whether explanations are meaningful from a human perspective; and
- how explanations could support learning and performance assessment.

## 7.2 Primary classification task

The default primary task is:

> **Classify the Kenpo movement or movement segment represented by an IMU sequence.**

Expected candidate classes include the American Kenpo Karate Blocking Set I movements and potentially a no-movement class. The exact label set must be verified during the data audit.

## 7.3 Conditional secondary task

A secondary user-performance or expertise classification task may be included only when:

- the dataset contains a valid performance or expertise label;
- the label definition is documented;
- each class has sufficient participant coverage;
- grouped evaluation is feasible; and
- the task does not compromise completion of the mandatory movement analysis.

The report must not infer expertise from participant identity, age, body mass, sequence length, or other confounds.

## 7.4 Dataset selection

The default dataset is KSAS.

Use of another dataset is allowed by the assignment but requires written justification. Changing dataset is a major scope decision and is permitted only if one of the following occurs:

- KSAS cannot be obtained or legally used;
- required metadata are missing and make the assignment questions impossible;
- the available XROCKET implementation is fundamentally incompatible and cannot reasonably be adapted;
- the professor explicitly recommends an alternative; or
- a superior inertial dataset is already available with documented labels, channels, sampling frequency, and participant groups.

A dataset change must be recorded in `docs/decision-log.md`.

---

# 8. Goals and non-goals

## 8.1 Goals

### G-01 — Valid data understanding

Create a verified data dictionary and sample manifest before model training.

### G-02 — Reproducible preprocessing

Convert raw recordings into a documented tensor representation with no hidden manual steps.

### G-03 — Leakage-resistant modeling

Evaluate models using participant-grouped splits and training-fold-only preprocessing.

### G-04 — XROCKET implementation

Train a fixed-kernel XROCKET representation that exposes kernel-channel and kernel-dilation metadata.

### G-05 — Sensor-axis explanation

Produce global, class-specific, and stability-aware channel contribution analyses.

### G-06 — Temporal-scale explanation

Convert dilations into effective temporal spans and characterize short-, medium-, and long-scale information.

### G-07 — Pattern interpretation

Map important transformed features back to representative signal intervals.

### G-08 — Biomechanical discussion

Relate evidence to plausible movement properties while documenting coordinate-frame and device-orientation limitations.

### G-09 — Reproducible technical report

Generate the required PDF with all central figures and tables from repository artifacts.

### G-10 — Transparent AI use

Maintain an auditable record of all generative-AI assistance.

## 8.2 Non-goals

The following are out of scope for the core submission:

- a mobile application;
- a real-time feedback interface;
- live smartphone data collection;
- computer-vision pose estimation;
- physiological sensing;
- multimodal fusion;
- a clinical diagnostic claim;
- a validated coaching recommendation system;
- a production web API;
- a graphical dashboard;
- cloud deployment;
- distributed training;
- a general-purpose XROCKET library;
- a new large-scale data-collection study.

These may appear only as future work.

---

# 9. Research questions

## 9.1 Required research questions

### RQ1 — Where is the information?

Which IMU sensor families, axes, or channel combinations contribute most to the selected classification task?

### RQ2 — At what temporal scale is the information?

Do the most informative XROCKET features emphasize short-duration local motion, long-duration global structure, or a mixture of scales?

### RQ3 — What do the discriminative patterns mean?

Which concrete signal patterns activate the most important features, and can they be interpreted in relation to the movement?

## 9.2 Grade-maximizing supporting questions

### RQ4 — Are the explanations stable?

How stable are axis rankings, dilation rankings, and selected patterns across participant-held-out folds, seeds, and classifier choices?

### RQ5 — Are the explanations causal or merely correlated?

Do channel ablation and permutation analyses agree with model-native importance?

### RQ6 — Are there confounds?

Could prediction be driven by participant identity, arm, sequence length, recording conditions, sensor orientation, or preprocessing artifacts?

### RQ7 — Are findings movement-specific?

Do different blocks rely on different channels and temporal scales?

### RQ8 — Does preprocessing alter the interpretation?

How do minimal preprocessing, normalization, smoothing, and sequence homogenization affect performance and explanation stability?

---

# 10. Tentative hypotheses

These are exploratory hypotheses, not facts. They should guide analysis but must not bias interpretation.

- **H1:** Rotational blocks may rely more strongly on gyroscope channels than on magnetometer channels.
- **H2:** Acceleration and angular-velocity channels may contribute complementary information.
- **H3:** Movement classification may combine short-scale transition information with longer-scale movement-shape information.
- **H4:** Different block classes may have different temporal-scale signatures.
- **H5:** Important axes may vary when device orientation or execution arm changes.
- **H6:** A stable explanation should remain broadly consistent under grouped folds and repeated seeds.
- **H7:** Model-native feature importance may overstate some channels unless supported by ablation or permutation tests.

Hypotheses must be revised after the data audit if the actual channels, labels, or recording protocol differ from assumptions.

---

# 11. Success criteria

## 11.1 Assignment success

The submission is assignment-complete when:

- the dataset is described;
- the full analytical process is described;
- implementation problems are documented;
- a repository or notebook link is included;
- Tasks 1.1–1.3 are answered directly;
- each task is supported by computed visualizations;
- generative-AI usage is disclosed; and
- the report is delivered as a readable PDF.

## 11.2 Scientific success

The analysis is scientifically defensible when:

- no participant appears in both train and test partitions;
- all preprocessing fitted from data is fitted on training folds only;
- baseline performance is reported;
- uncertainty is reported;
- label-shuffle or equivalent negative-control behavior is checked;
- explanation rankings are evaluated for stability;
- sensor-axis claims are supported by at least two complementary importance approaches;
- dilation values are translated into real temporal spans when sampling frequency is known;
- biomechanical claims acknowledge sensor coordinate limitations; and
- limitations and confounds are explicit.

## 11.3 Engineering success

The repository is technically complete when:

- installation is documented;
- dependencies are locked;
- data acquisition or placement is documented;
- raw data remain immutable;
- experiments are configuration-driven;
- run outputs include configuration, seed, environment, and Git commit;
- tests cover critical data and split logic;
- a smoke pipeline runs in CI;
- report artifacts are generated by scripts; and
- no central result depends on hidden notebook state.

## 11.4 Quality targets

These targets are independent of model accuracy.

| Target | Acceptance threshold |
|---|---:|
| Participant overlap between train and test | 0 |
| Report figures generated from code | 100% of core figures |
| Core experiment runs with stored configuration | 100% |
| Core results with stored random seed | 100% |
| Core figures with title, axes, units, and caption | 100% |
| Required assignment questions answered explicitly | 3 of 3 |
| Generative-AI interactions used in final work logged | 100% |
| Critical tests passing before submission | 100% |
| Broken repository paths in README | 0 |
| Undocumented manual preprocessing steps | 0 |

No minimum accuracy is specified before the dataset is audited. A performance target may be added later as a descriptive benchmark, not as a condition for honest reporting.

---

# 12. Scope tiers

## 12.1 Mandatory scope

- KSAS data audit and documentation.
- Participant-safe sample manifest.
- Reproducible preprocessing.
- At least one simple baseline.
- XROCKET transform and classifier.
- Grouped evaluation.
- Task 1.1 channel analysis.
- Task 1.2 dilation analysis.
- Task 1.3 pattern interpretation.
- Core visualizations.
- Technical PDF.
- Repository link.
- AI-use disclosure.
- Problems and limitations.

## 12.2 Recommended scope

- Nested grouped cross-validation.
- Multiple baseline models.
- Model-native importance plus permutation and ablation.
- Fold- and seed-level explanation stability.
- Class-specific explanations.
- Arm-stratified error analysis.
- Negative-control label shuffle.
- Automated report generation.
- Unit and integration tests.
- Reproduction command.
- Data and run manifests.

## 12.3 Stretch scope

- Coordinate-representation comparison.
- Quaternion or sensor-fusion-derived channels, if metadata support them.
- Raw Cartesian versus magnitude-channel comparison.
- SHAP-based local explanations as a secondary method.
- Participant-level bootstrap confidence intervals.
- Counterfactual masking of discriminative intervals.
- Alternative fixed-kernel classifiers.
- Lightweight Docker or dev-container environment.
- An appendix with full per-fold explanation rankings.
- Optional expertise task when labels are valid.

Stretch work must be removed or deferred if it threatens the core report.

---

# 13. Critical decision gates

The following questions must be resolved before final modeling.

| Gate | Question | Required evidence | Decision deadline |
|---|---|---|---|
| DG-02 | What files and labels are present? | Data audit and manifest | Before preprocessing |
| DG-03 | What is the sampling frequency? | Dataset documentation or timestamp analysis | Before dilation interpretation |
| DG-04 | What are the channel names, units, and coordinate conventions? | Data dictionary | Before axis analysis |
| DG-05 | Are recordings already segmented? | File and metadata inspection | Before segmentation code |
| DG-06 | Are both arms included and how is phone orientation defined? | Protocol documentation or signal audit | Before arm handling |
| DG-07 | Are expertise or performance labels available and valid? | Label provenance and participant counts | Before secondary-task commitment |
| DG-08 | Which XROCKET implementation will be used? | Executable prototype, license, metadata access | Before main experiments |
| DG-09 | What fixed sequence length is required? | Implementation constraint and length audit | Before preprocessing freeze |
| DG-10 | Which evaluation split preserves class coverage? | Grouped split diagnostics | Before hyperparameter tuning |
| DG-11 | How are multi-channel kernels attributed? | Documented aggregation policy | Before explanation figures |
| DG-12 | Can all mandatory experiments run on available hardware? | Timed smoke benchmark | Before full experiment grid |

Unresolved gates must appear in `docs/open-questions.md`.

---

# 14. Data requirements

## 14.1 Raw-data immutability

- Raw source files must never be modified in place.
- Raw files must be stored outside Git when licensing or size requires it.
- Retrieval date, source URL, commit or release identifier, and checksums must be recorded.
- Any corrections must produce new files in `data/interim/` with a transformation log.

## 14.2 Data provenance

The project must record:

- dataset source;
- dataset version;
- license;
- download date;
- file checksums;
- original file names;
- participant identifiers;
- recording identifiers;
- movement labels;
- arm or handedness metadata;
- sensor channels;
- units;
- timestamps or sampling frequency;
- known exclusions;
- preprocessing already applied upstream; and
- any discrepancies between documentation and observed files.

## 14.3 Sample definition

The preferred unit of analysis is:

```text
one participant × one arm × one movement execution
```

If the dataset structure differs, the manifest must define the actual unit explicitly.

## 14.4 Sample manifest

A canonical sample manifest must be stored in Parquet or CSV. Required candidate fields are:

| Field | Type | Requirement |
|---|---|---|
| `sample_id` | string | Unique and stable |
| `participant_id` | string | Pseudonymous group identifier |
| `recording_id` | string | Original recording source |
| `movement_label` | categorical | Primary target candidate |
| `performance_label` | categorical/nullable | Secondary target candidate |
| `arm` | categorical/nullable | Left, right, unknown |
| `sensor_location` | categorical/nullable | Smartphone or body location |
| `device_orientation` | string/nullable | Protocol description |
| `sampling_rate_hz` | float/nullable | Required for physical time |
| `original_length` | integer | Number of source samples |
| `processed_length` | integer | Number of model samples |
| `channel_names` | string/list | Canonical ordered channels |
| `source_path` | string | Relative source path |
| `split_group` | string | Usually participant ID |
| `quality_flag` | categorical | Valid, warning, excluded |
| `exclusion_reason` | string/nullable | Required when excluded |

## 14.5 Tensor contract

The default model tensor convention is:

```text
X.shape == [n_samples, n_channels, n_timesteps]
y.shape == [n_samples]
groups.shape == [n_samples]
```

Requirements:

- canonical dtype: `float32`, unless reference implementation requires `float64`;
- labels stored separately from signals;
- channel order stored in metadata and never inferred from column position alone;
- no NaN or infinite values after the final preprocessing stage;
- channel units preserved in the data dictionary;
- the time axis must remain ordered;
- any padding mask must be retained if padding is used;
- transformations must not reorder participants or labels independently.

## 14.6 Participant identity

- Original personally identifying information must not enter result files.
- Participant identifiers must be pseudonymous.
- Participant ID must not be used as a predictive feature.
- Group-split assertions must fail the run when participant overlap is detected.

## 14.7 Label quality

The audit must check:

- missing labels;
- contradictory labels;
- class imbalance;
- class coverage per participant;
- participants with only one class;
- duplicate sequences;
- suspiciously identical signals;
- sequence length by class;
- recording order;
- class encoding consistency; and
- whether “no movement” examples are genuinely comparable to movement examples.

## 14.8 Data exclusions

Any exclusion must be:

- rule-based;
- decided without looking at test outcomes;
- documented;
- counted in a flow diagram or table; and
- reproducible from code.

Manual deletion without a recorded reason is prohibited.

---

# 15. Data audit requirements

The first analytical milestone must generate:

1. a dataset inventory;
2. a data dictionary;
3. class counts;
4. participant counts;
5. participant-by-class coverage;
6. arm distribution;
7. channel list and units;
8. sampling-rate evidence;
9. missing-value counts;
10. sequence-length distribution;
11. duplicate and near-duplicate checks;
12. representative raw traces;
13. outlier warnings;
14. exclusion proposals;
15. a documented recommendation for the primary target;
16. a documented recommendation for the validation scheme; and
17. a decision on whether the secondary performance task is feasible.

The audit must be committed before the main model results.

---

# 16. Preprocessing requirements

## 16.1 Preprocessing philosophy

The core analysis must start from a minimally transformed signal representation so that short-scale evidence is not accidentally removed.

Every preprocessing operation must be:

- motivated;
- configurable;
- deterministic;
- fitted on training data when applicable;
- logged; and
- evaluated for its impact on interpretation.

## 16.2 Required preprocessing stages

### PRE-01 — Schema loading

Load data using explicit column or channel mappings. Unknown channels must raise a warning or error.

### PRE-02 — Ordering

Sort observations by timestamp or source index. Detect non-monotonic timestamps.

### PRE-03 — Missing values

Use a documented policy:

- reject severely corrupted sequences;
- interpolate only short internal gaps when justified;
- never silently fill entire channels;
- record every affected sample.

### PRE-04 — Duplicate handling

Detect duplicate files and identical signal sequences. Retain or remove according to documented provenance.

### PRE-05 — Segmentation

If raw recordings contain multiple movements:

- use official segmentation when available;
- otherwise use protocol-informed segmentation;
- store segment boundaries;
- avoid using test labels to tune boundaries;
- visually inspect a sample of segments.

If recordings are already segmented, do not re-segment unnecessarily.

### PRE-06 — Sequence-length homogenization

XROCKET or the selected implementation may require equal-length sequences.

The chosen method must be one of:

- interpolation/resampling to a fixed length;
- documented zero or edge padding with mask;
- truncation at a justified boundary; or
- an implementation supporting unequal lengths.

The decision must consider interpretability. Resampling changes the relationship between dilation and real time; therefore the mapping must use the post-resampling time base.

### PRE-07 — Normalization

The default candidates are:

- no amplitude normalization;
- training-fold per-channel standardization;
- training-fold per-channel min-max scaling.

Normalization statistics must be learned on training folds and applied unchanged to validation and test folds.

The final report must state whether importance refers to physical units or normalized units.

### PRE-08 — Filtering and smoothing

Smoothing is not mandatory in the main pipeline.

When used:

- filter type and parameters must be specified;
- the filter must not use future data in a way that invalidates interpretation, unless offline zero-phase processing is explicitly declared;
- raw and filtered examples must be compared;
- high-frequency information loss must be discussed;
- filtering must be included as a sensitivity experiment rather than silently applied.

### PRE-09 — Derived channels

Possible derived channels include:

- acceleration magnitude;
- gyroscope magnitude;
- magnetometer magnitude;
- jerk;
- orientation estimates;
- force proxies; and
- coordinate transforms.

Derived channels are recommended or stretch analyses, not replacements for raw-axis analysis.

Every derived channel must have:

- a formula;
- units;
- required source channels;
- missing-data behavior;
- biomechanical rationale; and
- separate reporting from raw axes.

### PRE-10 — Arm handling

Arm may be a confound.

The project must:

- inspect class balance by arm;
- report performance by arm when possible;
- check whether important axes change by arm;
- avoid using arm as a target proxy;
- document phone placement and orientation;
- consider a left/right canonicalization sensitivity experiment only when the physical mapping is defensible.

### PRE-11 — Data leakage safeguards

The code must enforce:

- participant grouping before train-test separation;
- training-only normalization;
- training-only feature selection;
- training-only hyperparameter tuning;
- no use of full-dataset class statistics in transformations that affect model fitting;
- no test-set-guided preprocessing decisions.

---

# 17. Modeling requirements

## 17.1 Model hierarchy

The project must include:

1. a chance or majority baseline;
2. at least one simple interpretable baseline;
3. the primary XROCKET model; and
4. at least one classifier or explanation sensitivity analysis.

## 17.2 Baseline models

Recommended baselines:

- majority class;
- stratified random predictor;
- simple statistical features plus logistic regression;
- simple statistical features plus Random Forest;
- standard ROCKET/MiniROCKET plus RidgeClassifierCV, when feasible.

Baselines must use the same outer grouped folds as XROCKET.

## 17.3 XROCKET requirements

The XROCKET component must:

- use fixed or reproducibly generated kernels;
- expose kernel metadata;
- expose channel or channel-combination metadata;
- expose dilation;
- expose kernel length;
- expose feature type when multiple transformed features are produced per kernel;
- preserve a mapping from transformed feature index to kernel metadata;
- support serialization;
- accept a fixed random seed when any stochastic step remains;
- be wrapped behind a stable project interface;
- include attribution and license notes if adapted from external code.

## 17.4 XROCKET implementation adapter

The repository should expose an interface conceptually equivalent to:

```python
class XRocketTransformer:
    def fit(self, X, y=None): ...
    def transform(self, X): ...
    def get_feature_metadata(self): ...
    def save_kernel_bank(self, path): ...
```

`get_feature_metadata()` must return one row per transformed feature with fields such as:

- `feature_index`;
- `kernel_id`;
- `feature_type`;
- `channel_set`;
- `sensor_family`;
- `axes`;
- `kernel_length`;
- `dilation`;
- `bias`;
- `padding`;
- `effective_samples`; and
- `effective_seconds`, when sampling frequency is available.

## 17.5 Primary classifier

Random Forest is the preferred primary downstream classifier because it supports nonlinear decision boundaries and global feature-importance aggregation.

Initial recommended settings, subject to grouped inner validation:

```yaml
classifier:
  type: random_forest
  n_estimators: 500
  class_weight: balanced
  max_features: sqrt
  random_state: 42
  n_jobs: -1
```

This is a starting configuration, not a guaranteed optimum.

## 17.6 Classifier sensitivity

At least one of the following should be compared:

- RidgeClassifierCV;
- logistic regression with regularization;
- linear SVM;
- calibrated linear classifier.

The purpose is to determine whether explanation conclusions depend strongly on the downstream model.

## 17.7 Hyperparameter tuning

Hyperparameters must be tuned only inside the training partition of each outer fold.

Preferred scheme:

```text
outer participant-grouped CV
    └── inner participant-grouped CV for tuning
```

If the number of participants makes nested cross-validation unreliable, use a small predeclared parameter grid and report the limitation.

## 17.8 Determinism

Every experiment must record:

- Python seed;
- NumPy seed;
- model seed;
- split seed;
- XROCKET kernel-bank identifier;
- library versions;
- Git commit;
- configuration file; and
- timestamp.

## 17.9 Model persistence

Save:

- preprocessing state;
- kernel bank;
- classifier;
- label encoder;
- channel order;
- training configuration;
- feature metadata; and
- environment manifest.

A saved model without its channel and feature metadata is invalid for explanation.

---

# 18. Validation and evaluation requirements

## 18.1 Grouped evaluation

The default outer split should be one of:

- `StratifiedGroupKFold`, when class coverage allows it;
- `GroupKFold`;
- `LeaveOneGroupOut`, as a sensitivity analysis; or
- a documented participant-level holdout.

Sample-level random K-fold is prohibited for reported final metrics.

## 18.2 Split diagnostics

For every fold, save:

- train participants;
- test participants;
- train class counts;
- test class counts;
- arm counts;
- number of samples;
- any absent classes; and
- an explicit participant-overlap assertion.

## 18.3 Metrics

Required metrics:

- macro F1;
- balanced accuracy;
- per-class precision;
- per-class recall;
- per-class F1;
- confusion matrix;
- support per class;
- fold mean;
- fold standard deviation.

Recommended metrics:

- Matthews correlation coefficient;
- macro one-vs-rest ROC AUC when probabilities are valid;
- calibration metrics when probability quality is discussed;
- participant-level accuracy distribution.

Ordinary accuracy may be reported but must not be the only headline metric.

## 18.4 Uncertainty

Use at least one:

- fold-level confidence intervals;
- participant bootstrap confidence intervals;
- repeated grouped splits;
- seed variation.

Do not treat fold values as fully independent observations without qualification.

## 18.5 Negative controls

At least one negative control is recommended:

- label permutation within the training process;
- random-kernel or random-feature comparison;
- participant-identity probe;
- sequence-length-only classifier.

A label-shuffled model should perform near chance. Unexpectedly high shuffled performance is a leakage warning.

## 18.6 Confound checks

The project should evaluate whether predictions correlate with:

- participant ID;
- arm;
- sequence length;
- missing-value pattern;
- sensor magnitude offset;
- acquisition order;
- sampling rate;
- device orientation; and
- upstream preprocessing.

## 18.7 Error analysis

The report must include qualitative or quantitative error analysis:

- most confused movement pairs;
- participant-level failures;
- arm-specific failures;
- low-confidence examples;
- representative false positives;
- representative false negatives;
- whether failures correspond to overlapping biomechanics or data-quality issues.

---

# 19. Explainability requirements

## 19.1 Explanation hierarchy

The project must provide explanations at four levels:

1. **Feature level:** transformed XROCKET feature importance.
2. **Channel level:** sensor family, axis, or channel combination.
3. **Temporal-scale level:** dilation and effective receptive field.
4. **Pattern level:** representative raw-signal intervals and kernel responses.

## 19.2 Importance methods

### Mandatory

- model-native global feature importance;
- channel aggregation;
- dilation aggregation.

### Recommended

- test-fold permutation importance;
- channel ablation;
- sensor-family ablation;
- seed and fold stability.

### Stretch

- SHAP on a limited, documented subset;
- counterfactual temporal masking;
- local surrogate explanation.

No single importance method should be presented as causal evidence.

## 19.3 Importance normalization

For comparison across folds:

1. set negative importance values to a documented policy, if the method can produce them;
2. preserve signed values when sign has meaning;
3. normalize absolute global importance within each fitted model to sum to one;
4. aggregate only after within-model normalization;
5. report mean, spread, and rank stability.

The implementation must not average raw importance values from models with incompatible scales.

## 19.4 Multi-channel kernel attribution

When a kernel uses multiple channels, produce two views.

### Combination view

Treat the complete channel set as one entity, for example:

```text
{acc_y, gyro_z}
```

### Marginal view

Allocate importance to constituent channels using a documented rule, initially equal allocation unless the XROCKET implementation supplies channel-specific weights.

The report must state that marginal allocation is an analytical convention. Combination-level importance must remain available to prevent misleading double counting.

## 19.5 Class-specific explanations

Global importance may hide movement-specific differences.

The project should produce class-specific evidence using one or more of:

- one-vs-rest models;
- class-specific permutation importance;
- class-conditional SHAP;
- differences in feature distributions for each class;
- top activating samples per class.

The chosen method must be described.

## 19.6 Stability analysis

Required stability outputs:

- rank correlation of axis importance across folds;
- top-k overlap for axes or channel combinations;
- rank correlation of dilation importance;
- top-k overlap for dilations;
- variance across seeds;
- identification of unstable findings.

Recommended statistics:

- Spearman rank correlation;
- Jaccard overlap;
- coefficient of variation;
- bootstrap interval.

An unstable explanation may still be reported, but must be labeled unstable.

---

# 20. Task 1.1 — Sensor-axis contribution specification

## 20.1 Required outputs

1. Global importance by sensor family.
2. Global importance by physical axis.
3. Global importance by sensor-axis channel.
4. Channel-combination importance when applicable.
5. Class-specific channel profiles.
6. Fold and seed uncertainty.
7. Channel or sensor-family ablation.
8. Biomechanical interpretation.
9. Coordinate-frame limitation.
10. A direct answer to the assignment question.

## 20.2 Required visualizations

- sensor-family importance bar chart;
- sensor-axis importance heatmap;
- ranked channel importance with confidence intervals;
- class-by-channel heatmap;
- ablation performance-drop chart;
- stability chart or matrix.

## 20.3 Biomechanical interpretation protocol

For every claimed relation:

1. identify the sensor and axis;
2. state the device-coordinate meaning, when known;
3. identify the movement phase or pattern;
4. describe the plausible body motion;
5. explain supporting evidence;
6. state uncertainty;
7. identify possible confounds.

Suggested interpretation table:

| Movement | Signal/channel | Temporal location | Plausible motion property | Evidence | Confidence | Caveat |
|---|---|---|---|---|---|---|

## 20.4 Prohibited interpretation

Do not state that:

- a phone axis is an anatomical axis without protocol evidence;
- an important feature proves a biomechanical mechanism;
- a correlation proves better technique;
- an accelerometer channel measures force directly;
- a magnetometer feature measures muscle activity;
- a model classifies expertise if only movement labels are available.

---

# 21. Task 1.2 — Dilation and temporal-scale specification

## 21.1 Required kernel metadata

For each feature:

- dilation;
- kernel length;
- feature type;
- post-preprocessing sampling rate;
- post-preprocessing sequence length;
- effective receptive field in samples;
- effective receptive field in seconds;
- relative span as a percentage of the movement sequence.

## 21.2 Effective temporal span

Use:

```text
effective_samples = 1 + dilation × (kernel_length − 1)
```

When sampling frequency is defined:

```text
effective_seconds = effective_samples / sampling_rate_hz
```

When sequences are resampled to preserve full movement duration:

```text
effective_seconds =
    effective_samples / processed_length × original_or_normalized_duration_seconds
```

The exact conversion must reflect the actual preprocessing pipeline.

## 21.3 Terminology requirement

Dilation is a receptive-field spacing parameter, not a direct Fourier frequency.

The report may use “short-duration/high-frequency-like” and “long-duration/low-frequency-like” only when it explains the relationship carefully.

## 21.4 Temporal-scale bins

Create data-driven bins after inspecting the distribution. A candidate scheme is:

- local: effective span below 20% of sequence;
- intermediate: 20%–50%;
- global: above 50%.

Alternative physically meaningful bins in seconds are preferred when the sampling rate and movement duration are reliable.

The binning rule must be fixed before examining class outcomes.

## 21.5 Required outputs

1. Importance-weighted dilation distribution.
2. Importance-weighted effective-duration distribution.
3. Local/intermediate/global contribution percentages.
4. Class-specific temporal-scale profiles.
5. Fold and seed stability.
6. Representative kernels at low and high dilation.
7. A direct conclusion: local, global, or mixed.
8. Explanation of why temporal scale matters for movement recognition.

## 21.6 Required visualizations

- dilation importance scatter plot;
- histogram or density of importance by dilation;
- importance by effective seconds;
- stacked contribution by temporal-scale band;
- class-by-scale heatmap;
- stability plot across folds;
- representative low- and high-dilation signal windows.

---

# 22. Task 1.3 — Discriminative-pattern specification

## 22.1 Pattern-selection rule

Representative patterns must be selected by a predeclared rule, such as:

- top global feature importance;
- top class-specific feature importance;
- highest stable feature across folds;
- highest average test-fold activation for a true class;
- representative median activator rather than extreme outlier.

Do not choose examples solely because they make a visually attractive story.

## 22.2 Pattern localization

For each selected feature:

1. identify the kernel;
2. identify the contributing channels;
3. compute the kernel response over time;
4. locate the strongest or most relevant activation;
5. map processed indices to original time;
6. display the raw or normalized signal;
7. display the response;
8. display class-level feature distributions;
9. compare true positives and errors;
10. document uncertainty.

## 22.3 Feature-type awareness

If XROCKET creates multiple feature types per kernel, such as maximum response or proportion of positive values, the interpretation must distinguish them.

A global proportion feature may not map to one unique interval. In that case:

- show the intervals contributing to the proportion;
- display response distribution over time;
- avoid pretending the feature has a single activation point.

## 22.4 Required outputs

- at least three discriminative pattern case studies;
- at least one pattern from a correctly classified sample;
- at least one pattern associated with a common confusion or failure;
- class distribution of selected features;
- human-meaningfulness assessment;
- potential learning or performance-assessment implication;
- limitations.

## 22.5 Required figure layout

Recommended multi-panel layout:

```text
A. raw multichannel signal
B. highlighted temporal interval
C. kernel response
D. transformed feature distribution by class
E. explanation and biomechanical interpretation
```

---

# 23. Biomechanics and human-meaningfulness requirements

## 23.1 Interpretation boundaries

The project may discuss:

- rotation;
- acceleration;
- deceleration;
- directional change;
- movement onset;
- peak action;
- transition;
- return to guard;
- rhythm;
- smoothness;
- consistency;
- coordination;
- abruptness;
- global movement shape;
- local micro-adjustment.

The project must avoid unsupported claims about:

- muscle activation;
- joint torque;
- injury risk;
- clinical condition;
- learning gain;
- expertise;
- force;
- balance; or
- correctness,

unless the dataset and measurement setup genuinely support them.

## 23.2 Device-frame caveat

Smartphone IMU axes describe the device coordinate frame. Their mapping to anatomical directions depends on:

- phone placement;
- phone orientation;
- arm;
- grip;
- coordinate convention;
- calibration;
- movement phase.

Every axis interpretation must acknowledge this.

## 23.3 Human-perspective evaluation

For each major explanation, classify it as:

- clearly meaningful;
- plausible but uncertain;
- difficult to interpret;
- likely confounded; or
- unstable.

This classification must be justified.

## 23.4 Learning and assessment implications

Potential implications may include:

- focusing feedback on movement transitions;
- identifying inconsistent rotational control;
- detecting timing differences;
- separating local execution quality from global sequence shape;
- selecting sensor channels for a simpler future device;
- designing expert-review visualizations.

These are design implications, not claims that the present system improves learning.

---

# 24. Functional requirements

The identifiers below should be referenced in issues, pull requests, and the final traceability matrix.

## 24.1 Data and ingestion

| ID | Requirement | Priority | Acceptance criterion |
|---|---|---|---|
| FR-DAT-001 | Acquire or locate the KSAS dataset reproducibly | Mandatory | Source and retrieval instructions documented |
| FR-DAT-002 | Verify local dataset access policy | Mandatory | Raw data remain local-only and untracked |
| FR-DAT-003 | Generate file checksums | Recommended | Manifest contains checksum per raw file |
| FR-DAT-004 | Parse all supported raw files | Mandatory | Loader reports complete inventory |
| FR-DAT-005 | Validate channel schema | Mandatory | Unknown or missing channels produce explicit error |
| FR-DAT-006 | Build sample manifest | Mandatory | One stable row per sample |
| FR-DAT-007 | Pseudonymize participant IDs | Mandatory | Result files contain no direct identifiers |
| FR-DAT-008 | Audit missing and duplicate data | Mandatory | Audit tables saved |
| FR-DAT-009 | Verify sampling frequency | Mandatory for seconds-based interpretation | Evidence recorded |
| FR-DAT-010 | Preserve raw data unchanged | Mandatory | No preprocessing writes to raw directory |

## 24.2 Preprocessing

| ID | Requirement | Priority | Acceptance criterion |
|---|---|---|---|
| FR-PRE-001 | Deterministically order temporal samples | Mandatory | Repeated load yields identical arrays |
| FR-PRE-002 | Handle missing values by policy | Mandatory | No silent fill; affected samples logged |
| FR-PRE-003 | Segment or validate segments | Mandatory | Boundaries or source segmentation documented |
| FR-PRE-004 | Homogenize sequence length | Mandatory if required by XROCKET | All model inputs have valid shape |
| FR-PRE-005 | Apply training-only normalization | Mandatory | Leakage test passes |
| FR-PRE-006 | Support channel selection by configuration | Mandatory | Axis and family ablations run without code edits |
| FR-PRE-007 | Support optional smoothing | Recommended | Raw and smoothed configurations separated |
| FR-PRE-008 | Support optional derived channels | Stretch | Formula and provenance stored |
| FR-PRE-009 | Preserve mapping to original time | Mandatory for Task 1.3 | Pattern indices map back to source sequence |
| FR-PRE-010 | Produce processed-data manifest | Mandatory | Every processed sample has transformation metadata |

## 24.3 Splitting and evaluation

| ID | Requirement | Priority | Acceptance criterion |
|---|---|---|---|
| FR-EVAL-001 | Split by participant group | Mandatory | Participant overlap assertion is zero |
| FR-EVAL-002 | Preserve class coverage where possible | Mandatory | Fold diagnostics saved |
| FR-EVAL-003 | Use same outer folds across models | Mandatory | Baseline and XROCKET fold IDs match |
| FR-EVAL-004 | Tune only within training data | Recommended | Nested or documented inner validation |
| FR-EVAL-005 | Compute required metrics | Mandatory | Metrics table contains all required fields |
| FR-EVAL-006 | Save fold predictions | Mandatory | One prediction row per test sample |
| FR-EVAL-007 | Run label-shuffle control | Recommended | Control performance reported |
| FR-EVAL-008 | Analyze errors by class and participant | Recommended | Error-analysis artifacts generated |
| FR-EVAL-009 | Estimate uncertainty | Mandatory | Fold spread or CI reported |
| FR-EVAL-010 | Compare at least one baseline | Mandatory | Baseline table in report |

## 24.4 Modeling

| ID | Requirement | Priority | Acceptance criterion |
|---|---|---|---|
| FR-MOD-001 | Implement or adapt XROCKET | Mandatory | End-to-end fit and transform succeeds |
| FR-MOD-002 | Expose fixed kernel metadata | Mandatory | Metadata row exists per feature |
| FR-MOD-003 | Persist kernel bank | Mandatory | Saved and reloadable |
| FR-MOD-004 | Train primary classifier | Mandatory | Grouped evaluation completed |
| FR-MOD-005 | Train classifier sensitivity model | Recommended | Comparison documented |
| FR-MOD-006 | Persist models and encoders | Recommended | Reloaded model reproduces predictions |
| FR-MOD-007 | Record all random seeds | Mandatory | Run manifest contains seeds |
| FR-MOD-008 | Support experiment configuration files | Mandatory | Core run uses no hard-coded experiment parameters |
| FR-MOD-009 | Support CPU execution | Mandatory | Core pipeline runs on documented machine |
| FR-MOD-010 | Detect incompatible input shape | Mandatory | Clear validation error |

## 24.5 Explainability

| ID | Requirement | Priority | Acceptance criterion |
|---|---|---|---|
| FR-EXP-001 | Map feature importance to channel metadata | Mandatory | Channel table generated |
| FR-EXP-002 | Map feature importance to dilation metadata | Mandatory | Dilation table generated |
| FR-EXP-003 | Normalize importance across folds | Mandatory | Aggregation method documented |
| FR-EXP-004 | Preserve channel-combination view | Mandatory | Combination table generated |
| FR-EXP-005 | Produce marginal axis view | Mandatory | Axis importance generated without double counting |
| FR-EXP-006 | Compute channel ablation | Recommended | Performance-drop table generated |
| FR-EXP-007 | Compute permutation importance | Recommended | Test-fold permutation results saved |
| FR-EXP-008 | Analyze explanation stability | Recommended | Rank and top-k stability outputs |
| FR-EXP-009 | Produce class-specific explanations | Recommended | At least one class-specific artifact |
| FR-EXP-010 | Localize important patterns | Mandatory | At least three pattern case studies |
| FR-EXP-011 | Map patterns to original time | Mandatory | Plot labels use source time or documented normalized time |
| FR-EXP-012 | Flag unstable explanations | Mandatory | Stability status included in tables |

## 24.6 Visualization and reporting

| ID | Requirement | Priority | Acceptance criterion |
|---|---|---|---|
| FR-VIS-001 | Generate dataset overview figures | Mandatory | Class, participant, length, signal figures exist |
| FR-VIS-002 | Generate evaluation figures | Mandatory | Confusion matrix and per-class metrics exist |
| FR-VIS-003 | Generate Task 1.1 figures | Mandatory | Axis and ablation visuals exist |
| FR-VIS-004 | Generate Task 1.2 figures | Mandatory | Dilation and temporal-span visuals exist |
| FR-VIS-005 | Generate Task 1.3 figures | Mandatory | Pattern panels exist |
| FR-VIS-006 | Export publication-quality formats | Mandatory | PDF/SVG or 300-DPI PNG available |
| FR-REP-001 | Generate technical report PDF | Mandatory | PDF builds without manual figure insertion |
| FR-REP-002 | Include repository link | Mandatory | Link present in final PDF |
| FR-REP-003 | Include problems encountered | Mandatory | Dedicated section present |
| FR-REP-004 | Include AI-use disclosure | Mandatory | Tool, purpose, and verification described |
| FR-REP-005 | Include limitations | Mandatory | Dedicated section present |
| FR-REP-006 | Include reproducibility instructions | Recommended | Commands and environment documented |
| FR-REP-007 | Include direct answers to each task | Mandatory | Clearly labeled Task 1.1–1.3 conclusions |

## 24.7 Operations and documentation

| ID | Requirement | Priority | Acceptance criterion |
|---|---|---|---|
| FR-OPS-001 | Provide one-command environment setup | Recommended | Documented command succeeds |
| FR-OPS-002 | Provide one-command core reproduction | Recommended | Core artifacts regenerate |
| FR-OPS-003 | Run lint and tests in CI | Recommended | Main branch CI passes |
| FR-OPS-004 | Save run configuration and commit | Mandatory | Every core run has manifest |
| FR-OPS-005 | Avoid absolute paths | Mandatory | Repository works from another location |
| FR-DOC-001 | Provide README | Mandatory | Setup, data, run, results sections present |
| FR-DOC-002 | Provide data dictionary | Mandatory | Channels, labels, units documented |
| FR-DOC-003 | Provide methodology document | Recommended | Evaluation and explanation policies recorded |
| FR-DOC-004 | Provide decision log | Recommended | Major decisions and rationale recorded |
| FR-DOC-005 | Provide AI-use log | Mandatory | Final assistance trace complete |

---

# 25. Non-functional requirements

| ID | Requirement | Priority | Acceptance criterion |
|---|---|---|---|
| NFR-001 | Reproducibility | Mandatory | Same config and seed reproduce materially identical outputs |
| NFR-002 | Traceability | Mandatory | Every report result maps to a run directory |
| NFR-003 | Portability | Recommended | Linux/macOS setup documented; no machine-specific paths |
| NFR-004 | Maintainability | Recommended | Reusable logic in package, not duplicated across notebooks |
| NFR-005 | Testability | Recommended | Critical split, preprocessing, and aggregation functions tested |
| NFR-006 | Explainability | Mandatory | Feature-to-signal lineage preserved |
| NFR-007 | Privacy | Mandatory | No direct participant identifiers in outputs |
| NFR-008 | Licensing | Mandatory | External code and data licenses documented |
| NFR-009 | Performance | Recommended | Core pipeline completes on a commodity laptop within a documented reasonable duration |
| NFR-010 | Memory efficiency | Recommended | Data loaded without unnecessary full copies |
| NFR-011 | Observability | Recommended | Structured logs and progress messages |
| NFR-012 | Failure clarity | Mandatory | Invalid data, splits, or shapes fail with actionable messages |
| NFR-013 | Figure accessibility | Mandatory | Legible labels, units, captions, and non-color-only distinctions |
| NFR-014 | Report portability | Mandatory | PDF embeds fonts and renders without external files |
| NFR-015 | Determinism | Mandatory | Kernel bank and splits reproducible |
| NFR-016 | Scientific caution | Mandatory | Conclusions distinguish evidence, interpretation, and speculation |
| NFR-017 | Configuration integrity | Recommended | Validated configuration schema |
| NFR-018 | Artifact integrity | Recommended | Checksums or manifest for final core artifacts |
| NFR-019 | Repository cleanliness | Mandatory | Raw data, caches, secrets, and large models excluded appropriately |
| NFR-020 | Documentation consistency | Recommended | README, PRD, report, and configs use consistent terminology |

---

# 26. Technical stack

## 26.1 Core language and environment

| Component | Choice | Role |
|---|---|---|
| Python | Python 3.11 | Main implementation language |
| Environment and lockfile | `uv` with `pyproject.toml` and `uv.lock` | Reproducible dependency management |
| Task runner | `Makefile` | Short, documented workflows |
| Shell | POSIX-compatible shell where possible | Automation |

Python 3.11 is selected as a conservative scientific-computing target. A different minor version may be used only after verifying all required packages and recording the change.

## 26.2 Data and scientific computing

| Package | Role |
|---|---|
| NumPy | Tensor operations and numerical utilities |
| pandas | Metadata, results, and tabular analysis |
| PyArrow | Parquet I/O |
| SciPy | Signal processing, interpolation, statistics |
| scikit-learn | Splits, classifiers, metrics, pipelines, permutation importance |
| joblib | Model and preprocessing serialization |
| PyYAML | Experiment configuration |
| Pydantic or dataclasses | Configuration and schema validation |
| Pandera, optional | Tabular data-contract validation |

## 26.3 Time-series modeling

| Component | Role |
|---|---|
| Local XROCKET adapter | Primary fixed-kernel explainable transform |
| `aeon` or `sktime`, subject to compatibility | ROCKET-family baseline and time-series utilities |
| Reference PhyUM code | Implementation reference, after license and behavior audit |

The project must not assume that a generic package exposes the exact XROCKET metadata required by the assignment. The adapter must preserve kernel and feature metadata even when using external components.

## 26.4 Visualization

| Package | Role |
|---|---|
| Matplotlib | Primary publication-quality plotting |
| Seaborn, optional | Statistical heatmaps and compact summaries |
| Plotly, optional | Exploratory inspection only |

Final report figures should use static formats.

## 26.5 Notebooks

| Component | Role |
|---|---|
| JupyterLab | Exploratory analysis |
| ipykernel | Project kernel |
| nbstripout or equivalent | Remove irrelevant notebook output |
| Papermill, optional | Parameterized notebook execution |

Notebooks are interfaces to reusable code, not the only implementation.

## 26.6 Reporting

| Component | Role |
|---|---|
| Quarto | Report authoring and executable document structure |
| Pandoc | Document conversion |
| TinyTeX/LaTeX | PDF rendering |
| BibTeX/CSL | References |

A Markdown-to-PDF alternative is acceptable if it supports reproducible figure inclusion and references.

## 26.7 Quality tooling

| Component | Role |
|---|---|
| pytest | Tests |
| pytest-cov | Coverage |
| Ruff | Formatting and linting |
| mypy | Static type checking |
| pre-commit | Local quality gates |
| GitHub Actions | Continuous integration |

## 26.8 Optional reproducibility tooling

| Component | Use condition |
|---|---|
| DVC | Only if data versioning materially improves reproducibility |
| Git LFS | Only for allowed large binary artifacts |
| Docker/devcontainer | Stretch portability layer |
| MLflow | Not required; use only if simple run directories become insufficient |

Avoid infrastructure that adds more failure modes than scientific value.

---

# 27. Repository structure

```text
ksas-xrocket-hmc/
+-- AGENT.md
├── README.md
├── LICENSE
├── CITATION.cff
├── CHANGELOG.md
├── pyproject.toml
├── uv.lock
├── Makefile
├── .gitignore
├── .pre-commit-config.yaml
�
+-- docs/
�   +-- PRD.md
�   +-- Roadmap.md
�   +-- Human_Motion_Computing_Lecture_Summary_and_Part_I_Assignment.md
│
├── configs/
│   ├── data/
│   │   └── ksas.yaml
│   ├── experiments/
│   │   ├── movement_xrocket_rf.yaml
│   │   ├── movement_xrocket_ridge.yaml
│   │   ├── baseline_statistical.yaml
│   │   ├── axis_ablation.yaml
│   │   ├── sensor_family_ablation.yaml
│   │   ├── preprocessing_sensitivity.yaml
│   │   └── performance_task.yaml
│   └── plotting/
│       └── report.yaml
│
├── data/
│   ├── README.md
│   ├── raw/
│   │   └── .gitkeep
│   ├── interim/
│   │   └── .gitkeep
│   ├── processed/
│   │   └── .gitkeep
│   └── manifests/
│       └── .gitkeep
│
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_preprocessing_validation.ipynb
│   ├── 03_baseline_models.ipynb
│   ├── 04_xrocket_training.ipynb
│   ├── 05_axis_contributions.ipynb
│   ├── 06_dilation_analysis.ipynb
│   ├── 07_discriminative_patterns.ipynb
│   ├── 08_error_and_stability_analysis.ipynb
│   └── 09_report_artifact_review.ipynb
│
├── src/
│   └── ksas_xrocket/
│       ├── __init__.py
│       ├── cli.py
│       │
│       ├── data/
│       │   ├── download.py
│       │   ├── load.py
│       │   ├── schema.py
│       │   ├── manifest.py
│       │   ├── validate.py
│       │   └── split.py
│       │
│       ├── preprocessing/
│       │   ├── channels.py
│       │   ├── missing.py
│       │   ├── filtering.py
│       │   ├── normalization.py
│       │   ├── resampling.py
│       │   ├── segmentation.py
│       │   ├── derived.py
│       │   └── pipeline.py
│       │
│       ├── models/
│       │   ├── baselines.py
│       │   ├── xrocket.py
│       │   ├── classifiers.py
│       │   ├── tuning.py
│       │   ├── train.py
│       │   └── predict.py
│       │
│       ├── explainability/
│       │   ├── feature_map.py
│       │   ├── importance.py
│       │   ├── axis_importance.py
│       │   ├── dilation_importance.py
│       │   ├── pattern_localization.py
│       │   ├── ablation.py
│       │   ├── permutation.py
│       │   └── stability.py
│       │
│       ├── evaluation/
│       │   ├── cross_validation.py
│       │   ├── metrics.py
│       │   ├── uncertainty.py
│       │   ├── controls.py
│       │   └── error_analysis.py
│       │
│       ├── visualization/
│       │   ├── dataset.py
│       │   ├── signals.py
│       │   ├── performance.py
│       │   ├── axes.py
│       │   ├── dilations.py
│       │   ├── patterns.py
│       │   └── style.py
│       │
│       └── utils/
│           ├── config.py
│           ├── io.py
│           ├── logging.py
│           ├── provenance.py
│           └── random.py
│
├── scripts/
│   ├── prepare_data.py
│   ├── audit_data.py
│   ├── run_experiment.py
│   ├── run_controls.py
│   ├── run_axis_analysis.py
│   ├── run_dilation_analysis.py
│   ├── run_pattern_analysis.py
│   ├── export_report_artifacts.py
│   └── verify_reproduction.py
│
├── tests/
│   ├── fixtures/
│   ├── test_data_schema.py
│   ├── test_manifest.py
│   ├── test_preprocessing.py
│   ├── test_group_splits.py
│   ├── test_no_leakage.py
│   ├── test_xrocket_metadata.py
│   ├── test_axis_aggregation.py
│   ├── test_dilation_mapping.py
│   ├── test_pattern_mapping.py
│   └── test_smoke_pipeline.py
│
├── results/
│   ├── README.md
│   ├── runs/
│   ├── metrics/
│   ├── predictions/
│   ├── explanations/
│   ├── controls/
│   ├── models/
│   └── manifests/
│
├── reports/
│   ├── technical_report.qmd
│   ├── references.bib
│   ├── figures/
│   ├── tables/
│   ├── generated/
│   └── final/
│
├── docs/
│   ├── data-dictionary.md
│   ├── methodology.md
│   ├── biomechanics.md
│   ├── reproducibility.md
│   ├── limitations.md
│   ├── ai-use.md
│   ├── decision-log.md
│   ├── open-questions.md
│   └── references.md
│
└── .github/
    └── workflows/
        └── ci.yml
```

---

# 28. Configuration specification

Core experiments must be defined in YAML.

Example:

```yaml
experiment:
  name: movement_xrocket_rf
  seed: 42
  tags:
    - core
    - movement
    - xrocket
    - random_forest

data:
  dataset: ksas
  manifest: data/manifests/ksas_samples.parquet
  target: movement_label
  group: participant_id
  channels: all_raw_imu
  arms: both
  exclusions: data/manifests/exclusions.csv

preprocessing:
  missing_policy: interpolate_short_gaps
  smoothing: none
  length_strategy: resample
  target_length: null
  normalization: train_fold_standardize
  dtype: float32

validation:
  outer:
    type: stratified_group_k_fold
    n_splits: 5
    shuffle: true
    random_state: 42
  inner:
    type: group_k_fold
    n_splits: 3

xrocket:
  kernel_bank: fixed_default
  kernel_seed: 42
  persist_metadata: true

classifier:
  type: random_forest
  n_estimators: 500
  class_weight: balanced
  max_features: sqrt
  random_state: 42
  n_jobs: -1

explainability:
  model_native_importance: true
  permutation_importance: true
  channel_ablation: true
  class_specific: true
  stability_top_k: [3, 5, 10]

outputs:
  save_model: true
  save_predictions: true
  save_feature_metadata: true
  save_figures: true
```

Configuration validation must reject:

- nonexistent channels;
- incompatible target and labels;
- zero folds;
- split groups identical to target;
- requested seconds-based outputs without time-base evidence;
- XROCKET runs without feature metadata persistence.

---

# 29. Run artifact contract

Each run directory must contain:

```text
results/runs/<timestamp>_<experiment_name>/
├── config.resolved.yaml
├── run_manifest.json
├── environment.txt
├── git_commit.txt
├── data_manifest_snapshot.parquet
├── split_manifest.parquet
├── fold_metrics.csv
├── aggregate_metrics.json
├── predictions.parquet
├── feature_metadata.parquet
├── feature_importance.parquet
├── channel_importance.parquet
├── dilation_importance.parquet
├── stability.json
├── model/
├── figures/
└── logs/
```

The run manifest must include:

- run ID;
- start and end time;
- status;
- machine and OS;
- Python version;
- dependency hash;
- data checksum;
- configuration hash;
- Git commit;
- seeds;
- elapsed time;
- warnings;
- exclusions;
- artifact paths.

---

# 30. Visualization inventory

## 30.1 Mandatory dataset figures

| File | Purpose |
|---|---|
| `dataset_class_distribution` | Show class balance |
| `dataset_participant_coverage` | Show participant-by-class coverage |
| `dataset_sequence_lengths` | Show variable-length structure and possible confound |
| `dataset_signal_examples` | Show representative raw channels |
| `dataset_pipeline_diagram` | Explain data-to-model flow |

## 30.2 Mandatory model figures

| File | Purpose |
|---|---|
| `model_confusion_matrix` | Show movement confusions |
| `model_per_class_metrics` | Show precision, recall, and F1 |
| `model_fold_variation` | Show uncertainty |
| `model_baseline_comparison` | Contextualize XROCKET |

## 30.3 Mandatory Task 1.1 figures

| File | Purpose |
|---|---|
| `axis_sensor_family_importance` | Compare accelerometer, gyroscope, magnetometer |
| `axis_channel_heatmap` | Compare sensor-axis channels |
| `axis_class_specific_heatmap` | Show movement-specific information |
| `axis_ablation_performance_drop` | Validate importance |
| `axis_stability` | Show robustness |

## 30.4 Mandatory Task 1.2 figures

| File | Purpose |
|---|---|
| `dilation_importance_distribution` | Show selected dilations |
| `dilation_effective_duration` | Convert to physical time |
| `dilation_scale_contribution` | Local/intermediate/global conclusion |
| `dilation_class_specific` | Show movement-specific temporal scales |
| `dilation_stability` | Show fold and seed robustness |

## 30.5 Mandatory Task 1.3 figures

| File | Purpose |
|---|---|
| `pattern_case_01` | Stable high-importance pattern |
| `pattern_case_02` | Different class or scale |
| `pattern_case_03` | Common failure or confusion |
| `pattern_feature_distributions` | Compare transformed values by class |
| `pattern_summary_table` | Human interpretation and caveats |

## 30.6 Figure standards

Every final figure must:

- have a descriptive title;
- label axes and units;
- identify whether values are raw, normalized, or relative;
- display uncertainty where relevant;
- use legible fonts at report size;
- avoid relying only on color;
- use consistent channel names;
- state the number of folds or samples;
- include a caption in the report;
- be generated from a run artifact;
- be exported as vector PDF/SVG or at least 300-DPI PNG.

---

# 31. Report requirements

## 31.1 Recommended report structure

1. Title page.
2. Abstract.
3. Introduction.
4. Human Motion Computing and explainability background.
5. Research questions.
6. Dataset.
7. Data audit.
8. Preprocessing.
9. Experimental design.
10. XROCKET and classifier.
11. Evaluation protocol.
12. Task 1.1 results.
13. Task 1.2 results.
14. Task 1.3 results.
15. Error and stability analysis.
16. Biomechanical discussion.
17. Implications for learning and performance assessment.
18. Problems encountered.
19. Limitations and threats to validity.
20. Reproducibility and repository.
21. Generative-AI disclosure.
22. Conclusion.
23. References.
24. Appendices.

## 31.2 Abstract requirements

The abstract should include:

- problem;
- dataset;
- method;
- evaluation protocol;
- principal performance result;
- principal axis result;
- principal temporal-scale result;
- principal interpretation;
- limitation.

## 31.3 Direct task answers

Each assignment task must end with a clearly marked response:

```text
Answer to Task 1.1
Answer to Task 1.2
Answer to Task 1.3
```

The evaluator should not need to infer the answer from several pages of plots.

## 31.4 Problems encountered

The report must document real issues, for example:

- data format ambiguity;
- missing metadata;
- sequence-length inconsistency;
- sampling-rate uncertainty;
- adaptation of XROCKET code;
- package compatibility;
- feature-metadata extraction;
- grouped split constraints;
- explanation instability;
- runtime;
- report-generation issues.

Problems must include the resolution or remaining limitation.

## 31.5 Limitations and threats to validity

At minimum discuss:

- sample size;
- participant diversity;
- label quality;
- participant-independent generalization;
- device-coordinate interpretation;
- arm and phone orientation;
- preprocessing sensitivity;
- model dependence of explanations;
- correlation versus biomechanical causation;
- absence of direct learning-outcome validation;
- generalization beyond the KSAS protocol.

## 31.6 Repository link

The final report must include:

- repository URL;
- commit or release corresponding to submission;
- reproduction command;
- data access instructions; and
- license statement.

## 31.7 AI disclosure

The report must state:

- tool name;
- date or period of use;
- purpose;
- whether generated text or code was retained;
- how outputs were reviewed;
- what was not delegated to AI.

---

# 32. README requirements

The README must contain:

```markdown
# Explainable Human Motion Computing with KSAS and XROCKET

## Overview
## Assignment scope
## Research questions
## Dataset
## Data access and licensing
## Repository structure
## Installation
## Quick start
## Data audit
## Running baselines
## Running XROCKET
## Running explanation analyses
## Reproducing figures
## Building the report
## Evaluation protocol
## Main results
## Limitations
## Generative-AI disclosure
## Citation
## License
```

The README must state prominently:

> This repository is an academic motion-analysis project. It is not a validated coaching, clinical, diagnostic, or injury-prevention system.

---

# 33. CLI and workflow requirements

The intended top-level workflow is:

```bash
uv sync
make data
make audit
make baseline
make train
make explain
make figures
make report
```

Recommended Make targets:

```text
install
data
validate-data
audit
baseline
train
controls
explain-axes
explain-dilations
explain-patterns
explain-stability
figures
report
test
lint
typecheck
reproduce
clean-generated
```

Recommended CLI:

```bash
uv run hmc prepare --config configs/data/ksas.yaml
uv run hmc audit --dataset ksas
uv run hmc train --config configs/experiments/movement_xrocket_rf.yaml
uv run hmc explain axes --run-dir results/runs/<run-id>
uv run hmc explain dilations --run-dir results/runs/<run-id>
uv run hmc explain patterns --run-dir results/runs/<run-id>
uv run hmc report --run-dir results/runs/<run-id>
```

Commands must:

- validate inputs;
- print the resolved configuration;
- create a run ID;
- log progress;
- fail with a nonzero exit code on critical errors;
- never overwrite a prior run silently.

---

# 34. Testing requirements

## 34.1 Unit tests

Required targets:

- channel mapping;
- sample ID uniqueness;
- manifest schema;
- time ordering;
- missing-value policy;
- resampling;
- normalization fit/apply separation;
- participant-group split;
- no-overlap assertion;
- effective receptive-field calculation;
- dilation-to-seconds calculation;
- feature-to-kernel mapping;
- multi-channel importance allocation;
- importance normalization;
- pattern index mapping.

## 34.2 Integration tests

- load a small fixture dataset;
- preprocess it;
- fit a small XROCKET kernel bank;
- train a classifier;
- generate predictions;
- produce channel and dilation tables;
- save and reload the model;
- generate one figure.

## 34.3 Regression tests

When the pipeline stabilizes, store expected outputs for a synthetic fixture:

- fixed split membership;
- feature metadata schema;
- deterministic predictions or tolerances;
- expected receptive-field values.

## 34.4 Scientific invariant tests

The code must assert:

- no participant overlap;
- no target leakage through metadata;
- no NaNs after preprocessing;
- feature count equals metadata row count;
- importance aggregation preserves total normalized importance;
- every report pattern maps to a valid sample and interval.

## 34.5 CI scope

CI should run:

```text
ruff
mypy
pytest
small synthetic smoke pipeline
```

CI should not require the private or large raw dataset.

---

# 35. Reproducibility requirements

## 35.1 Environment

Commit:

- `pyproject.toml`;
- `uv.lock`;
- optional system requirements;
- Quarto/LaTeX setup instructions.

## 35.2 Data

Record:

- source;
- license;
- version;
- checksum;
- placement instructions;
- preprocessing manifest.

## 35.3 Code

Tag the submission commit, for example:

```text
v1.0-submission
```

## 35.4 Experiments

Every report table and figure must identify its source run.

## 35.5 Reproduction command

Provide one documented command that regenerates core report artifacts from prepared data:

```bash
make reproduce
```

If full reproduction is too expensive, provide:

```bash
make reproduce-core
make reproduce-full
```

## 35.6 Deterministic report

The report build must not rely on network access or external mutable assets.

---

# 36. Generative-AI policy and disclosure

Generative AI is permitted to support software-tool usage for Part I, but its use must be declared.

## 36.1 Required log

Maintain `docs/ai-use.md` with:

| Date | Tool | Prompt purpose | Output used | Verification | Files affected |
|---|---|---|---|---|---|

## 36.2 Permitted uses

- repository planning;
- code scaffolding;
- debugging;
- test design;
- configuration design;
- explanation of library errors;
- documentation drafting;
- report-language editing;
- visualization-code assistance.

## 36.3 Human verification

Every retained output must be checked through one or more of:

- code execution;
- tests;
- source documentation;
- manual inspection;
- comparison with dataset evidence;
- independent rewriting.

## 36.4 Data protection

Do not send raw participant data, identifiable metadata, secrets, or restricted files to external generative-AI systems.

Use:

- synthetic snippets;
- schema-only examples;
- aggregate statistics;
- redacted errors.

## 36.5 Final disclosure wording

The final report should include a concise statement such as:

> Generative-AI tools were used to support repository planning, software implementation, debugging, test design, and language editing. All generated code and prose retained in the project were reviewed, executed, tested, and revised by the author. Raw participant data and personally identifying information were not supplied to generative-AI tools.

The exact statement must reflect actual use.

---

# 37. Ethics, privacy, and responsible interpretation

## 37.1 Privacy

- Minimize copied raw data.
- Pseudonymize participant IDs.
- Do not publish private metadata.
- Respect the course-provided data access boundary and keep raw participant data untracked.
- Avoid embedding raw paths containing names.

## 37.2 Fairness and confounding

Check whether model behavior differs by available participant attributes, arm, or recording condition. Do not make demographic claims without appropriate data.

## 37.3 Safety

Do not generate prescriptive coaching or injury advice from this model.

## 37.4 Transparency

State:

- data limitations;
- model limitations;
- explanation method limitations;
- uncertainty;
- alternative explanations.

## 37.5 Human oversight

The report should frame XROCKET explanations as decision support for a human instructor or researcher, not as an autonomous judgment of a learner.

---

# 38. Licensing requirements

Before submission:

1. keep the course-provided KSAS raw data local and untracked;
2. verify the XROCKET reference implementation license;
3. preserve attribution notices;
4. document adapted files and upstream commits;
5. choose a repository license compatible with reused code;
6. avoid redistributing data when the license does not permit it;
7. include a third-party notices section when necessary.

Do not select an open-source license by assumption.

---

# 39. Performance and hardware requirements

## 39.1 Target environment

Core experiments should run on a typical laptop or desktop CPU.

GPU support is not required.

## 39.2 Runtime reporting

Record:

- data preparation time;
- transform fit time;
- transform time;
- classifier time;
- explanation time;
- peak memory when practical.

## 39.3 Runtime budget

Before launching the complete grid:

- benchmark one fold;
- estimate total runtime;
- reduce nonessential grid size if the estimate threatens the schedule.

## 39.4 Efficiency priorities

Prioritize:

1. scientific correctness;
2. mandatory analyses;
3. reproducibility;
4. reasonable runtime;
5. optional model breadth.

---

# 40. Experiment matrix

## 40.1 Core experiments

| ID | Representation | Classifier | Validation | Purpose |
|---|---|---|---|---|
| EXP-001 | Simple statistical features | Logistic regression or RF | Grouped CV | Baseline |
| EXP-002 | XROCKET raw channels | Random Forest | Grouped CV | Primary model |
| EXP-003 | XROCKET raw channels | Ridge/logistic | Same folds | Classifier sensitivity |
| EXP-004 | XROCKET raw channels | RF | Label-shuffle control | Leakage check |
| EXP-005 | XROCKET channel ablations | RF | Same folds | Task 1.1 validation |
| EXP-006 | XROCKET | RF | Multiple seeds/folds | Stability |

## 40.2 Recommended preprocessing sensitivity

| ID | Change | Question |
|---|---|---|
| EXP-101 | No normalization | Does amplitude carry class information? |
| EXP-102 | Training-fold standardization | Are results scale-dependent? |
| EXP-103 | Minimal smoothing | Does noise reduction alter short-scale evidence? |
| EXP-104 | Alternate sequence length strategy | Are explanations created by resampling? |
| EXP-105 | Left/right stratification | Are axis findings arm-dependent? |

## 40.3 Stretch representation experiments

| ID | Change | Precondition |
|---|---|---|
| EXP-201 | Sensor magnitudes | Raw sensor axes verified |
| EXP-202 | Spherical/cylindrical transforms | Geometric rationale and channel units verified |
| EXP-203 | Quaternion orientation | Required sensors and reliable sampling available |
| EXP-204 | Jerk channels | Noise handling validated |
| EXP-205 | Expertise classification | Valid performance labels and group coverage |

---

# 41. Statistical-analysis requirements

## 41.1 Model comparison

Compare models on identical outer folds.

Report:

- per-fold metric differences;
- mean difference;
- uncertainty interval;
- practical interpretation.

Avoid overemphasis on a single p-value.

## 41.2 Importance comparison

For channel or dilation comparisons:

- use fold-level normalized values;
- report distribution, not only mean;
- use rank-based stability measures;
- avoid treating thousands of transformed features as independent observations.

## 41.3 Multiple comparisons

When testing many channels or dilations inferentially:

- define the family of comparisons;
- use correction or clearly label analysis exploratory;
- prefer effect size and uncertainty.

## 41.4 Participant bootstrap

If implemented, resample participants, not individual sequences, to preserve dependency structure.

---

# 42. Documentation requirements

## 42.1 `docs/data-dictionary.md`

Must define:

- file structure;
- participant IDs;
- movement labels;
- performance labels;
- channels;
- units;
- coordinate frame;
- sampling rate;
- arms;
- sequence boundaries;
- exclusions.

## 42.2 `docs/methodology.md`

Must define:

- preprocessing;
- split strategy;
- metrics;
- tuning;
- importance normalization;
- multi-channel allocation;
- temporal conversion;
- pattern selection;
- uncertainty;
- controls.

## 42.3 `docs/biomechanics.md`

Must separate:

- known protocol facts;
- reasonable biomechanical interpretation;
- speculative interpretation;
- unsupported claims to avoid.

## 42.4 `docs/reproducibility.md`

Must include:

- environment setup;
- data setup;
- experiment commands;
- expected runtime;
- report build;
- artifact locations.

## 42.5 `docs/limitations.md`

Maintain limitations throughout implementation, not only at the end.

## 42.6 `docs/decision-log.md`

Each entry should include:

- date;
- decision;
- alternatives;
- rationale;
- evidence;
- consequences;
- revisit condition.

---

# 43. Risks and mitigations

| Risk | Probability | Impact | Mitigation | Trigger |
|---|---|---|---|---|
| XROCKET reference code unavailable or incompatible | Medium | High | Prototype adapter early; keep alternative fixed-kernel plan | Cannot fit toy data by milestone |
| XROCKET metadata inaccessible | Medium | Critical | Modify adapter to persist metadata; do not use opaque transform | Feature indices cannot map to kernels |
| Dataset access boundary unclear | Medium | High | Do not redistribute; confirm with professor | No explicit course guidance found |
| Sampling frequency unknown | Medium | High | Inspect timestamps/docs; report normalized time if unresolved | No defensible Hz value |
| Small participant count | High | High | Grouped CV, uncertainty, restrained tuning | Folds lack class coverage |
| Severe class imbalance | Medium | High | Macro metrics, class weights, grouped stratification | Minority class absent in folds |
| Participant leakage | Medium | Critical | Automated assertions and manifest tests | Any participant overlap |
| Arm/device orientation confound | High | High | Stratified analysis, protocol review, cautious axis interpretation | Axis importance reverses by arm |
| Sequence length becomes target proxy | Medium | High | Length-only baseline and sensitivity | Length-only model performs well |
| Smoothing removes discriminative information | Medium | High | Minimal core preprocessing; sensitivity comparison | High-dilation conclusion changes |
| Importance unstable across folds | High | Medium | Stability analysis and qualified conclusions | Low rank correlation |
| Random Forest impurity bias | Medium | Medium | Permutation and ablation validation | Native and permutation rankings disagree |
| Overly broad scope delays report | High | Critical | Scope tiers and milestone gates | Core task incomplete by July 9 |
| Report toolchain fails | Medium | Medium | Build skeleton early; keep fallback PDF route | Quarto PDF not building by July 5 |
| External code license conflict | Medium | High | Audit before adaptation; document notices | License incompatible |
| Misleading biomechanical claims | Medium | High | Interpretation protocol and caveat review | Claim lacks coordinate evidence |
| AI-use disclosure incomplete | Low | Medium | Log use continuously | Missing prompt/use history |
| Repository cannot reproduce on clean machine | Medium | High | CI, clean-environment test | Setup requires manual fixes |

---

# 44. Milestones and schedule

The schedule is designed around the **20 July 2026** deadline.

| Milestone | Target dates | Exit criteria |
|---|---|---|
| M0 — Repository foundation | 10–12 June | Repo, PRD, environment, CI skeleton |
| M1 — Data acquisition and audit | 13–18 June | Manifest, data dictionary, split recommendation |
| M2 — Preprocessing and baselines | 19–24 June | Reproducible tensors, grouped baseline results |
| M3 — XROCKET integration | 25 June–1 July | Fixed kernels, metadata map, primary model |
| M4 — Task 1.1 and 1.2 analyses | 2–7 July | Axis, ablation, dilation, time-scale figures |
| M5 — Task 1.3 and robustness | 8–11 July | Pattern case studies, stability, errors, controls |
| M6 — Report drafting | 12–15 July | Complete first PDF with all mandatory sections |
| M7 — Reproduction and review | 16–17 July | Clean-run reproduction, citation and figure audit |
| M8 — Final buffer | 18–19 July | Corrections only; no new research scope |
| Submission | 20 July | PDF and repository release/tag |

## 44.1 Scope freeze

No new stretch experiment should be added after **11 July 2026** unless all mandatory report sections already contain final evidence.

## 44.2 Report-first safeguard

A report skeleton must build successfully before the end of M2. This prevents a final-week document-toolchain failure.

---

# 45. Quality gates

## Gate A — Data readiness

Pass when:

- local raw-data policy recorded;
- sample manifest valid;
- channels and labels documented;
- sampling-rate status known;
- participant groups valid;
- no unresolved critical data corruption.

## Gate B — Evaluation readiness

Pass when:

- participant-safe folds created;
- class coverage reviewed;
- leakage tests pass;
- baseline executes.

## Gate C — XROCKET readiness

Pass when:

- transformer fits;
- features transform;
- feature metadata align;
- saved model reloads;
- one fold produces channel and dilation tables.

## Gate D — Explanation readiness

Pass when:

- importance normalization validated;
- multi-channel attribution policy implemented;
- pattern index mapping validated;
- stability pipeline executes.

## Gate E — Report readiness

Pass when:

- all mandatory figures exist;
- Tasks 1.1–1.3 have direct answers;
- limitations and AI-use sections exist;
- repository link and reproduction command exist.

## Gate F — Submission readiness

Pass when:

- clean environment builds core results;
- tests pass;
- PDF opens and all figures render;
- references resolve;
- submission commit is tagged;
- no secrets or prohibited data are committed.

---

# 46. Definition of done

The project is done only when all of the following are true.

## Research

- [ ] Primary classification target is justified.
- [ ] Participant-independent evaluation is complete.
- [ ] Baseline comparison is complete.
- [ ] Task 1.1 is answered with importance, uncertainty, ablation, and biomechanics.
- [ ] Task 1.2 is answered with dilation, effective time, and scale interpretation.
- [ ] Task 1.3 is answered with localized patterns and human interpretation.
- [ ] Explanation stability is reported or explicitly identified as a limitation.
- [ ] Confounds and errors are analyzed.
- [ ] Conclusions do not exceed evidence.

## Engineering

- [ ] Environment is locked.
- [ ] Data setup is documented.
- [ ] Raw data are immutable.
- [ ] Sample manifest is committed or reproducibly generated.
- [ ] Core experiments are configuration-driven.
- [ ] Run provenance is saved.
- [ ] Critical tests pass.
- [ ] CI passes.
- [ ] Report artifacts are code-generated.
- [ ] Clean reproduction is tested.

## Submission

- [ ] Technical report PDF exists.
- [ ] PDF contains dataset description.
- [ ] PDF contains methodological steps.
- [ ] PDF contains problems encountered.
- [ ] PDF contains repository link.
- [ ] PDF contains required visualizations.
- [ ] PDF contains AI-use disclosure.
- [ ] PDF contains limitations.
- [ ] Repository README is complete.
- [ ] Submission commit is tagged.
- [ ] Final files are backed up.

---

# 47. Requirements traceability matrix

| Assignment requirement | PRD coverage | Core artifact |
|---|---|---|
| Apply XROCKET to inertial data | Sections 17, 24, 40 | Model run and metadata |
| Identify contributing sensor signals | Sections 19–20 | Axis importance and ablation |
| Relate signals to biomechanics | Sections 20, 23 | Interpretation table |
| Analyze dilation values | Sections 19, 21 | Dilation table |
| Determine short/global/mixed scale | Section 21 | Scale contribution figure |
| Explain effect of temporal scale | Section 21 | Report discussion |
| Interpret discriminative patterns | Section 22 | Pattern case studies |
| Assess human meaningfulness | Sections 22–23 | Meaningfulness classification |
| Discuss learning/performance support | Section 23 | Implications subsection |
| Describe dataset | Sections 14–15, 31 | Dataset and audit sections |
| Describe steps | Sections 16–22, 31 | Methods section |
| Report implementation problems | Section 31.4 | Problems encountered section |
| Link repository/notebook | Sections 31.6, 32 | PDF and README |
| Include visualizations | Section 30 | Generated figure inventory |
| Disclose generative-AI use | Section 36 | AI-use log and report statement |

---

# 48. Minimum viable submission versus excellent submission

## 48.1 Minimum viable valid submission

- documented KSAS data;
- reproducible preprocessing;
- participant-grouped evaluation;
- one baseline;
- XROCKET model;
- axis ranking;
- dilation ranking;
- three pattern examples;
- direct task answers;
- PDF;
- repository link;
- AI disclosure.

## 48.2 Strong submission

Adds:

- nested grouped validation;
- ablation and permutation importance;
- class-specific explanations;
- fold and seed stability;
- error analysis;
- negative control;
- automated report;
- tests and CI;
- detailed limitations.

## 48.3 Excellent submission

Adds only where time and evidence permit:

- carefully justified coordinate or derived-channel analysis;
- robust participant bootstrap;
- classifier-independence analysis;
- pattern counterfactual masking;
- comprehensive appendix;
- polished reproduction release;
- a clear connection between technical explanation and future personalized psychomotor support.

An excellent submission is not the one with the most experiments. It is the one with the clearest evidence chain and the fewest unexamined assumptions.

---

# 49. Open questions for the initial audit

1. What exact files are present in the current KSAS repository?
2. Is the dataset already segmented into each block?
3. Are there five movement classes, six classes including no movement, or another label scheme?
4. Are both arms included in every participant?
5. Is handedness recorded?
6. What smartphone placement and orientation were used?
7. What are the exact accelerometer, gyroscope, and magnetometer units?
8. Is the sampling frequency constant?
9. Are timestamps available?
10. Were the signals previously smoothed or normalized?
11. Are expertise, correctness, or performance labels available?
12. Are participant demographics present and permitted for use?
13. What is the official course data access policy?
14. Which XROCKET implementation is executable and licensed?
15. Does the implementation support multivariate channel combinations?
16. Which features are generated per kernel?
17. Does the implementation require fixed length?
18. Can a kernel response be localized in time?
19. Can the selected grouped folds preserve every movement class?
20. What hardware and runtime are available?

These questions must be answered or explicitly marked unresolved before final interpretation.

---

# 50. Reference material

The assignment identifies the following starting material:

1. KSAS Dataset: course-provided local dataset.
2. Time series classification with XROCKET:  
   `https://dida.do/blog/explainable-time-series-classification-with-x-rocket`
3. UMAP 2026 paper: `umap26-45.pdf`, when available.
4. Sleep analysis paper:  
   `https://link.springer.com/article/10.1007/s10796-026-10736-0`
5. Aikido paper:  
   `https://link.springer.com/article/10.1007/s11257-024-09393-2`
6. Code for sleep analysis:  
   `https://github.com/Physical-User-Modeling-PhyUM/EADS`
7. Code for Aikido analysis:  
   `https://github.com/Physical-User-Modeling-PhyUM/UMAP26_SP1`

External code is support material and is explicitly described by the assignment as not exhaustively tested. It must be audited before reliance.

---

# 51. Glossary

| Term | Meaning |
|---|---|
| HMC | Human Motion Computing |
| IMU | Inertial Measurement Unit |
| Accelerometer | Sensor measuring acceleration in the device frame |
| Gyroscope | Sensor measuring angular velocity |
| Magnetometer | Sensor measuring magnetic field, often used for orientation |
| MTS | Multivariate time series |
| ROCKET | Random Convolutional Kernel Transform for time-series classification |
| XROCKET | Explainable/fixed-kernel ROCKET approach used to expose channel and dilation relevance |
| Kernel | Convolutional pattern detector |
| Dilation | Spacing between kernel elements |
| Receptive field | Temporal span covered by a kernel |
| Axis contribution | Importance associated with a sensor-axis channel |
| Channel combination | Set of channels used jointly by a kernel |
| Ablation | Removal of a channel, sensor family, or feature group to measure impact |
| Permutation importance | Performance reduction after shuffling a feature or channel |
| Grouped CV | Cross-validation that keeps all samples from a participant in one partition |
| Macro F1 | Unweighted mean F1 across classes |
| Balanced accuracy | Mean recall across classes |
| Confound | Variable correlated with target that may produce a misleading model |
| Data leakage | Use of information unavailable in a genuine unseen-data setting |
| Biomechanical interpretation | Cautious relation between signal evidence and physical movement |
| Run manifest | Record of configuration, data, code, environment, and artifacts |
| Core artifact | Result required for the assignment or direct task answer |

---

# 52. Final product statement

The final repository and report should demonstrate a complete evidence chain:

```text
verified KSAS recordings
→ reproducible multivariate sequences
→ participant-independent XROCKET model
→ validated feature importance
→ sensor-axis contribution
→ dilation and temporal-scale contribution
→ localized discriminative patterns
→ biomechanical interpretation
→ implications, uncertainty, and limitations
```

The project should be judged not only by predictive performance, but by the clarity, stability, reproducibility, and human meaning of that chain.
