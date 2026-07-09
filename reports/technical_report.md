---
title: "Explainable Human Motion Computing with KSAS and XROCKET"
author: "Luca Foresti"
date: "20 July 2026"
lang: en
---

# Abstract

This report applies XROCKET to the course-provided KSAS smartphone inertial
dataset to study American Kenpo Karate Blocking Set I. The goal is not only to
classify movement labels, but to answer where the information appears in the
sensor axes, at what temporal scale it appears, and whether discriminative
patterns can be interpreted from a human motion computing perspective. The final
evaluation uses participant-grouped folds to avoid train-test participant
leakage. The primary XROCKET random-forest model reached mean macro F1 0.883
over five folds; a logistic-regression sensitivity model reached mean macro F1
0.884. The strongest stable explanation evidence points to gravity and
device-frame z-axis structure, long temporal spans in the padded 56-sample
representation, and representative PPV pattern intervals that are plausible but
not sufficient for validated coaching or biomechanical causation claims.

# Introduction

Human Motion Computing treats movement as both an action and an information
source. In this project, the action is a set of karate blocking movements, and
the information source is a smartphone inertial measurement stream. The selected
assignment scope is Part I - Human Motion Computing with Inertial Signals. The
technical aim is to apply XROCKET, an explainable time-series classifier, to the
KSAS dataset and provide evidence for three required questions:

1. Which sensor axes contribute most to movement classification?
2. Which temporal scales are most informative?
3. Which discriminative patterns appear meaningful from a human perspective?

XROCKET was selected because it exposes feature metadata such as channel,
dilation, kernel length, and threshold [@xrocket_blog; @xrocket_code]. This
metadata makes it possible to trace model behavior back to sensor-coordinate
evidence instead of reporting only aggregate accuracy.

# Dataset

The dataset is the course-provided KSAS smartphone IMU dataset
[@ksas_dataset]. The audited local copy contains 240 CSV recordings: 20
participants, two arms, and six movement labels with 40 samples per label. The
labels are no movement, upward block, hammering inward block, extended outward
block, outward downward block, and rear elbow block. Each CSV contains 18
channels: accelerometer, gravity, gyroscope, linear acceleration, game rotation
vector, and magnetic field, each with x, y, and z device-frame axes.

The audit found no missing values, no duplicate rows, no duplicate files, and no
schema errors. Sequence lengths vary from 18 to 56 samples, with median length
33. Raw CSVs are kept under `data/raw/KSAS-Dataset/` locally but are ignored by
Git and are not redistributed in the repository.

# Methods

The repository builds a manifest from the audited raw files, converts the 240
recordings into a tensor with shape `(240, 18, 56)`, and right-pads shorter
recordings with zeros. The tensor contract records sample order, channel order,
labels, participant IDs, arm metadata, original lengths, and valid-timestep
masks.

Evaluation uses five participant-grouped folds. All samples from a participant
remain in the same fold, and each fold contains 48 held-out samples with eight
samples per class. The reported metrics therefore test participant-independent
generalization rather than sample-level interpolation.

Baselines use masked statistical features with majority, logistic-regression,
and random-forest classifiers. The primary XROCKET experiment fits encoder
thresholds on training folds only, extracts 9,072 PPV features per fold, and
trains a random forest. A logistic-regression model on the same transformed
features is retained as a sensitivity check.

The XROCKET implementation is pinned to `dida-do/xrocket` commit
`1511e810c59d0c42f6431ef2f1f9fa57c71e9b2f` [@xrocket_code]. The public
upstream repository had no public license file during this project, so this
report treats its use as course-authorized academic support and does not claim a
general open-source license for the upstream code.

# Performance Summary

The statistical random-forest baseline reached mean macro F1 0.909, while the
primary XROCKET random forest reached mean macro F1 0.883. This means the
selected XROCKET representation is not the strongest classifier in the
repository, but it is the main explainable model because it provides traceable
channel and dilation metadata.

| Model | Mean macro F1 | Mean balanced accuracy | Notes |
|---|---:|---:|---|
| Majority baseline | 0.048 | 0.167 | Sanity baseline |
| Statistical logistic regression | 0.886 | 0.888 | Masked statistical features |
| Statistical random forest | 0.909 | 0.908 | Strongest baseline |
| XROCKET random forest | 0.883 | 0.883 | Primary explainable model |
| XROCKET logistic regression | 0.884 | 0.883 | Sensitivity model |

The primary XROCKET random forest was weakest for labels 2 and 3, with recalls
0.750 and 0.775. These are the hammering inward block and extended outward
block, and their mutual confusion is discussed as a limitation.

# Task 1.1 - Sensor-Axis Contribution

## Answer to Task 1.1

The strongest stable sensor-axis evidence comes from the gravity sensor family
and the pooled device-frame z axis. Native feature importance ranked gravity
first among sensor families with mean normalized importance 0.200, followed by
gyroscope at 0.175 and accelerometer at 0.168. The pooled z axis ranked first
with mean normalized importance 0.347. At the individual-channel level,
`gravity_z`, `gravity_x`, `gravity_y`, `gyros_x`, `game_rot_vec_z`, and
`accelerometer_z` were among the leading channels, but channel-level claims are
weaker because M7 showed top-channel instability across seed/fold cases.

![Task 1.1 sensor-family contribution. Gravity has the highest native
importance and is also supported by grouped permutation evidence.](figures/task_1_1_sensor_family_contribution.pdf)

![Task 1.1 device-frame channel and axis heatmap. Values are normalized within
folds and should be read as relative model-use evidence.](figures/task_1_1_axis_channel_contribution_heatmap.pdf)

![Task 1.1 class-specific channel profiles. These one-vs-rest diagnostics show
movement-specific channel emphasis but are secondary evidence.](figures/task_1_1_class_specific_channel_profiles.pdf)

Validation evidence is mixed but useful. Grouped test-set permutation produced
the largest sensor-family macro-F1 drop for gravity, followed by game rotation
vector and gyroscope. Feature-group ablation produced small or negative average
drops, which indicates redundancy in the transformed feature bank rather than a
simple one-sensor dependency.

![Task 1.1 ablation impact. Small and sometimes negative drops indicate
redundancy among transformed features.](figures/task_1_1_ablation_impact.pdf)

![Task 1.1 fold stability. Stable family-level findings are stronger than
individual-channel claims.](figures/task_1_1_fold_stability.pdf)

Biomechanically, gravity and rotation-related evidence is plausible for blocking
motions because the phone is worn on the forearm and the movements change
forearm orientation and acceleration. However, all axes are Android
device-frame axes. Without anatomical calibration, left/right mirroring checks,
or expert-labeled movement phases, this report avoids claims about exact joint
mechanics, muscle activation, force, or skill quality.

# Task 1.2 - Temporal-Scale Analysis

## Answer to Task 1.2

The saved padded XROCKET representation relies primarily on long-duration
patterns. With kernel length 9, the available dilations map to receptive-field
spans of 9, 17, 25, 33, 41, and 49 samples. Long spans, defined as 41 and 49
samples in the 56-sample padded window, contributed mean normalized importance
0.655. Intermediate spans contributed 0.224, and short spans contributed 0.122.
The leading dilation was 6, corresponding to a 49-sample receptive field.

![Task 1.2 dilation importance. Dilation 6 is the strongest contributor in the
saved padded XROCKET representation.](figures/task_1_2_dilation_importance.pdf)

![Task 1.2 temporal-scale contribution. Long spans dominate the normalized
importance profile.](figures/task_1_2_temporal_scale_contribution.pdf)

Approximate seconds are reported only as a nominal conversion. The KSAS app
requested Android `SENSOR_DELAY_GAME`, treated here as approximately 50 Hz, but
the CSV exports do not retain event timestamps. Android sensor delays are
requests rather than guaranteed exact acquisition intervals [@android_sensors].
Therefore, the primary temporal evidence is in samples: 41 and 49 samples, not
verified physical seconds.

![Task 1.2 class-specific temporal-scale profiles. Most classes show long-scale
dominance, with class 4 closest to mixed evidence.](figures/task_1_2_class_specific_scale_profiles.pdf)

![Task 1.2 fold stability. The long-scale conclusion is stable at the main
scale-bin level.](figures/task_1_2_fold_stability.pdf)

The result suggests that recognition depends more on broad movement structure
than on very local impulses. This is plausible for karate blocking gestures,
where the full arm trajectory and orientation transition may distinguish
classes. The caveat is important: XROCKET has no valid-timestep mask, so the
temporal explanation describes the padded representation. Padding diagnostics
showed prediction and feature sensitivity, so a mask-aware or timestamped
representation would be needed before making stronger time-domain claims.

# Task 1.3 - Discriminative-Pattern Interpretation

## Answer to Task 1.3

The most discriminative XROCKET patterns can be mapped to representative signal
intervals, but the localization is approximate. The selected features are PPV
features: each transformed value is the proportion of convolution responses
above a learned threshold. The interval shown in each case is therefore the
strongest representative above-threshold segment, not a unique causal instant.

The selected stable features were mostly long-span patterns at dilation 5 or 6.
The first cases involved `lin_accel_y`, `gravity_z`, and `lin_accel_z`, with
associated movement labels upward block and hammering inward block. Two cases
were labeled plausible from a human perspective, and two were labeled ambiguous.

![Task 1.3 pattern case 1. Representative correct case for a high-importance
PPV feature.](figures/task_1_3_pattern_case_01.pdf)

![Task 1.3 pattern case 2. A second correctly classified representative pattern
case.](figures/task_1_3_pattern_case_02.pdf)

![Task 1.3 pattern case 3. A correctly classified case with more ambiguous human
meaning.](figures/task_1_3_pattern_case_03.pdf)

![Task 1.3 failure or ambiguous case. Retaining this case makes the explanation
limits visible.](figures/task_1_3_pattern_case_failure_or_ambiguous.pdf)

![Task 1.3 feature distributions. Class separation supports the selected
patterns but does not prove causation.](figures/task_1_3_pattern_feature_distributions.pdf)

![Task 1.3 pattern summary table. Human-meaningfulness labels are cautious
report aids, not expert biomechanical validation.](figures/task_1_3_pattern_summary_table.pdf)

The patterns appear meaningful as sensor-coordinate evidence of broad movement
shape, orientation change, and acceleration structure. They could support a
future human-in-the-loop review workflow by highlighting signal regions used by
the model. They do not validate a coaching system, performance scoring,
expertise assessment, force estimate, or learning-gain claim.

# Robustness, Confounds, And Problems Encountered

M7 controls found no unresolved evidence of participant leakage or obvious label
leakage. Label-shuffle controls stayed low: the strongest mean macro F1 was
0.1605 and the maximum fold/seed macro F1 was 0.3151. Metadata-only controls
using sequence length, padding fraction, and arm code also stayed low, with the
strongest mean macro F1 0.2158.

The main problems encountered were:

- raw CSVs had no timestamps, so realized sampling rate and jitter are unknown;
- sequence lengths varied, requiring right-padding to 56 samples;
- XROCKET has no valid-timestep mask, so padding affects some predictions and
  temporal explanations;
- feature metadata required a local adapter and tests because flattened feature
  order had to be aligned with saved feature columns;
- channel-level rankings were less stable than family, axis, and temporal-scale
  rankings;
- labels 2 and 3 were the weakest classes and are mutually confused;
- Quarto was not installed locally, so the final report uses Pandoc with
  MiKTeX/pdflatex instead of a Quarto pipeline.

# Limitations And Threats To Validity

The dataset is small: 240 samples from 20 participants. Results are descriptive
and assignment-focused, not a clinical, coaching, diagnostic, or
injury-prevention validation. Participant-independent folds reduce leakage, but
they do not prove generalization beyond the KSAS protocol.

Axis interpretation is limited by phone placement and device coordinates. The
phone was intended to be worn along the forearm, but participant-specific
placement and orientation consistency were not available. Arm-stratified checks
found that the top pooled axis differs by arm code, so pooled z-axis evidence
must be read with an arm-orientation caveat.

Temporal interpretation is limited by missing timestamps and padding. Receptive
fields are reported in samples first, with approximate nominal-50-Hz seconds
only as context. Pattern interpretations are heuristic and must not be treated
as validated biomechanics.

# Reproducibility And Repository

Repository URL: <https://github.com/Forest904/ksas-xrocket-hmc.git>

Submission identifier: Git tag `v1.0-submission` on branch `main`.

The canonical Python environment is `pyproject.toml` plus `uv.lock` targeting
Python 3.11. Raw KSAS data must be placed locally under
`data/raw/KSAS-Dataset/`; raw participant CSVs are ignored by Git.

The compact final checks are:

```bash
uv sync --locked
make reproduce
make figures
make report
pdftoppm -png reports/ksas_xrocket_hmc_report.pdf tmp/pdfs/ksas_report
```

The full artifact rebuild sequence is documented in `docs/reproducibility.md`.
Some commands require `--overwrite` if rerunning over existing result
directories. The committed `results/` directory contains the final artifacts
used by this report.

No new repository license is added in this submission. Original project code is
provided for academic assignment review, raw KSAS data are excluded, and
XROCKET is attributed as pinned course-authorized upstream software without a
public upstream license file.

# Generative-AI Disclosure

This Generative-AI disclosure covers retained repository, software, and report
work. Codex was used between 9 June and 9 July 2026 to support repository
planning, software implementation, debugging, test design, documentation
updates, report planning, and language editing. Generated code and prose
retained in the project were reviewed, executed, tested, and revised by the
author. Raw participant data, secrets, and personally identifying information
were not supplied to generative-AI tools. The author retained responsibility for
final decisions, interpretation, verification, and submission.

# Conclusion

This project produced a reproducible XROCKET analysis and final PDF report for
the KSAS inertial dataset. The main scientific conclusion is cautious: the model
uses gravity and device-frame orientation/acceleration structure, relies most
strongly on long spans in the padded XROCKET representation, and exposes
representative pattern intervals that can guide expert review. The evidence is
useful for explainable human motion computing, but it remains sensor-coordinate
and protocol-specific rather than validated biomechanical or pedagogical truth.

# References
