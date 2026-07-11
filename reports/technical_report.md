---
title: "Explainable Human Motion Computing with KSAS and XROCKET"
author: "Luca Foresti"
date: "11 July 2026"
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

| Model                           | Mean macro F1 | Mean balanced accuracy | Notes                       |
| ------------------------------- | ------------: | ---------------------: | --------------------------- |
| Majority baseline               |         0.048 |                  0.167 | Sanity baseline             |
| Statistical logistic regression |         0.886 |                  0.888 | Masked statistical features |
| Statistical random forest       |         0.909 |                  0.908 | Strongest baseline          |
| XROCKET random forest           |         0.883 |                  0.883 | Primary explainable model   |
| XROCKET logistic regression     |         0.884 |                  0.883 | Sensitivity model           |

The primary XROCKET random forest was weakest for labels 2 and 3, with recalls
0.750 and 0.775. These are the hammering inward block and extended outward
block, and their mutual confusion is discussed as a limitation.

# Task 1.1 - Sensor-Axis Contribution

## Answer to Task 1.1

My answer is that the most useful signal group is gravity, and the strongest
pooled axis is the device-frame z axis. Gravity has mean normalized importance
0.200, ahead of gyroscope at 0.175 and accelerometer at 0.168. When I group all
sensors by axis, z is highest at 0.347. This does not mean that only one axis
matters, but it tells me where the model found the most stable information.

Figure 1 combines the two main pieces of evidence for this answer. On the left,
the bar chart ranks the sensor families, where gravity is clearly first. On the
right, the heatmap shows how each family splits across the x, y, and z
device-frame axes; the brightest gravity cell is on z. I use this panel for the
main answer because it shows both "which sensor" and "which axis" in one place.

\begin{figure}[H]
\centering
\includegraphics[width=0.96\textwidth]{reports/figures/task_1_1_core_evidence.pdf}
\caption{Task 1.1 core evidence. Gravity is the strongest sensor family, and the pooled device-frame z axis carries the strongest axis evidence.}
\end{figure}

Figure 2 is more detailed and should be read as a secondary diagnostic. It shows
that different movement classes emphasize different channels, but I do not use
it for the strongest claim because individual channel ranks were less stable
than the broader family and axis results.

\begin{figure}[H]
\centering
\includegraphics[width=0.95\textwidth]{reports/figures/task_1_1_class_specific_channel_profiles.pdf}
\caption{Task 1.1 class-specific channel profiles. These one-vs-rest diagnostics show movement-specific channel emphasis but are secondary evidence.}
\end{figure}

I checked the ranking against validation tests instead of trusting one score
only. The grouped permutation test again points to gravity as the most important
sensor family, followed by game rotation vector and gyroscope. The ablation
plot is less direct: removing one feature group often causes only a small drop,
or even a negative drop. I interpret this as redundancy. XROCKET creates many
related transformed features, so the model can sometimes compensate when one
group is removed.

Figure 3 puts the validation checks together. The ablation panel is a warning
against a simple "one sensor causes the classification" story. The stability
panel shows why I trust broad family and axis conclusions more than claims about
one exact channel such as `gravity_z`.

\begin{figure}[H]
\centering
\includegraphics[width=0.96\textwidth]{reports/figures/task_1_1_validation_checks.pdf}
\caption{Task 1.1 validation checks. Small ablation drops indicate feature redundancy, while fold stability shows that broad conclusions are safer than individual-channel claims.}
\end{figure}

My biomechanical interpretation is cautious. The result makes sense because a
phone on the forearm should capture changes in orientation, rotation, and
acceleration during karate blocks. Gravity and rotation-related channels are
therefore plausible sources of movement information. At the same time, these
are Android device-frame axes, not anatomical axes. Without calibration to the
body, left/right mirroring checks for every movement phase, or expert-labeled
biomechanical phases, I should not claim that the model has identified exact
joint mechanics, muscle activation, force, or skill quality.

\FloatBarrier

# Task 1.2 - Temporal-Scale Analysis

## Answer to Task 1.2

My answer is that the model relies mainly on long-duration movement structure,
not only on short impulses. With kernel length 9, the XROCKET dilations map to
spans of 9, 17, 25, 33, 41, and 49 samples. The long spans, 41 and 49 samples in
the padded 56-sample window, account for mean normalized importance 0.655. The
intermediate spans account for 0.224, and the short spans account for 0.122.
The single strongest dilation is 6, which corresponds to a 49-sample receptive
field.

Figure 4 combines the dilation and scale evidence. The left panel shows that
dilation 6 is much higher than the shorter dilations. The right panel simplifies
the same result into short, intermediate, and long bins: the long-scale bin
dominates. This supports the idea that XROCKET is recognizing the overall shape
of the block, not just a quick spike at one instant.

\begin{figure}[H]
\centering
\includegraphics[width=0.96\textwidth]{reports/figures/task_1_2_scale_evidence.pdf}
\caption{Task 1.2 scale evidence. Dilation 6 and the long temporal-scale bin dominate the saved padded XROCKET representation.}
\end{figure}

I keep the time interpretation in samples first. The KSAS app requested Android
`SENSOR_DELAY_GAME`, which I treat as about 50 Hz only for rough context, but
the exported CSV files do not contain timestamps. Android sensor delays are also
requests rather than guaranteed exact sampling intervals [@android_sensors].
For that reason, the strong evidence is "41 and 49 samples", not a verified
number of physical seconds.

Figure 5 adds two checks. The left panel shows that most classes have their
highest importance in the long-scale column. The right panel shows that the
dilation ranking is stable across folds, so the long-scale conclusion is not
coming from one unusual split.

\begin{figure}[H]
\centering
\includegraphics[width=0.96\textwidth]{reports/figures/task_1_2_class_and_stability.pdf}
\caption{Task 1.2 class and stability checks. Most classes rely most on long spans, and dilation-rank correlations stay high across folds.}
\end{figure}

My interpretation is that these karate blocks are distinguished by broad arm
trajectory and orientation changes. That fits the assignment question because
the temporal scale affects what the model can see: short spans would emphasize
brief impacts or local changes, while long spans capture a larger part of the
movement. The main caveat is padding. XROCKET does not use the valid-timestep
mask, and padding diagnostics showed that padding can affect some predictions
and features. A timestamped or mask-aware version would be needed before making
stronger claims about real physical timing.

\FloatBarrier

# Task 1.3 - Discriminative-Pattern Interpretation

## Answer to Task 1.3

My answer is that the selected XROCKET patterns are meaningful as cautious
sensor-coordinate explanations. They show broad movement shape, orientation
change, and acceleration structure. They do not prove a biomechanical cause, and
they are not enough to build a coaching or performance-scoring system by
themselves.

The important detail is how PPV features work. A PPV feature is the proportion
of convolution responses above a learned threshold. So the highlighted interval
in each case is the strongest representative interval for that pattern, not a
single exact moment that caused the prediction. Most selected patterns are
long-span patterns at dilation 5 or 6. The clearest examples involve
`lin_accel_y`, `gravity_z`, and `lin_accel_z`, mainly for upward block and
hammering inward block.

Figures 6 and 7 show the four pattern cases as compact cards. Each case card has
the raw same-family sensor traces on top, the XROCKET response under it, and a
compact metrics strip below. The orange bands show the representative interval
used for interpretation. Figure 6 contains the plausible upward-block examples;
Figure 7 keeps the ambiguous and failure cases visible so the explanation limits
are not hidden.

\begin{figure}[H]
\centering
\includegraphics[width=0.96\textwidth]{reports/figures/task_1_3_case_cards_plausible.pdf}
\caption{Task 1.3 plausible pattern case cards. Correct upward-block examples show long-span linear acceleration and gravity patterns used by XROCKET.}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.96\textwidth]{reports/figures/task_1_3_case_cards_ambiguous.pdf}
\caption{Task 1.3 ambiguous and failure-pattern checks. These cases keep the interpretation limits visible, especially when padding or misclassification affects the pattern.}
\end{figure}
\FloatBarrier

Figure 8 checks whether the selected features separate classes in the held-out
data and summarizes the four case studies. The boxplots support the selected
patterns, but they still do not prove causation. The summary table makes the
mixed result explicit: two cases are plausible and two are ambiguous. I treat
that as useful because explainability should also show where the model evidence
is hard to interpret.

\begin{figure}[H]
\centering
\includegraphics[width=0.96\textwidth]{reports/figures/task_1_3_distribution_summary.pdf}
\caption{Task 1.3 distribution and case summary. Feature distributions support the selected cases, while the table keeps the human-meaningfulness labels explicit.}
\end{figure}

My final interpretation is that XROCKET can help a human reviewer see which
parts of the sensor signal the model is using. That could support a future
human-in-the-loop learning tool, because an instructor or researcher could
inspect highlighted regions instead of only seeing a class label. However, this
project does not validate coaching feedback, expertise assessment, force
estimation, or learning gains. Those would require extra ground truth and
separate evaluation.

\FloatBarrier

# Robustness, Confounds, And Problems Encountered

In the leakage checks, I did not find signs that the reported
results were being driven by participant leakage or simple label leakage.
Label-shuffle controls stayed low: the strongest mean macro F1 was
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
- labels 2 and 3 were the weakest classes and are mutually confused.

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

Repository URL: [https://github.com/Forest904/ksas-xrocket-hmc.git](https://github.com/Forest904/ksas-xrocket-hmc.git)

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

# Generative-AI disclosure

I used Codex to support software setup, debugging, tests, report drafting, and
language editing. I reviewed and verified the retained code, results, and text.

# Conclusion

This project produced a reproducible XROCKET analysis and final PDF report for
the KSAS inertial dataset. The main scientific conclusion is cautious: the model
uses gravity and device-frame orientation/acceleration structure, relies most
strongly on long spans in the padded XROCKET representation, and exposes
representative pattern intervals that can guide expert review. The evidence is
useful for explainable human motion computing, but it remains sensor-coordinate
and protocol-specific rather than validated biomechanical or pedagogical truth.

# References
