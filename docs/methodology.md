# Methodology

This document records the modeling workflow and decisions that affect the
interpretation of the KSAS XROCKET project.

## M2 Preprocessing

The M2 primary target is six-class movement classification using
`movement_label_id` values `0` through `5`, including `0 = no movement`. This is
consistent with the audited class balance and the course KSAS setup.

`data/manifests/samples.csv` is the source of truth for sample order, labels,
participant groups, arm metadata, raw source paths, and channel order. The
preprocessing command reads the manifest in row order and writes model-ready
tensors under `data/processed/ksas_m2_raw_padded/`. It verifies each raw CSV
checksum against the manifest before tensor creation.

The primary tensor contract is:

- axis order: sample, channel, time;
- shape: 240 samples, 18 channels, 56 timesteps for the audited full dataset;
- dtype: `float32`;
- channel order: the audited `EXPECTED_CHANNELS` order from `hmc audit`;
- sequence order: raw CSV row order;
- padding: right-pad shorter sequences with `0.0`;
- valid timesteps: stored in `valid_mask`;
- labels and metadata: stored separately from the signal tensor.

M2 deliberately does not smooth, normalize, derive channels, filter samples, or
truncate sequences. Statistical baselines ignore padded timesteps through the
saved mask.

## M2 Splits And Baselines

Participant-safe folds are generated with `StratifiedGroupKFold`, five splits,
shuffle enabled, and random seed `42`. The group is `split_group`, which equals
the pseudonymous participant ID. Every fold must have zero participant overlap
between train and test and complete class coverage.
Baseline training validates that saved split rows still align with tensor sample
indices, labels, participants, and arms before fitting any model.

M2 baselines use the exact saved grouped folds:

- majority-class predictor;
- masked statistical features with logistic regression;
- masked statistical features with random forest.

Saved metrics include macro F1, balanced accuracy, per-class precision, recall,
F1, support, predictions, confusion matrices, and provenance.

## M3 XROCKET Experiment

The primary encoder is the course-authorized `dida-do/xrocket` implementation,
pinned in `pyproject.toml` and `uv.lock` to Git commit
`1511e810c59d0c42f6431ef2f1f9fa57c71e9b2f`. The original ROCKET repository is
not required. XROCKET accepts tensors in sample, channel, time order.

The completed primary experiment uses:

- 18 input channels;
- `combination_order=1` and additive channel mixing;
- kernel length 9;
- maximum kernel span 56 and feature cap 10,000;
- the existing participant-grouped folds;
- explicit threshold fitting on training-fold data only; and
- a 500-tree balanced random forest plus standardized logistic-regression
  classifier sensitivity.

The fitted bank contains 84 patterns, six dilations (`1` through `6`), 18
single-channel combinations, one threshold, and 9,072 PPV features. The adapter
persists one ordered metadata row per feature, including exact thresholds,
kernel and pattern IDs, device-channel names, dilation, per-side convolution
padding, and receptive-field spans.

The upstream `feature_names` property repeats pattern labels in an order that
does not match the encoder's kernel-major flattened output when multiple
channels are present. The local adapter therefore derives pattern order from
the convolution weights and flattening structure, and cross-checks feature
count, dilation, channel, and threshold order against upstream state.

Across five grouped folds, random forest achieved mean macro F1 `0.8829`
(`0.0675` population standard deviation) and logistic regression achieved
`0.8838` (`0.0699`). Total recorded experiment runtime was 35.58 seconds on the
development machine.

## M3 Padding Diagnostic

Only two sequences naturally contain 56 observations. For every fold, the
training-fitted encoder transformed both the primary right-padded sequence and
the same sequence cropped to its recorded original length. Mean absolute PPV
feature differences ranged from approximately `0.115` to `0.165` across
dilations. Predictions changed for 61 of 240 random-forest test cases (25.4%)
and 57 of 240 logistic-regression test cases (23.8%).

Threshold effects are also substantial: dilation 1 thresholds were exactly
zero in every fold-feature combination, while dilation 2 had an average zero
threshold fraction of about 87.8%. Therefore M3 performance is valid for the
declared padded representation, but later channel, dilation, and pattern
explanations must not be presented as padding-independent motion evidence.

## Temporal Interpretation Policy

The KSAS application requests `SENSOR_DELAY_GAME`, nominally 20 ms or 50 Hz.
Android does not guarantee that requested delay, and the dataset contains no
event timestamps. Temporal-scale outputs will therefore report:

1. dilation and effective receptive field in samples;
2. approximate seconds using nominal 50 Hz; and
3. an explicit warning that the realized sampling rate and jitter are unknown.

Dilation is not a Fourier frequency. Interpretive language will distinguish
short-duration/high-frequency-like patterns from
long-duration/low-frequency-like structure.

## M4 Task 1.1 Sensor-Axis Explanation

Task 1.1 is generated by:

```bash
uv run hmc explain --config configs/explanations/task_1_1_xrocket_raw_padded.yaml
```

The workflow reads the saved M3 XROCKET fold artifacts rather than refitting the
encoder. Random-forest native feature importance is normalized within each fold
to sum to one, then aggregated by sensor family, device-frame axis,
sensor-axis channel, family-axis channel, and channel combination. Because the
primary M3 run uses `combination_order=1`, channel combinations are identical
to individual channels for this milestone.

Fold uncertainty is reported from the five participant-held-out folds using
mean, population standard deviation, min, max, mean rank, rank standard
deviation, pairwise Spearman rank correlations, and top-k overlap. Validation
uses saved XROCKET feature groups:

- ablation retrains the random forest after removing each channel or sensor
  family feature group;
- permutation shuffles each channel or sensor family feature group on the
  test fold and measures signed macro-F1 and balanced-accuracy drops;
- negative validation drops are preserved as signed evidence, while positive
  drops are clipped and normalized only for cross-method ranking.

Class-specific profiles are produced with one-vs-rest random forests trained
on the same saved transformed features and grouped folds. These profiles are
secondary explanation artifacts; they do not replace the multiclass primary
random forest.

M4 generated `results/explanations/task_1_1/`. The native family ranking was
gravity, gyroscope, accelerometer, game rotation vector, magnetic field, and
linear acceleration. The device-frame `z` axis ranked highest, followed by `y`
and `x`. The top native channels were `gravity_z`, `gravity_x`, `gravity_y`,
`gyros_x`, `game_rot_vec_z`, and `accelerometer_z`. Grouped permutation
supported gravity as the largest sensor-family performance drop, while
feature-group ablation produced very small or negative mean drops, indicating
redundancy in the transformed feature bank.

## M5 Task 1.2 Temporal-Scale Explanation

Task 1.2 is generated by:

```bash
uv run hmc explain --config configs/explanations/task_1_2_xrocket_raw_padded.yaml
```

The workflow reads the saved M3 XROCKET fold artifacts and does not refit the
encoder. It validates each feature's temporal metadata with
`effective_span = 1 + dilation * (kernel_length - 1)`, normalizes
random-forest native feature importance within each fold, and aggregates by
dilation and by a predeclared temporal-scale bin.

Scale bins are defined before inspecting final importance rankings, using the
56-sample padded window:

- short: span / 56 <= 1/3, covering dilations 1-2 and spans 9-17 samples;
- intermediate: 1/3 < span / 56 <= 2/3, covering dilations 3-4 and spans
  25-33 samples;
- long: span / 56 > 2/3, covering dilations 5-6 and spans 41-49 samples.

Approximate seconds are shown only as nominal 50 Hz conversions. The Android
app requested this rate, but the CSV exports do not retain realized sensor
timestamps or jitter.

The report classification rule is deterministic: call the evidence
`short-scale` or `long-scale` only if that bin has mean normalized importance
of at least 0.50 and at least 1.5 times the next-largest bin. Otherwise report
mixed temporal-scale evidence and name the largest bin.

M5 generated `results/explanations/task_1_2/`. Native importance is strongly
long-scale in the saved padded representation: long spans average `0.6545`
normalized importance, intermediate spans `0.2237`, and short spans `0.1218`.
The top dilations are 6 and 5, corresponding to 49 and 41 samples, or
approximately 0.98 and 0.82 seconds at nominal 50 Hz. Dilation ranks are stable
across the five participant-held-out folds, with dilation 6 ranked first and
dilation 5 second in every fold.

## M6 Task 1.3 Discriminative-Pattern Explanation

Task 1.3 is generated by:

```bash
uv run hmc explain --config configs/explanations/task_1_3_xrocket_raw_padded.yaml
```

The workflow reads the saved M3 fold artifacts and does not refit XROCKET. It
selects candidate features by stable native random-forest importance: feature
importance is normalized within fold, candidates must have nonzero importance
in every available fold, and ties are resolved by higher mean normalized
importance, lower fold standard deviation, then lower feature index.

The selected XROCKET features are PPV features. For each case study, the
workflow reconstructs the per-position kernel response from the saved adapter
with `block.conv(x)` followed by `block.mix(...)`, verifies that the proportion
above the fitted threshold exactly matches the saved transformed feature value,
and localizes the contiguous above-threshold segment containing the maximum
response margin. This interval is a representative strongest response segment,
not a unique causal instant, because PPV pools over all response positions.

Kernel footprints are mapped from response index to processed sample index with
`response_index - padding_per_side + arange(kernel_length) * dilation`, clipped
to the 56-sample processed window and the recorded original sequence length.
The output records whether the localized footprint touches convolution edge
padding or right-padded timesteps.

M6 generated `results/explanations/task_1_3/`. The top stable features are
long-span PPV patterns, led by `lin_accel_y`, `gravity_z`, and `lin_accel_z` at
dilation 6. The documented case studies include three correctly classified
high-activation examples and one failure or ambiguous example. Human
meaningfulness is classified with a cautious rubric based on one-vs-rest
held-out AUC and padding overlap; the generated cases are plausible or
ambiguous rather than treated as validated coaching evidence.
