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
