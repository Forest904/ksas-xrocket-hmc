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
