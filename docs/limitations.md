# Limitations

Initial M0 limitations:

- The selected XROCKET repository is authorized for this course project and is
  pinned by commit, but the public upstream repository contains no license
  file. The project records attribution and does not claim a general
  open-source license for that code.
- Figure and report workflow commands are implemented for M8 through Pandoc and
  MiKTeX/pdflatex. Quarto remains out of the critical path because it was not
  installed in the final local environment.

M1 data limitations:

- Raw KSAS data are audited locally under `data/raw/KSAS-Dataset/`, but the raw CSVs
  remain ignored and are not redistributed by this repository.
- The KSAS CSVs do not include timestamps.
- The app requested a nominal 50 Hz sensor rate, but Android treats the request
  as a hint and no event timestamps were retained. Realized rate, jitter, and
  exact cross-sensor synchronization cannot be verified.
- The phone was intended to be worn screen-outward along the forearm, but
  participant-level placement consistency is unavailable. Axis-level
  interpretation remains device-frame based and must consider arm mirroring.
- The dataset README does not fully specify whether any preprocessing occurred
  before the published per-movement CSV files.

M3 limitations:

- XROCKET has no valid-timestep mask and uses convolutional zero-padding.
  Cropped-sequence diagnostics changed approximately 25.4% of random-forest
  predictions and 23.8% of logistic-regression predictions. Dilation 1
  thresholds were all zero and dilation 2 thresholds were predominantly zero.
  Primary metrics therefore describe the padded representation, and downstream
  explanations require an explicit padding caveat and sensitivity evidence.
- The upstream multichannel `feature_names` pattern ordering does not match its
  flattened feature order. The adapter reconstructs pattern order from encoder
  tensors; this local correction is tested but remains tied to the pinned
  upstream revision.
- Approximate seconds use nominal 50 Hz rather than measured timestamps.

M4 Task 1.1 limitations:

- Sensor-axis rankings describe the saved padded M3 XROCKET representation, not
  a padding-independent or raw-sensor-only model.
- Random-forest impurity importance can favor features with convenient split
  structure. M4 therefore compares native importance with feature-group
  ablation and grouped permutation, but those methods answer different
  questions.
- Feature-group ablation removes saved transformed features and retrains the
  random forest. It does not rerun XROCKET with raw sensors removed and should
  not be described as proof that a raw sensor is or is not physically necessary.
- M4 ablation produced very small or negative mean macro-F1 drops at the sensor
  family level, while permutation produced positive drops led by gravity. This
  conflict suggests redundancy and compensation in the transformed feature
  bank; claims should emphasize agreement between native importance and
  permutation, and treat ablation as a redundancy check.
- Class-specific profiles are one-vs-rest random-forest explanations fitted on
  transformed features. They are useful diagnostics, but they are not native
  class-specific explanations from the original multiclass random forest.
- Only five participant-held-out folds are available, so uncertainty estimates
  are descriptive rather than inferentially strong.
- All biomechanical interpretation remains device-frame based. No anatomical
  coordinate transform, participant-specific phone placement check, or
  arm-orientation calibration is available.

M5 Task 1.2 limitations:

- Dilation is interpreted as effective temporal span, not Fourier frequency.
  Labels such as short-duration/high-frequency-like and
  long-duration/low-frequency-like are descriptive analogies.
- Spans in seconds are approximate nominal-50-Hz conversions. Because Android
  event timestamps were not retained, realized sampling rate, jitter, and
  cross-sensor timing cannot be verified.
- Temporal-scale rankings describe the saved padded M3 XROCKET representation.
  They inherit the padding sensitivity documented in M3 rather than proving
  that the same dilation profile would hold for a mask-aware or resampled
  representation.
- The short/intermediate/long bins are defined from the finite 56-sample
  receptive-field mapping before inspecting final importance rankings. They
  are appropriate for this padded model but should be redefined if a later
  representation changes sequence length or available dilations.
- Class-specific temporal-scale profiles are one-vs-rest random-forest
  diagnostics fitted on transformed features. They are secondary evidence, not
  native class-specific explanations from the original multiclass model.

M6 Task 1.3 limitations:

- The selected XROCKET features are PPV features. A PPV value is a proportion
  over response positions, so the localized interval is only a representative
  strongest above-threshold segment rather than a unique causal instant.
- Important selected features are mostly long-span dilation-5 or dilation-6
  patterns. Their receptive fields can cover much of a short KSAS recording
  and may overlap right-padding or convolution edge padding.
- Human-meaningfulness labels are heuristic report aids based on feature
  separation and padding overlap. They are not expert biomechanical validation.
- Pattern interpretations remain in Android device coordinates and cannot
  establish muscle activation, joint mechanics, force, expertise, coaching
  quality, or learning gains.

M7 robustness limitations:

- M7 seed stability retrains classifiers on saved XROCKET features but does
  not rerun the XROCKET encoder or refit encoder thresholds across seeds. Full
  encoder variability remains future work.
- Label-shuffle controls stayed below the configured leakage thresholds: the
  strongest mean macro F1 was `0.1605`, with maximum fold/seed macro F1
  `0.3151`. This does not prove leakage is impossible, but it found no obvious
  label leakage in the saved grouped evaluation setup.
- Metadata controls using sequence length, padding fraction, and arm code
  stayed below the configured confound threshold: the strongest mean macro F1
  was `0.2158`. Sequence length differs by up to `6.65` samples across labels,
  so it remains a documented weak confound risk rather than an explanation for
  the main XROCKET result.
- Seed-stability checks found stable top sensor-family, pooled-axis, dilation,
  and temporal-scale rankings, but channel-level top ranks changed across
  seed/fold cases. The final report should make stronger claims at family,
  axis, and temporal-scale levels than at individual-channel level.
- Arm-stratified axis rankings differ: right-arm code `d` ranks device-frame
  `x` highest on average, while left-arm code `i` ranks `z` highest. Pooled
  axis claims must include an arm-orientation caveat.
- Primary random-forest per-class recall is weakest for labels `2` and `3`
  (`0.7500` and `0.7750`). These movement-specific weaknesses should be
  reported directly, especially because labels `2` and `3` are confused with
  each other.
