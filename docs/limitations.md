# Limitations

Initial M0 limitations:

- The selected XROCKET repository is authorized for this course project and is
  pinned by commit, but the public upstream repository contains no license
  file. The project records attribution and does not claim a general
  open-source license for that code.
- Explanation, figure, and report workflow commands remain placeholders until
  their relevant milestones implement them.
- Quarto and LaTeX/TinyTeX availability has not yet been validated.

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
