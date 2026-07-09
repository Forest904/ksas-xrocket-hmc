# Task 1.3 Answer: Discriminative-Pattern Interpretation

## Direct Answer

The most discriminative saved XROCKET patterns can be linked to representative
signal intervals, but the localization is approximate. The selected features
are PPV features, so each transformed value is the fraction of response
positions above a fitted threshold. The reported interval is therefore the
strongest representative above-threshold segment, not a unique causal instant.

## Selection Rule

Features were selected by stable random-forest native importance across the
participant-held-out folds: importance was normalized within fold, features had
to be nonzero in every fold, and ties were resolved by higher mean importance,
lower fold variation, then lower feature index.

Top selected patterns:

| pattern_rank | feature_index | channel_name   | dilation | mean_normalized_importance | class_separation_auc | associated_class_label |
| ------------ | ------------- | -------------- | -------- | -------------------------- | -------------------- | ---------------------- |
| 1            | 8164          | lin_accel_y    | 6        | 0.0032                     | 0.6821               | upward block           |
| 2            | 8429          | gravity_z      | 6        | 0.0032                     | 0.9714               | upward block           |
| 3            | 7931          | lin_accel_z    | 6        | 0.0031                     | 0.8220               | hammering inward block |
| 4            | 6206          | game_rot_vec_z | 5        | 0.0030                     | 0.9685               | outward downward block |
| 5            | 8463          | gravity_x      | 6        | 0.0030                     | 0.8904               | outward downward block |
| 6            | 8405          | magn_field_z   | 6        | 0.0030                     | 0.8932               | rear elbow block       |

## Case Studies

| case_id                           | case_type            | movement_label         | channel_name | feature_value_ppv | human_meaningfulness |
| --------------------------------- | -------------------- | ---------------------- | ------------ | ----------------- | -------------------- |
| pattern_case_01                   | correct              | upward block           | lin_accel_y  | 0.5536            | plausible            |
| pattern_case_02                   | correct              | upward block           | gravity_z    | 0.6607            | plausible            |
| pattern_case_03                   | correct              | hammering inward block | lin_accel_z  | 0.6429            | ambiguous            |
| pattern_case_failure_or_ambiguous | failure_or_ambiguous | hammering inward block | lin_accel_z  | 0.6250            | ambiguous            |

Human-meaningfulness labels across these cases were: clear=0,
plausible=2, ambiguous=2, not meaningful=0.

## Interpretation

The documented cases support cautious sensor-coordinate interpretations such as
broad movement shape, transition timing, sustained orientation structure,
device-frame acceleration, angular velocity, or magnetic-field variation. They
do not validate a coaching system, expertise assessment, force estimate, joint
mechanics claim, or learning-gain claim. A future learning or performance
assessment system could use this style of explanation to identify which signal
regions a model used and to guide expert review, but that would require separate
validation with pedagogical or biomechanical ground truth.

## Limitation

The analysis inherits the M3 padded-representation caveat. Several important
features use long spans and may overlap right-padding or convolution edge
padding. Those cases are retained because they clarify model behavior, but their
human meaning is marked as ambiguous or not meaningful when padding dominates.
