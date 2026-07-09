# Task 1.1 Answer: Sensor-Axis Contribution

## Direct Answer

The M4 evidence indicates that the most useful signals are the top-ranked
sensor families and channels below. These values are normalized within each
participant-held-out fold before aggregation, so they describe relative model
use in the saved padded XROCKET representation rather than physical effect
sizes.

## Native Importance Evidence

Top sensor families:

| group_value   | importance_mean | importance_std | mean_rank |
| ------------- | --------------- | -------------- | --------- |
| gravity       | 0.2000          | 0.0103         | 1.0000    |
| gyros         | 0.1748          | 0.0048         | 2.0000    |
| accelerometer | 0.1676          | 0.0036         | 3.4000    |

Top device-frame axes:

| group_value | importance_mean | importance_std | mean_rank |
| ----------- | --------------- | -------------- | --------- |
| z           | 0.3472          | 0.0082         | 1.0000    |
| y           | 0.3292          | 0.0073         | 2.2000    |
| x           | 0.3236          | 0.0027         | 2.8000    |

Top sensor-axis channels:

| group_value     | importance_mean | importance_std | mean_rank |
| --------------- | --------------- | -------------- | --------- |
| gravity_z       | 0.0698          | 0.0030         | 1.0000    |
| gravity_x       | 0.0653          | 0.0017         | 3.8000    |
| gravity_y       | 0.0649          | 0.0064         | 3.8000    |
| gyros_x         | 0.0624          | 0.0046         | 5.2000    |
| game_rot_vec_z  | 0.0617          | 0.0016         | 5.2000    |
| accelerometer_z | 0.0616          | 0.0025         | 4.6000    |

## Validation Evidence

Largest sensor-family macro-F1 drops under feature-group ablation:

| group_value   | macro_f1_drop_mean | positive_macro_f1_drop_mean |
| ------------- | ------------------ | --------------------------- |
| lin_accel     | -0.0001            | 0.0042                      |
| accelerometer | -0.0122            | 0.0042                      |
| game_rot_vec  | -0.0007            | 0.0038                      |

Largest sensor-family macro-F1 drops under grouped test-set permutation:

| group_value  | macro_f1_drop_mean | positive_macro_f1_drop_mean |
| ------------ | ------------------ | --------------------------- |
| gravity      | 0.0311             | 0.0337                      |
| game_rot_vec | 0.0243             | 0.0250                      |
| gyros        | 0.0163             | 0.0211                      |

## Class-Specific Profiles

Top one-vs-rest channels per class:

| class_label | group_value     | importance_mean | importance_std |
| ----------- | --------------- | --------------- | -------------- |
| 0           | magn_field_z    | 0.0764          | 0.0102         |
| 0           | accelerometer_y | 0.0670          | 0.0081         |
| 1           | gravity_z       | 0.0891          | 0.0085         |
| 1           | magn_field_x    | 0.0839          | 0.0038         |
| 2           | gravity_x       | 0.0757          | 0.0106         |
| 2           | gravity_z       | 0.0680          | 0.0076         |
| 3           | gravity_x       | 0.0777          | 0.0059         |
| 3           | accelerometer_z | 0.0732          | 0.0066         |
| 4           | gravity_z       | 0.0879          | 0.0040         |
| 4           | gravity_y       | 0.0863          | 0.0141         |
| 5           | gravity_x       | 0.0948          | 0.0077         |
| 5           | accelerometer_z | 0.0830          | 0.0097         |

## Biomechanical Interpretation

The observed rankings support a device-frame sensor contribution claim, not a
direct causal biomechanics claim. Channels with high native importance and
validation support are plausible indicators of forearm acceleration, angular
velocity, orientation-related gravity/rotation structure, or magnetic-field
variation during the Kenpo blocks. Because the phone is mounted on the forearm
in an Android device coordinate frame and both arms are included, anatomical
meaning must be treated as a protocol-based inference and checked against
arm-stratified diagnostics before any strong left/right or flexion/rotation
claim is made.

## Agreement And Caveats

Native, ablation, and permutation evidence should be read together. Conflict
flags were raised for 15 channel or sensor-family rows, so claims
should emphasize groups that are consistently high across native importance and
at least one validation method. The analysis remains specific to the padded M3
XROCKET representation, and feature-group ablation validates transformed
feature reliance rather than proving raw-sensor necessity.
