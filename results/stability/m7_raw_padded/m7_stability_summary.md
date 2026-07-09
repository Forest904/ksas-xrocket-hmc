# M7 Stability Summary

## Metric Stability

Retraining classifiers on saved XROCKET features across seeds produced primary
mean macro F1 `0.8842` with standard
deviation `0.0675` across seed/fold
cases.

## Explanation Stability

Top seed-stability groups: sensor family `gravity`, axis `z`, and
dilation `6`. These summarize random-forest feature importance after
refitting only the classifier layer on the saved XROCKET representation.

## Arm-Stratified Axis Check

| status    | arm_code | top_axis | top_axis_importance_mean | axis_rank_std_max | arm_top_axis_conflict |
| --------- | -------- | -------- | ------------------------ | ----------------- | --------------------- |
| completed | d        | x        | 0.3391                   | 0.9798            | True                  |
| completed | i        | z        | 0.3489                   | 0.0000            | True                  |

## Per-Class Weaknesses

| class_label | recall | precision | top_confused_as | top_confused_count |
| ----------- | ------ | --------- | --------------- | ------------------ |
| 2           | 0.7500 | 0.7895    | 3               | 5                  |
| 3           | 0.7750 | 0.7750    | 2               | 2                  |

## Triggered Flags

| flag                       | observed                                  | threshold                                    | message                                                |
| -------------------------- | ----------------------------------------- | -------------------------------------------- | ------------------------------------------------------ |
| rank_correlation_min       | 0.5                                       | 0.75                                         | Seed/fold explanation rank correlation below threshold |
| channel_top_rank_changed   | True                                      | top group should be stable for strong claims | Top channel changed across seed/fold cases             |
| movement_specific_weakness | class_2_recall_0.750;class_3_recall_0.775 | 0.8                                          | At least one movement class has low recall             |
