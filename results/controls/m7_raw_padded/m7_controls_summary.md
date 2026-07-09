# M7 Controls Summary

## Negative Controls

The strongest label-shuffle control was `xrocket_logistic_regression_label_shuffle` with mean
macro F1 `0.1605` and max fold/seed macro
F1 `0.3151`.

The strongest metadata-only control was `metadata_random_forest` with mean
macro F1 `0.2158`.

## Confound Check

Mean original sequence length differs by up to `6.65` samples across
movement labels. Arm coverage is balanced in the audited dataset, and metadata
controls remain the formal check for whether length or arm can explain labels.

## Triggered Flags

| flag                       | observed                                  | threshold                                    | message                                                |
| -------------------------- | ----------------------------------------- | -------------------------------------------- | ------------------------------------------------------ |
| rank_correlation_min       | 0.5                                       | 0.75                                         | Seed/fold explanation rank correlation below threshold |
| channel_top_rank_changed   | True                                      | top group should be stable for strong claims | Top channel changed across seed/fold cases             |
| movement_specific_weakness | class_2_recall_0.750;class_3_recall_0.775 | 0.8                                          | At least one movement class has low recall             |
