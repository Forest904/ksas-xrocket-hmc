# Task 1.2 Answer: Temporal-Scale Analysis

## Direct Answer

The Task 1.2 evidence classifies the saved padded XROCKET representation as
**long-scale**. Importance is normalized within each participant-held-out
fold before aggregation. Dilation is interpreted as an effective temporal span,
not as a Fourier frequency.

## Temporal Span Mapping

| dilation | kernel_length | effective_receptive_field_samples | effective_receptive_field_seconds_nominal | relative_span_of_padded_window | temporal_scale_bin |
| -------- | ------------- | --------------------------------- | ----------------------------------------- | ------------------------------ | ------------------ |
| 1        | 9             | 9                                 | 0.1800                                    | 0.1607                         | short              |
| 2        | 9             | 17                                | 0.3400                                    | 0.3036                         | short              |
| 3        | 9             | 25                                | 0.5000                                    | 0.4464                         | intermediate       |
| 4        | 9             | 33                                | 0.6600                                    | 0.5893                         | intermediate       |
| 5        | 9             | 41                                | 0.8200                                    | 0.7321                         | long               |
| 6        | 9             | 49                                | 0.9800                                    | 0.8750                         | long               |

The seconds column is approximate at nominal 50 Hz. The KSAS Android app
requested this rate, but realized sensor timing and jitter were not retained in
the CSV exports.

## Native Importance Evidence

Dilation ranking:

| group_value | importance_mean | importance_std | mean_rank |
| ----------- | --------------- | -------------- | --------- |
| 6           | 0.4051          | 0.0059         | 1.0000    |
| 5           | 0.2494          | 0.0059         | 2.0000    |
| 4           | 0.1440          | 0.0039         | 3.0000    |
| 3           | 0.0797          | 0.0069         | 4.0000    |
| 1           | 0.0620          | 0.0024         | 5.4000    |
| 2           | 0.0598          | 0.0036         | 5.6000    |

Temporal-scale ranking:

| group_value  | importance_mean | importance_std | mean_rank |
| ------------ | --------------- | -------------- | --------- |
| long         | 0.6545          | 0.0103         | 1.0000    |
| intermediate | 0.2237          | 0.0083         | 2.0000    |
| short        | 0.1218          | 0.0053         | 3.0000    |

## Class-Specific Profiles

Secondary one-vs-rest temporal-scale profiles:

| class_label | classification                                            | largest_scale | largest_importance |
| ----------- | --------------------------------------------------------- | ------------- | ------------------ |
| 0           | long-scale                                                | long          | 0.9509             |
| 1           | long-scale                                                | long          | 0.6065             |
| 2           | long-scale                                                | long          | 0.5322             |
| 3           | long-scale                                                | long          | 0.5298             |
| 4           | mixed temporal-scale evidence; largest contribution: long | long          | 0.5116             |
| 5           | long-scale                                                | long          | 0.6968             |

## Padding Caveat

Existing M3 padding diagnostics are retained as a guardrail: across the available temporal diagnostic rows, the mean absolute feature delta was 0.1274, and the mean zero-threshold fraction was 0.3160. This supports reporting temporal-scale evidence with a padded-representation caveat.

These temporal claims are traceable to saved XROCKET metadata:
`dilation`, `kernel_length`, and `effective_receptive_field_samples`. They
should still be read as evidence about the padded M3 feature representation,
not as timing-verified claims about the original Android sensor stream.
