# Confusion And Per-Class Error Review

Aggregated primary-model per-class errors:

| class_label | support | recall | precision | top_confused_as | top_confused_count |
| ----------- | ------- | ------ | --------- | --------------- | ------------------ |
| 0           | 40      | 1.0000 | 0.9756    |                 | 0                  |
| 1           | 40      | 0.9250 | 0.9737    | 3               | 1                  |
| 2           | 40      | 0.7500 | 0.7895    | 3               | 5                  |
| 3           | 40      | 0.7750 | 0.7750    | 2               | 2                  |
| 4           | 40      | 0.9250 | 0.9250    | 5               | 2                  |
| 5           | 40      | 0.9250 | 0.8605    | 2               | 2                  |

Classes with lower recall should be treated as movement-specific weaknesses in
the final report rather than averaged away by the overall macro score.
