# Biomechanics Notes

This document separates verified protocol facts, cautious biomechanical
interpretations, speculative ideas, and unsupported claims.

## Verified Protocol Facts

- KSAS records executions from both left and right arms.
- The smartphone is worn on the executing forearm using a runner-style band.
- The protocol image places the screen outward and the phone's long axis along
  the forearm, with the top toward the hand.
- Channels remain in the Android device coordinate frame; no anatomical
  coordinate transformation is documented.

## Interpretation Rules

- Name important channels exactly, for example `gyros_z`, before proposing a
  physical interpretation.
- Treat mapping from device axes to anatomical directions as a
  protocol-supported inference, not a directly measured anatomical frame.
- Compare or stratify by execution arm because mirroring can reverse anatomical
  meaning while leaving device-axis names unchanged.
- Treat XROCKET importance as predictive association, not biomechanical
  causation.
- Report temporal spans in samples first and approximate seconds at nominal
  50 Hz second.

## Evidence Sources

- Lecture slides 22-23 and 33-35 in `Slides_HMC_Santos2026.pdf`.
- KSAS application instructions and bundled `wearable.jpg`.
- KSAS Android sensor-registration and export source.
- Dataset README and the local M1 audit.

## M4 Task 1.1 Interpretation

The M4 sensor-axis evidence is stored in `results/explanations/task_1_1/`.
Native random-forest importance ranks gravity highest among sensor families
(`0.2000 +/- 0.0103` normalized fold importance), followed by gyroscope
(`0.1748 +/- 0.0048`), accelerometer (`0.1676 +/- 0.0036`), game rotation
vector (`0.1661 +/- 0.0037`), magnetic field (`0.1535 +/- 0.0049`), and linear
acceleration (`0.1379 +/- 0.0057`). The device-frame `z` axis is highest
(`0.3472 +/- 0.0082`), with `y` (`0.3292 +/- 0.0073`) and `x`
(`0.3236 +/- 0.0027`) close behind.

The strongest native channels are `gravity_z`, `gravity_x`, `gravity_y`,
`gyros_x`, `game_rot_vec_z`, and `accelerometer_z`. This suggests that the
classifier relies heavily on forearm orientation relative to gravity, angular
velocity, orientation-related rotation-vector structure, and acceleration
patterns. In movement terms, these are plausible correlates of block direction,
forearm rotation, and end-position or transition differences between the Kenpo
blocks.

This is an evidence-supported interpretation, not a direct anatomical
measurement. The axes remain Android device-frame axes; the intended forearm
mount supports cautious interpretation, but no participant-specific anatomical
calibration was recorded. Because both arms are included, left/right mirroring
can change anatomical meaning even when the device-axis channel name is the
same. The report should therefore phrase these findings as device-channel
contributions and use arm-stratified diagnostics before promoting any
arm-specific biomechanical claim.

Validation is mixed. Grouped permutation supports gravity as the largest
sensor-family drop and also gives positive evidence for game rotation vector,
gyroscope, accelerometer, magnetic field, and linear acceleration. Feature-group
ablation, however, shows very small or negative mean macro-F1 drops, suggesting
that other transformed features can compensate when a group is removed and the
random forest is retrained. The safest biomechanical statement is therefore:
gravity and related orientation/rotation channels are consistently associated
with classification, while ablation does not prove that any single raw sensor
family is strictly necessary.

## M5 Task 1.2 Interpretation

The M5 temporal-scale evidence is stored in
`results/explanations/task_1_2/`. It classifies the saved padded XROCKET
representation as long-scale: long receptive fields contribute `0.6545` mean
normalized native importance, compared with `0.2237` for intermediate spans
and `0.1218` for short spans. The strongest dilations are 6 and 5, with
effective spans of 49 and 41 samples, approximately 0.98 and 0.82 seconds at
nominal 50 Hz.

Biomechanically, this suggests that the classifier relies more on movement
structure across much of each short KSAS execution than on isolated local
sample-to-sample events. Plausible human-motion correlates include the overall
block trajectory, transition timing, orientation evolution, and end-position
organization. This should still be phrased as a device-signal interpretation:
dilations identify temporal span in the transformed XROCKET representation,
not a directly validated motor-control timescale.

Short spans are not absent; they retain about `0.1218` normalized importance.
However, the dominant evidence is long-duration/low-frequency-like in the
padded model. The approximate seconds should be treated only as nominal
context because the dataset does not retain measured Android sensor timing.
