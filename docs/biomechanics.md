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
