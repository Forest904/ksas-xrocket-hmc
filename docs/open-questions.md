# Decision-Gate Status

This file tracks PRD decision gates. All gates needed to enter M3 now have an
operational decision. Remaining measurements are milestone work, not blockers.

| Gate | Question | Status | Decision and evidence | Required follow-through |
|---|---|---|---|---|
| DG-03 | What is the sampling frequency? | Resolved with measurement caveat | The KSAS app requests Android `SENSOR_DELAY_GAME`, nominally 20 ms or 50 Hz. CSVs and app exports retain no event timestamps, so realized rate and jitter cannot be verified. | Report samples as primary units and approximate seconds at nominal 50 Hz; do not claim measured Hz. |
| DG-04 | What are the coordinate conventions? | Resolved for analysis | Channels use Android device coordinates. The protocol image shows the phone screen outward, long axis along the forearm, top toward the hand. | Keep formal claims in device-frame axes and treat anatomical mapping as protocol-based inference. |
| DG-06 | Are both arms included and how is phone orientation defined? | Resolved for analysis | Both arms have complete coverage. App instructions attach the phone to the forearm in a runner-style band; the reference image defines the intended placement. | Preserve arm metadata and test whether mirrored placement changes axis rankings. |
| DG-08 | Which XROCKET implementation will be used? | Resolved | Use course-authorized `dida-do/xrocket`, pinned to commit `1511e810c59d0c42f6431ef2f1f9fa57c71e9b2f`. It exposes pattern, dilation, channels, and threshold metadata. The original ROCKET repository is not required. | Wrap the encoder locally, retain attribution, and record that upstream publishes no license file despite course permission. |
| DG-09 | What fixed sequence length is required? | Resolved for M2 | Raw sequence lengths range from 18 to 56 samples; M2 pads to 56 and saves masks/original lengths. | Revisit in M3 only if the selected XROCKET adapter requires another representation. |
| DG-10 | Which evaluation split preserves class coverage? | Resolved for M2 | Participant-arm-label coverage is complete for all 20 participants; M2 uses 5-fold stratified grouped splits. | Revisit if later sensitivity analyses need leave-one-participant-out or a fixed holdout. |
| DG-11 | How are multi-channel kernels attributed? | Resolved | Primary XROCKET uses `combination_order=1`, so every feature maps directly to one channel. Optional order-2 work reports combinations and equal-allocation marginal importance. | Encode the policy in the adapter and explanation tests. |
| DG-12 | Can all mandatory experiments run on available hardware? | Resolved for M3 entry | The dependency is CPU-capable and the dataset is small. Exact runtime is a measurement, not an entry gate. | Benchmark one fold in M3, then set batch size and experiment breadth from the measurement. |

Resolved in M1:

- DG-02: File inventory and labels validated by `hmc audit`.
- DG-05: Published CSVs are treated as already segmented movement instances.
