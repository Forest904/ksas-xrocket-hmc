# Limitations

Initial M0 limitations:

- The selected XROCKET repository is authorized for this course project and is
  pinned by commit, but the public upstream repository contains no license
  file. The project records attribution and does not claim a general
  open-source license for that code.
- README workflow commands are placeholders until the relevant milestones implement them.
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

M3 readiness limitations:

- XROCKET has no valid-timestep mask and uses convolutional zero-padding.
  Existing right-padding may influence thresholds and high-dilation features;
  this requires a padding/length sensitivity check.
- Runtime has not yet been benchmarked on the project machine. This controls
  experiment breadth but does not block the one-fold M3 prototype.
