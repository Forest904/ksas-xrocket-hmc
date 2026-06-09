# Limitations

Placeholder for limitations discovered during implementation.

Initial M0 limitations:

- XROCKET implementation selection, licensing, and metadata access are deferred to M3.
- README workflow commands are placeholders until the relevant milestones implement them.
- Quarto and LaTeX/TinyTeX availability has not yet been validated.

M1 data limitations:

- Raw KSAS data are audited locally under `data/raw/KSAS-Dataset/`, but the raw CSVs
  remain ignored and are not redistributed by this repository.
- The KSAS CSVs do not include timestamps.
- Sampling frequency is unknown after README and CSV inspection, so time spans must
  be reported in samples unless later evidence establishes a defensible Hz value.
- Phone placement and device orientation are not fully specified, so axis-level
  biomechanical interpretation must be cautious.
- The dataset README does not fully specify whether any preprocessing occurred
  before the published per-movement CSV files.
