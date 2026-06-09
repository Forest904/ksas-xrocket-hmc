# Data Directory

This directory holds local and generated data artifacts for the KSAS XROCKET HMC project.

- `raw/`: immutable local source data. Raw KSAS files are intentionally ignored by Git.
- `interim/`: intermediate validation or conversion outputs.
- `processed/`: model-ready tensors and derived datasets.
- `manifests/`: committed or reproducibly generated inventories, splits, and audit summaries.

Do not edit upstream raw CSV files in place. The root `KSAS-Dataset/` copy should remain untracked until a later milestone moves or copies it under `data/raw/KSAS-Dataset/`.
