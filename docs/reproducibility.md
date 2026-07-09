# Reproducibility

## Environment

The canonical Python environment is defined by `pyproject.toml` and `uv.lock`.

```bash
uv sync
```

The project targets Python 3.11. Quarto and LaTeX/TinyTeX are expected system tools for final report rendering and are not installed as Python dependencies.

XROCKET is installed directly from `dida-do/xrocket` at immutable Git commit
`1511e810c59d0c42f6431ef2f1f9fa57c71e9b2f`. `pyproject.toml` supplies corrected
dependency metadata because upstream version 0.1 declares the obsolete package
name `pytorch`; the imported runtime package is `torch`.

Verify the locked encoder:

```bash
uv sync --locked
uv run python -c "from xrocket.encoder import XRocket; print(XRocket)"
```

## Data

Raw KSAS data must remain immutable and untracked. Local raw files belong under `data/raw/KSAS-Dataset/` after the M1 data-placement step.

## Workflow

The M2 preprocessing, baseline, and M3 model workflow is:

```bash
uv run hmc prepare --config configs/preprocessing_m2_raw_padded.yaml
uv run hmc baseline --config configs/baseline_m2_raw_padded.yaml
uv run hmc train --config configs/experiments/m3_xrocket_raw_padded.yaml
```

`hmc prepare` writes padded tensors and participant-grouped split manifests.
`hmc baseline` trains the statistical baselines on those exact folds. `hmc
train` fits a separate XROCKET encoder on each training fold and saves features,
metadata, models, metrics, predictions, confusion matrices, runtime, provenance,
and padding diagnostics. It refuses to replace a non-empty output directory
unless `--overwrite` is passed explicitly.

Each M3 fold directory contains the fitted adapter and classifiers. The runner
reloads them before completing and verifies that features and smoke-sample
predictions are unchanged.
