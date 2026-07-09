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

The M2 preprocessing and baseline workflow is:

```bash
uv run hmc prepare --config configs/preprocessing_m2_raw_padded.yaml
uv run hmc baseline --config configs/baseline_m2_raw_padded.yaml
```

`hmc prepare` writes padded tensors and participant-grouped split manifests.
`hmc baseline` trains the majority, statistical logistic-regression, and
statistical random-forest baselines on those exact folds. Both commands save
resolved configs and provenance for the generated artifacts.

M3 and later runs must additionally save the XROCKET Git revision, encoder
hyperparameters, ordered feature metadata, threshold-fitting fold, PyTorch
version, device, runtime, and padding strategy.
