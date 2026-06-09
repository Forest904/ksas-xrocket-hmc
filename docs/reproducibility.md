# Reproducibility

Placeholder for M0-M8.

## Environment

The canonical Python environment is defined by `pyproject.toml` and `uv.lock`.

```bash
uv sync
```

The project targets Python 3.11. Quarto and LaTeX/TinyTeX are expected system tools for final report rendering and are not installed as Python dependencies.

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
