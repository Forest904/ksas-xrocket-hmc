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

The M0 workflow commands exist as documented placeholders. Later milestones will replace them with real data, modeling, explanation, and report steps.
