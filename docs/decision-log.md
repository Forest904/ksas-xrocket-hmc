# Decision Log

| Date | Decision | Alternatives | Rationale | Evidence | Consequences | Revisit condition |
|---|---|---|---|---|---|---|
| 2026-06-09 | Use `pyproject.toml` plus committed `uv.lock` as the canonical environment. | `pyproject.toml` without lockfile; another package manager. | Matches README, PRD, and reproducibility goals. | `docs/PRD.md` technical stack and M0 planning decision. | `uv sync --locked` becomes the CI setup path. | Revisit only if required dependencies cannot support Python 3.11 through `uv`. |
| 2026-06-09 | Keep M0 workflow commands as successful documented placeholders. | Fail-fast stubs; full CLI implementation. | Lets README commands resolve without starting data or model work too early. | M0 plan and roadmap objective. | Later milestones must replace placeholders with real implementations. | Revisit when each milestone begins implementation. |
| 2026-06-09 | Run lint, formatting, typing, tests, importability, and CLI smoke checks in CI. | Lint/tests only; import-only CI. | Matches PRD recommendation without requiring raw data. | `docs/PRD.md` CI scope and M0 planning decision. | CI stays data-free until later smoke fixtures exist. | Revisit when a synthetic pipeline smoke test is implemented. |
