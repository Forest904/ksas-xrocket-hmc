.PHONY: install lint format-check typecheck test smoke data audit baseline train explain figures report reproduce

install:
	uv sync

lint:
	uv run ruff check .

format-check:
	uv run ruff format --check .

typecheck:
	uv run mypy src

test:
	uv run pytest

smoke:
	uv run python -c "import ksas_xrocket"
	uv run python -c "from xrocket.encoder import XRocket; print(XRocket)"
	uv run hmc --version
	uv run hmc prepare

data:
	uv run hmc prepare

audit:
	uv run hmc audit

baseline:
	uv run hmc baseline

train:
	uv run hmc train

explain:
	uv run hmc explain

figures:
	uv run hmc figures

report:
	uv run hmc report

reproduce: lint format-check typecheck test smoke
