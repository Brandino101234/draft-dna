.PHONY: setup lint typecheck test dq check data transform models report refresh

setup:  ## Install the environment
	uv sync

lint:
	uv run ruff check .
	uv run ruff format --check .

typecheck:
	uv run mypy

test:  ## Unit tests (no network, no built database)
	uv run pytest

dq:  ## Data-quality tests against the built database
	uv run pytest -m dq

check: lint typecheck test

data:  ## Rebuild all data from source (reuses cached raw pulls)
	uv run draft-dna build

transform:  ## Rebuild staging/modeled/DuckDB from cached raw data (no network)
	uv run draft-dna transform

models:  ## Phase 3: backtest all models and build projections + comps
	uv run draft-dna backtest
	uv run draft-dna project

report:  ## Regenerate all reports/ charts and tables
	uv run draft-dna report

refresh:  ## In-season update: new games, regrade, regenerate cards
	uv run draft-dna refresh
