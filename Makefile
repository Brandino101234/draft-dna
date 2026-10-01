.PHONY: setup lint typecheck test dq check data refresh

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

refresh:  ## In-season update: new games, regrade, regenerate cards
	uv run draft-dna refresh
