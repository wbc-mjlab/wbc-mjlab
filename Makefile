.PHONY: sync sync-cpu format lint smoke check test test-fast build publish publish-test

sync:
	uv sync --extra cu128 --group dev

sync-cpu:
	uv sync --extra cpu --group dev

format:
	uv run ruff format
	uv run ruff check --fix

lint:
	uv run ruff format --check
	uv run ruff check

smoke:
	uv run wbc-mjlab-list-envs

check: lint smoke

test:
	uv run pytest

test-fast:
	uv run pytest -m "not slow"

build:
	uv build
	uv run --isolated --no-project --with dist/*.whl tests/smoke_test.py
	uv run --isolated --no-project --with dist/*.tar.gz tests/smoke_test.py
	@echo "Build and import test successful"

publish-test: build
	uv publish --publish-url https://test.pypi.org/legacy/

publish: build
	uv publish
