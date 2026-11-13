.PHONY: sync sync-cpu format lint smoke check test test-fast build publish publish-test docs docs-watch docs-clean docker-build docker-up docker-shell docker-run docker-studio docker-stop

# Match bind-mount ownership on Linux; Compose uses these as image build arguments.
export WBC_UID ?= $(shell id -u)
export WBC_GID ?= $(shell id -g)
DOCKER_COMPOSE ?= docker compose
DOCKER_EXEC_FLAGS ?=
CMD ?= wbc-mjlab-list-envs

.PHONY: docker-gpus docker-gpu-list
docker-gpus:
	python3 src/wbc_mjlab/gpu.py --compose '$(DOCKER_COMPOSE)' $(if $(GPU_IDS),--set '$(GPU_IDS)',--select)

docker-gpu-list:
	python3 src/wbc_mjlab/gpu.py

docker-build:
	$(DOCKER_COMPOSE) build dev

docker-up:
	$(DOCKER_COMPOSE) up -d --wait --wait-timeout 1200 dev

docker-shell: docker-up
	$(DOCKER_COMPOSE) exec dev bash

docker-run: docker-up
	$(DOCKER_COMPOSE) exec $(DOCKER_EXEC_FLAGS) dev $(CMD)

docker-studio: docker-up
	$(DOCKER_COMPOSE) exec $(DOCKER_EXEC_FLAGS) dev wbc-mjlab-studio --host 0.0.0.0 --no-open-browser

docker-stop:
	$(DOCKER_COMPOSE) down

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
	MUJOCO_GL=disable uv run --isolated --no-project --with dist/*.whl tests/smoke_test.py
	MUJOCO_GL=disable uv run --isolated --no-project --with dist/*.tar.gz tests/smoke_test.py
	@echo "Build and import test successful"

publish-test: build
	uv publish --publish-url https://test.pypi.org/legacy/

publish: build
	uv publish

docs:
	uv run --group docs sphinx-build -j auto docs docs/_build

docs-watch:
	uv run --group docs sphinx-autobuild -j auto docs docs/_build --watch src

docs-clean:
	rm -rf docs/_build
