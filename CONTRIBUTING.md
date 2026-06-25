# Contributing

Bug fixes and documentation improvements are welcome. For new **tasks**, MDP changes, or paper reproductions, [open an issue](https://github.com/wbc-mjlab/wbc-mjlab/issues) first—we keep a shared MDP in `env/` and express paper differences as registered tasks (see README).

Planned themes: [docs/ROADMAP.md](docs/ROADMAP.md).

## Setup

```bash
git clone https://github.com/wbc-mjlab/wbc-mjlab.git && cd wbc-mjlab
make sync
uv run wbc-mjlab-list-envs
```

Uses [uv](https://docs.astral.sh/uv/) with a locked environment (`uv.lock`), matching
the [mjlab](https://github.com/mujocolab/mjlab) workflow:

- `make sync` — `uv sync --extra cu128 --group dev` (CUDA PyTorch)
- `make sync-cpu` — CPU-only / macOS evaluation

See [docs/INSTALLATION.md](docs/INSTALLATION.md) for pip, PyPI, and local mjlab
checkout options.

Optional: `uvx pre-commit install` after `make sync`.

## Pull requests

1. Fork, branch from `main`, keep PRs focused.
2. Run `make check` (lint + smoke) or at minimum `make format` and `make smoke`.
3. Smoke-test what you touched (short train/play on `--dataset samples` when relevant).
4. Link the issue: `Closes #N`.

## Adding a paper task

1. Env builder: `robots/<id>/configs/<method>.py`
2. Register: `WbcTaskConfig` in `robots/<id>/configs/__init__.py`
3. Neutral names in code; cite the paper in the task description / docstring
4. Update [docs/TASKS.md](docs/TASKS.md) paper ↔ task table
5. Example command: `uv run wbc-mjlab-train --task Wbc-... --dataset ...`

## Help

- **Issues:** https://github.com/wbc-mjlab/wbc-mjlab/issues
- **mjlab:** https://github.com/mujocolab/mjlab/issues

## License

Contributions are Apache 2.0 (see [LICENSE](LICENSE)).
