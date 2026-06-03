# Contributing

Bug fixes and documentation improvements are welcome. For new **tasks**, MDP changes, or paper reproductions, [open an issue](https://github.com/simeon-ned/wbc_mjlab/issues) first—we keep a shared MDP in `env/` and express paper differences as registered tasks (see README).

Planned themes: [docs/ROADMAP.md](docs/ROADMAP.md).

## Setup

```bash
git clone https://github.com/simeon-ned/wbc_mjlab.git && cd wbc_mjlab
pip install -e .
wbc-mjlab-list-envs
```

Requires [mjlab](https://github.com/mujocolab/mjlab) and a CUDA-capable PyTorch build for GPU training. **uv**, Makefile, and CI checks are planned (ROADMAP); until then use `pip install -e .` and ruff locally.

## Pull requests

1. Fork, branch from `main`, keep PRs focused.
2. Format and lint: `ruff format src/` · `ruff check src/ --fix`
3. Smoke-test what you touched (`wbc-mjlab-list-envs`, short train/play on a small dataset).
4. Link the issue: `Closes #N`.

No full test suite yet—PRs should at least pass ruff and the smoke steps above.

## Adding a paper task

1. Env builder: `robots/<id>/configs/<method>.py`
2. Register: `WbcTaskConfig` in `robots/<id>/configs/__init__.py`
3. Neutral names in code; cite the paper in the task description / docstring
4. Update README paper ↔ task table
5. Example command: `wbc-mjlab-train --task Wbc-... --dataset ...`

## Help

- **Issues:** https://github.com/simeon-ned/wbc_mjlab/issues
- **mjlab:** https://github.com/mujocolab/mjlab/issues

## License

Contributions are Apache 2.0 (see [LICENSE](LICENSE)).
