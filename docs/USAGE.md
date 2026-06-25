# Usage

## CLI commands

| Command | Purpose |
|---------|---------|
| `wbc-mjlab-train` | Train (wraps `mjlab.scripts.train`) |
| `wbc-mjlab-play` | Play / eval |
| `wbc-mjlab-list-envs` | List registered tasks |
| `wbc-mjlab-data-to-npz` | Build motion NPZ from CSV / PKL (`--batch-size N` for parallel GPU conversion) |
| `wbc-mjlab-export-tracking-params` | Write `config.yaml` |
| `wbc-mjlab-data-vis` | Play motion NPZ clips in Viser (browser) |

## Train / play

Pick a **task** ([tasks & papers](TASKS.md)), then pass a **dataset**:

```bash
uv run wbc-mjlab-train --task Wbc-G1-Zest --dataset lafan
uv run wbc-mjlab-play --task Wbc-G1-Zest --dataset lafan
```

**Preferred:** `--task` + dataset (robot inferred from task → `data/g1/…`):

```bash
uv run wbc-mjlab-train --task Wbc-G1 --dataset lafan
uv run wbc-mjlab-train --task Wbc-G1 --dataset lafan --cache-motion-bundle
```

Training loads `npz/*.npz` in memory by default. Pass **`--cache-motion-bundle`** to write or reuse `data/g1/lafan/lafan.npz` on disk.

**Shorthand** (default task `Wbc-G1` for robot `g1`):

```bash
uv run wbc-mjlab-train --robot g1 --dataset lafan
uv run wbc-mjlab-train --robot g1 --no-state-estimation --dataset lafan   # → Wbc-G1-Zest
```

Motion source (pick one):

| Flag | Resolves to |
|------|-------------|
| `--dataset lafan` | load `data/g1/lafan/npz/*.npz` in memory (or `lafan.npz` with `--cache-motion-bundle`) |
| `--dataset-path /path/to/lafan/` | load `npz/*.npz` in that folder |
| `--dataset-path /path/to/custom.npz` | that file |
| `--motion-file …` | explicit NPZ or dataset directory |
| `--cache-motion-bundle` | write/read `<dataset>/<dataset>.npz` on disk |

## Motion conversion

Conversion needs **`--robot`** (MuJoCo asset / scene), not a task id:

```bash
uv run wbc-mjlab-data-to-npz --robot g1 --dataset lafan
uv run wbc-mjlab-data-to-npz --robot g1 --dataset lafan --batch-size 128
```

`--batch-size N` runs **N parallel FK workers** on the GPU. `--render` preview is only supported with `--batch-size 1`.

See [data/README.md](../data/README.md) for layout and download links.

## Visualize motion NPZ

```bash
uv run wbc-mjlab-data-vis --robot g1 --dataset lafan
uv run wbc-mjlab-data-vis --motion-file data/g1/lafan/npz/walk1_subject1.npz
```

When a dataset has `npz/*.npz` clips, the GUI lists them in a **Motion** dropdown.

## WBC tracking params YAML

Checkpoint saves write `params/config.yaml` alongside `params/policy.onnx`.

```bash
uv run wbc-mjlab-export-tracking-params --task Wbc-G1-Zest --out /path/to/config.yaml
```

Train → export → run on hardware with [wbc-g1-deploy](https://github.com/wbc-mjlab/wbc-g1-deploy).

## Repo layout

```
data/g1/<dataset>/      # retargeted clips (see data/g1/README.md)
src/wbc_mjlab/
  env/                  # Shared WBC MDP (rewards, RSI, motion command, …)
  robots/<id>/configs/  # G1_WBC_TASKS + per-method env builders
  tasks/                # WbcTaskConfig type + mjlab registration
  export/  rl/  motion/  scripts/
```

## Add a robot or paper task

1. **Robot:** `robots/<id>/configs/`, `rl_cfg.py`, assets; register in `robots/env.py` and `tasks/__init__.py`.
2. **Paper task:** env builder in `robots/<id>/configs/<method>.py`, `WbcTaskConfig` in `configs/__init__.py`.
3. Update [TASKS.md](TASKS.md) and train: `uv run wbc-mjlab-train --task Wbc-<ID> --dataset <name>`

See [CONTRIBUTING.md](../CONTRIBUTING.md) for PR workflow.

## Development

```bash
make sync          # CUDA env (uv sync --extra cu128 --group dev)
make format        # ruff format + fix
make lint          # ruff check
make smoke         # wbc-mjlab-list-envs
make test          # pytest
make build         # wheel + smoke import test
```

Optional: `uvx pre-commit install` (after `make sync`).
