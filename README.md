# wbc_mjlab

**Accumulate whole-body control (WBC) methods and make them easy to reproduce.**

This repo is an [mjlab](https://github.com/mujocolab/mjlab) extension for **universal motion tracking**: one shared MDP, many **registered tasks** that turn paper-specific knobs (RSI, observations, similarity metrics) into runnable experiments. The goal is not a pile of one-off reproduction scripts—it is a growing library where **Zest**, **BeyondMimic**, **Sonic** / WBC-style tracking, and future papers plug into the same training stack, data layout, and deploy export.

Train with `--task`, swap motion datasets, export ONNX + `wbc_tracking_params.yaml` for real robots ([wbc_g1_deploy](../wbc_g1_deploy) for G1).

## Philosophy

| Principle | What it means here |
|-----------|-------------------|
| **Shared MDP** | Rewards, terminations, motion command, assistive wrench, and reference-residual actions live in `env/` once. Robots wire assets and task configs. |
| **Tasks, not forks** | Each paper’s distinguishing choices become a **task** (`Wbc-G1-Zest`, …) with an env builder in `robots/<id>/configs/`—same CLI, same logs layout, comparable runs. |
| **Neutral code, cited methods** | Implementation names stay generic (`similarity_ema`, `binary_failure`). Paper links live in task descriptions and module docstrings. |
| **Reproducible data path** | Motions under `data/<robot>/<dataset>/`; conversion scripts; optional cached bundles. See [data/README.md](data/README.md). |
| **Deploy parity** | `Wbc-G1-NoSE` matches deploy-style observations; play exports policy artifacts for `wbc_g1_deploy`. |

## Paper ↔ task map (G1)

Tasks are starting points for reproduction—not guaranteed bit-for-bit matches to every ablation in each paper. Add new env builders in `configs/` as methods land.

| Method / paper | Task id | What differs in this repo |
|----------------|---------|---------------------------|
| **WBC** (default stack) | `Wbc-G1` | Deploy-style obs, joint-only RSI, assistive wrench, **actor history = 10** |
| **Zest** | `Wbc-G1-Zest` | Same RSI/obs as deploy stack, no history |
| **Deploy / no SE** | `Wbc-G1-NoSE` | Same env as Zest (for export parity) |
| **BeyondMimic** | `Wbc-G1-BinaryFailure` | Full obs, whole-body RSI + binary failure resampling |
| **Sonic** and others | *(add task)* | Shared motion command stack; add a builder + `WbcTaskConfig` entry |

RSI logic: `env/mdp/sampling.py`. Motion command + reference stack: `env/mdp/commands.py`.

## Layout

```
data/g1/<dataset>/      # retargeted clips (see data/g1/README.md)
src/wbc_mjlab/
  env/                  # Shared WBC MDP (rewards, RSI, motion command, …)
  robots/<id>/configs/  # G1_WBC_TASKS + per-method env builders
  robots/ids.py         # robot aliases (motion conversion, optional --robot)
  robots/env.py         # env/RL builders (used when registering tasks)
  tasks/                # WbcTaskConfig type + mjlab registration
  export/
  rl/
  motion/
  scripts/
```

## Contributing

[CONTRIBUTING.md](CONTRIBUTING.md) · [docs/ROADMAP.md](docs/ROADMAP.md)

## Install

```bash
pip install -e .
```

## CLI commands

| Command | Purpose |
|---------|---------|
| `wbc-mjlab-train` | Train (wraps `mjlab.scripts.train`) |
| `wbc-mjlab-play` | Play / eval |
| `wbc-mjlab-list-envs` | List registered tasks |
| `wbc-mjlab-csv-to-npz` | Build motion bundles from CSV |
| `wbc-mjlab-pkl-to-npz` | Build motion bundles from PKL |
| `wbc-mjlab-export-tracking-params` | Write `wbc_tracking_params.yaml` |

## Tasks (G1)

| Task id | Logs under | Purpose |
|---------|------------|---------|
| `Wbc-G1` | `logs/rsl_rl/wbc_g1/` | Joint-only RSI, deploy-style obs, actor history 10 |
| `Wbc-G1-NoSE` | `logs/rsl_rl/wbc_g1_nose/` | Deploy export (same env as Zest) |
| `Wbc-G1-Zest` | `logs/rsl_rl/wbc_g1_zest/` | Zest reproduction, no history |
| `Wbc-G1-BinaryFailure` | `logs/rsl_rl/wbc_g1_binary/` | BeyondMimic-style binary failure RSI |

```bash
wbc-mjlab-list-envs
```

## Train / play

Pick a **task** to match the paper you are reproducing, then pass a **dataset**:

```bash
wbc-mjlab-train --task Wbc-G1-Zest --dataset lafan
wbc-mjlab-train --task Wbc-G1-BinaryFailure --dataset lafan
wbc-mjlab-play --task Wbc-G1-NoSE --dataset lafan
```

**Preferred:** `--task` + dataset (robot inferred from task → `data/g1/…`):

```bash
wbc-mjlab-train --task Wbc-G1 --dataset lafan
wbc-mjlab-train --task Wbc-G1 --dataset lafan --cache-motion-bundle
```

Training stacks `npz/*.npz` at startup (temp file). Pass **`--cache-motion-bundle`** to also write `data/g1/lafan/lafan.npz` for faster reruns.

**Shorthand** (default task `Wbc-G1` for robot `g1`):

```bash
wbc-mjlab-train --robot g1 --dataset lafan
wbc-mjlab-train --robot g1 --no-state-estimation --dataset lafan   # → Wbc-G1-NoSE
```

Motion source (pick one):

| Flag | Resolves to |
|------|-------------|
| `--dataset lafan` | stack `data/g1/lafan/npz/*.npz` (or reuse fresh `lafan.npz`) |
| `--dataset-path /path/to/lafan/` | stack `npz/*.npz` in that folder |
| `--dataset-path /path/to/custom.npz` | that file |
| `--motion-file …` | explicit NPZ |
| `--cache-motion-bundle` | also write `<dataset>/<dataset>.npz` when stacking |

## Motion conversion

Conversion needs **`--robot`** (MuJoCo asset / scene), not a task id:

```bash
wbc-mjlab-csv-to-npz --robot g1 --dataset lafan
```

## WBC tracking params YAML

Checkpoint saves write `params/wbc_tracking_params.yaml` alongside `params/latest.onnx`.

```bash
wbc-mjlab-export-tracking-params --task Wbc-G1-NoSE --out /path/to/wbc_tracking_params.yaml
```

## Add a robot or paper task

1. **Robot:** `robots/<id>/configs/`, `rl_cfg.py`, assets; register env builder in `robots/env.py` and tasks in `tasks/__init__.py`.
2. **Paper task:** add env builder in `robots/<id>/configs/<method>.py`, add a `WbcTaskConfig` entry in `configs/__init__.py`.
3. Train: `wbc-mjlab-train --task Wbc-<ID> --dataset <name>`

## Deploy reference

Example real-robot stack: [wbc_g1_deploy](../wbc_g1_deploy).
