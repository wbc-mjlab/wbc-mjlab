# wbc_mjlab

One policy. Any motion. Train universal whole-body tracking in mjlab—swap clips, not checkpoints.

**One shared WBC MDP** — **training presets** are mjlab **tasks** (`Wbc-G1`, `Wbc-G1-NoSE`, …). Use `--task` for the preset; `--robot` selects hardware / data layout (`data/g1/…`).

The CLI injects the task id automatically; use `--task` or shorthand flags below.

## Layout

```
data/g1/<dataset>/      # retargeted clips + <dataset>.npz (see data/README.md)
src/wbc_mjlab/
  env/                  # Shared WBC MDP
  robots/<id>/          # env_cfg, rl_cfg, presets.py
  robots/ids.py         # ``--robot`` aliases + data paths
  robots/env.py         # env/RL builders (used when registering tasks)
  tasks/                # mjlab task registration from presets
  export/
  rl/
  motion/
  scripts/
```

## Install

```bash
pip install -e .
```

## CLI commands

| Command | Purpose |
|---------|---------|
| `wbc-mjlab-train` | Train (wraps `mjlab.scripts.train`) |
| `wbc-mjlab-play` | Play / eval |
| `wbc-mjlab-list-envs` | List tasks + robots |
| `wbc-mjlab-csv-to-npz` | Build motion bundles from CSV |
| `wbc-mjlab-pkl-to-npz` | Build motion bundles from PKL |
| `wbc-mjlab-export-tracking-params` | Write `wbc_tracking_params.yaml` |

## Task presets (G1)

| Task id | Logs under | Purpose |
|---------|------------|---------|
| `Wbc-G1` | `logs/rsl_rl/wbc_g1/` | Full actor obs, whole-body RSI (default) |
| `Wbc-G1-NoSE` | `logs/rsl_rl/wbc_g1_nose/` | Deploy-style obs (no anchor position) |
| `Wbc-G1-Zest` | `logs/rsl_rl/wbc_g1_zest/` | No SE + joint-only RSI (Zest-style) |
| `Wbc-G1-BinaryFailure` | `logs/rsl_rl/wbc_g1_binary/` | Whole-body RSI + binary failure resampling |

```bash
wbc-mjlab-list-envs    # prints the table above + mjlab registry
```

## Usage

```bash
wbc-mjlab-csv-to-npz --robot g1 --dataset lafan
wbc-mjlab-train --robot g1 --dataset lafan
wbc-mjlab-train --task Wbc-G1-Zest --robot g1 --dataset lafan
wbc-mjlab-train --robot g1 --no-state-estimation --dataset lafan   # → Wbc-G1-NoSE
wbc-mjlab-play --task Wbc-G1-NoSE --robot g1 --dataset lafan
```

Motion source for train/play (pick one):

| Flag | Resolves to |
|------|-------------|
| `--dataset lafan` | `data/g1/lafan/lafan.npz` |
| `--dataset-path /path/to/lafan/` | `<dir>/<name>.npz` or a single `*.npz` in the folder |
| `--dataset-path /path/to/custom.npz` | that file |
| `--motion-file …` | explicit NPZ |

## WBC tracking params YAML

Checkpoint saves write `params/wbc_tracking/wbc_tracking_params.yaml`.

```bash
wbc-mjlab-export-tracking-params --task Wbc-G1-NoSE --robot g1 --out /path/to/wbc_tracking_params.yaml
```

## Add a robot

1. `robots/<id>/env_cfg.py`, `rl_cfg.py`, `presets.py`, assets  
2. Register builders in `robots/env.py` and ids in `robots/ids.py`  
3. Extend `tasks/__init__.py` `_load_all_presets()` and motion spec in `motion/robot_assets.py`  
4. `wbc-mjlab-train --robot <id> --task Wbc-<ID> --dataset <name>`

## Deploy reference

Example real-robot stack: [wbc_g1_deploy](../wbc_g1_deploy).
