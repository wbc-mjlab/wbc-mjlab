# wbc_mjlab

**Accumulate whole-body control (WBC) methods and make them easy to reproduce.**

This repo is an [mjlab](https://github.com/mujocolab/mjlab) extension for **universal motion tracking**: one shared MDP, many **registered tasks** that turn paper-specific knobs (RSI, observations, similarity metrics) into runnable experiments. The goal is not a pile of one-off reproduction scripts—it is a growing library where **Zest**, **BeyondMimic**, **Sonic** / WBC-style tracking, and future papers plug into the same training stack, data layout, and deploy export.

Train with `--task`, point at a motion library (`--dataset` / `--motion-file`), export ONNX + `config.yaml` for real robots ([wbc_g1_deploy](../wbc_g1_deploy) for G1).

## Philosophy

| Principle | What it means here |
|-----------|-------------------|
| **Shared MDP** | Rewards, terminations, motion command, assistive wrench, and reference-residual actions live in `env/` once. Robots wire assets and task configs. |
| **Tasks, not forks** | Each paper’s distinguishing choices become a **task** (`Wbc-G1-Zest`, …) with an env builder in `robots/<id>/configs/`—same CLI, same logs layout, comparable runs. |
| **Neutral code, cited methods** | Implementation names stay generic (`similarity_ema`, `binary_failure`). Paper links live in task descriptions and module docstrings. |
| **Reproducible data path** | Motion libraries under `data/<robot>/<dataset>/`; conversion scripts; optional cached bundles. See [data/README.md](data/README.md). |
| **Deploy parity** | `Wbc-G1` / `Wbc-G1-Zest` use deploy-style obs (no SE); play exports policy artifacts for `wbc_g1_deploy`. |

## Paper ↔ task map (G1)

Tasks are starting points for reproduction—not guaranteed bit-for-bit matches to every ablation in each paper. Add new env builders in `configs/` as methods land.

| Method / paper | Task id | What differs in this repo |
|----------------|---------|---------------------------|
| **WBC** (default stack) | `Wbc-G1` | Zest core + EE z resets, foot slip, anti-shake, deploy obs (history=1) |
| **Zest** | `Wbc-G1-Zest` | Paper repro: Table S4, reward-aligned RSI, no SE |
| **Zest + SE** | `Wbc-G1-Zest-SE` | Zest + full ref pose, anchor pos/ori tracking error, `base_lin_vel` |
| **BeyondMimic** | `Wbc-G1-BinaryFailure` | Full obs, whole-body RSI + binary failure resampling |
| **Sonic** and others | *(add task)* | Shared motion command stack; add a builder + `WbcTaskConfig` entry |

RSI logic: `env/mdp/sampling.py`. Motion playback + RSI: `env/mdp/commands.py`. Actor reference features are obs terms in `env/wbc_env_cfg.py` (non-SE template); SE tasks add measurements via `env/se_actor_obs.py`.

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
| `wbc-mjlab-export-tracking-params` | Write `config.yaml` |
| `wbc-mjlab-vis-data` | Play motion NPZ clips in Viser (browser) |

## Tasks (G1)

| Task id | Logs under | Purpose |
|---------|------------|---------|
| `Wbc-G1` | `logs/rsl_rl/wbc_g1/` | Default WBC stack |
| `Wbc-G1-Zest` | `logs/rsl_rl/wbc_g1_zest/` | Zest paper repro (no SE) |
| `Wbc-G1-Zest-SE` | `logs/rsl_rl/wbc_g1_zest_se/` | Zest + anchor pos / base lin vel obs |
| `Wbc-G1-BinaryFailure` | `logs/rsl_rl/wbc_g1_binary/` | BeyondMimic-style binary failure RSI |

```bash
wbc-mjlab-list-envs
```

## Train / play

Pick a **task** to match the paper you are reproducing, then pass a **dataset**:

```bash
wbc-mjlab-train --task Wbc-G1-Zest --dataset lafan
wbc-mjlab-train --task Wbc-G1-BinaryFailure --dataset lafan
wbc-mjlab-play --task Wbc-G1-Zest --dataset lafan
```

**Preferred:** `--task` + dataset (robot inferred from task → `data/g1/…`):

```bash
wbc-mjlab-train --task Wbc-G1 --dataset lafan
wbc-mjlab-train --task Wbc-G1 --dataset lafan --cache-motion-bundle
```

Training loads `npz/*.npz` in memory by default (no temp stack). Pass **`--cache-motion-bundle`** to write or reuse `data/g1/lafan/lafan.npz` on disk.

**Shorthand** (default task `Wbc-G1` for robot `g1`):

```bash
wbc-mjlab-train --robot g1 --dataset lafan
wbc-mjlab-train --robot g1 --no-state-estimation --dataset lafan   # → Wbc-G1-Zest
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
wbc-mjlab-csv-to-npz --robot g1 --dataset lafan
```

## Visualize motion NPZ

Play clips in the browser using mjlab's ``MjlabViserScene`` (same stack as
``viz-nan``). Uses the same ``--robot`` / ``--dataset`` / ``--motion-file`` flags
as train/play. NPZ files with ``qpos`` play directly; WBC training NPZ
(``joint_pos`` + ``body_*``) is also supported.

```bash
wbc-mjlab-vis-data --robot g1 --dataset lafan
wbc-mjlab-vis-data --motion-file data/g1/lafan/npz/walk1_subject1.npz
wbc-mjlab-vis-data --dataset-path data/g1/lafan
```

When a dataset has ``npz/*.npz`` clips, the GUI lists them in a **Motion** dropdown.

## WBC tracking params YAML

Checkpoint saves write `params/config.yaml` alongside `params/policy.onnx`.

```bash
wbc-mjlab-export-tracking-params --task Wbc-G1-Zest --out /path/to/config.yaml
```

## Add a robot or paper task

1. **Robot:** `robots/<id>/configs/`, `rl_cfg.py`, assets; register env builder in `robots/env.py` and tasks in `tasks/__init__.py`.
2. **Paper task:** add env builder in `robots/<id>/configs/<method>.py`, add a `WbcTaskConfig` entry in `configs/__init__.py`.
3. Train: `wbc-mjlab-train --task Wbc-<ID> --dataset <name>`

## Deploy reference

Example real-robot stack: [wbc_g1_deploy](../wbc_g1_deploy).
