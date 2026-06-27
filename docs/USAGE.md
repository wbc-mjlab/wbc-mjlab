# Usage

## CLI commands

| Command | Purpose |
|---------|---------|
| `wbc-mjlab-train` | Train (wraps `mjlab.scripts.train`) |
| `wbc-mjlab-play` | Play / eval |
| `wbc-mjlab-list-envs` | List registered tasks |
| `wbc-mjlab-data-to-npz` | Build motion NPZ from CSV / PKL (`--batch-size N` for parallel GPU conversion) |
| `wbc-mjlab-export-tracking-params` | Write `config.yaml` |
| `wbc-mjlab-export-web-reference` | Emit a wbc-demo live-policy folder (ONNX + config + per-clip reference-command `.bin`s + manifest) |
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

## Web demo live-policy folder

`wbc-mjlab-export-web-reference` emits a ready-to-drop
[wbc-demo](https://github.com/wbc-mjlab/wbc-demo) `policies/<policy-id>/` folder
for the **in-browser live policy** (issue wbc-mjlab-ow5): `policy.onnx`,
`config.yaml`, `motion_library.yaml`, `reference/index.json` + per-clip
`reference/<clipId>.bin`, `thumb.png`, and a `policy.yaml` manifest stub
(conforms to `policies/policy.schema.json`).

```bash
uv run wbc-mjlab-export-web-reference --task Wbc-G1 --dataset samples \
  --run logs/rsl_rl/wbc_g1/<run> --out /path/to/wbc-demo/policies
```

The browser runs the policy ONNX live and builds the full actor observation by
concatenating live proprioception (from its own sim) with the **reference
command** shipped per clip — so this exporter ships only the reference stream,
**not** full-body render poses (far smaller than the shelved render bundle).

Each `reference/<clipId>.bin` is the `wbc_reference_stream_v1` wire format: raw
little-endian Float32, frame-major `frames × 39` (no header). The 39 dims follow
the deploy `config.yaml` `tracking.reference_observation_names` order:

| Term | Dims | Meaning |
|------|------|---------|
| `ref_base_height` | 1 | anchor (`torso_link`) world z (env origin z = 0 on export) |
| `ref_base_lin_vel_b` | 3 | anchor linear velocity, rotated into the anchor frame |
| `ref_base_ang_vel_b` | 3 | anchor angular velocity, rotated into the anchor frame |
| `ref_gravity_b` | 3 | gravity unit vector `(0,0,-1)`, rotated into the anchor frame |
| `ref_joint_pos` | 29 | reference joint positions (config `joint_names` order) |

The anchor world state is read straight from each clip NPZ's `body_*_w` arrays at
the `torso_link` index — no forward kinematics. The math matches mjlab's
`MotionCommand` (`env/mdp/commands.py`) and the deploy C++ `WbcMotionLoader`
(`wbc-g1-deploy/src/WbcMotionLoader.cpp`) term-for-term. `reference/index.json`
(schema `wbc_reference_stream_v1`) carries `commandDim`, `fps`, the `refTerms`
layout, and the per-clip listing (`id`, `name`, `file`, `frames`, `durationSec`,
`tags`). `--run` / `--checkpoint-file` points at a run whose `params/` holds
`policy.onnx` + `config.yaml`; `config.yaml` is regenerated from the task config
if missing. Edit the `policy.yaml` metadata before opening the wbc-demo PR. The
JS consumer contract is documented in `REFERENCE_STREAM.md` in the wbc-demo repo.

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
