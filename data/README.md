# Motion data

Motion libraries for WBC tracking live under **`data/<robot>/<dataset_name>/`**.

A small **`samples/`** folder is version-controlled under each robot (e.g.
[`g1/samples/`](g1/samples/)) with a few clips from public datasets for smoke
tests. All other dataset folders stay local — download full libraries from
Hugging Face (see robot guides) or add your own retargeted clips.

Only READMEs, `samples/`, and `.gitkeep` placeholders are version-controlled;
other clip and bundle files stay gitignored.

## Robot guides

| Robot | Guide |
|-------|--------|
| **G1** | [g1/README.md](g1/README.md) — downloads, G1 joint order, example datasets |

## Directory layout

```
data/<robot>/<dataset>/
  *.csv / *.pkl          # source clips (or under raw/) — what training reads
  .cache/motion_<key>.npz  # packed source clips, rebuilt when a clip changes
  npz/<clip>.npz         # optional per-clip exports (wbc-mjlab-data-to-npz)
  <dataset>.npz          # optional stacked npz/ (--cache-motion-bundle)
```

- Put source clips in the dataset folder or in **`raw/`** — `raw/` is preferred when it contains `.csv` or `.pkl` files.
- Training reads the source clips directly: resampling, velocities, mirroring (`--mirror`), height debias (`--debias-z`) and body kinematics are computed on the GPU. No conversion step.
- **`.cache/`**, **`npz/`** and **`*.npz`** are **never committed**. `npz/` is only for tools that want precomputed body poses (`wbc-mjlab-data-to-npz`); a dataset with only `npz/` still trains from it.
- **`params/motion_library.yaml`** is written automatically on **play** from the loaded motion library — no sidecar manifest in the dataset folder.

## Supported formats

Source layouts are defined in `wbc_mjlab.motion.motion_formats` (canonical fields:
`base_pos`, `base_rot_xyzw`, `dof_pos`, `fps`). The format is inferred from
the file extension; `--env.commands.motion.source-format` (train) or `--format` (export) picks a registered name explicitly.

| Format key | Extension | Role |
|------------|-----------|------|
| `default` | `.csv` | LAFAN / retarget CSV (no header, m + quat xyzw + rad) |
| `gmr_pkl` | `.pkl` | GMR / `bvh_to_robot` pickle dict |
| **NPZ** | train / play | Optional per-clip exports or a pre-stacked bundle |

### CSV `default` (LAFAN / retarget)

Comma-separated rows, no header:

```
root_pos(3) + root_rot_xyzw(4) + joint_pos(n_dof)   # meters, radians
```

Joint count must match the target robot. Default input rate is 30 Hz unless overridden with `--input-fps`.

### PKL (`gmr_pkl`)

Python dict with at least:

```
fps, root_pos (T×3), root_rot (T×4, xyzw), dof_pos (T×n_dof)
```

Optional keys: `joint_names`, `dof_joint_names`, or `joint_order` to override DOF ordering.

## Workflow

```bash
# 1. Add clips under data/<robot>/<dataset>/ (see robot README for where to download)
uv run wbc-mjlab-data-to-npz --robot <robot> --dataset <dataset>
# optional: --format default | gmr_pkl
# large libraries: parallel FK workers on GPU
uv run wbc-mjlab-data-to-npz --robot <robot> --dataset <dataset> --batch-size 8

# 2. Train (loads npz/*.npz in memory; no stacked file unless cached)
uv run wbc-mjlab-train --task Wbc-<Robot> --dataset <dataset>

# 3. Optional: write <dataset>/<dataset>.npz for faster startup
uv run wbc-mjlab-train --task Wbc-<Robot> --dataset <dataset> --cache-motion-bundle
```

`--batch-size N` keeps **N MuJoCo envs** on GPU and converts clips in parallel
(idle workers take the next file from the queue). Use `--batch-size 1` for
`--render` preview.

Conversion requires **`--robot`** (MuJoCo asset). Train/play normally use **`--task`** (robot is inferred from the task config).

## Train / play motion source

Pick one:

| Flag | Resolves to |
|------|-------------|
| `--dataset <name>` | `data/<robot>/<name>/` (loads `npz/*.npz` in memory) |
| `--dataset-path <dir>` | load `npz/*.npz` in that folder |
| `--dataset-path <file>.npz` | that file directly |
| `--motion-file <file>.npz` | explicit NPZ path |
| `--motion-file <dataset-dir>` | dataset folder with `npz/*.npz` |
| `--cache-motion-bundle` | write/read `<dataset>/<dataset>.npz` instead of in-memory stack |

Conversion accepts **`--dataset-path <dir>`** as the input/output root (same role as `data/<robot>/<name>/`).

## Version control

Everything under `data/` is gitignored except:

- `README.md` and `.gitkeep` files
- **`samples/`** — bundled example clips (see `g1/samples/README.md` for manifest and credits)

See repo `.gitignore`.
