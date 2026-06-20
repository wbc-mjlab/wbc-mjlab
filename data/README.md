# Motion data

Motion libraries for WBC tracking live under **`data/<robot>/<dataset_name>/`**.

**Nothing is bundled in the repo yet** — download datasets locally (see robot guides below) or convert your own clips. We plan to add a small **`samples/`** folder later with a few simple motions from one of the public datasets so you can smoke-test convert / train / play without a full download.

Until then, only READMEs and `.gitkeep` placeholders are version-controlled; all other clip and bundle files stay gitignored.

## Robot guides

| Robot | Guide |
|-------|--------|
| **G1** | [g1/README.md](g1/README.md) — downloads, G1 joint order, example datasets |

## Directory layout

```
data/<robot>/<dataset>/
  *.csv / *.pkl          # source clips (or under raw/)
  npz/<clip>.npz         # per-clip exports (source of truth for training)
  <dataset>.npz          # optional cached stack (--cache-motion-bundle)
```

- Put source clips in the dataset folder or in **`raw/`** — converters prefer `raw/` when it contains `.csv` or `.pkl` files.
- **`npz/`** is written by the conversion tools; train/play load clips from there in memory by default.
- **`<dataset>.npz`** is optional: written only when you pass **`--cache-motion-bundle`** on train/play.
- **`params/motion_library.yaml`** is written automatically on **play** from the loaded motion bundle — no sidecar manifest in the dataset folder.

## Supported formats

Source layouts are defined in `wbc_mjlab.motion.motion_formats` (canonical fields:
`base_pos`, `base_rot_xyzw`, `dof_pos`, `fps`). The converter infers format from
the file extension; pass `--format` to pick a registered name explicitly.

| Format key | Extension | Role |
|------------|-----------|------|
| `default` | `.csv` | LAFAN / retarget CSV (no header, m + quat xyzw + rad) |
| `gmr_pkl` | `.pkl` | GMR / `bvh_to_robot` pickle dict |
| **NPZ** | train / play | Per-clip exports or a pre-stacked training bundle |

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
wbc-mjlab-data-to-npz --robot <robot> --dataset <dataset>
# optional: --format default | gmr_pkl

# 2. Train (loads npz/*.npz in memory; no stacked file unless cached)
wbc-mjlab-train --task Wbc-<Robot> --dataset <dataset>

# 3. Optional: write <dataset>/<dataset>.npz for faster startup
wbc-mjlab-train --task Wbc-<Robot> --dataset <dataset> --cache-motion-bundle
```

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
- **`samples/`** — reserved for a future small set of bundled example clips (not populated yet)

See repo `.gitignore`.
