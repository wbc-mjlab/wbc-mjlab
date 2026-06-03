# Motion data

Motion clips for WBC tracking live under **`data/<robot>/<dataset_name>/`**.

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
- **`npz/`** is written by the conversion tools; train/play stack from there at startup.
- **`<dataset>.npz`** is optional: created when you pass **`--cache-motion-bundle`** on train/play.

## Supported formats

| Format | Tool | Role |
|--------|------|------|
| **CSV** | `wbc-mjlab-csv-to-npz` | Retargeted robot motion, one file per clip |
| **PKL** | `wbc-mjlab-pkl-to-npz` | Pickle dict from retargeting pipelines (e.g. GMR) |
| **NPZ** | train / play | Per-clip exports or a pre-stacked training bundle |

### CSV

Comma-separated rows, no header:

```
root_pos(3) + root_rot_xyzw(4) + joint_pos(n_dof)
```

Joint count and order must match the target robot’s mjlab asset. Default input rate is 30 Hz; conversion resamples to 50 Hz.

### PKL

Python dict with at least:

```
fps, root_pos (T×3), root_rot (T×4, xyzw), dof_pos (T×n_dof)
```

Optional keys: `joint_names`, `dof_joint_names`, or `joint_order` to override DOF ordering.

## Workflow

```bash
# 1. Add clips under data/<robot>/<dataset>/ (see robot README for where to download)
wbc-mjlab-csv-to-npz --robot <robot> --dataset <dataset>
# or
wbc-mjlab-pkl-to-npz --robot <robot> --dataset <dataset>

# 2. Train (stacks npz/*.npz at startup; temp bundle unless cached)
wbc-mjlab-train --task Wbc-<Robot> --dataset <dataset>

# 3. Optional: write <dataset>/<dataset>.npz for faster reruns
wbc-mjlab-train --task Wbc-<Robot> --dataset <dataset> --cache-motion-bundle
```

Conversion requires **`--robot`** (MuJoCo asset). Train/play normally use **`--task`** (robot is inferred from the task preset).

## Train / play motion source

Pick one:

| Flag | Resolves to |
|------|-------------|
| `--dataset <name>` | `data/<robot>/<name>/npz/*.npz` (or reuse fresh `<name>.npz`) |
| `--dataset-path <dir>` | stack `npz/*.npz` in that folder |
| `--dataset-path <file>.npz` | that file directly |
| `--motion-file <file>.npz` | explicit NPZ path |
| `--cache-motion-bundle` | also write `<dataset>/<dataset>.npz` when stacking |

Conversion accepts **`--dataset-path <dir>`** as the input/output root (same role as `data/<robot>/<name>/`).

## Version control

Everything under `data/` is gitignored except:

- `README.md` and `.gitkeep` files
- **`samples/`** — reserved for a future small set of bundled example clips (not populated yet)

See repo `.gitignore`.
