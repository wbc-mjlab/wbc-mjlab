# Motion data

Tracking datasets live under **`data/<robot>/<dataset_name>/`** (e.g. `data/g1/lafan/`).

## Layout (G1 example)

```
data/g1/lafan/
  walk1_subject1.csv    # retargeted clips (or place under raw/)
  walk2_subject1.csv
  lafan.npz             # stacked bundle for training
  npz/walk1_subject1.npz
```

Optional: put source clips in `data/g1/<dataset>/raw/` — converters prefer `raw/` when it contains `.csv` or `.pkl` files.

## Workflow

```bash
# 1. Add CSV/PKL clips under data/g1/my_dataset/ (or raw/)
wbc-mjlab-csv-to-npz --robot g1 --dataset my_dataset

# 2. Train on the bundle data/g1/my_dataset/my_dataset.npz
wbc-mjlab-train --robot g1 --dataset my_dataset
```

Train/play motion source (one of):

- `--dataset <name>` → `data/<robot>/<name>/<name>.npz`
- `--dataset-path <dir>` → `<dir>/<dirname>.npz` or the only `*.npz` in the folder
- `--dataset-path <file>.npz` → that bundle directly
- `--motion-file <file>.npz` → explicit path

Conversion also accepts `--dataset-path <dir>` as the output/input root.

Dataset directories are gitignored (`data/*`); keep only `.gitkeep` / this README in version control.
