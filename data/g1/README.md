# G1 motion data

G1 datasets live in **`data/g1/<dataset_name>/`** (e.g. `lafan`, `seed`).  
General layout, formats, and CLI flags are in [../README.md](../README.md).

No motion files are shipped in this repo yet — use the Hugging Face links below or your own retargeting pipeline. A future **`data/g1/samples/`** folder will hold a few short clips for quick end-to-end tests.

## G1 CSV / PKL expectations

- **29 DoF** (no hands), joint order must match the G1 mjlab asset (`qpos[7:]`).
- **CSV:** `root_pos(3) + root_rot_xyzw(4) + joint_pos(29)` per row @ 30 Hz input → 50 Hz after conversion.
- Pre-retargeted CSV from the Hugging Face sets below use the standard G1 order: root XYZ + quaternion XYZW, then leg → waist → arm joints.

## Recommended datasets (Hugging Face)

### LAFAN1 — starter set (~40 clips)

**[lvhaidong/LAFAN1_Retargeting_Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset)**

LAFAN1 mocap retargeted to Unitree G1 as **CSV @ 30 FPS**. Small, diverse set (walk, run, dance, fight, …) — good default for first training runs.

```bash
pip install huggingface_hub

huggingface-cli download lvhaidong/LAFAN1_Retargeting_Dataset \
  --repo-type dataset \
  --local-dir /tmp/lafan1_retarget

mkdir -p data/g1/lafan/raw
cp /tmp/lafan1_retarget/g1/*.csv data/g1/lafan/raw/
# HF layout may vary — copy all G1 *.csv

wbc-mjlab-csv-to-npz --robot g1 --dataset lafan
wbc-mjlab-train --task Wbc-G1 --dataset lafan
```

LAFAN1 content: [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) (see dataset card).

### BONES-SEED — large-scale (~142k clips)

**[bones-studio/seed](https://huggingface.co/datasets/bones-studio/seed)**

Everyday human motion with **Unitree G1 MuJoCo-compatible CSV** at `g1/csv/{date}/{motion_name}.csv`. Accept the [BONES-SEED license](https://huggingface.co/datasets/bones-studio/seed) on Hugging Face before download.

```bash
huggingface-cli download bones-studio/seed \
  --repo-type dataset \
  --local-dir /tmp/bones-seed

tar -xzf /tmp/bones-seed/g1.tar.gz -C /tmp/bones-seed

mkdir -p data/g1/seed/raw
# subset via metadata/seed_metadata_v003.parquet, then copy CSVs
rsync -a /tmp/bones-seed/g1/csv/ data/g1/seed/raw/

wbc-mjlab-csv-to-npz --robot g1 --dataset seed
wbc-mjlab-train --task Wbc-G1 --dataset seed
```

See the [dataset card](https://huggingface.co/datasets/bones-studio/seed) for packages (Locomotion, Dances, …), metadata schema, and Python download snippets.

## Other G1 sources

### Retarget yourself (GMR)

1. Raw BVH — [Ubisoft LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset) or other mocap.
2. Retarget with [GMR](https://github.com/YanjieZe/GMR) to **`unitree_g1`**:
   ```bash
   python scripts/bvh_to_robot.py \
     --bvh_file path/to/clip.bvh \
     --robot unitree_g1 \
     --format lafan1 \
     --save_path path/to/clip.pkl

   python scripts/batch_gmr_pkl_to_csv.py --folder path/to/pkls/
   ```
3. Copy into `data/g1/<your_dataset>/`.

AMASS / OMOMO and similar sources work if the pipeline outputs compatible CSV or PKL.

### Quick smoke test

mjlab demo NPZ: https://storage.googleapis.com/mjlab_beta/lafan_dance1_subject1.npz

```bash
wbc-mjlab-play --task Wbc-G1 --motion-file /path/to/lafan_dance1_subject1.npz
```

## Example layout (`lafan`)

```
data/g1/lafan/
  raw/dance1_subject1.csv
  npz/dance1_subject1.npz
  lafan.npz              # optional (--cache-motion-bundle)
```
