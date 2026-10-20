# G1 sample motions

A small, **version-controlled** subset of public retargeted G1 clips for smoke-testing
train and play without downloading full datasets.

**10 clips** ship as source CSV in this folder (7 from LAFAN1 retargeting, 3 retargeted
from video). Training reads them as they are:

```bash
uv run wbc-mjlab-train --task Wbc-G1 --dataset samples --mirror
uv run wbc-mjlab-play --task Wbc-G1 --dataset samples
uv run wbc-mjlab-data-to-npz --robot g1 --dataset samples   # optional: npz/ for other tools
```

## Layout

```
data/g1/samples/
  README.md
  *.csv              # source clips (bundled)
  .cache/            # packed clips for training (local, rebuilt automatically)
  npz/<clip>.npz     # optional export — wbc-mjlab-data-to-npz (never committed)
```

## Bundled clips

### LAFAN1 retarget (7 clips)

From [lvhaidong/LAFAN1_Retargeting_Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset)
— LAFAN1 mocap retargeted to Unitree G1, CSV @ 30 Hz.

| File | Motion |
|------|--------|
| `walk1_subject1.csv` | Walking |
| `run1_subject2.csv` | Running |
| `sprint1_subject2.csv` | Sprinting |
| `dance1_subject1.csv` | Dance |
| `fallAndGetUp1_subject1.csv` | Fall and get up |
| `fight1_subject2.csv` | Fight / kick |
| `fightAndSports1_subject1.csv` | Fight and sports combo |

### Video retarget (3 clips)

`dance.csv`, `gainer_flip.csv`, and `backflip.csv` are retargeted from video with
GVHMR and UMR.

| File | Motion |
|------|--------|
| `dance.csv` | Dance |
| `gainer_flip.csv` | Gainer flip |
| `backflip.csv` | Backflip |

The bundled checkpoint [`demos/wbc_g1/model.pt`](../../../demos/README.md) was trained
on LAFAN1 and on BONES-SEED. The BONES-SEED motion files are not in this folder.

## Credits and licenses

### LAFAN1 retargeting (7 clips)

| | |
|---|---|
| **Mocap** | [Ubisoft LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset) (CC BY-NC-ND 4.0) |
| **G1 retarget** | [lvhaidong/LAFAN1_Retargeting_Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset) on Hugging Face |
| **Format** | CSV @ 30 Hz, 29-DoF G1 joint order |

LAFAN1 motion content is licensed under
[CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) (non-commercial,
no derivatives). The bundled clips are **unmodified subsets** for tutorial and
reproducibility; for the full set or commercial use, download from Hugging Face and
follow the dataset card.

### Video retarget (3 clips)

Recovered from video with GVHMR and retargeted to G1 with UMR: `dance.csv`,
`gainer_flip.csv`, `backflip.csv`.

### This repository

Sample motion **files** remain under their respective dataset licenses above.
The `wbc_mjlab` **code** is Apache-2.0 (see repo root `LICENSE`).

When you publish results trained on these clips, cite the original datasets (see
**Attribution** above) in addition to
`wbc_mjlab` and [mjlab](https://github.com/mujocolab/mjlab).
