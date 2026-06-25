# G1 sample motions

A small, **version-controlled** subset of public retargeted G1 clips for smoke-testing
convert, train, play, and `wbc-mjlab-data-vis` without downloading full datasets.

**13 clips** ship as source CSV in this folder (8 from LAFAN1 retargeting, 5 from
BONES-SEED). Run conversion once to populate `npz/`, then train or visualize:

```bash
wbc-mjlab-data-to-npz --robot g1 --dataset samples
wbc-mjlab-train --task Wbc-G1 --dataset samples
wbc-mjlab-play --task Wbc-G1 --dataset samples
wbc-mjlab-data-vis --robot g1 --dataset samples
```

## Layout

```
data/g1/samples/
  README.md
  *.csv              # source clips (bundled)
  npz/<clip>.npz     # written by wbc-mjlab-data-to-npz (not in git until converted)
```

## Bundled clips

### LAFAN1 retarget (8 clips)

From [lvhaidong/LAFAN1_Retargeting_Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset)
— LAFAN1 mocap retargeted to Unitree G1, CSV @ 30 Hz.

| File | Motion |
|------|--------|
| `walk1_subject1.csv` | Walking |
| `walk1_subject2.csv` | Walking (alternate performer) |
| `run1_subject2.csv` | Running |
| `sprint1_subject2.csv` | Sprinting |
| `dance1_subject1.csv` | Dance |
| `fallAndGetUp1_subject1.csv` | Fall and get up |
| `fight1_subject2.csv` | Fight / kick |
| `fightAndSports1_subject1.csv` | Fight and sports combo |

### BONES-SEED (5 clips)

From [bones-studio/seed](https://huggingface.co/datasets/bones-studio/seed) — acrobatic
flips retargeted to G1, stored in LAFAN-style CSV (no header, meters, quat xyzw,
radians) after conversion from the native SEED header format.

| File | Motion |
|------|--------|
| `flip_090_001__A304.csv` | 90° flip |
| `flip_090_002__A304.csv` | 90° flip (variant) |
| `flip_090_003__A304_M.csv` | 90° flip (mirrored) |
| `flip_360_009__A416.csv` | 360° flip |
| `flip_360_011__A416.csv` | 360° flip (variant) |

See [data/g1/README.md](../README.md) for full BONES-SEED download and conversion notes.

## Credits and licenses

### LAFAN1 retargeting (8 clips)

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

### BONES-SEED (5 clips)

| | |
|---|---|
| **Dataset** | [bones-studio/seed](https://huggingface.co/datasets/bones-studio/seed) on Hugging Face |
| **Publisher** | [Bones Studio](https://bones.studio/datasets/seed) |
| **Format** | Unitree G1 CSV, LAFAN-style layout in this folder |

BONES-SEED requires accepting the
[dataset license](https://huggingface.co/datasets/bones-studio/seed) on Hugging Face
before download. The samples here are a small excerpt; see `LICENSE.md` in the full
dataset for terms. SOMA and G1 retargets in BONES-SEED were contributed with
acknowledgment to NVIDIA (Kimodo project) — see the
[dataset card](https://huggingface.co/datasets/bones-studio/seed).

### This repository

Sample motion **files** remain under their respective dataset licenses above.
The `wbc_mjlab` **code** is Apache-2.0 (see repo root `LICENSE`).

When you publish results trained on these clips, cite the original datasets (see
main [README](../../../README.md#sample-data--attribution)) in addition to
`wbc_mjlab` and [mjlab](https://github.com/mujocolab/mjlab).
