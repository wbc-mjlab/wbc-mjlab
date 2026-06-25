# Demo

Bundled **G1 checkpoint** for interactive play (`demos/wbc_g1/model.pt`, ~44 MB).

Motions come from the version-controlled **samples** CSVs — convert once to NPZ, then run the demo.

## Run

```bash
# 1. Convert bundled samples (one time)
uv run wbc-mjlab-data-to-npz --robot g1 --dataset samples --batch-size 16

# 2. Interactive viewer (8 envs, uniform clip sampling)
uv run wbc-mjlab-demo
```

Opens the Viser viewer in the browser. Play also writes deploy artifacts under `demos/wbc_g1/params/` (gitignored).

## Manual play

```bash
uv run wbc-mjlab-play --task Wbc-G1 --dataset samples \
  --checkpoint-file demos/wbc_g1/model.pt --viewer viser --num-envs 8
```

## Checkpoint

| File | Source |
|------|--------|
| `demos/wbc_g1/model.pt` | `Wbc-G1` training on samples + LAFAN/SEED mix (`model_140000.pt` from local run) |

More checkpoints may move to Git LFS later.
