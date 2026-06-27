# WBC-Mjlab: Whole Body Control in MuJoCo Lab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wbc-mjlab/wbc-mjlab/blob/main/notebooks/demo.ipynb)
[![Demo MJ-WASM](https://img.shields.io/badge/Demo-MJ--WASM-007ec6?labelColor=555555)](https://wbc-mjlab.github.io/wbc-demo/)
[![PyPI](https://img.shields.io/pypi/v/wbc-mjlab)](https://pypi.org/project/wbc-mjlab/)

**One shared MDP for whole-body motion tracking on [mjlab](https://github.com/mujocolab/mjlab) — train once on a motion library, deploy one policy for many skills.**

Recent humanoid WBC work ([ZEST](https://arxiv.org/abs/2602.00401), [BeyondMimic](https://beyondmimic.github.io/), [SONIC](https://arxiv.org/abs/2511.07820), [OmniXtreme](https://arxiv.org/abs/2602.23843), …) often ships as **separate stacks per method or per skill**. **wbc-mjlab** is a single training surface: a shared motion-tracking MDP where paper choices are **`--task` switches**, and on deploy **one ONNX policy + a clip library** — swap trajectories at runtime (walk, jog, run, fight, flips, …) instead of retraining per motion.
![WBC G1 sim collage](assets/wbc_g1_collage.gif)

<!-- **Sim2sim preview** (Unitree MuJoCo) — idle · dance · fight · jog · flip: -->



- **Multi-motion by design** — train on **multi-clip datasets** (LAFAN, SEED, custom NPZ libraries); one controller generalizes across the library. At runtime, pick a clip from `manifest.yaml` — no checkpoint change.
- **Shared MDP** — rewards, terminations, motion command, RSI, and playback live in `env/` once; robots and papers plug in via task configs.
- **Tasks, not forks** — ZEST, BeyondMimic-style RSI, deploy obs, etc. are **`--task` switches** (`Wbc-G1`, `Wbc-G1-Zest`, `Wbc-G1-BinaryFailure`, …) with the same CLI and log layout for fair comparison.
- **Motion data pipeline** — versioned libraries under `data/`, GMR PKL ingest, **batch GPU CSV→NPZ**, optional motion-bundle cache ([data/README.md](data/README.md)).
- **Building blocks** — small env builders per method (`robots/g1/configs/`); add a paper setup or tune your own WBC without forking the core MDP.
- **One policy, many skills** — one policy for walk, jog, run, crawl, fight, get-up, lie-down, flips, and more.
- **Sim → real** — train/play export `policy.onnx` + `config.yaml` aligned with the deploy runtime.

Details: [docs/TASKS.md](docs/TASKS.md) · [CONTRIBUTING.md](CONTRIBUTING.md)

## Quick start

Requires [mjlab](https://github.com/mujocolab/mjlab) (≥ 1.4) and an NVIDIA GPU for training.

```bash
git clone https://github.com/wbc-mjlab/wbc-mjlab.git && cd wbc-mjlab
uv run wbc-mjlab-list-envs
```

`uv run` syncs from `uv.lock` on first use. For CUDA/CPU PyTorch and dev deps: `make sync` / `make sync-cpu`. See [docs/INSTALLATION.md](docs/INSTALLATION.md).

**Try bundled samples** (13 source CSVs — convert to NPZ locally, then train; [manifest & credits](data/g1/samples/README.md)):

```bash
uv run wbc-mjlab-data-to-npz --robot g1 --dataset samples
uv run wbc-mjlab-train --task Wbc-G1 --dataset samples
uv run wbc-mjlab-play --task Wbc-G1 --dataset samples
```

**Demo** — [live web demo](https://wbc-mjlab.github.io/wbc-demo/) (trained policy in the browser); local play with bundled checkpoint (convert samples first):

```bash
uv run wbc-mjlab-data-to-npz --robot g1 --dataset samples --batch-size 16
uv run wbc-mjlab-demo
```

See [demos/README.md](demos/README.md). **Colab:** [demo](https://colab.research.google.com/github/wbc-mjlab/wbc-mjlab/blob/main/notebooks/demo.ipynb)

## Docs

| Doc | Contents |
|-----|----------|
| [docs/TASKS.md](docs/TASKS.md) | Paper references, philosophy, G1 task map |
| [docs/USAGE.md](docs/USAGE.md) | CLI, train/play, motion conversion, layout |
| [docs/INSTALLATION.md](docs/INSTALLATION.md) | uv, pip, PyPI, local mjlab |
| [docs/ROADMAP.md](docs/ROADMAP.md) | Planned work (SONIC, infra, …) |
| [data/README.md](data/README.md) | Motion library layout & downloads |
| [CONTRIBUTING.md](CONTRIBUTING.md) | PRs, adding tasks |

Full Sphinx docs and a project page are planned; the README stays a short landing page until then.

## Related repos

| Repo | Role |
|------|------|
| [wbc-mjlab/wbc-mjlab](https://github.com/wbc-mjlab/wbc-mjlab) | Training library (this repo) |
| [wbc-mjlab/wbc-g1-deploy](https://github.com/wbc-mjlab/wbc-g1-deploy) | Optional G1 runtime (ONNX + motion clips) |
| [mujocolab/mjlab](https://github.com/mujocolab/mjlab) | Simulation and RL stack |
