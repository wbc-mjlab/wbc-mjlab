# WBC-Mjlab: Whole Body Control in MuJoCo Lab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wbc-mjlab/wbc-mjlab/blob/main/notebooks/demo.ipynb)
[![PyPI](https://img.shields.io/pypi/v/wbc-mjlab)](https://pypi.org/project/wbc-mjlab/)

**One shared MDP for whole-body motion tracking — compare and reproduce recent methods on [mjlab](https://github.com/mujocolab/mjlab).**

Recent work has pushed humanoid WBC toward **large-scale motion tracking** ([ZEST](https://arxiv.org/abs/2602.00401), [BeyondMimic](https://beyondmimic.github.io/), [SONIC](https://arxiv.org/abs/2511.07820), [OmniXtreme](https://arxiv.org/abs/2602.23843), …). Each paper ships its own stack, which makes fair comparison and sim-to-real export painful.

**wbc_mjlab** unifies that line of work on **one training surface**: a shared motion-tracking MDP with paper-specific choices as **`--task` switches** (RSI, observations, rewards, DR). Export ONNX + `config.yaml` for optional G1 deploy ([wbc-g1-deploy](https://github.com/wbc-mjlab/wbc-g1-deploy)).

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

**Demo** (bundled checkpoint + samples clip library — convert samples first):

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
