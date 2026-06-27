# WBC-Mjlab: Whole Body Control in MuJoCo Lab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wbc-mjlab/wbc-mjlab/blob/main/notebooks/demo.ipynb)
[![Demo MJ-WASM](https://img.shields.io/badge/Demo-MJ--WASM-007ec6?labelColor=555555)](https://wbc-mjlab.github.io/wbc-demo/)
[![PyPI](https://img.shields.io/pypi/v/wbc-mjlab)](https://pypi.org/project/wbc-mjlab/)

**One shared MDP for whole-body motion tracking on [mjlab](https://github.com/mujocolab/mjlab) — train once on a motion library, deploy one policy for many skills.**

![WBC G1 sim collage](assets/wbc_g1_collage.gif)

Recent works ([ZEST](https://arxiv.org/abs/2602.00401), [BeyondMimic](https://beyondmimic.github.io/), [SONIC](https://arxiv.org/abs/2511.07820), [OmniXtreme](https://arxiv.org/abs/2602.23843)) is all **WBC / large-scale tracking**, with overlapping ideas (keybody rewards, adaptive sampling, multi-clip training) but **different design choices** — each still tends to ship as its own codebase. In wbc-mjlab, paper-specific knobs are **`--task` switches** on a shared stack:

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

**Convert trajectory samples** (13 source CSVs [manifest & credits](data/g1/samples/README.md)) to npz - calculating FK for body targets, velocities etc:

```bash
uv run wbc-mjlab-data-to-npz --robot g1 --dataset samples --batch-size 16
```

**Demo** — [live web demo](https://wbc-mjlab.github.io/wbc-demo/) (trained policy in the browser); local play with bundled checkpoint (convert samples first):

```bash
uv run wbc-mjlab-demo
```

**Train** on converted npz library (check mjlab train args for resuming, number of envs etc):

```bash
uv run wbc-mjlab-train --task Wbc-G1 --dataset samples
```

**Evaluation** of last exported log on library (check args for viewer, choosing chekpoint, motion etc):
```bash
uv run wbc-mjlab-play --task Wbc-G1 --dataset samples
```

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
