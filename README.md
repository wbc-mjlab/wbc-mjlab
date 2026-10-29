# WBC-Mjlab: Whole Body Control in MuJoCo Lab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wbc-mjlab/wbc-mjlab/blob/main/notebooks/demo.ipynb)
[![Demo MJ-WASM](https://img.shields.io/badge/Demo-MJ--WASM-007ec6?labelColor=555555)](https://wbc-mjlab.github.io/wbc-demo/)
[![Docs](https://github.com/wbc-mjlab/wbc-mjlab/actions/workflows/docs.yml/badge.svg)](https://wbc-mjlab.github.io/wbc-mjlab/)
[![Project page](https://img.shields.io/badge/Project-page-007ec6?labelColor=555555)](https://wbc-mjlab.github.io/)
[![License](https://img.shields.io/github/license/wbc-mjlab/wbc-mjlab)](https://github.com/wbc-mjlab/wbc-mjlab/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/wbc-mjlab)](https://pypi.org/project/wbc-mjlab/)
[![PyPI downloads](https://img.shields.io/pypi/dm/wbc-mjlab?color=blue)](https://pypistats.org/packages/wbc-mjlab)

**One shared MDP for whole-body motion tracking on [mjlab](https://github.com/mujocolab/mjlab) — train once on a motion library, deploy one policy for many skills.**

![WBC G1 sim collage](assets/wbc_g1_collage.gif)

Recent works ([ZEST](https://arxiv.org/abs/2602.00401), [BeyondMimic](https://arxiv.org/abs/2508.08241), [SONIC](https://arxiv.org/abs/2511.07820), [OmniXtreme](https://arxiv.org/abs/2602.23843)) is all **WBC / large-scale tracking**, with overlapping ideas (keybody rewards, adaptive sampling, multi-clip training) but **different design choices** — each still tends to ship as its own codebase. In wbc-mjlab, paper-specific knobs are **`--task` switches** on a shared stack:

- **Multi-motion by design** — train on **multi-clip datasets** (LAFAN, SEED, custom NPZ libraries); one controller generalizes across the library. At runtime, provide reference - via playing clip, teleop, or higher level policy.
- **Shared MDP** — rewards, terminations, motion command, RSI, and playback live in `env/` once; robots and papers plug in via task configs.
- **Tasks, not forks** — ZEST, BeyondMimic-style RSI, deploy obs, etc. are **`--task` switches** (`Wbc-G1`, `Wbc-G1-Zest`, `Wbc-G1-BinaryFailure`, …) with the same CLI and log layout for fair comparison.
- **Motion data pipeline** — versioned libraries under `data/`, GMR PKL ingest, **batch GPU CSV→NPZ**, optional motion-bundle cache.
- **Building blocks** — paper presets in `presets/` and G1 task builders in `robots/g1/tasks.py`; add a paper setup or tune your own WBC without forking the core MDP.
- **Plug-in robots** — add a humanoid from its MJCF / URDF [without code](https://wbc-mjlab.github.io/wbc-mjlab/source/extensions/custom_robots.html), or ship an extension package via `register_wbc_extension` ([H2 example](https://github.com/wbc-mjlab/wbc-mjlab-extension-h2)).
- **PPO or a world model** — any task also trains through a learned, differentiable world model on one GPU (`--algo wm`, after [SuperMimic](https://arxiv.org/abs/2610.28480)): [world-model training](https://wbc-mjlab.github.io/wbc-mjlab/source/concepts/world_model.html).
- **[WBC Studio](https://wbc-mjlab.github.io/wbc-mjlab/source/studio.html)** — the whole pipeline in a browser: robots, clip QA, presets, training jobs, playback on many envs.
- **One policy, many skills** — one policy for walk, jog, run, crawl, fight, get-up, lie-down, flips, and more.
- **Sim → real** — train/play export `policy.onnx` + `config.yaml` aligned with the deploy runtime.

Details: [Documentation](https://wbc-mjlab.github.io/wbc-mjlab/) · [CONTRIBUTING.md](CONTRIBUTING.md)

## Quick start

Requires [mjlab](https://github.com/mujocolab/mjlab) (≥ 1.5) and an NVIDIA GPU for training ([installation](https://wbc-mjlab.github.io/wbc-mjlab/source/installation.html)).

```bash
git clone https://github.com/wbc-mjlab/wbc-mjlab.git && cd wbc-mjlab
uv run wbc-mjlab-list-envs
```

**Demo** — [live web demo](https://wbc-mjlab.github.io/wbc-demo/), or locally with the bundled checkpoint: `uv run wbc-mjlab-demo` (convert the samples first).

### In the browser: WBC Studio

```bash
uv run wbc-mjlab-studio      # open the printed link (also from other machines on the network)
```

Five steps — **Robot → Data → Preset → Train → Play**: pick or [add a robot](https://wbc-mjlab.github.io/wbc-mjlab/source/extensions/custom_robots.html) from its MJCF / URDF, import and check clips (training reads them as they are), tune observations / rewards / sampling / terrain / PPO as presets with reusable modules, launch and follow training jobs, play policies on many envs with reference ghosts. Options, files and the HTTP API: [Studio guide](https://wbc-mjlab.github.io/wbc-mjlab/source/studio.html).

### From the command line

```bash
# train a task (Wbc-G1, Wbc-G1-Zest, Wbc-G1-Terrain-Scan, …) on source clips (CSV / PKL):
# resampled, mirrored and run through FK on the GPU while training — nothing to convert
uv run wbc-mjlab-train --task Wbc-G1 --dataset samples --mirror

# the same task through a learned world model instead of PPO (one GPU)
uv run wbc-mjlab-train --task Wbc-G1 --dataset samples --algo wm

# play / evaluate the last run (exports policy.onnx + config.yaml)
uv run wbc-mjlab-play --task Wbc-G1 --dataset samples --viewer viser
```

More: [quickstart](https://wbc-mjlab.github.io/wbc-mjlab/source/workflows/quickstart.html) · [tasks](https://wbc-mjlab.github.io/wbc-mjlab/source/tasks/index.html) · [motion data](https://wbc-mjlab.github.io/wbc-mjlab/source/data.html) · [training & resume](https://wbc-mjlab.github.io/wbc-mjlab/source/workflows/training.html) · [deploy](https://wbc-mjlab.github.io/wbc-mjlab/source/workflows/deploy.html) · [all commands](https://wbc-mjlab.github.io/wbc-mjlab/source/usage.html).

## Related repos

| Repo | Role |
|------|------|
| [wbc-mjlab/wbc-mjlab](https://github.com/wbc-mjlab/wbc-mjlab) | Training library (this repo) — shared MDP, presets, extension API |
| [wbc-mjlab/wbc-mjlab-extension-h2](https://github.com/wbc-mjlab/wbc-mjlab-extension-h2) | Reference robot extension (Unitree H2) — plug-in WBC without forking core |
| [wbc-mjlab/wbc-g1-deploy](https://github.com/wbc-mjlab/wbc-g1-deploy) | Optional G1 runtime (ONNX + motion clips) |
| [mujocolab/mjlab](https://github.com/mujocolab/mjlab) | Simulation and RL stack |
