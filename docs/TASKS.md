# Tasks & papers

wbc_mjlab exposes paper-specific choices as **registered tasks** (`--task`) on one shared motion-tracking MDP. Tasks are **starting points for reproduction** — not bit-for-bit replicas of every ablation.

## Philosophy

| Principle | What it means here |
|-----------|-------------------|
| **Shared MDP** | Rewards, terminations, motion command, assistive wrench, and reference-residual actions live in `env/` once. Robots wire assets and task configs. |
| **Tasks, not forks** | Each paper’s distinguishing choices become a **task** (`Wbc-G1-Zest`, …) with an env builder in `robots/<id>/configs/`—same CLI, same logs layout, comparable runs. |
| **Neutral code, cited methods** | Implementation names stay generic (`similarity_ema`, `binary_failure`). Paper links live in task descriptions and module docstrings. |
| **Reproducible data path** | Motion libraries under `data/<robot>/<dataset>/`; conversion scripts; optional cached bundles. See [data/README.md](../data/README.md). |
| **Deploy parity** | `Wbc-G1` / `Wbc-G1-Zest` use deploy-style obs (no SE); play exports policy artifacts for [wbc-g1-deploy](https://github.com/wbc-mjlab/wbc-g1-deploy). |

## References

| Paper | Link | In this repo |
|-------|------|--------------|
| **ZEST** — Zero-shot Embodied Skill Transfer | [arXiv:2602.00401](https://arxiv.org/abs/2602.00401) | **`Wbc-G1-Zest`**: Table S4 tracking rewards, reward-aligned RSI, assistive wrench, Zest-style terminations, joint/torque limits, per-joint action scales + DR. **`Wbc-G1`** builds on the same core with deploy-oriented extras. |
| **BeyondMimic** | [Project](https://beyondmimic.github.io/) | **`Wbc-G1-BinaryFailure`**: binary-failure adaptive RSI, whole-body keybody similarity, BeyondMimic-style reward mix, full actor observations. |
| **SONIC** — supersized motion tracking | [arXiv:2511.07820](https://arxiv.org/abs/2511.07820) | **Planned** (`Wbc-G1-Sonic`): time-binned RSI, failure-rate caps, command jitter — [roadmap](ROADMAP.md). |
| **OmniXtreme** — high-dynamic scalable tracking | [arXiv:2602.23843](https://arxiv.org/abs/2602.23843) | **Partial**: G1 Unitree torque–speed envelope clipping in sim (`UnitreeActuator`). Power-safety rewards and a dedicated task are not wired yet. |

Shared infrastructure (all tasks): motion command + playback (`env/mdp/commands.py`), RSI / adaptive sampling (`env/mdp/sampling.py`), motion libraries + NPZ conversion (`motion/`), deploy export (`export/`).

If you use a specific method, please **cite the original paper** in addition to this repo and [mjlab](https://github.com/mujocolab/mjlab).

## Paper ↔ task map (G1)

| Method | Task id | What differs here |
|--------|---------|-------------------|
| **ZEST** (default + deploy tweaks) | `Wbc-G1` | Zest Table S4 + RSI, EE z resets, foot slip, anti-shake, deploy obs (`history=1`) |
| **ZEST + state estimation** | `Wbc-G1-SE` | `Wbc-G1` + anchor pose tracking error, `base_lin_vel` |
| **ZEST** (paper repro) | `Wbc-G1-Zest` | Table S4 + reward-aligned RSI, no SE, Zest terminations |
| **ZEST + SE** | `Wbc-G1-Zest-SE` | Zest repro + SE actor obs |
| **BeyondMimic** | `Wbc-G1-BinaryFailure` | Binary-failure RSI, full obs, body-focused rewards |
| **SONIC** | *(planned)* | Binned RSI, DR, command noise — [roadmap](ROADMAP.md) |

```bash
uv run wbc-mjlab-list-envs
uv run wbc-mjlab-train --task Wbc-G1-Zest --dataset lafan
uv run wbc-mjlab-train --task Wbc-G1-BinaryFailure --dataset lafan
```

## Task ids & log dirs

| Task id | Logs under | Purpose |
|---------|------------|---------|
| `Wbc-G1` | `logs/rsl_rl/wbc_g1/` | Default WBC stack |
| `Wbc-G1-SE` | `logs/rsl_rl/wbc_g1_se/` | Wbc-G1 + SE actor obs |
| `Wbc-G1-Zest` | `logs/rsl_rl/wbc_g1_zest/` | Zest paper repro (no SE) |
| `Wbc-G1-Zest-SE` | `logs/rsl_rl/wbc_g1_zest_se/` | Zest + anchor pos / base lin vel obs |
| `Wbc-G1-BinaryFailure` | `logs/rsl_rl/wbc_g1_binary/` | BeyondMimic-style binary failure RSI |
