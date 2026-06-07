# Roadmap

**Backlog:** [GitHub Issues](https://github.com/simeon-ned/wbc_mjlab/issues) (source of truth). Check off or remove items when filed/merged (`Closes #N`).

| | |
|---|---|
| **Issues** | One issue ≈ one PR |
| **Labels** | `area:env`, `area:tasks`, `paper:zest`, `paper:sonic`, `type:enhancement`, … |
| **Milestones** | `foundation`, `zest-parity`, `sonic-tracker`, `v0.2-dx` |

Paper references: `wbc/references/` (zest, sonic, beyond_mimick PDFs).

## Tasks

### Foundation

- [x] Move G1-specific params from `env/wbc_env_cfg.py` → `robots/g1/configs/base.py`
- [x] Export `wbc_tracking_params.yaml` next to `params/latest.onnx` (train + play)
- [x] Fix / merge `mjlab_entry` circular import if still open
- [ ] Split command in to configurable terms similar to other observations (like projected gravity, etc)

### Zest parity (`Wbc-G1-Zest`)

- [ ] Dwell on last frame at clip end, then timeout reset (not failure)
- [ ] Enable joint position limit reward; add joint torque limit (Table S4)
- [ ] Optional: L_max-normalized similarity EMA (§S5)
- [ ] Clarify `Wbc-G1` actor history=10 vs Zest (no history)

### SONIC tracker (§3.1 — not universal token / VLA)

- [ ] Epic: `Wbc-G1-Sonic` — 1 s bins, failure-rate cap, BeyondMimic-style rewards, DR
- [ ] Introduce motion command jitter (noise) to commands observation (Table 2)
- [ ] README bib links

### Paper repro

- [ ] SONIC universal policy (FSQ, multi-encoder, multi-frame command)
- [ ] Zest per-joint action scales and DR (Table S5)
- [ ] BeyondMimic gaps beyond `Wbc-G1-BinaryFailure`
- [ ] Additional robots

### Handy Utilities
- [x] Dataset visualizer (to play npz)

### Developer experience (`area:infra`)

- [ ] uv + `uv.lock`, `uv_build`, `RELEASING.md`, PyPI
- [ ] Makefile: `sync`, `format`, `type`, `test`, `build`, `docs`
- [ ] Dockerfile: CUDA + uv + `MUJOCO_GL=egl`
- [ ] Sphinx docs + GitHub Pages
- [ ] `CITATION.cff` + README citing section

Epics (e.g. full SONIC stack): one issue with a checklist, then sub-issues per PR.
