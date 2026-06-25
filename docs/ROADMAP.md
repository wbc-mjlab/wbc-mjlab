# Roadmap

**Backlog:** [GitHub Issues](https://github.com/simeon-ned/wbc_mjlab/issues) (source of truth).
Check off or remove items when filed/merged (`Closes #N`).

| | |
|---|---|
| **Issues** | One issue ≈ one PR |
| **Labels** | `area:env`, `area:tasks`, `area:infra`, `paper:zest`, `paper:sonic`, `type:enhancement`, … |
| **Milestones** | `foundation`, `v0.1-public`, `sonic-tracker`, `v0.2-dx` |

## Foundation

- [x] Move G1-specific params from `env/wbc_env_cfg.py` → `robots/g1/configs/base.py`
- [x] Export `config.yaml` next to `params/policy.onnx` (train + play)
- [x] Fix / merge `mjlab_entry` circular import
- [x] Configurable motion command terms
- [x] Bundled `data/g1/samples/` (LAFAN1 + BONES-SEED excerpts) + credits

## Zest parity (`Wbc-G1-Zest`)

- [x] Dwell on last frame at clip end, then timeout reset (not failure)
- [x] Joint position limit reward; joint torque limit (Table S4)
- [x] Optional L_max-normalized similarity EMA (§S5)
- [x] SE task obs: anchor pose tracking error + base lin vel
- [x] Per-joint action scales and DR (Table S5)

## SONIC tracker (§3.1 — not universal token / VLA)

- [ ] Epic: `Wbc-G1-Sonic` — 1 s bins, failure-rate cap, BeyondMimic-style rewards, DR
- [ ] Motion command jitter in commands observation (Table 2)
- [ ] README / docs bib links for shipped tasks

## Paper repro

- [x] `Wbc-G1-BinaryFailure` (BeyondMimic-style binary failure RSI)
- [ ] BeyondMimic gaps beyond BinaryFailure
- [ ] Additional robots

## Utilities

- [x] Dataset visualizer (`wbc-mjlab-data-vis`)
- [x] Motion conversion pipeline (`wbc-mjlab-data-to-npz`, parallel batch)

## Developer experience (`area:infra`)

Public launch checklist (CI, `CITATION.cff`, docs site, PyPI, …) is tracked
locally — not in this file.

- [ ] `.github/workflows/ci.yml` — ruff + `wbc-mjlab-list-envs`
- [ ] `CITATION.cff` + README citing section
- [ ] Sphinx docs + GitHub Pages
- [ ] `uv` + `uv.lock`, `RELEASING.md`, PyPI
- [ ] Makefile: `sync`, `format`, `test`, `build`, `docs`
- [ ] Dockerfile: CUDA + uv + `MUJOCO_GL=egl`
- [ ] Smoke tests in `tests/`

Epics (e.g. full SONIC stack): one issue with a checklist, then sub-issues per PR.
