.. _roadmap:

Roadmap
=======

**Backlog:** `GitHub Issues <https://github.com/wbc-mjlab/wbc-mjlab/issues>`_ (source of truth).
Check off or remove items when filed/merged (``Closes #N``).

.. list-table::
   :widths: 30 70

   * - **Issues**
     - One issue ≈ one PR
   * - **Labels**
     - ``area:env``, ``area:tasks``, ``area:infra``, ``paper:zest``, ``paper:sonic``, …
   * - **Milestones**
     - ``foundation``, ``v0.1-public``, ``sonic-tracker``, ``v0.2-dx``

Foundation
----------

- ✅ Move G1-specific params from ``env/wbc_env_cfg.py`` → ``robots/g1/constants.py`` + ``base.py``
- ✅ Export ``config.yaml`` next to ``params/policy.onnx`` (train + play)
- ✅ Fix / merge ``mjlab_entry`` circular import
- ✅ Configurable motion command terms
- ✅ Bundled ``data/g1/samples/`` (LAFAN1 + BONES-SEED excerpts) + credits

Zest parity (``Wbc-G1-Zest``)
-----------------------------

- ✅ Dwell on last frame at clip end, then timeout reset (not failure)
- ✅ Joint position limit reward; joint torque limit (Table S4)
- ✅ Optional L_max-normalized similarity EMA (§S5)
- ✅ SE task obs: anchor pose tracking error + base lin vel
- ✅ Per-joint action scales and DR (Table S5)

SONIC tracker (§3.1 — not universal token / VLA)
------------------------------------------------

- ☐ Epic: ``Wbc-G1-Sonic`` — 1 s bins, failure-rate cap, BeyondMimic-style rewards, DR
- ☐ Motion command jitter in commands observation (Table 2)
- ☐ README / docs bib links for shipped tasks

Paper repro
-----------

- ✅ ``Wbc-G1-BinaryFailure`` (BeyondMimic-style binary failure RSI)
- ☐ BeyondMimic gaps beyond BinaryFailure
- ☐ Additional robots

Utilities
---------

- ✅ Dataset visualizer (``wbc-mjlab-data-vis``)
- ✅ Motion conversion pipeline (``wbc-mjlab-data-to-npz``, parallel batch)

Developer experience (``area:infra``)
-------------------------------------

- ✅ ``uv`` + ``uv.lock``, ``Makefile``, ``RELEASING.md``
- ☐ ``.github/workflows/ci.yml`` — ruff + ``wbc-mjlab-list-envs``
- ☐ ``CITATION.cff`` + README citing section
- ✅ Sphinx docs + GitHub Pages (`#30 <https://github.com/wbc-mjlab/wbc-mjlab/issues/30>`_)
- ☐ PyPI publish ``wbc-mjlab``
- ☐ Dockerfile: CUDA + uv + ``MUJOCO_GL=egl``
- ☐ Smoke tests in ``tests/`` (beyond import smoke)

Epics (e.g. full SONIC stack): one issue with a checklist, then sub-issues per PR.
