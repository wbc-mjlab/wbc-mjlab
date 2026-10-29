.. _architecture:

Architecture
============

How a registered ``--task`` becomes a runnable ``ManagerBasedRlEnvCfg``. For the
layer model (MDP, robot, preset, task), start with :doc:`concepts/index`.

Config pipeline
---------------

Every task cfg is assembled in the same order:

.. code-block:: text

   make_base_wbc_env_cfg()       # shared WBC template (env/wbc_env_cfg.py)
        ↓
   <robot>_base_cfg()            # registered robot entity
        ↓
   apply_<preset>(...)           # paper / deploy recipe (presets/*.py)
        ↓
   [optional] apply_se_actor()   # actor obs swap (wbc / zest / end_effector)
        ↓
   WbcTaskConfig.build_env_cfg   # --task id

Preset ↔ task concepts: :doc:`concepts/presets_and_tasks`. Code examples:
:doc:`tasks/index`.

Task registration
-----------------

``WbcTaskConfig`` (``tasks/config.py``) binds:

- ``task_id`` → CLI ``--task``
- ``robot_id`` → registered entity + ``data/<robot>/``
- ``experiment_name`` → ``logs/rsl_rl/<name>/``
- ``build_env_cfg`` → full env cfg callable

Robots added in the Studio (``robots/custom.py``) follow the same pipeline: their
``<robot>_base_cfg`` is built from ``robot.yaml`` (model, PD gains, body names) and their
tasks stack ``apply_wbc`` (+ ``apply_se_actor`` for ``-SE``). Studio presets add one more
step after ``build_env_cfg``: modules, then the preset's own overrides
(``studio/presets.py``), registered as ``<task>--<name>``.

Play / eval mode
----------------

Task builders accept ``play=True`` to disable training-only features: long episodes,
obs corruption, motion DR, assistive wrench, push events. Same pattern for every
registered robot.

WBC Studio
----------

``src/wbc_mjlab/studio/`` — a UI-agnostic core with services, and two frontends:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Module
     - Role
   * - ``core.py``
     - ``StudioCore``: selected robot (MuJoCo model + CPU kinematics), scene state, event
       bus, one background task at a time, a 50 Hz tick loop (playback, job polling)
   * - ``robots.py`` · ``robot_import.py``
     - Robot catalogue, wiring, pose · add / edit robots from model files
   * - ``data.py`` · ``clips.py``
     - Datasets, clip index / search, raw + NPZ preview, clip QA, conversion and import
   * - ``presets.py`` · ``train_preset.py``
     - Studio presets and modules (schema, apply, register, run snapshots) · training entry
   * - ``recipes.py``
     - Training recipes: a method (:data:`~wbc_mjlab.rl.algos.ALGORITHMS`) and its
       settings, saved apart from presets
   * - ``training.py`` · ``jobs.py`` · ``tensorboard.py`` · ``algo_check.py``
     - Detached train / convert jobs (``logs/studio/jobs``), runs (sizes, delete, prune
       checkpoints), curves, TensorBoard ·
       whether a method can train a preset (:data:`~wbc_mjlab.rl.algos.ALGORITHMS`)
   * - ``play.py``
     - A checkpoint on N envs in-process: clip pinning, reference ghost, reward terms
   * - ``web/``
     - FastAPI server (JSON API + binary websocket frames) and the three.js UI
   * - ``viser_app.py``
     - The same core as a viser app

Related
-------

- :doc:`concepts/index` — modularity, RSI, robots, presets
- :doc:`mdp/index` — Shared MDP (design and term catalogs)
- :doc:`extensions/index` — extension how-to
- :doc:`usage` — CLI
- :doc:`studio` — WBC Studio
