.. _architecture:

Architecture overview
=====================

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
   [optional] apply_se_actor()   # actor obs swap
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

Play / eval mode
----------------

Task builders accept ``play=True`` to disable training-only features: long episodes,
obs corruption, motion DR, assistive wrench, push events. Same pattern for every
registered robot.

Related
-------

- :doc:`concepts/index` — modularity, RSI, robots, presets
- :doc:`mdp/index` — Shared MDP (design and term catalogs)
- :doc:`extensions/index` — extension how-to
- :doc:`usage` — CLI
