Extension packages
==================

External robots ship as **separate Python packages** that call
``register_wbc_extension`` at import time. Core stays robot-agnostic; extensions add
ids, builders, data roots, and tasks.

API overview
------------

``register_wbc_extension(robot, tasks)`` (``extension.py``) performs:

1. ``register_robot_id`` — ``--robot`` resolution and ``data/<id>/`` paths
2. ``register_robot_builders`` — ``make_env_cfg`` / ``make_rl_cfg`` for train/play
3. ``register_robot_motion_spec`` — FK scene for ``wbc-mjlab-data-to-npz``
4. ``register_project_root`` — extension-owned ``data/<robot>/`` tree
5. ``register_wbc_tasks`` — ``--task`` ids
6. ``register_robot_symmetry_config`` (optional) — when ``symmetry_config`` is set on the spec

``WbcRobotSpec`` fields
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Field
     - Purpose
   * - ``robot_id``
     - Canonical id (lowercase), e.g. ``h2``, ``mybot``
   * - ``aliases``
     - Extra names for ``--robot`` (``unitree_h2``, …)
   * - ``make_env_cfg``
     - ``(play=False, task_id=…) -> ManagerBasedRlEnvCfg``
   * - ``make_rl_cfg``
     - RL runner cfg (PPO hyperparameters, …)
   * - ``motion_spec``
     - ``RobotMotionSpec`` — scene fn, foot debias, MJCF path for FK
   * - ``project_root``
     - Root containing ``data/<robot_id>/``
   * - ``mjcf_path``
     - Optional shortcut when motion_spec omits mjcf_path
   * - ``symmetry_config``
     - Optional :class:`~wbc_mjlab.robots.symmetry.RobotSymmetryConfig` for ``--mirror`` on ``wbc-mjlab-data-to-npz``

Minimal registration
--------------------

.. code-block:: python

   from wbc_mjlab.extension import WbcRobotSpec, register_wbc_extension
   from wbc_mjlab.motion.robot_assets import RobotMotionSpec
   from wbc_mjlab.tasks.config import WbcTaskConfig
   from my_robot_wbc.robots.mybot.symmetry import MYBOT_SYMMETRY_CONFIG

   ROOT = Path(__file__).resolve().parents[2]

   def mybot_wbc_env_cfg() -> ManagerBasedRlEnvCfg:
     cfg = mybot_base_cfg()
     apply_wbc(cfg, motion_body_names=MY_MOTION_BODIES, ...)
     return cfg

   register_wbc_extension(
     WbcRobotSpec(
       robot_id="mybot",
       aliases=("vendor_mybot",),
       project_root=ROOT,
       make_env_cfg=make_mybot_wbc_env_cfg,
       make_rl_cfg=mybot_wbc_rl_cfg,
       motion_spec=RobotMotionSpec(
         scene_cfg_fn=lambda: mybot_base_cfg().scene,
         mjcf_path=MYBOT_XML,
       ),
       symmetry_config=MYBOT_SYMMETRY_CONFIG,
     ),
     WbcTaskConfig(
       task_id="Wbc-Mybot",
       robot_id="mybot",
       description="Default WBC stack (apply_wbc).",
       experiment_name="wbc_mybot",
       build_env_cfg=mybot_wbc_env_cfg,
     ),
   )

Package layout (recommended)
----------------------------

.. code-block:: text

   my-robot-wbc/
     pyproject.toml          # mjlab.tasks entry point
     data/<robot_id>/        # datasets (versioned or git-lfs)
     src/my_robot_wbc/
       mjlab_entry.py        # register_wbc_extension(...)
       robots/<id>/
         constants.py        # body names, XML paths
         symmetry.py         # RobotSymmetryConfig for --mirror (optional)
         base.py             # <robot>_base_cfg()
         actuators.py
         tasks.py            # WbcTaskConfig table
         rl_cfg.py

Discovery via mjlab
-------------------

Add to ``pyproject.toml``:

.. code-block:: toml

   [project.entry-points."mjlab.tasks"]
   my_robot_wbc = "my_robot_wbc.mjlab_entry"

When the package is installed, mjlab imports the entry module and registration runs
before ``wbc-mjlab-list-envs`` or train/play.

Typical task stack
------------------

Most extensions expose **one default task** with ``apply_wbc`` only. Paper-specific
variants (Zest, binary failure, SE) can be added by stacking additional presets in
``tasks.py`` — same pattern as core ``robots/g1/tasks.py``.

Data paths
----------

With ``project_root`` set, CLI resolves:

.. code-block:: text

   <project_root>/data/<robot_id>/<dataset>/

Motion conversion uses ``--robot <robot_id>`` (entity + FK), not ``--task``.

Reference implementation
------------------------

See :doc:`example_extension` for a walkthrough of the published H2 extension layout
(`wbc-mjlab-extension-h2 <https://github.com/wbc-mjlab/wbc-mjlab-extension-h2>`_).

Related: :doc:`robot_entity`, :doc:`../data`, :doc:`../usage`.
