Example: H2 extension
=====================

This page walks through the **reference extension** for Unitree H2. Treat it as a
blueprint — your robot replaces MJCF paths, body tuples, and actuators; the
**registration pattern stays the same**.

Repository: `wbc-mjlab-extension-h2 <https://github.com/wbc-mjlab/wbc-mjlab-extension-h2>`_

Why a separate package
----------------------

H2 is **not** built into wbc-mjlab core. It demonstrates:

- Robot entity in an external repo (MJCF, actuators, sensors)
- ``register_wbc_extension`` wiring
- Extension-owned ``data/h2/`` via ``project_root``
- A single default task ``Wbc-H2`` with ``apply_wbc``

Install alongside core:

.. code-block:: bash

   git clone https://github.com/wbc-mjlab/wbc-mjlab-extension-h2.git
   cd wbc-mjlab-extension-h2 && uv sync
   uv run wbc-mjlab-list-envs    # includes Wbc-H2

Registration entry point
------------------------

``mjlab_entry.py`` calls ``register_wbc_extension`` once:

.. code-block:: python

   register_wbc_extension(
     WbcRobotSpec(
       robot_id="h2",
       aliases=("unitree_h2",),
       project_root=_H2_PROJECT_ROOT,
       make_env_cfg=make_h2_wbc_env_cfg,
       make_rl_cfg=h2_wbc_rl_cfg,
       motion_spec=RobotMotionSpec(
         scene_cfg_fn=lambda: make_h2_wbc_env_cfg().scene,
         foot_body_names=MOTION_Z_DEBIAS_FOOT_BODY_NAMES,
         foot_sole_z=MOTION_Z_DEBIAS_FOOT_SOLE_Z,
         mjcf_path=H2_XML,
       ),
     ),
     H2_WBC_TASKS,
   )

``pyproject.toml`` exposes ``wbc_mjlab_h2.mjlab_entry`` under
``[project.entry-points."mjlab.tasks"]``.

Robot entity (``h2_base_cfg``)
------------------------------

Follows the generic checklist in :doc:`robot_entity`:

- ``make_base_wbc_env_cfg()`` → attach H2 entity, action scale, IMU wiring
- Set ``motion_cmd.anchor_body_name`` and ``motion_cmd.body_names`` from H2 constants
- Contact sensors for feet, self-collision, keybody ground contact

Task builder
------------

``h2_wbc_env_cfg()`` = ``h2_base_cfg()`` + ``apply_wbc(...)`` with H2 body tuples.
One registered task:

.. code-block:: python

   WbcTaskConfig(
     task_id="Wbc-H2",
     robot_id="h2",
     experiment_name="wbc_h2",
     build_env_cfg=h2_wbc_env_cfg,
   )

End-to-end commands
-------------------

.. code-block:: bash

   uv run wbc-mjlab-data-to-npz --robot h2 --dataset samples --batch-size 6
   uv run wbc-mjlab-train --task Wbc-H2 --dataset samples
   uv run wbc-mjlab-play --task Wbc-H2 --dataset samples --viewer viser

Logs: ``logs/rsl_rl/wbc_h2/<run>/``

Adapting for your robot
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - H2-specific
     - Replace with
   * - ``H2_XML``, ``get_h2_robot_cfg()``
     - Your MJCF + entity cfg
   * - ``H2_MOTION_BODY_NAMES``, anchor / EE tuples
     - Your kinematic tracking set
   * - ``H2_ACTION_SCALE``, actuators
     - Your motor model
   * - ``robot_id="h2"``, ``Wbc-H2``
     - Your ids (``mybot``, ``Wbc-Mybot``)
   * - ``data/h2/``
     - ``data/<your_robot_id>/``

Keep ``apply_wbc`` (or other core presets) unchanged — only pass your body name tuples.

Related: :doc:`extensions`, :doc:`robot_entity`, :doc:`../tasks/index`.
