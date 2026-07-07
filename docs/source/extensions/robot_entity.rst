The robot entity
================

A **robot entity** is the mjlab scene object (MJCF + actuators + sensors) plus the
wiring that connects it to the shared WBC MDP. It is **not** part of ``env/`` — each
platform owns its entity in ``robots/<id>/`` (core) or an extension package.

Registration is mandatory for any robot beyond the built-in ``g1`` id. Until
``register_robot`` runs, ``--robot <id>`` and ``data/<id>/`` resolution will fail.

Responsibilities of ``<robot>_base_cfg()``
------------------------------------------

Your base builder starts from ``make_base_wbc_env_cfg()`` and must set:

Scene & actuators
~~~~~~~~~~~~~~~~~

- ``cfg.scene.entities["robot"]`` — mjlab entity cfg (MJCF path, initial state)
- ``cfg.actions["joint_pos"].scale`` — per-DoF or grouped action scale
- Actuator model (motor limits, gear ratios) in robot-specific ``actuators.py``

Motion command wiring
~~~~~~~~~~~~~~~~~~~~~

On ``cfg.commands["motion"]`` (``MotionCommandCfg``):

- ``anchor_body_name`` — body for anchor-frame errors and assistive wrench
- ``body_names`` — tuple of **motion keybodies** tracked in rewards / RSI / critic obs
- ``actuated_joint_names`` (optional) — subset for joint metrics

Sensors
~~~~~~~

Typical contact sensors (names are convention, patterns are robot-specific):

- Feet ↔ ground contact
- Self-collision on pelvis / torso subtree
- Per-keybody ground contact (for catastrophic termination)

IMU / SE layouts
~~~~~~~~~~~~~~~~

If tasks use ``apply_se_actor``, provide ``wire_<robot>_imu_sensors(cfg)`` to bind
``base_lin_vel`` / ``base_ang_vel`` builtin sensor names.

Constants module
----------------

Keep body name tuples in ``constants.py`` — presets receive them as arguments:

.. code-block:: python

   apply_wbc(
     cfg,
     motion_body_names=MY_ROBOT_MOTION_BODY_NAMES,
     ee_termination_bodies=MY_ROBOT_EE_TERMINATION_BODY_NAMES,
   )

This keeps ``env/mdp/`` free of robot-specific strings.

Symmetry config (optional, for ``--mirror``)
--------------------------------------------

If you want ``wbc-mjlab-data-to-npz --mirror`` for your robot, add
``robots/<id>/symmetry.py`` with a :class:`~wbc_mjlab.robots.symmetry.RobotSymmetryConfig`.
The config lists **joint pairs** (left name, right name, sign) used to mirror
``joint_pos`` and ``joint_vel`` in exported NPZs.

**Sign convention:** ``+1`` for pitch-like joints, ``-1`` for roll/yaw-like joints.
Central joints (waist) use the same name for both sides and typically ``-1`` on
yaw/roll, ``+1`` on pitch.

In-tree G1 example (``robots/g1/symmetry.py``):

.. code-block:: python

   from wbc_mjlab.robots.symmetry import (
     JointSymmetryEntry,
     RobotSymmetryConfig,
     register_robot_symmetry_config,
   )

   MYBOT_SYMMETRY_CONFIG = RobotSymmetryConfig(
     joints=(
       JointSymmetryEntry("left_hip_roll_joint", "right_hip_roll_joint", -1.0),
       JointSymmetryEntry("left_hip_pitch_joint", "right_hip_pitch_joint", 1.0),
       # ... all actuated pairs ...
       JointSymmetryEntry("waist_yaw_joint", "waist_yaw_joint", -1.0),
     ),
     mirror_suffix="_mirror",
   )

   register_robot_symmetry_config("mybot", MYBOT_SYMMETRY_CONFIG)

For core robots, call ``register_robot_symmetry_config`` at import time (G1 does
this in ``robots/g1/__init__.py`` via ``robots/g1/symmetry.py``). For extensions,
pass ``symmetry_config=...`` on :class:`~wbc_mjlab.extension.WbcRobotSpec` or call
``register_robot_symmetry_config`` from ``mjlab_entry.py``.

Body kinematics (``body_pos_w``, ``body_quat_w``, velocities) use a shared
left/right **name swap** plus world-frame reflection; only the joint table is
robot-specific.

What the robot entity does *not* do
-----------------------------------

- Define new reward or RSI **implementations** (use presets + shared ``env/mdp/``)
- Hard-code paper choices (those belong in presets and task builders)
- Own the MDP template (``make_base_wbc_env_cfg`` stays in core)

In-tree reference
-----------------

The core ``robots/g1/`` package is the canonical example of a fully wired entity.
Use it as a template, not as documentation's primary subject — your extension
follows the same structure with different MJCF and body names.

Next: :doc:`extensions` to register the entity and tasks.
