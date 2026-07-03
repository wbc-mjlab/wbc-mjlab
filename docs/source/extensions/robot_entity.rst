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
