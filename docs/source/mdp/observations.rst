Observations
============

Manager: **ObservationManager** · Module: ``env/mdp/observations.py``

Groups
------

**Actor** (``wbc_env_cfg.py`` template + preset tweaks):

- **Reference command terms** — base height, gravity, joint pos/vel, base vel in body
  frame (all read from ``MotionCommand`` via ``command_name="motion"``)
- **Proprioception** — projected gravity, joint pos/vel, last action
- Optional corruption noise per term

**Critic** — actor terms without noise, plus **privileged** features: anchor/body
pose errors, reference body kinematics, assistive wrench state, motion segment phase,
per-step tracking reward breakdown.

Presets and robot wiring
------------------------

- ``apply_wbc`` / ``apply_zest`` may drop ``ref_joint_vel`` or set ``history_length``
- ``apply_se_actor`` swaps height/gravity proxies for anchor pose error + ``base_lin_vel``
- Robot entities call ``wire_<robot>_imu_sensors`` when SE layouts need named IMU sensors

Body lists and anchor frame come from the robot's ``<robot>_base_cfg()`` — not from
observation modules directly.

API: :doc:`../api/mdp` (Observations).

Related: :doc:`motion_command`, :doc:`../tasks/index`, :doc:`../extensions/robot_entity`.
