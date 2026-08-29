Rewards
=======

Manager: **RewardManager** · Module: ``env/mdp/rewards.py``

Tracking terms
--------------

All primary terms use ``command_name="motion"`` and compare robot state to the
reference from ``MotionCommand``:

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Term
     - Tracks
   * - ``motion_global_root_pos`` / ``motion_global_root_ori``
     - Root pose vs reference
   * - ``motion_body_pos`` / ``motion_body_ori``
     - Keybody pose (body list from **cfg params** — set by preset + robot)
   * - ``motion_body_lin_vel`` / ``motion_body_ang_vel``
     - Keybody velocities
   * - ``motion_joint_pos`` / ``motion_joint_vel``
     - Joint state vs reference

Kernel: ``exp(−κ · error² / σ²)``. Presets such as ``apply_wbc`` set κ, σ, and
**which bodies** participate. ``apply_end_effector`` does not retune rewards.

Regularizers
------------

Base template also defines ``survival``, ``action_rate_l1``, ``joint_acc``,
``joint_limit``, ``actuator_torque_soft_limit``, ``foot_slip``, ``angular_momentum``.
Presets set weights — many are **0** on default WBC stacks.

Reward-aligned RSI
------------------

When ``RsiCfg.similarity_from_rewards=True``, per-step RSI similarity is computed
from the **same active** ``motion_*`` terms and weights — see :doc:`rsi`.

API: :doc:`../api/mdp` (Rewards). Callable names in code use the
``motion_*_error_exp`` form (cfg term ids may be shorter aliases in presets).

Related: :doc:`../tasks/index`, :doc:`../extensions/robot_entity`.
