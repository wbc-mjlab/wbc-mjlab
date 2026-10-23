Wbc-G1-Lookahead — the SuperMimic paper's task
==============================================

Task id: ``Wbc-G1-Lookahead`` · In-tree reference on robot ``g1``.

A policy that looks ahead: it sees 10 steps of proprioception and the next 10 reference
frames, with BeyondMimic's tracking reward, actions about the default pose and episode starts
where it earns low return. It is the task of the SuperMimic paper (Appendix A and D.1); the
name is the task's, not the method's. Trained with the world model
(:doc:`../concepts/world_model`) it is SuperMimic; with PPO, its baseline.

- **Preset:** ``apply_lookahead`` (``presets/lookahead.py``)
- **Builder:** ``g1_wbc_lookahead_env_cfg()``
- **Paper:** `arXiv:2610.28480 <https://arxiv.org/abs/2610.28480>`_

.. code-block:: bash

   uv run wbc-mjlab-train --task Wbc-G1-Lookahead --dataset lafan_seed --algo wm   # SuperMimic
   uv run wbc-mjlab-train --task Wbc-G1-Lookahead --dataset lafan_seed   # PPO

Logs: ``logs/rsl_rl/wbc_g1_lookahead/<run>/``

.. note::

   The reference terms look 10 frames ahead (:ref:`observations-ahead`): the deploy
   parameters record ``frames: 10`` for each, so a robot runtime has to feed the next 10
   reference frames. A preset can change the window per term (fewer frames, a stride, other
   reference terms) like any other observation.

Observation (Table A.1)
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Term
     - Noise
     - Notes
   * - ``ref_joint_pos`` · ``ref_joint_vel``
     - —
     - tracked joint positions · velocities of the reference, 10 frames from the current
       one (``frames: 10``)
   * - ``ref_anchor_ori_b``
     - —
     - the reference anchor's orientation in the robot anchor frame (6D), 10 frames
   * - ``ref_anchor_shift_b``
     - —
     - the reference anchor's displacement from where the reference is now, in the
       robot anchor frame, 10 frames
   * - ``joint_pos`` · ``joint_vel``
     - ±0.01 · ±0.5
     - relative to the default pose; 10-step history
   * - ``base_ang_vel`` · ``projected_gravity``
     - ±0.2 · ±0.05
     - IMU; 10-step history
   * - ``actions``
     - —
     - previous action; 10-step history

The actor needs only the robot's joints and IMU (no state estimation). Actions:
``q_target = q_default + scale · a``.

Reward (Table A.2)
------------------

All terms ``exp(−mean_b ‖e_b‖² / σ²)`` over the tracked bodies, × the control period.

.. list-table::
   :header-rows: 1

   * - Term
     - σ
     - Weight
   * - ``motion_body_pos`` · ``motion_body_ori``
     - 0.3 · 0.4
     - 1.0
   * - ``motion_body_lin_vel`` · ``motion_body_ang_vel``
     - 1.0 · 3.14
     - 1.0
   * - ``motion_global_root_pos`` · ``motion_global_root_ori``
     - 0.3 · 0.4
     - 0.5
   * - ``action_rate_l2`` · ``joint_limit``
     - —
     - −0.1 · −2

Termination, randomization, starts
----------------------------------

- **Termination:** anchor height off by more than 0.25 m, or tilt (gravity z in the anchor
  frame) off by more than 0.8; no end-effector termination.
- **Randomization (Table A.3):** foot friction U[0.3, 1.2], encoder offsets ±0.01 rad,
  torso center-of-mass offset; no pushes, no assistive wrench.
- **Starts (Appendix D.1):** 1 s bins, each with a running estimate of the discounted
  return earned from it; starts are 50 % uniform, 40 % ``softmax(−25 G / G_max)``, 10 % in
  proportion to each bin's past softmax weight (``rsi.strategy = "discounted_return"``,
  :doc:`../mdp/rsi`).
