State-estimation observations
=============================

Task id: ``Wbc-G1-SE`` · In-tree reference on robot ``g1``.

**Default WBC stack + state-estimation actor observations** — anchor pose tracking
error and base linear velocity on top of :doc:`wbc-g1`.

- **Presets:** ``apply_wbc`` + ``apply_se_actor`` + ``wire_g1_imu_sensors``
- **Builder:** ``g1_wbc_se_env_cfg()``

What this task adds
-------------------

On top of :doc:`wbc-g1`, ``apply_se_actor`` (``presets/se_actor.py``):

**Removes** from actor: ``ref_base_height``, ``ref_gravity_b``, ``projected_gravity``

**Adds:** ``motion_anchor_pos_error_w``, ``motion_anchor_ori_error``, ``base_lin_vel``
(IMU wiring in the robot entity builder)

Train & play
------------

.. code-block:: bash

   uv run wbc-mjlab-train --task Wbc-G1-SE --dataset samples
   uv run wbc-mjlab-play --task Wbc-G1-SE --dataset samples --viewer viser

Logs: ``logs/rsl_rl/wbc_g1_se/<run>/``

Tips
----

- For deploy export, prefer :doc:`wbc-g1` unless your runtime implements the same SE
  observation terms.
- Check ``env.yaml`` for IMU sensor names after training.
