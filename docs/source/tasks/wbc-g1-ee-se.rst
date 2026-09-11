End-effector tracking + state-estimation obs
============================================

Task id: ``Wbc-G1-EE-SE`` · In-tree reference on robot ``g1``.

**EE actor command with state-estimation observations** — same WBC rewards
as :doc:`wbc-g1` / :doc:`wbc-g1-ee`, plus anchor pose error and base linear
velocity.

- **Presets:** ``apply_wbc`` + ``apply_end_effector`` + ``apply_se_actor`` + IMU
- **Builder:** ``g1_wbc_ee_se_env_cfg()``

What this task adds
-------------------

On top of :doc:`wbc-g1-ee`, ``apply_se_actor`` (``presets/se_actor.py``):

**Removes** from actor: ``ref_base_height``, ``ref_gravity_b``, ``projected_gravity``

**Adds:** ``motion_anchor_pos_error_w``, ``motion_anchor_ori_error``, ``base_lin_vel``

Dim rules: :ref:`reference-obs-dims`.

Train & play
------------

.. code-block:: bash

   uv run wbc-mjlab-train --task Wbc-G1-EE-SE --dataset samples
   uv run wbc-mjlab-play --task Wbc-G1-EE-SE --dataset samples --viewer viser

Logs: ``logs/rsl_rl/wbc_g1_ee_se/<run>/``
