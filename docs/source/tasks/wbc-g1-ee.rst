End-effector actor command
==========================

Task id: ``Wbc-G1-EE`` · In-tree reference on robot ``g1``.

**Same WBC rewards / RSI / terminations as** :doc:`wbc-g1`. The actor drops
``ref_joint_pos`` and instead sees the existing keybody pose reference
(``ref_body_pos`` / ``ref_body_ori``) on wrists and feet. The critic and
joint-tracking rewards still see the motion joint reference during training.

- **Presets:** ``apply_wbc`` → ``apply_end_effector``
- **Builder:** ``g1_wbc_ee_env_cfg()``

What this task changes
----------------------

Actor observations
~~~~~~~~~~~~~~~~~~

Removes ``ref_joint_pos`` / ``ref_joint_vel``. Adds the **already-defined**
critic terms ``ref_body_pos`` / ``ref_body_ori`` (same functions, no MDP
changes). These are reference keybody poses — not current robot state.

Critic and rewards
~~~~~~~~~~~~~~~~~~

Unchanged from :doc:`wbc-g1`.

Actions
~~~~~~~

Switches to default-offset actions (``q_cmd = q_default + scale · a``). After
RSI, ``a`` is seeded from ``q_ref`` for the first frame only. Deploy does not
need the joint stream.

Dim rules: :ref:`reference-obs-dims`.

Train & play
------------

.. code-block:: bash

   uv run wbc-mjlab-train --task Wbc-G1-EE --dataset samples
   uv run wbc-mjlab-play --task Wbc-G1-EE --dataset samples --viewer viser

Logs: ``logs/rsl_rl/wbc_g1_ee/<run>/``
