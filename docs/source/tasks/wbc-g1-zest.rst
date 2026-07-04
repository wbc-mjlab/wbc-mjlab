ZEST reproduction
=================

Task id: ``Wbc-G1-Zest`` · In-tree reference on robot ``g1``.

**ZEST paper reproduction** — Table S4 on end-effector bodies only, reward-aligned
RSI, no EE height termination.

- **Preset:** ``apply_zest`` (``presets/zest.py``)
- **Builder:** ``g1_wbc_zest_env_cfg()``
- **Paper:** `arXiv:2602.00401 <https://arxiv.org/abs/2602.00401>`_

Preset wiring
-------------

The task builder calls the preset with robot-specific body tuples:

.. code-block:: python

   def g1_wbc_zest_env_cfg() -> ManagerBasedRlEnvCfg:
     cfg = g1_base_cfg()
     apply_zest(
       cfg,
       reward_body_names=G1_ENDEFFECTOR_BODY_NAMES,  # 4 EE bodies
       contact_body_names=G1_MOTION_BODY_NAMES,
     )
     return cfg

What ``apply_zest`` mutates on ``cfg`` — see :doc:`index` for annotated preset
source and the preset ↔ task model.

Summary
-------

Rewards
~~~~~~~

- Same Table S4 **σ/κ** as :doc:`wbc-g1`, but body pos/ori on **4 EE bodies** only
- Disables extra vel terms and ``foot_slip`` / ``angular_momentum`` (weight **0**)

RSI
~~~

Identical reward-aligned settings to :doc:`wbc-g1` — :doc:`../mdp/rsi`.

Terminations
~~~~~~~~~~~~

- **Removes** ``ee_body_pos`` (no EE height cutoff)
- Keeps ``anchor_pos`` + ``keybody_ground_contact_force`` on full keybody set

Actor observations
------------------

- Removes ``ref_joint_vel`` (same as :doc:`wbc-g1`)

Dim rules: :ref:`reference-obs-dims`.

Train & play
------------

.. code-block:: bash

   uv run wbc-mjlab-train --task Wbc-G1-Zest --dataset samples
   uv run wbc-mjlab-train --robot g1 --no-state-estimation --dataset samples
   uv run wbc-mjlab-play --task Wbc-G1-Zest --dataset samples --viewer viser

Logs: ``logs/rsl_rl/wbc_g1_zest/<run>/``

Tips
----

- Inspect RSI bins in Viser during play.
- SE variant: :doc:`wbc-g1-zest-se`.
