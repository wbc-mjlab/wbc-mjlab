Default WBC stack
=================

Task id: ``Wbc-G1`` · In-tree reference on robot ``g1``.

**Default WBC + deploy stack** — ZEST Table S4 tracking on all motion keybodies,
reward-aligned RSI, deploy-style actor obs.

- **Preset:** ``apply_wbc`` (``presets/wbc.py``) — see :doc:`index`
- **Builder:** ``g1_wbc_env_cfg()`` = ``g1_base_cfg()`` + ``apply_wbc(...)``
- **Paper link:** ZEST Table S4 (with deploy extras) — `arXiv:2602.00401 <https://arxiv.org/abs/2602.00401>`_

What this task changes
----------------------

Rewards
~~~~~~~

- Table S4 on **all 14** motion keybodies (``G1_MOTION_BODY_NAMES`` in core)
- Extra: ``motion_body_lin_vel`` / ``motion_body_ang_vel`` @ **0.5**
- ``motion_joint_vel``, ``foot_slip``, ``angular_momentum`` @ **0**
- Regularizers: ``survival`` 1.0, ``action_rate_l1`` −0.1, ``joint_acc`` −5e−6,
  ``joint_limit`` −1.0, ``actuator_torque_soft_limit`` −0.1

RSI
~~~

Reward-aligned ``similarity_ema`` — :doc:`../mdp/rsi`.

Terminations
~~~~~~~~~~~~

- ``anchor_pos`` z **0.35**
- ``ee_body_pos`` z-only on ankles + wrists (**0.25**)
- ``keybody_ground_contact_force`` > 2000 N

Actor observations
------------------

- Removes ``ref_joint_vel`` (deploy-style)

Dim rules and preset deltas: :ref:`reference-obs-dims`.

Train & play
------------

.. code-block:: bash

   uv run wbc-mjlab-train --task Wbc-G1 --dataset samples
   uv run wbc-mjlab-train --robot g1 --dataset samples   # shorthand
   uv run wbc-mjlab-play --task Wbc-G1 --dataset samples --viewer viser

Logs: ``logs/rsl_rl/wbc_g1/<run>/`` · ``params/policy.onnx`` · ``params/config.yaml``

Tips
----

- Best starting point for multi-clip training and hardware deploy export.
- Compare against :doc:`wbc-g1-zest` for ZEST paper repro (4 EE bodies, no EE-z term).
- Add ``--cache-motion-bundle`` on train/play for faster restarts.
- See :doc:`../workflows/deploy` for export handoff.
