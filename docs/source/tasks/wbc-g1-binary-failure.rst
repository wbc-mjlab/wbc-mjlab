BeyondMimic-style RSI
=====================

Task id: ``Wbc-G1-BinaryFailure`` · In-tree reference on robot ``g1``.

**BeyondMimic-style adaptive RSI** — body pose/vel rewards, ``binary_failure`` RSI,
full base actor obs (includes ``ref_joint_vel``).

- **Preset:** ``apply_binary_failure`` (``presets/binary_failure.py``)
- **Builder:** ``g1_wbc_binary_failure_env_cfg()``
- **Paper:** `arXiv:2508.08241 <https://arxiv.org/abs/2508.08241>`_

What this task changes
----------------------

Mutates **rewards + RSI only**; keeps base terminations and full actor layout from
``g1_base_cfg()`` (including ``ee_body_pos`` and ``ref_joint_vel``).

Actor dim rules: :ref:`reference-obs-dims`.

Rewards
~~~~~~~

.. list-table::
   :header-rows: 1

   * - Term
     - Weight
   * - ``motion_global_root_pos`` / ``motion_global_root_ori``
     - 0.5
   * - ``motion_body_pos`` / ``motion_body_ori`` / ``motion_body_lin_vel`` / ``motion_body_ang_vel``
     - 1.0 (all motion keybodies)
   * - ``motion_joint_pos`` / ``motion_joint_vel``
     - **0**
   * - ``survival``
     - **0**
   * - ``action_rate_l1`` / ``joint_acc``
     - −0.1 / −2e−6

RSI
~~~

- ``strategy="binary_failure"`` — failure on early termination
- ``similarity_terms = keybody_similarity_preset()``

See :doc:`../mdp/rsi`.

Train & play
------------

.. code-block:: bash

   uv run wbc-mjlab-train --task Wbc-G1-BinaryFailure --dataset samples
   uv run wbc-mjlab-play --task Wbc-G1-BinaryFailure --dataset samples --viewer viser

Logs: ``logs/rsl_rl/wbc_g1_binary/<run>/``

Tips
----

- Watch adaptive RSI panels in Viser — binary-failure bins differ from ZEST RSI.
