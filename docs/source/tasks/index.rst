.. _tasks:
.. _which-task:

Choose a task
=============

A **task** is what you train: a robot plus a stack of presets (rewards, sampling,
observations, terminations), registered under one ``--task`` id with its own log folder.
All tasks share the same MDP, CLI and log layout, so results are directly comparable.

.. code-block:: bash

   uv run wbc-mjlab-list-envs                                  # every registered task
   uv run wbc-mjlab-train --task <TaskId> --dataset samples

By goal
-------

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: :octicon:`star` Default stack
      :link: wbc-g1
      :link-type: doc

      ``Wbc-G1`` — the starting point: tracking on all keybodies, adaptive sampling,
      deploy-ready actor. **Start here.**

   .. grid-item-card:: :octicon:`beaker` ZEST reproduction
      :link: wbc-g1-zest
      :link-type: doc

      ``Wbc-G1-Zest`` — the ZEST paper setup: tracking on end-effectors, reward-aligned
      sampling, no EE-height termination.

   .. grid-item-card:: :octicon:`beaker` BeyondMimic-style sampling
      :link: wbc-g1-binary-failure
      :link-type: doc

      ``Wbc-G1-BinaryFailure`` — adaptive sampling driven by binary failures.

   .. grid-item-card:: :octicon:`beaker` Success-guided sampling
      :link: wbc-g1-sgs
      :link-type: doc

      ``Wbc-G1-SGS`` — ``Wbc-G1`` with SGS: episodes start on clip bins near a target
      success rate instead of the hardest ones.

   .. grid-item-card:: :octicon:`location` End-effector command
      :link: wbc-g1-ee
      :link-type: doc

      ``Wbc-G1-EE`` — the policy sees hand and foot targets instead of joint references;
      deploy without a joint-space reference.

   .. grid-item-card:: :octicon:`pulse` State estimation
      :link: wbc-g1-se
      :link-type: doc

      ``Wbc-G1-SE``, ``Wbc-G1-Zest-SE``, ``Wbc-G1-EE-SE`` — the actor gets anchor pose error
      and IMU velocity instead of height / gravity references.

   .. grid-item-card:: :octicon:`milestone` Rough terrain
      :link: wbc-g1-terrain-scan
      :link-type: doc

      ``Wbc-G1-Terrain-Scan`` — curriculum terrain, a height scan in the actor,
      terrain-aware terminations.

Tasks of other robots follow the same names: ``Wbc-H2`` (extension), ``Wbc-<Id>``,
``Wbc-<Id>-SE`` and ``Wbc-<Id>-Terrain-Scan`` for robots added in the Studio.

Compare
-------

.. list-table::
   :header-rows: 1
   :stub-columns: 1
   :widths: 20 20 20 20 20

   * -
     - ``Wbc-G1``
     - ``Wbc-G1-Zest``
     - ``Wbc-G1-BinaryFailure``
     - ``Wbc-G1-EE``
   * - Presets
     - ``apply_wbc``
     - ``apply_zest``
     - ``apply_binary_failure``
     - ``apply_wbc`` + ``apply_end_effector``
   * - Body rewards
     - all keybodies
     - end-effectors only
     - all keybodies
     - all keybodies
   * - Actor command
     - ``ref_joint_pos``
     - ``ref_joint_pos``
     - ``ref_joint_*``
     - ``ref_body_pos`` / ``ref_body_ori``
   * - Action
     - residual on ``q_ref``
     - residual on ``q_ref``
     - residual on ``q_ref``
     - offset from default pose
   * - Adaptive sampling
     - reward-aligned ``similarity_ema``
     - reward-aligned ``similarity_ema``
     - ``binary_failure``
     - reward-aligned ``similarity_ema``
   * - EE height termination
     - yes
     - no
     - yes
     - yes

Variants stack on these: ``-SE`` adds ``apply_se_actor``, ``-Terrain-Scan`` adds
``apply_terrain_scan``. How presets compose: :doc:`../concepts/presets_and_tasks`.

Papers
------

.. list-table::
   :header-rows: 1
   :widths: 18 26 56

   * - Paper
     - Link
     - Tasks
   * - ZEST
     - `arXiv:2602.00401 <https://arxiv.org/abs/2602.00401>`_
     - ``Wbc-G1-Zest`` (reproduction), ``Wbc-G1`` (its tracking with deploy extras)
   * - BeyondMimic
     - `arXiv:2508.08241 <https://arxiv.org/abs/2508.08241>`_
     - ``Wbc-G1-BinaryFailure``
   * - SGS
     - `arXiv:2610.12465 <https://arxiv.org/abs/2610.12465>`_
     - ``Wbc-G1-SGS``; the ``sgs`` Studio module on any task
   * - SONIC
     - `arXiv:2511.07820 <https://arxiv.org/abs/2511.07820>`_
     - planned — :doc:`../roadmap`

Cite the method papers when you reproduce or compare against them: :doc:`../research`.

Tune a task without code
------------------------

In the :doc:`Studio <../studio>`, *Preset* starts from any task and edits observations,
rewards, terminations, sampling, terrain and PPO; the result trains as ``<task>--<name>``.
To add a task in code: :doc:`adding`.

.. toctree::
   :hidden:

   wbc-g1
   wbc-g1-zest
   wbc-g1-zest-se
   wbc-g1-binary-failure
   wbc-g1-sgs
   wbc-g1-se
   wbc-g1-ee
   wbc-g1-ee-se
   wbc-g1-terrain-scan
   adding
