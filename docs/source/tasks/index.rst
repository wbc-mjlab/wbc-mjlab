.. _tasks:
.. _presets:

Tasks and presets
=================

A **preset** is a pure Python function ``apply_*(cfg) -> None`` that **mutates** an
existing ``ManagerBasedRlEnvCfg`` in place. A **task** is a **registered name** for
one specific stack: robot entity + preset(s) + optional wiring.

Not sure which task to run first? See :ref:`which-task` on the Quickstart.

Presets never add new MDP term *implementations* — they reweight rewards, swap RSI
flags, remove observation terms, or adjust termination thresholds on slots already
defined in ``make_base_wbc_env_cfg()``. Tasks are **not** separate MDP forks.

How they connect
----------------

.. code-block:: text

   make_base_wbc_env_cfg()          # shared MDP template (env/wbc_env_cfg.py)
            ↓
   <robot>_base_cfg()               # robot entity: MJCF, body names, sensors
            ↓
   apply_<preset>(cfg, ...)         # paper / deploy recipe (presets/*.py)
            ↓
   [optional] apply_se_actor(cfg)   # composable obs swap
            ↓
   WbcTaskConfig(                   # CLI registration (tasks/config.py)
     task_id="Wbc-…",
     build_env_cfg=<builder above>,
   )

**Preset** = recipe (reusable across robots). **Task** = one named builder + log dir.
The same preset can back multiple tasks (e.g. ``apply_wbc`` on ``g1`` and ``h2``).

Available presets (core)
------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   * - Function
     - Module
     - Typical use
   * - ``apply_wbc``
     - ``presets/wbc.py``
     - Default WBC + deploy extras (Table S4 on all keybodies)
   * - ``apply_zest``
     - ``presets/zest.py``
     - ZEST paper repro (Table S4 on EE bodies only, no EE-z termination)
   * - ``apply_binary_failure``
     - ``presets/binary_failure.py``
     - BeyondMimic-style ``binary_failure`` RSI
   * - ``apply_se_actor``
     - ``presets/se_actor.py``
     - Actor obs swap (compose on top of wbc/zest)

Preset → task map (in-tree example)
-----------------------------------

.. list-table::
   :header-rows: 1
   :widths: 28 36 36

   * - Task id
     - Preset stack
     - Builder (``robots/g1/tasks.py``)
   * - ``Wbc-G1``
     - ``apply_wbc``
     - ``g1_wbc_env_cfg``
   * - ``Wbc-G1-SE``
     - ``apply_wbc`` → ``apply_se_actor`` + IMU wiring
     - ``g1_wbc_se_env_cfg``
   * - ``Wbc-G1-Zest``
     - ``apply_zest``
     - ``g1_wbc_zest_env_cfg``
   * - ``Wbc-G1-Zest-SE``
     - ``apply_zest`` → ``apply_se_actor`` + IMU wiring
     - ``g1_wbc_zest_se_env_cfg``
   * - ``Wbc-G1-BinaryFailure``
     - ``apply_binary_failure``
     - ``g1_wbc_binary_failure_env_cfg``

Extension tasks (e.g. ``Wbc-H2``) use the same pattern — see
:doc:`../extensions/extensions`.

Philosophy
----------

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Principle
     - Meaning
   * - **Shared MDP**
     - One ``env/``; presets reweight cfg
   * - **Tasks, not forks**
     - Paper differences = ``presets/*.py`` + task builders
   * - **Robot entity separate**
     - Each platform registered via :doc:`../extensions/robot_entity`
   * - **Neutral code**
     - RSI strategies named generically; cite papers from :doc:`../research`

Quick start (example)
---------------------

.. code-block:: bash

   uv run wbc-mjlab-data-to-npz --robot g1 --dataset samples --batch-size 8
   uv run wbc-mjlab-list-envs
   uv run wbc-mjlab-train --task Wbc-G1 --dataset samples

Paper ↔ task map (in-tree example)
-----------------------------------

.. list-table::
   :header-rows: 1
   :widths: 18 22 60

   * - Paper
     - Link
     - Example task ids
   * - **ZEST**
     - `arXiv:2602.00401 <https://arxiv.org/abs/2602.00401>`_
     - ``Wbc-G1-Zest``, ``Wbc-G1``, SE variants
   * - **BeyondMimic**
     - `arXiv:2508.08241 <https://arxiv.org/abs/2508.08241>`_
     - ``Wbc-G1-BinaryFailure``
   * - **SONIC**
     - `arXiv:2511.07820 <https://arxiv.org/abs/2511.07820>`_
     - Planned — :doc:`../roadmap`

Preset comparison (in-tree example)
-----------------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 26 26 26

   * -
     - ``Wbc-G1``
     - ``Wbc-G1-Zest``
     - ``Wbc-G1-BinaryFailure``
   * - Preset
     - ``apply_wbc``
     - ``apply_zest``
     - ``apply_binary_failure``
   * - Body rewards
     - all keybodies
     - EE bodies only
     - all keybodies
   * - RSI
     - reward-aligned ``similarity_ema``
     - reward-aligned ``similarity_ema``
     - ``binary_failure``
   * - ``ee_body_pos`` term
     - yes
     - **removed**
     - yes (base)

Example: ``apply_zest`` (preset code)
-------------------------------------

``presets/zest.py`` — mutates **rewards**, **RSI**, **observations**, and
**terminations** on the cfg object passed in. Robot-specific body tuples come from
**arguments** (supplied by the task builder):

.. code-block:: python

   def apply_zest(
     cfg: ManagerBasedRlEnvCfg,
     *,
     reward_body_names: tuple[str, ...],
     contact_body_names: tuple[str, ...],
     force_threshold: float = 2000.0,
   ) -> None:
     rw = cfg.rewards

     # Table S4 tracking kernels on selected bodies
     rw["motion_body_pos"].weight = 1.0
     rw["motion_body_pos"].params["sigma_per_keybody"] = 0.2
     rw["motion_body_pos"].params["body_names"] = reward_body_names
     rw["motion_body_ori"].params["sigma_per_keybody"] = 0.4
     rw["motion_body_ori"].params["body_names"] = reward_body_names
     # ... root + joint terms, kappa=0.25 ...

     # Disable extras not in ZEST Table S4
     rw["motion_body_lin_vel"].weight = 0.0
     rw["motion_joint_vel"].weight = 0.0
     rw["foot_slip"].weight = 0.0

     # Reward-aligned RSI
     motion_cmd = cfg.commands["motion"]
     motion_cmd.rsi = replace(
       motion_cmd.rsi,
       similarity_from_rewards=True,
       bin_width_s=4.0,
       similarity_norm_by_remaining_clip=True,
       min_bin_span_ratio=0.5,
       persist_failure_levels=True,
     )

     # Actor: drop ref_joint_vel
     cfg.observations["actor"].terms.pop("ref_joint_vel", None)

     # Terminations: ZEST style — no EE height cutoff
     cfg.terminations.pop("ee_body_pos", None)
     cfg.terminations["anchor_pos"].params["threshold"] = 0.35
     cfg.terminations["keybody_ground_contact_force"] = TerminationTermCfg(
       func=mdp.excessive_keybody_ground_contact_force,
       params={"body_names": contact_body_names, "force_threshold": force_threshold, ...},
     )

Full source: ``src/wbc_mjlab/presets/zest.py``.

Example: task builder + registration
------------------------------------

.. code-block:: python

   def g1_wbc_zest_env_cfg() -> ManagerBasedRlEnvCfg:
     cfg = g1_base_cfg()
     apply_zest(
       cfg,
       reward_body_names=G1_ENDEFFECTOR_BODY_NAMES,
       contact_body_names=G1_MOTION_BODY_NAMES,
     )
     return cfg

   G1_WBC_TASKS = (
     WbcTaskConfig(
       task_id="Wbc-G1-Zest",
       robot_id="g1",
       experiment_name="wbc_g1_zest",
       build_env_cfg=g1_wbc_zest_env_cfg,
     ),
     # ...
   )

Training uses the **task id**, not the preset name:

.. code-block:: bash

   uv run wbc-mjlab-train --task Wbc-G1-Zest --dataset samples

Composing presets
-----------------

.. code-block:: python

   def g1_wbc_zest_se_env_cfg() -> ManagerBasedRlEnvCfg:
     cfg = g1_wbc_zest_env_cfg()
     apply_se_actor(cfg)
     wire_g1_imu_sensors(cfg)
     return cfg

Order matters: base cfg → primary paper preset → optional obs/IMU tweaks.

When to add a preset vs a task
------------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Change
     - Where
   * - New reward/RSI **implementation**
     - ``env/mdp/`` (shared; needs review)
   * - New paper **weights / RSI flags / term toggles**
     - New ``presets/<method>.py``
   * - New ``--task`` id for a robot
     - Task builder + ``WbcTaskConfig`` in ``robots/<id>/tasks.py`` or extension
   * - Same preset, different robot
     - New task builder with that robot's body name tuples — preset unchanged

Task guides (in-tree)
---------------------

.. list-table::
   :header-rows: 1

   * - Task
     - Logs
     - Guide
   * - ``Wbc-G1``
     - ``logs/rsl_rl/wbc_g1/``
     - :doc:`wbc-g1`
   * - ``Wbc-G1-SE``
     - ``logs/rsl_rl/wbc_g1_se/``
     - :doc:`wbc-g1-se`
   * - ``Wbc-G1-Zest``
     - ``logs/rsl_rl/wbc_g1_zest/``
     - :doc:`wbc-g1-zest`
   * - ``Wbc-G1-Zest-SE``
     - ``logs/rsl_rl/wbc_g1_zest_se/``
     - :doc:`wbc-g1-zest-se`
   * - ``Wbc-G1-BinaryFailure``
     - ``logs/rsl_rl/wbc_g1_binary/``
     - :doc:`wbc-g1-binary-failure`

.. toctree::
   :maxdepth: 1
   :hidden:

   wbc-g1
   wbc-g1-se
   wbc-g1-zest
   wbc-g1-zest-se
   wbc-g1-binary-failure
   adding

Related: :doc:`adding`, :doc:`../mdp/index`, :doc:`../architecture`.
