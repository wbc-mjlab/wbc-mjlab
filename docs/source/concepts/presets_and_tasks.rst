.. _presets:

Presets and tasks
=================

Papers on whole-body tracking differ in reward weights, sampling strategy, observation layout
and termination rules — rarely in the MDP machinery itself. wbc-mjlab encodes each method as
a **preset** and registers each robot + preset combination as a **task**, so methods are
switches on one codebase instead of forks.

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Concept
     - What it is
   * - **Preset**
     - A function ``apply_*(cfg, …) -> None`` that mutates an env config in place:
       reweights rewards, sets sampling flags, drops or swaps observation terms, changes
       termination thresholds. Reusable across robots.
   * - **Task**
     - A registered name (``--task Wbc-G1-Zest``) for one robot + preset stack, with its
       own log folder.
   * - **Studio preset**
     - A YAML diff on top of a task, made in the Studio — tuning without code (below).

How a task is built
-------------------

.. code-block:: text

   make_base_wbc_env_cfg()          shared MDP template — every term slot  (env/wbc_env_cfg.py)
            ↓
   <robot>_base_cfg()               robot: model, actuators, sensors, body names
            ↓
   apply_<preset>(cfg, …)           method recipe                          (presets/*.py)
            ↓
   apply_se_actor / apply_terrain_scan / …   optional overlays
            ↓
   WbcTaskConfig(task_id=…, build_env_cfg=…) registered --task id          (tasks/config.py)

Presets never add new term *implementations* — they configure the slots the template already
defines. New reward or sampling **callables** belong in ``env/mdp/``.

Core presets
------------

.. list-table::
   :header-rows: 1
   :widths: 24 22 54

   * - Function
     - Module
     - Recipe
   * - ``apply_wbc``
     - ``presets/wbc.py``
     - Default stack: ZEST Table S4 tracking on all keybodies + deploy extras
   * - ``apply_zest``
     - ``presets/zest.py``
     - ZEST reproduction: Table S4 on end-effectors, no EE-height termination
   * - ``apply_binary_failure``
     - ``presets/binary_failure.py``
     - BeyondMimic-style ``binary_failure`` sampling
   * - ``apply_end_effector``
     - ``presets/end_effector.py``
     - Overlay: actor command ``ref_body_*``, default-offset actions
   * - ``apply_se_actor``
     - ``presets/se_actor.py``
     - Overlay: state-estimation actor observations
   * - ``apply_terrain_scan``
     - ``presets/terrain.py``
     - Overlay: curriculum terrain, height scan, terrain-aware terminations

Signatures: :doc:`../api/presets`. The G1 tasks built from them: :doc:`../tasks/index`.

Anatomy of a preset
-------------------

Robot-specific names come in as **arguments** from the task builder, so the same preset works
on any robot:

.. code-block:: python

   def apply_zest(cfg, *, reward_body_names, contact_body_names, force_threshold=2000.0):
     rw = cfg.rewards
     rw["motion_body_pos"].params["body_names"] = reward_body_names   # Table S4 kernels
     rw["motion_body_pos"].params["sigma_per_keybody"] = 0.2
     rw["motion_body_lin_vel"].weight = 0.0                           # not in Table S4

     motion = cfg.commands["motion"]                                  # reward-aligned sampling
     motion.rsi = replace(motion.rsi, similarity_from_rewards=True, bin_width_s=4.0)

     cfg.observations["actor"].terms.pop("ref_joint_vel", None)       # deploy-style actor
     cfg.terminations.pop("ee_body_pos", None)                        # ZEST: no EE-height cutoff

And a task wires a robot to it:

.. code-block:: python

   def g1_wbc_zest_env_cfg():
     cfg = g1_base_cfg()
     apply_zest(cfg, reward_body_names=G1_ENDEFFECTOR_BODY_NAMES,
                contact_body_names=G1_MOTION_BODY_NAMES)
     return cfg

   WbcTaskConfig(task_id="Wbc-G1-Zest", robot_id="g1",
                 experiment_name="wbc_g1_zest", build_env_cfg=g1_wbc_zest_env_cfg)

Composing presets
-----------------

Overlays stack on a primary preset, in order: base config → primary preset → overlays →
robot-specific wiring (e.g. IMU sensor names).

.. code-block:: python

   def g1_wbc_ee_se_env_cfg():
     cfg = g1_wbc_env_cfg()        # apply_wbc
     apply_end_effector(cfg, body_names=G1_ENDEFFECTOR_BODY_NAMES)
     apply_se_actor(cfg)
     wire_g1_imu_sensors(cfg)
     return cfg

The same preset backs tasks on different robots (``apply_wbc`` → ``Wbc-G1``, ``Wbc-H2``,
``Wbc-<Id>``); one robot exposes many tasks by stacking different presets.

Studio presets and modules
--------------------------

For tuning without code, the :doc:`Studio <../studio>` adds a file-based layer on top of
tasks:

- a **Studio preset** — a YAML diff on a registered task (``base: Wbc-G1``): observation
  terms, reward weights and parameters, terminations, sampling, randomization, terrain, PPO;
  registered as ``<base>--<name>``;
- a **module** — a partial diff with no base, switched on per Studio preset
  (``modules: [se]``). The built-in ``se`` module is read off the ``<task>-SE`` /
  ``<task>`` pair — the same change ``apply_se_actor`` makes in code.

Order: task (code presets) → modules → the Studio preset's own changes. Each training run
keeps the resolved preset in ``params/studio_preset.yaml``. File format:
:ref:`reference-preset-file`.

Neutral code, cited methods
---------------------------

Code uses generic names (``similarity_ema``, ``binary_failure``); paper names live in task
descriptions and these docs. Cite the method papers when you use a task: :doc:`../research`.

.. seealso::

   - :doc:`../tasks/adding` — add a preset or a task
   - :doc:`modularity` — the layer model behind this
   - :doc:`../mdp/observations` — why reference observations are modular terms
