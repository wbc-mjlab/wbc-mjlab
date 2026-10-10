.. _reference-configuration:

Configuration and files
=======================

Every file and directory wbc-mjlab reads or writes, and the formats of the YAML files you
may edit by hand.

.. contents:: On this page
   :local:
   :depth: 1

Directory layout
----------------

.. code-block:: text

   <repo>/
     data/<robot>/<dataset>/       motion clips — sources (*.csv, *.pkl) and npz/ after conversion
     logs/rsl_rl/<experiment>/     training runs (one folder per run)
     logs/studio/jobs/             Studio jobs: command, log, status (*.json, *.log)
     demos/                        bundled demo checkpoint

   ~/.config/wbc-mjlab/            per user, never in the repo
     presets/<name>.yaml           Studio presets
     modules/<name>.yaml           Studio modules
     robots/<id>/                  robots added from model files (robot.yaml + model)

A training run:

.. code-block:: text

   logs/rsl_rl/<experiment>/<timestamp>_<run_name>/
     model_<iteration>.pt          checkpoints
     events.out.tfevents.*         TensorBoard
     params/
       env.yaml, agent.yaml        the configs as trained
       policy.onnx                 deploy policy (actor only)
       config.yaml                 deploy tracking params (wbc_tracking_params_v1)
       rsi_bin_stats.npz           adaptive-sampling state (when persisted)
       studio_preset.yaml          Studio presets only: the preset as trained

``<experiment>`` comes from the task (``Wbc-G1`` → ``wbc_g1``); a Studio preset
``<task>--<name>`` logs to ``<experiment>--<name>``. ``config.yaml`` schema:
:doc:`../api/export`. Data layout: :doc:`data_formats`.

Environment variables
---------------------

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Variable
     - Effect
   * - ``WBC_MJLAB_CONFIG_DIR``
     - Per-user config directory (default ``$XDG_CONFIG_HOME/wbc-mjlab``, i.e.
       ``~/.config/wbc-mjlab``)
   * - ``MUJOCO_GL``
     - Rendering backend; ``egl`` on headless machines, ``disable`` for smoke tests
   * - ``CUDA_VISIBLE_DEVICES``
     - GPUs visible to training; ``--gpu-ids`` indexes into this list
   * - ``WBC_MJLAB_SKIP_MJLAB_REGISTER``
     - ``1`` skips task registration on import (docs builds, tooling)

.. _reference-preset-file:

Studio preset file
------------------

``presets/<name>.yaml`` — a diff on top of a registered task, written by the Studio
(*Preset → Save as*) or by hand. Registered as task ``<base>--<name>``.

.. code-block:: yaml

   base: Wbc-G1                  # registered task the preset starts from
   robot: g1                     # filled in on save
   modules: [se]                 # applied in order, before the overrides
   description: SE + keybody refs, history 2
   overrides:
     observations:
       actor:
         history_length: 2
         enable_corruption: true
         terms:
           ref_body_pos: {add: true, bodies: [pelvis, left_wrist_yaw_link]}
           joint_vel: {noise_scale: 2.0}
           projected_gravity: {enabled: false}
     rewards:
       motion_body_pos: {weight: 2.5, params: {sigma_per_keybody: 0.15}}
       anti_shake: {enabled: false}
     terminations:
       anchor_pos: {params: {threshold: 0.5}}
     command:
       rsi.bin_width_s: 2.0
       env.episode_length_s: 20.0
     events:
       push_robot: {params: {velocity_range.x: [-1.0, 1.0]}}
     terrain:
       curriculum.start_row: 2
       rough.noise_range: [0.0, 0.15]
     agent:
       algorithm.learning_rate: 5.0e-4
       actor.hidden_dims: [512, 256, 128]

.. list-table:: ``overrides`` sections
   :header-rows: 1
   :widths: 18 82

   * - Section
     - Keys
   * - ``observations``
     - Per group (``actor``, ``critic``): ``history_length`` (0 = none),
       ``enable_corruption``, and ``terms.<name>`` with ``enabled: false`` (drop),
       ``add: true`` (add from the robot's term library), ``noise_scale`` (multiplies the
       term's noise; 0 = off), ``history_length``, ``bodies`` (keybody subset; ``[]`` = all)
   * - ``rewards``
     - ``<term>: {enabled, weight, params: {<param>: value}}``; dict-valued parameters as
       ``<param>.<key>``
   * - ``terminations``, ``events``
     - ``<term>: {enabled, params: {…}}``
   * - ``command``
     - Motion-command fields by dotted path (``rsi.bin_width_s``,
       ``assistive_beta_max``, ``pose_range.x``); ``env.<field>`` for env fields
       (``env.episode_length_s``)
   * - ``terrain``
     - Procedural-terrain tasks only: ``curriculum.start_row``, ``curriculum.num_rows``,
       ``curriculum.difficulty_range``, ``<sub_terrain>.<field>``
       (``rough.noise_range``, ``perlin.proportion`` …)
   * - ``agent``
     - Runner fields by dotted path (``algorithm.learning_rate``, ``num_steps_per_env``,
       ``actor.hidden_dims``)

The Studio's *Preset* step lists every key a task accepts, with its default
(``GET /api/presets/schema?base=<task>``).

Studio module file
------------------

``modules/<name>.yaml`` — the same ``overrides`` without a ``base``, switched on in any
preset (``modules: [<name>]``).

.. code-block:: yaml

   description: actor without the last action
   robot: g1
   overrides:
     observations:
       actor:
         terms:
           actions: {enabled: false}

The built-in module ``se`` is computed from the robot's ``<task>-SE`` / ``<task>`` pair and
can't be edited (copy it with *Save as module…*).

Run snapshot
------------

``params/studio_preset.yaml`` — written when a Studio preset trains: the preset file with
``resolved_modules`` (each module's ``name`` and ``overrides``) inlined. Play rebuilds the
run from it, so later edits to the preset or its modules don't change how the run replays.

.. _reference-robot-file:

Robot spec (``robot.yaml``)
---------------------------

``robots/<id>/robot.yaml`` — a robot added from its model files
(:doc:`../extensions/custom_robots`).

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Field
     - Meaning
   * - ``id``
     - Robot id: lowercase letters, digits, ``_``; names the tasks (``h1`` → ``Wbc-H1``)
   * - ``model``
     - Source MJCF / URDF, relative to the robot folder (``wbc_robot.xml`` is derived from
       it on save)
   * - ``anchor_body``
     - Body the reference is expressed in (usually the torso)
   * - ``keybodies``
     - Tracked bodies (rewards, RSI, terminations)
   * - ``ee_bodies``
     - End-effectors (EE height termination, the EE task's command)
   * - ``feet_bodies``
     - Feet (contact, air time, conversion height correction)
   * - ``foot_sites``
     - Sites under the soles for the foot-slip reward (empty: reward dropped)
   * - ``foot_geoms``
     - Named foot collision geoms for friction randomization (empty: not randomized)
   * - ``foot_sole_z``
     - Foot body height above the floor when standing (m)
   * - ``init_height``, ``init_joint_pos``
     - Root height (m) and joint defaults (rad) of the initial pose
   * - ``joints``
     - Per joint: ``effort`` (N·m), ``stiffness``, ``damping``, ``armature`` (kg·m²)
   * - ``action_scale``
     - Policy action = ``action_scale`` × effort / stiffness (default 0.25)
   * - ``nconmax``, ``njmax``
     - Contact / constraint buffers per env
   * - ``description``
     - Shown in task listings

Python API: :ref:`api-custom-robots`.

.. seealso::

   :doc:`../studio` (where these files are edited) · :doc:`../usage` · :doc:`studio_api`
