.. _observations:
.. _reference-obs-dims:

Observations
============

Manager: **ObservationManager** · Module: ``env/mdp/observations.py``

Groups
------

**Actor** (``wbc_env_cfg.py`` template + preset tweaks):

- **Reference command terms** (``ref_*``) — slices of the active ``MotionCommand``
  (``command_name="motion"``): base height, gravity, joint pos/vel, base vel in body
  frame. Exposed as **separate observation terms**, not one fixed vector — you pick
  which fields the policy sees, reorder them, or attach
  per-term corruption noise (see below).
- **Proprioception** — projected gravity, joint pos/vel, last action
- Optional corruption noise per term

**Critic** — actor terms without noise, plus **privileged** features: anchor/body
pose errors, reference body kinematics, assistive wrench state, motion segment phase,
per-step tracking reward breakdown.

Modular reference terms
-----------------------

Reference observations are **part of the motion command** but wired as independent
``ObservationTermCfg`` entries in ``cfg.observations["actor"].terms``. Each ``ref_*``
function reads one slice from ``MotionCommand``; mjlab concatenates active terms in
cfg order.

Why modularity matters
~~~~~~~~~~~~~~~~~~~~~~

Modular ``ref_*`` terms exist for two related reasons:

**Different papers and deploy stacks.** Published trackers do not agree on a single
actor observation vector — which reference fields the policy sees, whether
``ref_joint_vel`` is included, or whether state estimation replaces global
height/gravity with anchor pose error. Hard-coding one layout in the MDP would fork
rewards, RSI, and ``MotionCommand`` for every reproduction.

**Per-channel corruption.** Splitting reference kinematics into named terms also
lets you attach **independent uniform noise** to each channel (height, gravity, joint
pos, body-frame reference velocity, …). That matches training practice in stacks
such as SONIC, and is awkward when reference fields are fused into one bundled
``command`` vector. The base template corrupts most ``ref_*`` terms by default;
presets can retune or disable noise per term when matching a published layout.

**Sensing stack.** What the robot can measure at deploy time also drives layout:
IMU-only platforms use height/gravity proxies; odometry or mocap can use anchor pose
error instead (:doc:`../tasks/wbc-g1-se`).

Examples in this repo:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Stack / task
     - Reference-related actor layout
   * - **ZEST / deploy** (``Wbc-G1``, ``Wbc-G1-Zest``)
     - Modular ``ref_*`` terms; **no** ``ref_joint_vel`` (deploy-style)
   * - **BeyondMimic-style** (``Wbc-G1-BinaryFailure``)
     - Full base template including ``ref_joint_vel``
   * - **State estimation** (``Wbc-G1-SE``, ``Wbc-G1-Zest-SE``)
     - Drops height/gravity proxies; adds anchor pose error + ``base_lin_vel``
   * - **Bundled command** (mjlab tracking, some deploy runtimes)
     - Single ``command`` vector instead of per-field ``ref_*`` terms

Presets assemble, drop, swap, and noise individual reference channels while sharing
the same rewards, RSI, and ``MotionCommand`` — see :doc:`../tasks/index`,
:doc:`../concepts/presets_and_tasks`, and :doc:`../research` for method citations.

Typical customizations (no MDP fork):

- **Drop a field** — e.g. ``apply_wbc`` / ``apply_zest`` remove ``ref_joint_vel`` for
  deploy-style stacks; add it back with ``terms["ref_joint_vel"] = ObservationTermCfg(...)``
- **Swap layout** — e.g. ``apply_se_actor`` replaces height/gravity proxies with anchor
  error terms
- **Set noise** — each term accepts ``noise=Unoise(...)``; the base template sets
  per-channel corruption on most ``ref_*`` terms (e.g. ``ref_joint_pos`` ±0.05,
  ``ref_joint_vel`` ±0.25 when present)

Example (from ``wbc_env_cfg.py``):

.. code-block:: python

   motion = {"command_name": "motion"}
   actor_terms = {
     "ref_joint_pos": ObservationTermCfg(
       func=mdp.ref_joint_pos,
       params=motion,
       noise=Unoise(n_min=-0.05, n_max=0.05),
     ),
     # pop or add terms in a task builder / preset:
     # cfg.observations["actor"].terms.pop("ref_joint_vel", None)
   }

Deploy export may also record a bundled ``command`` vector (``10+J``) for runtimes that
prefer a single reference input — that is separate from this modular sim layout. See
:ref:`obs-config-yaml`.

Reference observation dimensions
--------------------------------

Actor observations are **modular**: ``make_base_wbc_env_cfg()`` defines named
**terms**; presets **remove or swap** reference terms without forking MDP code.
Each term contributes a fixed **shape rule**; robot-specific sizes use:

- **``J``** — number of actuated / tracked joints (robot model)
- **``B``** — number of motion keybodies (``MotionCommandCfg.body_names``)

Dim rules match ``wbc_mjlab.export.tracking_params_yaml._observation_dim`` (used when
writing ``config.yaml``). For a **trained task**, the exported file is ground truth
— see :ref:`obs-config-yaml`.

Default actor reference terms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

From ``wbc_env_cfg.py`` (before preset tweaks):

.. list-table::
   :header-rows: 1
   :widths: 28 10 62

   * - Term
     - Dim
     - Meaning
   * - ``ref_base_height``
     - 1
     - Reference anchor height (z) relative to env origin
   * - ``ref_base_lin_vel_b``
     - 3
     - Reference anchor linear velocity in anchor (body) frame
   * - ``ref_base_ang_vel_b``
     - 3
     - Reference anchor angular velocity in anchor frame
   * - ``ref_gravity_b``
     - 3
     - Reference gravity direction in anchor frame
   * - ``ref_joint_pos``
     - ``J``
     - Reference joint positions (tracked DoFs)
   * - ``ref_joint_vel``
     - ``J``
     - Reference joint velocities (often removed for deploy — see below)

Optional reference terms (tasks / critic)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Not in the default **actor** template, but supported in export dim rules or used on
**critic** / SE layouts:

.. list-table::
   :header-rows: 1
   :widths: 28 10 62

   * - Term
     - Dim
     - Typical use
   * - ``ref_anchor_pos_w``
     - 3
     - Reference anchor position in world frame (SE / alternate layouts)
   * - ``ref_anchor_ori_6d``
     - 6
     - Reference anchor orientation (6D rotation)
   * - ``motion_anchor_pos_error_w``
     - 3
     - Anchor position error (SE actor — :doc:`../tasks/wbc-g1-se`)
   * - ``motion_anchor_ori_error``
     - 3
     - Anchor orientation error (axis-angle)
   * - ``ref_body_pos`` / ``body_pos``
     - ``3×B``
     - Keybody positions (critic / privileged)
   * - ``ref_body_ori`` / ``body_ori``
     - ``6×B``
     - Keybody orientations (critic / privileged)
   * - ``command``
     - ``10+J``
     - Bundled reference command vector (deploy runtimes; rarely in sim actor)

Default actor **proprio** terms (same dim rules, not ``ref_*``):

.. list-table::
   :header-rows: 1
   :widths: 28 10 62

   * - Term
     - Dim
     - Meaning
   * - ``base_ang_vel``
     - 3
     - Robot anchor angular velocity (IMU)
   * - ``projected_gravity``
     - 3
     - Gravity in robot base frame
   * - ``joint_pos`` / ``joint_vel`` / ``actions``
     - ``J``
     - Proprio joint state and last action

Preset changes to reference obs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Preset / task
     - Reference-related change
   * - ``apply_wbc``, ``apply_zest``
     - Remove ``ref_joint_vel`` from actor (deploy-style)
   * - ``apply_se_actor`` (``Wbc-G1-SE``, ``Wbc-G1-Zest-SE``)
     - Remove ``ref_base_height``, ``ref_gravity_b``, ``projected_gravity``; add
       ``motion_anchor_pos_error_w``, ``motion_anchor_ori_error``, ``base_lin_vel``
   * - Default ``Wbc-G1-BinaryFailure``
     - Keeps full base actor including ``ref_joint_vel``

Task-specific details: :doc:`../tasks/index`.

Total actor observation size
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Terms are **concatenated in cfg order** (``cfg.observations["actor"].terms``):

.. code-block:: text

   actor_dim = sum(dim(term) for term in active_actor_terms) × history_length

``history_length`` may be set on the observation **group** or on individual terms
(mjlab ``ObservationManager``). Presets such as ``apply_zest`` can set actor
``history_length`` > 1.

Reference command size for deploy (``wbc_command_dim``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Separate from the raw actor vector: export records how much reference motion the
**deploy runtime** must feed when playing clips.

- If actor includes a bundled ``command`` term: ``wbc_command_dim = 10 + J``
- Otherwise: sum of dims for active ``ref_*`` terms listed in
  ``REFERENCE_OBS_TERM_NAMES`` (see :doc:`../api/export`)

.. _obs-config-yaml:

Ground truth: ``config.yaml``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After train or play, open ``logs/rsl_rl/<experiment>/<run>/params/config.yaml``:

.. code-block:: yaml

   actor_observations:
     ref_base_height:
       dim: 1
       scale: [1.0]
       params: {command_name: motion}
     ref_joint_pos:
       dim: 29          # example — equals J for your robot
       scale: [1.0, ...]
   tracking:
     reference_observation_names: [ref_base_height, ref_joint_pos, ...]
     actor_observation_names: [...]   # full actor stack + order
     wbc_command_dim: 39              # example — depends on active ref terms
     has_state_estimation: false

Do not hard-code G1 numbers in your runtime — read them from the checkpoint's
``config.yaml``. More: :doc:`../workflows/deploy` · :doc:`../api/export`.

Presets and robot wiring
------------------------

- ``apply_wbc`` / ``apply_zest`` drop ``ref_joint_vel`` from the actor
- ``apply_se_actor`` swaps height/gravity proxies for anchor pose error + ``base_lin_vel``
- Robot entities call ``wire_<robot>_imu_sensors`` when SE layouts need named IMU sensors

Body lists and anchor frame come from the robot's ``<robot>_base_cfg()`` — not from
observation modules directly.

API: :doc:`../api/mdp` (Observations).

Related: :doc:`motion_command`, :doc:`../tasks/index`, :doc:`../extensions/robot_entity`.
