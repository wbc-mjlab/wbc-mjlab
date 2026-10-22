.. _custom-robots:

Add a robot from model files
============================

A humanoid's **MJCF or URDF with its meshes** is enough to train on it: the Studio's
**Robot → Add robot** (:ref:`studio`) turns the model files into a registered robot with
tasks, data folder and motion conversion — no extension package. The result is a folder
with a YAML spec; :mod:`wbc_mjlab.robots.custom` registers it in every process (Studio,
``wbc-mjlab-train``, ``wbc-mjlab-data-to-npz``, ``wbc-mjlab-list-envs``).

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * -
     - Added in the Studio
     - Extension package (:doc:`index`)
   * - **Setup**
     - Upload the model, check the suggested wiring, save
     - Python package: constants, actuators, ``register_wbc_extension``
   * - **Lives in**
     - ``~/.config/wbc-mjlab/robots/<id>/`` (per user)
     - Its own repo, installed next to wbc-mjlab
   * - **Tasks**
     - ``Wbc-<Id>`` (default WBC stack), ``Wbc-<Id>-SE``, ``Wbc-<Id>-Terrain-Scan``
     - Any task table
   * - **Actuators**
     - Ideal PD per joint
     - Anything mjlab supports (grouped motors, torque-speed envelopes …)
   * - **Mirroring**
     - Not available (no symmetry config)
     - ``RobotSymmetryConfig``
   * - **Best for**
     - Trying a robot, research on many embodiments
     - Shipping a platform, deploy-grade actuator models

Start in the Studio; move to an extension when the robot needs more than the spec below —
the saved ``robot.yaml`` holds the numbers the extension's constants need.

What *Add robot* does
---------------------

**Prepare the model.** The source model stays untouched; ``wbc_robot.xml`` is written next to
it, which is what wbc-mjlab loads:

- the model's own **actuators** are removed (their force limits become the suggested
  efforts) — wbc-mjlab adds its own PD actuators;
- **keyframes**, world geoms and props (floor, scene objects) are removed — the task adds its
  terrain;
- the root body gets a **free joint** if it has none (fixed-base robots aren't supported);
- the sensors the WBC tasks read are added when missing: IMU ``imu_ang_vel`` /
  ``imu_lin_vel`` (gyro and velocimeter at the model's ``imu`` site, or the root) and
  ``root_angmom`` (subtree angular momentum of the root);
- URDF meshes are resolved around the URDF (``package://`` paths included). MuJoCo reads
  STL, OBJ and MSH meshes only — convert ``.dae`` first.

**Suggest the wiring** from body names and the kinematic tree — on G1 the suggestions match
the hand-written constants exactly:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Item
     - Suggestion
   * - Anchor
     - A torso / chest / trunk body, else the root
   * - Feet
     - The deepest ankle / foot / toe body per side, else the two lowest leaves
   * - End-effectors
     - The feet + the deepest wrist / hand / palm body per side, else the deepest leaf per
       side (heads and sensors excluded)
   * - Tracked keybodies
     - Root, anchor, and per limb a proximal (hip / shoulder), middle (knee / elbow) and end
       body — 13–14 bodies on a typical humanoid
   * - Foot sites, foot geoms
     - Sites under the feet (foot-slip reward) and named collision geoms of the feet
       (friction randomization)
   * - Root height, sole height
     - Root height that puts the lowest collision geom on the floor; foot body height above
       its sole (conversion height correction)

**Suggest PD gains.** Per joint: armature and effort from the model when it has them (joint
``armature``, ``actuatorfrcrange``, actuator force / control ranges); otherwise G1's values for
that kind of joint (hip, knee, ankle, waist, shoulder / elbow, wrist, head) scaled by robot
mass. Gains then follow the H2 extension's rule: stiffness = armature·ω², damping =
2ζ·armature·ω with ω = 2π·10 Hz, ζ = 2. Policy action scale = 0.25 · effort / stiffness.

**Check.** Drops the robot on a floor in its initial pose and holds the pose with these gains
for 0.5 s (CPU MuJoCo, no policy). Fails: the simulation blows up, the robot starts inside
the floor, it lands on bodies other than its feet, torque limits saturate. Warns: it sags
(plain PD isn't expected to balance for long — G1 with its own gains sags after about a
second; the policy does the balancing).

**Save.** Validates the names, writes ``wbc_robot.xml`` and ``robot.yaml`` and registers the
robot immediately with three tasks:

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Task
     - Stack
   * - ``Wbc-<Id>``
     - ``apply_wbc`` — the default WBC tracking stack with RSI
   * - ``Wbc-<Id>-SE``
     - also ``apply_se_actor`` (state-estimation actor, IMU base velocity)
   * - ``Wbc-<Id>-Terrain-Scan``
     - also ``apply_terrain_scan``: curriculum terrain, a height scan hanging from the root body
       (offset by the root height), terrain-aware anchor / end-effector terminations — as
       :doc:`../tasks/wbc-g1-terrain-scan`

Data goes to ``data/<id>/``, runs to ``logs/rsl_rl/wbc_<id>/`` (``wbc_<id>_se``,
``wbc_<id>_terrain_scan``).

After that the robot is like any other:

.. code-block:: bash

   uv run wbc-mjlab-list-envs                     # Robots: g1, h1 … Wbc-H1, Wbc-H1-SE, Wbc-H1-Terrain-Scan
   uv run wbc-mjlab-train --task Wbc-H1 --dataset my_clips --debias-z

Clips must be retargeted to the robot (joint order of the model, see :doc:`../data`).

``robot.yaml``
--------------

Written by the Studio; hand edits are picked up on the next start (or re-save in the
Studio). Every field: :ref:`reference-robot-file`.

.. code-block:: yaml

   # ~/.config/wbc-mjlab/robots/h1/robot.yaml
   id: h1
   model: h1.xml                  # source model, relative to this folder
   anchor_body: torso_link
   keybodies: [pelvis, torso_link, left_hip_roll_link, left_knee_link, left_ankle_link, …]
   ee_bodies: [left_ankle_link, right_ankle_link, left_elbow_link, right_elbow_link]
   feet_bodies: [left_ankle_link, right_ankle_link]
   foot_sites: []                 # none: the foot-slip reward is dropped
   foot_geoms: []                 # none: foot friction isn't randomized
   foot_sole_z: 0.06
   init_height: 1.05
   init_joint_pos: {left_knee_joint: 0.3, …}
   joints:
     left_knee_joint: {effort: 274.8, stiffness: 195.9, damping: 12.5, armature: 0.0496}
     …
   action_scale: 0.25
   description: Unitree H1 from its URDF
   nconmax: 188                   # contact / constraint buffers per env
   njmax: 564

Task ids: ``h1`` → ``Wbc-H1``, ``h2x`` → ``Wbc-H2X``, ``my_bot`` → ``Wbc-My-Bot`` (short parts
with a digit upper-case, the rest capitalised).

Python API
----------

.. code-block:: python

   import mjlab  # registers tasks, including Studio robots
   from wbc_mjlab.robots import custom

   info = custom.inspect_model(custom.robots_dir() / "h1" / "h1.xml")   # suggestions
   spec = custom.load_spec("h1")
   report = custom.settle_test(spec)            # {"ok": …, "checks": [...]}
   custom.save_spec(spec)                       # validate, write, (re)register

Reference: :ref:`api-custom-robots`.
