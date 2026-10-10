.. _reference-data-formats:

Motion data formats
===================

Source clip formats, the per-clip NPZ written by conversion, the dataset layout and the
mirroring transform. How to prepare a dataset: :doc:`../data`.

.. contents:: On this page
   :local:
   :depth: 1

Dataset layout
--------------

.. code-block:: text

   data/<robot>/<dataset>/
     *.csv | *.pkl          source clips (or under raw/, preferred when it has clips)
     npz/<clip>.npz         converted clips — what training loads
     npz/<clip>_mirror.npz  mirrored copies (--mirror)
     <dataset>.npz          optional stacked bundle (--cache-motion-bundle)

Only READMEs, ``samples/`` and ``.gitkeep`` files under ``data/`` are version-controlled;
converted ``npz/`` folders and bundles stay local. Extension robots keep their data under
the extension's ``project_root``.

Source formats
--------------

Formats are defined in ``wbc_mjlab.motion.motion_formats``; the converter picks one from the
file extension, or ``--format <name>``.

.. list-table::
   :header-rows: 1
   :widths: 16 14 70

   * - Format
     - Extension
     - Content
   * - ``default``
     - ``.csv``
     - LAFAN / retarget CSV, no header, one row per frame:
       ``root_pos(3) + root_rot_xyzw(4) + joint_pos(n_dof)`` — metres, radians.
       Default rate 30 Hz (``--input-fps``).
   * - ``gmr_pkl``
     - ``.pkl``
     - `GMR <https://github.com/YanjieZe/GMR>`_ / ``bvh_to_robot`` pickle: a dict with
       ``fps``, ``root_pos`` (T×3), ``root_rot`` (T×4, xyzw), ``dof_pos`` (T×n_dof);
       optional ``joint_names`` / ``dof_joint_names`` / ``joint_order``

Joint values follow the robot model's joint order and count; a mismatch fails conversion
(``expected N joint DOFs … got M``).

Converted clip (NPZ)
--------------------

``wbc-mjlab-data-to-npz`` runs forward kinematics on the robot model and writes one NPZ per
source clip. Arrays (time ``T``, joints ``J``, model bodies ``B``):

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Key
     - Shape · meaning
   * - ``fps``
     - scalar — export rate (default 50 Hz, ``--output-fps``)
   * - ``joint_pos`` / ``joint_vel``
     - ``(T, J)`` — rad · rad/s
   * - ``body_pos_w``
     - ``(T, B, 3)`` — body positions, world frame (m)
   * - ``body_quat_w``
     - ``(T, B, 4)`` — body orientations, **xyzw**
   * - ``body_lin_vel_w`` / ``body_ang_vel_w``
     - ``(T, B, 3)`` — m/s · rad/s, world frame

Bodies follow the model's body order. Training selects the tracked keybodies through
``MotionCommandCfg.body_names``. A stacked bundle (``--cache-motion-bundle``) concatenates the
clips and stores segment starts and lengths so adaptive sampling can draw across the library.

With ``--debias-z``, each clip is shifted vertically so the lowest foot sole touches the
floor (the robot's ``foot_body_names`` / ``foot_sole_z``).

.. _reference-mirroring:

Mirroring
---------

``--mirror`` writes ``<clip>_mirror.npz`` next to each clip: the sagittal reflection, computed
after FK (``motion/motion_mirror.py``) from the robot's ``RobotSymmetryConfig`` (joint swap
pairs and signs; G1: ``robots/g1/symmetry.py``).

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Channel
     - Transform
   * - ``joint_pos``, ``joint_vel``
     - Swap left / right joint pairs; negate roll / yaw axes
   * - ``body_pos_w``, ``body_lin_vel_w``
     - Swap left / right bodies; flip world **y**
   * - ``body_quat_w`` (xyzw)
     - Swap bodies; negate quaternion **x** and **z**
   * - ``body_ang_vel_w``
     - Swap bodies; negate **x** and **z** (axial vector under reflection)
   * - ``fps``, ``joint_names``, ``robot``
     - Unchanged

Sources whose name already ends in ``_mirror`` are skipped, so re-converting doesn't produce
``_mirror_mirror`` clips. Robots without a symmetry config (including robots added in the
Studio) can't be mirrored.

Motion source flags (train / play)
----------------------------------

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - Flag
     - Loads
   * - ``--dataset <name>``
     - ``data/<robot>/<name>/npz/*.npz`` (robot from the task)
   * - ``--dataset-path <dir>``
     - ``<dir>/npz/*.npz``
   * - ``--dataset-path <file>.npz`` · ``--motion-file <file>.npz``
     - That file
   * - ``--cache-motion-bundle``
     - Write / reuse ``<dataset>/<dataset>.npz``

.. seealso::

   :doc:`../data` · :doc:`../usage` · :doc:`../visualization`
