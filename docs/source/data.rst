.. _data:

Motion data
===========

Motion libraries for WBC tracking live under **`data/<robot>/<dataset_name>/`**.

.. tip::

   First time? Convert the bundled samples and train with ``Wbc-G1`` — see
   :doc:`workflows/quickstart`. Errors like empty ``npz/``:
   :doc:`troubleshooting`.

A small **`samples/`** folder is version-controlled under each robot (e.g.
``g1/samples/``) with a few clips from public datasets for smoke tests. All other
dataset folders stay local — download full libraries from Hugging Face (see robot
guides) or convert your own clips.

Only READMEs, ``samples/``, and ``.gitkeep`` placeholders are version-controlled;
other clip and bundle files stay gitignored.

Robot guides
------------

.. list-table::
   :header-rows: 1

   * - Robot
     - Guide
   * - **g1** (in-tree example)
     - ``data/g1/README.md`` — downloads, joint order, example datasets

Directory layout
----------------

.. code-block:: text

   data/<robot>/<dataset>/
     *.csv / *.pkl          # source clips (or under raw/)
     npz/<clip>.npz         # per-clip exports (source of truth for training)
     <dataset>.npz          # optional cached stack (--cache-motion-bundle)

- Put source clips in the dataset folder or in **`raw/`** — converters prefer ``raw/`` when it contains ``.csv`` or ``.pkl`` files.
- **`npz/`** and **`*.npz`** are **never committed** — run ``wbc-mjlab-data-to-npz`` after adding source clips.
- **`<dataset>.npz`** is optional: written only when you pass **`--cache-motion-bundle`** on train/play.
- **`params/motion_library.yaml`** is written automatically on **play** from the loaded motion bundle.

Per-clip NPZ schema
-------------------

``wbc-mjlab-data-to-npz`` writes one ``.npz`` per source clip under ``npz/``.
``MotionCommand`` / ``MotionLoader`` load either a directory of those clips or a
pre-stacked bundle.

Required arrays (time axis ``T``, joints ``J``, bodies ``B``):

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Key
     - Shape / meaning
   * - ``fps``
     - Scalar or length-1 array — export rate (default **50** Hz)
   * - ``joint_pos``
     - ``(T, J)`` — joint positions (rad)
   * - ``joint_vel``
     - ``(T, J)`` — joint velocities (rad/s)
   * - ``body_pos_w``
     - ``(T, B, 3)`` — body positions in world frame (m)
   * - ``body_quat_w``
     - ``(T, B, 4)`` — body orientations, **xyzw**
   * - ``body_lin_vel_w``
     - ``(T, B, 3)`` — body linear velocities (m/s)
   * - ``body_ang_vel_w``
     - ``(T, B, 3)`` — body angular velocities (rad/s)

Bodies ``B`` follow the robot model body order from FK. Training selects a
**keybody subset** via ``MotionCommandCfg.body_names`` (robot entity / preset).

Optional stacked bundle (``--cache-motion-bundle``) concatenates clips and stores
segment start/length metadata so RSI can sample across the library.

Supported formats
-----------------

Source layouts are defined in ``wbc_mjlab.motion.motion_formats``. The converter
infers format from the file extension; pass ``--format`` to pick a registered name.

.. list-table::
   :header-rows: 1

   * - Format key
     - Extension
     - Role
   * - ``default``
     - ``.csv``
     - LAFAN / retarget CSV (no header, m + quat xyzw + rad)
   * - ``gmr_pkl``
     - ``.pkl``
     - GMR / ``bvh_to_robot`` pickle dict
   * - **NPZ**
     - train / play
     - Per-clip exports or a pre-stacked training bundle

CSV ``default`` (LAFAN / retarget)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Comma-separated rows, no header::

   root_pos(3) + root_rot_xyzw(4) + joint_pos(n_dof)   # meters, radians

Joint count must match the target robot. Default input rate is 30 Hz unless
``--input-fps`` is set.

PKL (``gmr_pkl``)
^^^^^^^^^^^^^^^^^

Python dict with at least::

   fps, root_pos (T×3), root_rot (T×4, xyzw), dof_pos (T×n_dof)

Optional keys: ``joint_names``, ``dof_joint_names``, or ``joint_order``.

Workflow
--------

.. code-block:: bash

   # 1. Add clips under data/<robot>/<dataset>/
   uv run wbc-mjlab-data-to-npz --robot <robot> --dataset <dataset>
   # large libraries: parallel FK on GPU
   uv run wbc-mjlab-data-to-npz --robot <robot> --dataset <dataset> --batch-size 8

   # 2. Train
   uv run wbc-mjlab-train --task Wbc-<Robot> --dataset <dataset>

   # 3. Optional cached bundle
   uv run wbc-mjlab-train --task Wbc-<Robot> --dataset <dataset> --cache-motion-bundle

``--batch-size N`` runs **N parallel FK workers** on GPU. Use ``--batch-size 1`` for ``--render`` preview.

Conversion requires **`--robot``** (MuJoCo asset). Train/play use **`--task``** (robot inferred).

Train / play motion source
--------------------------

.. list-table::
   :header-rows: 1

   * - Flag
     - Resolves to
   * - ``--dataset <name>``
     - ``data/<robot>/<name>/`` (loads ``npz/*.npz`` in memory)
   * - ``--dataset-path <dir>``
     - load ``npz/*.npz`` in that folder
   * - ``--dataset-path <file>.npz``
     - that file directly
   * - ``--motion-file <file>.npz``
     - explicit NPZ path
   * - ``--cache-motion-bundle``
     - write/read ``<dataset>/<dataset>.npz`` on disk

Version control
---------------

Everything under ``data/`` is gitignored except READMEs, ``samples/``, and
``.gitkeep`` files. See the repository ``.gitignore``.
