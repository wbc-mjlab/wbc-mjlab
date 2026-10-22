.. _data:

Prepare motion data
===================

Training needs a **motion library**: clips retargeted to your robot. Training reads the
source files directly — there is no conversion step: the clips are resampled to the control
rate, differentiated, mirrored and run through forward kinematics on the GPU while training
(:ref:`data-online`). This guide takes clips from source files to a dataset ready to train
on.

.. admonition:: You need
   :class: tip

   Clips already retargeted to the robot's joints — CSV or GMR pickle. Retarget human motion
   with any tool, e.g. `GMR <https://github.com/YanjieZe/GMR>`_. To try things first, use the
   bundled ``samples`` (10 G1 clips, credits in ``data/g1/samples/README.md``).

1. Create a dataset
-------------------

A dataset is a folder of clips under ``data/<robot>/<dataset>/``.

.. tab-set::

   .. tab-item:: Studio

      *Data → Library → Import*: pick a folder of CSV / PKL clips (copy it, or link it without
      copying), or upload files. The new dataset appears in the library.

   .. tab-item:: Command line

      .. code-block:: bash

         mkdir -p data/g1/my_clips
         cp /path/to/retargeted/*.csv data/g1/my_clips/

Clips may also sit in a ``raw/`` subfolder. Full layout and formats:
:doc:`reference/data_formats`. Large public libraries for G1 (LAFAN and others):
``data/g1/README.md``.

2. Check the clips
------------------

Look at clips before spending GPU hours on them: retargeting errors (feet in the floor,
joint flips, limit violations) are much cheaper to catch here.

.. tab-set::

   .. tab-item:: Studio

      Click a clip in *Data*: it plays **straight from the source file**. The *Clip* tab
      lists quality issues — joint-limit violations, joint-speed
      spikes, feet below or above the ground, time airborne; clicking one opens its curves
      in the Plots window (root, every joint against its limits, foot heights). *Grid* plays
      up to 49 clips of a search side by side.

   .. tab-item:: Command line

      ``wbc-mjlab-data-vis`` plays exported NPZ clips (:ref:`cli-convert`); for source clips
      use the Studio.

3. Train on it
--------------

.. tab-set::

   .. tab-item:: Studio

      *Train*: pick the dataset; *Motion* sets mirroring and height debias.

   .. tab-item:: Command line

      .. code-block:: bash

         uv run wbc-mjlab-train --task Wbc-G1 --dataset my_clips

``--dataset`` resolves to ``data/<robot>/<name>/`` with the robot taken from the task. Motion
options (each also ``--env.commands.motion.<name>``):

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Option
     - Use it when
   * - ``--mirror``
     - Doubling the library with left-right mirrored copies — needs the robot's symmetry
       config (:ref:`reference-mirroring`). Clips already named ``*_mirror`` are not
       mirrored again.
   * - ``--debias-z``
     - Clips float above or sink into the floor — shifts each clip so its lowest foot sits
       on the ground
   * - ``--env.commands.motion.source-fps``
     - Source rate isn't the format's (CSV: 30 Hz)
   * - ``--env.commands.motion.source-format``
     - The extension doesn't identify the format (e.g. ``gmr_pkl``)
   * - ``--env.commands.motion.motion-storage host``
     - The library is too big for GPU memory (source frames in pinned host memory, ~10 %
       slower steps)

Play applies the motion options a run was trained with (``params/env.yaml``). Other sources
(a folder or file anywhere): :doc:`reference/data_formats`.

.. _data-online:

How the library is built
------------------------

The first run on a dataset packs its source clips into one array —
``<dataset>/.cache/motion_<key>.npz``, base pose and joint angles at the source rate. The key
covers every clip's name, size and modification time, so adding or editing a clip rebuilds it
(seconds for hundreds of clips, ~30 s for 15 000). Each step, for every env, the reference is
computed from it on the GPU:

- **Resampling** to the control rate — linear for positions and joints, slerp for the base
  rotation;
- **velocities** by central differences (one-sided at a clip's ends);
- **mirroring** as virtual clips (``<clip>_mirror``: the coordinates are mirrored before FK,
  nothing is stored twice);
- **height debias** — a per-clip shift found with one FK pass when the library is built;
- **forward kinematics** of the tracked bodies (``mujoco_warp``, one CUDA graph);
- **anchor accelerations** from the anchor velocity over the source frames.

The library takes 144 bytes per source frame (about 0.5 GB for 30 hours at 30 Hz, against
~20 GB of converted NPZ); the reference adds ~0.5 ms per step at 4096 envs (2 %). Clips match
the old converter's NPZ within 1.5 mm / 0.15° (the old converter's own float32 time steps;
the online path is checked against MuJoCo to 0.1 mm).

Converted NPZ (optional)
------------------------

Tools that want body poses precomputed — the web reference export, ``wbc-mjlab-data-vis``,
external code — read ``npz/<clip>.npz``. ``wbc-mjlab-data-to-npz`` (Studio: *Data → Export*)
bakes them with the same code path as training:

.. code-block:: bash

   uv run wbc-mjlab-data-to-npz --robot g1 --dataset my_clips --mirror

A dataset with only ``npz/`` (no source clips), or a ``.npz`` bundle, still trains as before
(``--env.commands.motion.motion-source npz`` forces it).

.. seealso::

   - :doc:`workflows/training` — train, resume, scale to many GPUs
   - :doc:`extensions/index` — data for a new robot
   - :doc:`reference/data_formats` — formats, NPZ arrays, mirroring
