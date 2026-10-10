.. _data:

Prepare motion data
===================

Training needs a **motion library**: clips retargeted to your robot, converted to NPZ with
forward kinematics (body poses and velocities the rewards and observations read). This guide
takes clips from source files to a dataset ready to train on.

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

      Click a clip in *Data*: it plays **straight from the source file**, before
      conversion. The *Clip* tab lists quality issues — joint-limit violations, joint-speed
      spikes, feet below or above the ground, time airborne — and plots root height and
      speed, foot heights and any joint against its limits. *Grid* plays up to 49 clips of a
      search side by side.

   .. tab-item:: Command line

      After conversion (next step):

      .. code-block:: bash

         uv run wbc-mjlab-data-vis --robot g1 --dataset my_clips

3. Convert to NPZ
-----------------

.. tab-set::

   .. tab-item:: Studio

      *Data → Convert*: choose *missing*, *all* or *selected* clips, set the options and
      **Convert**. The job runs in the background with progress and log.

   .. tab-item:: Command line

      .. code-block:: bash

         uv run wbc-mjlab-data-to-npz --robot g1 --dataset my_clips --batch-size 8

Output: ``data/g1/my_clips/npz/<clip>.npz``. Options:

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Option
     - Use it when
   * - ``--batch-size N``
     - Always for more than a few clips — N parallel FK workers on the GPU
   * - ``--input-fps`` · ``--output-fps``
     - Source rate isn't 30 Hz (CSV) · you want another training rate than 50 Hz
   * - ``--debias-z``
     - Clips float above or sink into the floor — shifts each clip so the feet touch it
   * - ``--mirror``
     - Doubling the library with left-right mirrored copies — needs the robot's symmetry
       config (:ref:`reference-mirroring`)
   * - ``--format gmr_pkl``
     - The extension doesn't identify the format

Conversion uses ``--robot`` (the model), not a task. Re-running converts again; the Studio's
*missing* option converts only new clips.

4. Train on it
--------------

.. code-block:: bash

   uv run wbc-mjlab-train --task Wbc-G1 --dataset my_clips

``--dataset`` resolves to ``data/<robot>/<name>/npz/`` with the robot taken from the task. For
big libraries, ``--cache-motion-bundle`` writes a stacked ``<dataset>.npz`` that loads faster
on the next run. Other sources (a folder or file anywhere): :doc:`reference/data_formats`.

.. seealso::

   - :doc:`workflows/training` — train, resume, scale to many GPUs
   - :doc:`extensions/index` — data for a new robot
   - :doc:`reference/data_formats` — formats, NPZ arrays, mirroring
