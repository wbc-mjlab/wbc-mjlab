.. _recipes:

How do I…?
==========

Short answers, grouped by stage of the pipeline. Each links to the full guide.

Get going
---------

.. list-table::
   :widths: 46 54

   * - Try it without a GPU
     - `Live demo <https://wbc-mjlab.github.io/wbc-demo/>`_ · :doc:`workflows/demo`
   * - Train my first policy
     - :doc:`workflows/quickstart`
   * - Do everything in a browser — also on a remote GPU server
     - :doc:`studio`

Motion data
-----------

.. list-table::
   :widths: 46 54

   * - Convert LAFAN / GMR / my own clips
     - :doc:`data`
   * - Spot bad clips before training
     - :doc:`data` (*Check the clips*) — Studio *Data → Clip*
   * - Double the library with mirrored clips
     - :doc:`data` (``--mirror``)
   * - Look up the NPZ arrays or CSV columns
     - :doc:`reference/data_formats`

Train
-----

.. list-table::
   :widths: 46 54

   * - Pick a task
     - :doc:`tasks/index`
   * - Tune rewards, observations, sampling or PPO without code
     - :doc:`studio` (*Preset*)
   * - Reuse a set of changes, e.g. state estimation
     - :doc:`studio` (*modules*)
   * - Resume, extend or pin a checkpoint
     - :doc:`workflows/training`
   * - Train on several GPUs
     - :doc:`workflows/training` (*Multi-GPU*)
   * - Train on rough terrain
     - :doc:`tasks/wbc-g1-terrain-scan`

Evaluate and deploy
-------------------

.. list-table::
   :widths: 46 54

   * - Watch a policy on many clips at once
     - :doc:`studio` (*Play*)
   * - Debug the reference and adaptive sampling in viser
     - :doc:`visualization`
   * - Export ONNX + ``config.yaml`` for the robot
     - :doc:`workflows/deploy`
   * - Look up observation sizes
     - :ref:`reference-obs-dims` · the run's ``params/config.yaml``

Extend
------

.. list-table::
   :widths: 46 54

   * - Add a robot from its MJCF / URDF
     - :doc:`extensions/custom_robots`
   * - Ship a robot as a package
     - :doc:`extensions/index`
   * - Add a paper's setup as a task
     - :doc:`tasks/adding`
   * - Script the Studio
     - :doc:`reference/studio_api`

Something broke
---------------

:doc:`troubleshooting` covers the common errors — missing tasks, empty motion library, DOF
mismatch, GPU / GL errors, resume, Studio access and robot checks.
