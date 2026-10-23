.. _usage:

Command-line reference
======================

Every command, with its options. Run any command with ``--help`` for the full list. With
pip, drop the ``uv run`` prefix.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Command
     - Does
   * - :ref:`wbc-mjlab-studio <cli-studio>`
     - The whole pipeline in a browser
   * - :ref:`wbc-mjlab-data-to-npz <cli-convert>`
     - Export source clips (CSV / PKL) as NPZ — optional, training reads the sources
   * - :ref:`wbc-mjlab-data-vis <cli-data-vis>`
     - Play exported NPZ clips (no policy)
   * - :ref:`wbc-mjlab-list-envs <cli-list>`
     - List robots and tasks
   * - :ref:`wbc-mjlab-train <cli-train>`
     - Train a task
   * - :ref:`wbc-mjlab-play <cli-train>`
     - Evaluate a checkpoint; exports ``policy.onnx`` + ``config.yaml``
   * - :ref:`wbc-mjlab-demo <cli-demo>`
     - Play the bundled demo checkpoint
   * - :ref:`wbc-mjlab-export-tracking-params <cli-export>`
     - Write a task's deploy ``config.yaml``
   * - :ref:`wbc-mjlab-plot-rsi-bins <cli-rsi>`
     - Plot saved adaptive-sampling state
   * - :ref:`python -m wbc_mjlab.studio.train_preset <cli-train-preset>`
     - Train a Studio preset

.. _cli-studio:

wbc-mjlab-studio
----------------

.. code-block:: bash

   uv run wbc-mjlab-studio [--host 0.0.0.0] [--port 8765] [--token SECRET] [--ui web|viser]

.. list-table::
   :widths: 30 70

   * - ``--host``
     - Interface to serve on (default ``0.0.0.0``: the network; ``127.0.0.1``: this machine)
   * - ``--port``
     - UI port (default 8765)
   * - ``--token``
     - Require this token from network clients (default: no access control)
   * - ``--robot`` · ``--device``
     - Robot to start with · torch device for playback and conversion
   * - ``--ui``
     - ``web`` (default) or ``viser``
   * - ``--no-open-browser``
     - Don't open a browser tab

Guide: :doc:`studio`.

.. _cli-convert:

wbc-mjlab-data-to-npz
---------------------

.. code-block:: bash

   uv run wbc-mjlab-data-to-npz --robot g1 --dataset my_clips

Training does not need it (:ref:`data-online`): it writes ``npz/<clip>.npz`` — body poses
and velocities — for tools that want them, baked with the training code path (batched GPU
forward kinematics).

.. list-table::
   :widths: 30 70

   * - ``--robot``
     - Robot model to run forward kinematics on (default ``g1``)
   * - ``--dataset`` · ``--dataset-path``
     - ``data/<robot>/<name>/`` · any folder; output goes to its ``npz/``
   * - ``--input-path`` · ``--output-dir``
     - A single file or folder in, an explicit folder out
   * - ``--format``
     - Source format (``default`` CSV, ``gmr_pkl``); default from the file extension
   * - ``--input-fps`` · ``--output-fps``
     - Source rate (CSV default 30) · export rate (default 50)
   * - ``--batch-size``
     - Frames per FK batch on the GPU (at least 4096)
   * - ``--debias-z`` · ``--debias-foot-sole-z``
     - Shift clips so the feet touch the floor · override the robot's sole height
   * - ``--mirror``
     - Also write left-right mirrored clips (needs a symmetry config)
   * - ``--device`` · ``--render`` · ``--line-range``
     - Torch device · step the clips through the sim and render them (frame by frame, batch
       size 1) · CSV row range

Formats and output: :doc:`reference/data_formats`.

.. _cli-data-vis:

wbc-mjlab-data-vis
------------------

.. code-block:: bash

   uv run wbc-mjlab-data-vis --robot g1 --dataset samples

Plays exported NPZ clips in viser with a timeline (source clips: the Studio). Options: ``--robot``, ``--dataset``,
``--dataset-path``, ``--motion-file``, ``--fps``, ``--cache-motion-bundle``.

.. _cli-list:

wbc-mjlab-list-envs
-------------------

Lists registered robots and tasks (built-in, extensions, robots added in the Studio) with
their log folders.

.. _cli-train:

wbc-mjlab-train · wbc-mjlab-play
--------------------------------

.. code-block:: bash

   uv run wbc-mjlab-train --task Wbc-G1 --dataset samples
   uv run wbc-mjlab-play  --task Wbc-G1 --dataset samples --viewer viser

wbc-mjlab options (parsed first):

.. list-table::
   :widths: 34 66

   * - ``--task <id>``
     - Task to train / play (robot taken from it) — :doc:`tasks/index`
   * - ``--dataset <name>``
     - Motion library ``data/<robot>/<name>/`` (source clips, played online)
   * - ``--dataset-path`` · ``--motion-file``
     - A dataset folder, a clip or an NPZ anywhere
   * - ``--mirror`` · ``--debias-z`` *(train)*
     - Add left-right mirrored clips · put each clip's lowest foot on the ground —
       :doc:`data`
   * - ``--algo ppo|wm`` *(train)*
     - Training algorithm: PPO (default) or the world model — :doc:`concepts/world_model`
   * - ``--cache-motion-bundle``
     - Converted-NPZ datasets only: write / reuse ``<dataset>/<dataset>.npz``
   * - ``--robot <id>`` · ``--no-state-estimation``
     - Shorthand instead of ``--task``: the robot's default task · its non-SE variant

Everything else passes through to mjlab (`mjlab CLI <https://mujocolab.github.io/mjlab/main/>`_):

.. list-table::
   :widths: 34 66

   * - ``--env.scene.num-envs N``
     - Parallel environments (default 8192)
   * - ``--agent.max-iterations N``
     - Training length
   * - ``--agent.run-name NAME``
     - Suffix of the run folder
   * - ``--agent.resume True`` · ``--agent.load-run`` · ``--agent.load-checkpoint``
     - Continue a run — :doc:`workflows/training`
   * - ``--agent.logger tensorboard|wandb``
     - Logger (default TensorBoard)
   * - ``--gpu-ids 0 1`` · ``all``
     - Train on several GPUs
   * - ``--checkpoint-file PATH`` *(play)*
     - A specific checkpoint (default: latest of the task)
   * - ``--viewer viser|native`` *(play)*
     - Viewer (viser when headless)

.. _cli-demo:

wbc-mjlab-demo
--------------

Plays the bundled checkpoint (``demos/wbc_g1``) on the bundled ``samples`` clips. Options include
``--agent trained|zero|random``, ``--num-envs``, ``--viewer``, ``--video``.

.. _cli-export:

wbc-mjlab-export-tracking-params
--------------------------------

.. code-block:: bash

   uv run wbc-mjlab-export-tracking-params --task Wbc-G1 --out params/config.yaml [--play]

Rewrites the deploy ``config.yaml`` of a task (training and play already write it). Schema:
:doc:`api/export`.

.. _cli-rsi:

wbc-mjlab-plot-rsi-bins
-----------------------

.. code-block:: bash

   uv run wbc-mjlab-plot-rsi-bins logs/rsl_rl/<experiment>/<run>/params/rsi_bin_stats.npz \
     [--trajectory N] [--output plot.png]

.. _cli-train-preset:

Training a Studio preset
------------------------

.. code-block:: bash

   uv run python -m wbc_mjlab.studio.train_preset --preset <name> --dataset samples [train options]

Registers ``<base>--<name>`` from ``~/.config/wbc-mjlab/presets/<name>.yaml`` and trains it;
accepts every ``wbc-mjlab-train`` option. Single GPU only. File format:
:ref:`reference-preset-file`.

.. seealso::

   :doc:`reference/configuration` — files, directories and environment variables
