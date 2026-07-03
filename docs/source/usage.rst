.. _usage:

Usage
=====

CLI commands
------------

.. list-table::
   :header-rows: 1

   * - Command
     - Purpose
   * - ``wbc-mjlab-train``
     - Train (wraps ``mjlab.scripts.train``)
   * - ``wbc-mjlab-play``
     - Play / eval
   * - ``wbc-mjlab-list-envs``
     - List registered tasks
   * - ``wbc-mjlab-data-to-npz``
     - Build motion NPZ from CSV / PKL (``--batch-size N`` for parallel GPU conversion)
   * - ``wbc-mjlab-export-tracking-params``
     - Write ``config.yaml``
   * - ``wbc-mjlab-data-vis``
     - Play motion NPZ clips in Viser (browser)
   * - ``wbc-mjlab-plot-rsi-bins``
     - Plot saved ``rsi_bin_stats.npz`` (offline)

See :doc:`visualization` for reference alignment, RSI panels, and Viser overlays.

Train / play
------------

Pick a **task** (:doc:`tasks/index`), then pass a **dataset**:

.. code-block:: bash

   uv run wbc-mjlab-train --task Wbc-G1-Zest --dataset lafan
   uv run wbc-mjlab-play --task Wbc-G1-Zest --dataset lafan

``wbc-mjlab-train`` wraps mjlab's train script — **all mjlab CLI flags pass through**
(``--agent.*``, ``--env.*``, ``--gpu-ids``, …). See :doc:`workflows/training` for
**resume** (``--agent.resume True``), **multi-GPU**, and log layout.

**Preferred:** ``--task`` + dataset (robot inferred from task → ``data/g1/…``):

.. code-block:: bash

   uv run wbc-mjlab-train --task Wbc-G1 --dataset lafan
   uv run wbc-mjlab-train --task Wbc-G1 --dataset lafan --cache-motion-bundle

Training loads ``npz/*.npz`` in memory by default. Pass **`--cache-motion-bundle`**
to write or reuse ``data/g1/lafan/lafan.npz`` on disk.

**Shorthand** (default task ``Wbc-G1`` for robot ``g1``):

.. code-block:: bash

   uv run wbc-mjlab-train --robot g1 --dataset lafan
   uv run wbc-mjlab-train --robot g1 --no-state-estimation --dataset lafan   # → Wbc-G1-Zest

Motion source (pick one):

.. list-table::
   :header-rows: 1

   * - Flag
     - Resolves to
   * - ``--dataset lafan``
     - load ``data/g1/lafan/npz/*.npz`` (or ``lafan.npz`` with ``--cache-motion-bundle``)
   * - ``--dataset-path /path/to/lafan/``
     - load ``npz/*.npz`` in that folder
   * - ``--dataset-path /path/to/custom.npz``
     - that file
   * - ``--motion-file …``
     - explicit NPZ or dataset directory
   * - ``--cache-motion-bundle``
     - write/read ``<dataset>/<dataset>.npz`` on disk

Motion conversion
-----------------

Conversion needs **`--robot``** (MuJoCo asset / scene), not a task id:

.. code-block:: bash

   uv run wbc-mjlab-data-to-npz --robot g1 --dataset lafan
   uv run wbc-mjlab-data-to-npz --robot g1 --dataset lafan --batch-size 128

``--batch-size N`` runs **N parallel FK workers** on the GPU. ``--render`` preview
is only supported with ``--batch-size 1``.

See :doc:`data` for layout and download links.

Visualize motion NPZ
--------------------

.. code-block:: bash

   uv run wbc-mjlab-data-vis --robot g1 --dataset lafan
   uv run wbc-mjlab-data-vis --motion-file data/g1/lafan/npz/walk1_subject1.npz

When a dataset has ``npz/*.npz`` clips, the GUI lists them in a **Motion** dropdown.

WBC tracking params YAML
------------------------

Checkpoint saves write ``params/config.yaml`` alongside ``params/policy.onnx``.

.. code-block:: bash

   uv run wbc-mjlab-export-tracking-params --task Wbc-G1-Zest --out /path/to/config.yaml

Train → export → run on hardware with a deploy runtime. See :doc:`workflows/deploy`
(`wbc-g1-deploy <https://github.com/wbc-mjlab/wbc-g1-deploy>`_ is a reference
implementation for one platform).

Repo layout
-----------

.. code-block:: text

   data/g1/<dataset>/      # retargeted clips
   source/wbc_mjlab/
     env/                  # Shared WBC MDP (wbc_env_cfg.py, mdp/*)
     presets/              # apply_wbc, apply_zest, …
     robots/<id>/          # base.py, tasks.py, assets
     tasks/                # WbcTaskConfig + registration
     extension.py          # register_wbc_extension API

Add a robot or paper task
-------------------------

**New robot:** publish an extension package — :doc:`extensions/index`.

**New paper task (in-tree or extension):**

1. Preset in ``presets/<method>.py`` (or stack existing ``apply_*``)
2. Task builder + ``WbcTaskConfig`` in ``robots/<id>/tasks.py``
3. Document under :doc:`tasks/index`

See :doc:`architecture` for the layer model and :doc:`contributing` for PR workflow.

Development commands
--------------------

.. code-block:: bash

   make sync          # CUDA env
   make format        # ruff format + fix
   make lint          # ruff check
   make smoke         # wbc-mjlab-list-envs
   make test          # pytest
   make build         # wheel + smoke import test

Optional: ``uvx pre-commit install`` (after ``make sync``).
