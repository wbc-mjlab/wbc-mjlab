.. _quickstart:

Quickstart: install → convert → train → play
============================================

End-to-end workflow on the bundled **`samples`** dataset (example clips under
``data/g1/samples/``). Requires Linux + NVIDIA GPU for training; CPU works for
conversion and short smoke tests.

**No GPU?** Try the `live web demo <https://wbc-mjlab.github.io/wbc-demo/>`_,
`Google Colab notebook <https://colab.research.google.com/github/wbc-mjlab/wbc-mjlab/blob/main/notebooks/demo.ipynb>`_,
or :doc:`demo` (``wbc-mjlab-demo``) — no training required.

1. Install
----------

.. code-block:: bash

   git clone https://github.com/wbc-mjlab/wbc-mjlab.git && cd wbc-mjlab
   make sync
   uv run wbc-mjlab-list-envs

You should see registered tasks (e.g. ``Wbc-G1``, ``Wbc-G1-Zest``, ``Wbc-G1-BinaryFailure``
on the in-tree ``g1`` entity).

2. Convert motion → NPZ
-----------------------

.. code-block:: bash

   uv run wbc-mjlab-data-to-npz --robot g1 --dataset samples --batch-size 8

Preview clips (optional):

.. code-block:: bash

   uv run wbc-mjlab-data-vis --robot g1 --dataset samples

3. Train
--------

Default task **`Wbc-G1`** (in-tree example):

.. code-block:: bash

   uv run wbc-mjlab-train --task Wbc-G1 --dataset samples

Logs: ``logs/rsl_rl/wbc_g1/<timestamp>/``. Training is long — resume with
``--agent.resume True`` (see :doc:`training`).

4. Play / evaluate
------------------

.. code-block:: bash

   uv run wbc-mjlab-play --task Wbc-G1 --dataset samples --viewer viser

Use ``--run logs/rsl_rl/wbc_g1/<timestamp>`` to pick a specific run.

5. What you get in ``params/``
------------------------------

.. list-table::
   :header-rows: 1

   * - File
     - Purpose
   * - ``policy.onnx``
     - Deployable policy
   * - ``config.yaml``
     - Tracking params (``schema_version: wbc_tracking_params_v1``)
   * - ``env.yaml`` / ``agent.yaml``
     - Full env + RL config snapshots
   * - ``motion_library.yaml``
     - Written on play from loaded clips

Copy ``policy.onnx`` + ``config.yaml`` into your deploy runtime — see :doc:`deploy`
(`wbc-g1-deploy <https://github.com/wbc-mjlab/wbc-g1-deploy>`_ is one reference
implementation).

Next steps
----------

- Other tasks: :doc:`../tasks/index`
- Long runs / resume / multi-GPU: :doc:`training`
- Full LAFAN libraries: :doc:`../data`
- Demo without training: :doc:`demo`
- Add your robot: :doc:`../usage` + `wbc-mjlab-extension-h2 <https://github.com/wbc-mjlab/wbc-mjlab-extension-h2>`_
