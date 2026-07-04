.. _quickstart:

Quickstart: install → convert → train → play
============================================

After this page you can train and play a policy on the bundled **``samples``**
clips. Requires Linux + NVIDIA GPU for training; CPU works for conversion and
short smoke tests.

**No GPU?** Try the `live web demo <https://wbc-mjlab.github.io/wbc-demo/>`_,
`Google Colab notebook <https://colab.research.google.com/github/wbc-mjlab/wbc-mjlab/blob/main/notebooks/demo.ipynb>`_,
or :doc:`demo` (``wbc-mjlab-demo``) — no training required.

.. _which-task:

Which task should I use?
------------------------

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Goal
     - Start with
   * - **First train** (default stack)
     - ``Wbc-G1`` + ``--dataset samples``
   * - **ZEST-style** paper stack
     - ``Wbc-G1-Zest``
   * - **Deploy-friendly** actor obs
     - ``Wbc-G1`` (``apply_wbc``)
   * - **BeyondMimic-style** binary-failure RSI
     - ``Wbc-G1-BinaryFailure``
   * - **State-estimation** actor terms
     - ``Wbc-G1-SE`` / ``Wbc-G1-Zest-SE``

Full catalog: :doc:`../tasks/index`. Stuck? :doc:`../troubleshooting`.

1. Install
----------

Follow :doc:`../installation` (``uv`` or ``pip`` from source), then verify:

.. code-block:: bash

   uv run wbc-mjlab-list-envs

You should see registered tasks (e.g. ``Wbc-G1``, ``Wbc-G1-Zest``, ``Wbc-G1-BinaryFailure``
on the in-tree ``g1`` entity).

2. Convert motion → NPZ
-----------------------

.. code-block:: bash

   uv run wbc-mjlab-data-to-npz --robot g1 --dataset samples --batch-size 8

.. tip::

   Preview clips before training: ``uv run wbc-mjlab-data-vis --robot g1 --dataset samples``.

3. Train
--------

Default task **``Wbc-G1``**:

.. code-block:: bash

   uv run wbc-mjlab-train --task Wbc-G1 --dataset samples

Logs: ``logs/rsl_rl/wbc_g1/<timestamp>/``.

.. tip::

   Full training is long. For a **smoke run** (pipeline check only):

   .. code-block:: bash

      uv run wbc-mjlab-train --task Wbc-G1 --dataset samples \
        --agent.max-iterations 1000

   Resume a real run later with ``--agent.resume True`` — see :doc:`training`.

4. Play / evaluate
------------------

.. code-block:: bash

   uv run wbc-mjlab-play --task Wbc-G1 --dataset samples --viewer viser

Play writes ``params/policy.onnx`` and ``params/config.yaml`` before the viewer
opens. Use ``--checkpoint-file /path/to/model_*.pt`` to pick a specific checkpoint.

5. What you get in ``params/``
------------------------------

.. code-block:: text

   train / play
        │
        ▼
   logs/rsl_rl/<experiment>/<run>/params/
        ├── policy.onnx      ← deploy policy
        ├── config.yaml      ← obs layout, joints, PD (wbc_tracking_params_v1)
        ├── env.yaml
        └── agent.yaml
        │
        ▼
   deploy runtime (e.g. wbc-g1-deploy)

Copy ``policy.onnx`` + ``config.yaml`` into your runtime — see :doc:`deploy`.

Next steps
----------

- :doc:`../recipes` — “How do I…?” index
- Long runs / resume / multi-GPU: :doc:`training`
- Full motion libraries: :doc:`../data`
- Add your robot: :doc:`../extensions/index`
