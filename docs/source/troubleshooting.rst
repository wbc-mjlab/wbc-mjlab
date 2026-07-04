.. _troubleshooting:

Troubleshooting
===============

Quick fixes for the errors people hit first. Still stuck? Open an
`issue <https://github.com/wbc-mjlab/wbc-mjlab/issues/new>`_.

No tasks in ``wbc-mjlab-list-envs``
----------------------------------

**Symptom:** empty list, or only unexpected ids.

**Checks:**

1. Install the package in the active env (``make sync`` / ``uv sync``).
2. Extension robots: install the extension editable and confirm
   ``[project.entry-points."mjlab.tasks"]`` points at your ``mjlab_entry``.
3. Re-run ``uv run wbc-mjlab-list-envs`` after install.

See :doc:`extensions/index`.

``No clip NPZs`` / empty motion library
---------------------------------------

**Symptom:** train/play fails looking for ``npz/*.npz``.

**Fix:** convert first:

.. code-block:: bash

   uv run wbc-mjlab-data-to-npz --robot g1 --dataset samples --batch-size 8

Confirm files exist under ``data/<robot>/<dataset>/npz/``. Layout: :doc:`data`.

Joint DOF mismatch on convert
-----------------------------

**Symptom:** ``expected N joint DOFs … got M``.

**Fix:** source CSV/PKL joint count must match the robot model. Check
``data/g1/README.md`` (or your extension) for joint order; use the correct
``--robot`` and ``--format``.

Unknown task / robot
--------------------

**Symptom:** ``Unknown WBC task …`` or ``No WBC env builder for robot …``.

**Fix:** use an id from ``wbc-mjlab-list-envs``. Prefer ``--task Wbc-G1`` over
guessing ``--robot``. Extension robots must call ``register_wbc_extension`` at
import time.

CUDA / MuJoCo GL errors on train
--------------------------------

**Symptom:** EGL / GL import failures on headless machines.

**Fix:** set ``MUJOCO_GL=egl`` (or ``disable`` for non-rendering smoke tests).
Training needs an NVIDIA GPU; for no-GPU try the
`live demo <https://wbc-mjlab.github.io/wbc-demo/>`_ or
`Colab <https://colab.research.google.com/github/wbc-mjlab/wbc-mjlab/blob/main/notebooks/demo.ipynb>`_.

Install: :doc:`installation`.

Resume loads the wrong run
--------------------------

**Symptom:** training continues from an unexpected checkpoint.

**Fix:** pin the run and file:

.. code-block:: bash

   uv run wbc-mjlab-train --task Wbc-G1 --dataset samples \
     --agent.resume True \
     --agent.load-run 2026-07-03_14-30-00 \
     --agent.load-checkpoint model_5000.pt

Details: :doc:`workflows/training`.

Play has no checkpoint
----------------------

**Symptom:** play fails or uses a random/zero agent.

**Fix:** train once, or pass ``--checkpoint-file /path/to/model_*.pt``. Default
play picks the latest ``model_*.pt`` under ``logs/rsl_rl/<experiment>/``.

Docs build / autodoc
--------------------

Local live reload: ``make docs-watch`` (watches ``docs/`` and ``src/``).
One-shot: ``make docs``. See ``docs/BUILDING.md``.
