.. _troubleshooting:

Troubleshooting
===============

Fixes for the errors people hit most, by area. Not here? Search the
`issues <https://github.com/wbc-mjlab/wbc-mjlab/issues>`_ or open one with the command and its
output.

.. contents:: On this page
   :local:
   :depth: 2

Install and registration
------------------------

No tasks in ``wbc-mjlab-list-envs``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The list is empty or misses expected ids.

1. Install the package in the active environment (``make sync`` or ``uv sync``).
2. Extension robots: install the extension (``uv pip install -e …``) and check its
   ``[project.entry-points."mjlab.tasks"]`` points at its entry module.
3. Robots added in the Studio: ``~/.config/wbc-mjlab/robots/<id>/robot.yaml`` exists; a broken
   one prints ``custom robot <id> not registered: …`` on startup.

``Unknown WBC task`` / ``No WBC env builder for robot``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use an id from ``wbc-mjlab-list-envs``; prefer ``--task Wbc-G1`` over ``--robot``. Extensions
must call ``register_wbc_extension`` at import time.

EGL / GL errors
~~~~~~~~~~~~~~~

On headless machines set ``MUJOCO_GL=egl`` (or ``disable`` for runs without rendering).
Training needs an NVIDIA GPU; without one, use the
`live demo <https://wbc-mjlab.github.io/wbc-demo/>`_ or :doc:`Colab <workflows/demo>`.

Motion data
-----------

``No clip NPZs`` / empty motion library
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Training looks for ``data/<robot>/<dataset>/npz/*.npz`` — convert first:

.. code-block:: bash

   uv run wbc-mjlab-data-to-npz --robot g1 --dataset samples --batch-size 8

``expected N joint DOFs … got M``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The clips weren't retargeted to this robot: the joint count must match the model. Check
``--robot``, the joint order (``data/g1/README.md`` for G1) and ``--format``.

``mirroring needs a left/right symmetry config``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Convert without ``--mirror``. Mirroring needs a ``RobotSymmetryConfig`` — G1 and extensions
that register one have it; robots added in the Studio don't.

Training and play
-----------------

Resume loads the wrong run
~~~~~~~~~~~~~~~~~~~~~~~~~~

Pin the run and the checkpoint:

.. code-block:: bash

   uv run wbc-mjlab-train --task Wbc-G1 --dataset samples \
     --agent.resume True --agent.load-run 2026-07-03_14-30-00 --agent.load-checkpoint model_5000.pt

Play finds no checkpoint
~~~~~~~~~~~~~~~~~~~~~~~~

Play loads the latest ``model_*.pt`` under ``logs/rsl_rl/<experiment>/``: train once, or pass
``--checkpoint-file /path/to/model_*.pt``.

An old run of a Studio preset replays differently
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Runs trained from the Studio keep their preset in ``params/studio_preset.yaml`` and replay
from it. Runs older than that snapshot use the preset file as it is now — restore the old
values (or a copy of the preset) to replay them as trained.

Studio
------

Not reachable from another machine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the LAN link the Studio prints, not ``127.0.0.1``; don't start it with
``--host 127.0.0.1``; open the port (``sudo ufw allow 8765/tcp``). With ``--token``, open
``/?token=…`` once or type the token on the sign-in page.

Port 8765 already in use
~~~~~~~~~~~~~~~~~~~~~~~~

Another Studio is still running — stop it with Ctrl+C in its terminal (the port is freed on
exit), or use ``--port 8766``. ``fuser -k 8765/tcp`` stops whatever holds the port.

Robots added from model files
-----------------------------

The check fails or the robot falls in Play
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Edit the robot (*Robot* → sliders icon) and fix what the check names:

- *starts inside the floor* — raise the root height;
- *touches the floor with …* — the root height or default pose puts other bodies on the
  floor, or the feet are wrong;
- *blew up* — lower the gains or raise the armature;
- *torque saturated* — raise the efforts (``*`` marks guessed values: check the datasheet).

*Sags* is only a warning — plain PD doesn't balance a humanoid; the policy does.

The model doesn't load
~~~~~~~~~~~~~~~~~~~~~~

MuJoCo reads STL, OBJ and MSH meshes only: convert ``.dae`` meshes to ``.stl`` / ``.obj``.
Pick the robot's model file, not a scene that includes it. Fixed-base robots aren't supported.
More: :doc:`extensions/custom_robots`.

Docs
----

Build locally with ``make docs`` (live reload: ``make docs-watch``) — ``docs/BUILDING.md``.
