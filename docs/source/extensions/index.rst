.. _extensions:
.. _development:

Add a robot
===========

Every robot is a **registered entity**: its model, actuators, body names and tasks plug into
the shared MDP without touching it. There are two ways to register one.

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: :octicon:`zap` From model files — no code
      :link: custom_robots
      :link-type: doc

      Upload the MJCF / URDF in the Studio. Wiring and PD gains are suggested, checked on a
      floor and saved per user; tasks ``Wbc-<Id>``, ``-SE`` and ``-Terrain-Scan`` appear at
      once.

      **Best for** trying a robot, research across embodiments.

   .. grid-item-card:: :octicon:`package` As an extension package
      :link: extensions
      :link-type: doc

      A Python package calling ``register_wbc_extension``: full control over actuator
      models, symmetry for mirroring, any task table; ships with your platform.

      **Best for** a robot you maintain and deploy.

Both paths give a robot that every command and the Studio treat like the built-in G1.
Starting with the Studio and moving to a package later is common — the saved ``robot.yaml``
holds the numbers the package's constants need.

.. _first-run-extension:

Walkthrough: extension → data → train
-------------------------------------

The extension path end to end, for a robot ``my_robot`` with clips ``my_data``.

1. **Create the package** following
   `wbc-mjlab-extension-h2 <https://github.com/wbc-mjlab/wbc-mjlab-extension-h2>`_
   (:doc:`example_extension` walks through it) and install it next to wbc-mjlab:

   .. code-block:: bash

      uv pip install -e ../my-robot-wbc
      uv run wbc-mjlab-list-envs          # lists Wbc-MyRobot

2. **Retarget clips** to the robot (e.g. with `GMR <https://github.com/YanjieZe/GMR>`_) and put
   them in the extension's data tree: ``<project_root>/data/my_robot/my_data/*.csv``.

3. **Convert, check, train:**

   .. code-block:: bash

      uv run wbc-mjlab-data-to-npz --robot my_robot --dataset my_data --batch-size 8
      uv run wbc-mjlab-data-vis --robot my_robot --dataset my_data
      uv run wbc-mjlab-train --task Wbc-MyRobot --dataset my_data

Checklist
---------

Before publishing an extension or opening a PR:

1. **No MDP forks** — rewards, sampling and terminations stay in ``env/mdp/``; use presets
2. **Registered** — ``register_wbc_extension`` runs at import, from a ``mjlab.tasks`` entry
   point
3. **Names explicit** — anchor, keybodies and end-effectors as constants
4. **Conversion wired** — a ``RobotMotionSpec`` so ``wbc-mjlab-data-to-npz --robot <id>`` works
5. **Tasks** — ``WbcTaskConfig`` entries with the matching ``robot_id``

.. toctree::
   :hidden:

   custom_robots
   extensions
   example_extension
   robot_entity

.. seealso::

   - :doc:`../concepts/robots_and_extensions` — why robots register
   - :doc:`../api/extension` — ``WbcRobotSpec``, ``register_wbc_extension``, ``RobotMotionSpec``
