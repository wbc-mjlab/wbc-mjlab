.. _extensions:
.. _development:

Extensions
==========

wbc-mjlab is modular by design: **core** owns the shared MDP and reference tasks;
**each robot is a separate entity** that must be **registered** before train, play,
or data conversion will work.

What you typically build
------------------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Artifact
     - Where it lives
   * - **Robot entity**
     - Extension package — MJCF, actuators, sensors, ``<robot>_base_cfg()``
   * - **Tasks**
     - ``WbcTaskConfig`` table — preset stack per ``--task`` id
   * - **Motion data**
     - ``data/<robot_id>/<dataset>/`` under extension ``project_root``
   * - **Optional deploy runtime**
     - Separate repo (not part of wbc-mjlab core)

Core ships **one in-tree robot** (``g1``) as a reference. New hardware platforms
should **not** fork ``env/`` — publish an extension package instead.

Modularity checklist
--------------------

Before opening a PR or publishing an extension, confirm:

1. **No MDP forks** — rewards/RSI/terminations stay in ``env/mdp/``; use presets
2. **Robot registered** — ``register_robot`` or ``register_wbc_extension`` at import
3. **Body names explicit** — anchor, motion keybodies, EE termination tuple in constants
4. **Motion FK wired** — ``RobotMotionSpec`` for ``wbc-mjlab-data-to-npz --robot <id>``
5. **Tasks registered** — ``WbcTaskConfig`` with matching ``robot_id``
6. **Entry point** — ``[project.entry-points."mjlab.tasks"]`` so mjlab discovers the package

Workflow
--------

.. code-block:: bash

   # 1. Scaffold extension (separate repo)
   # 2. Implement <robot>_base_cfg + register_wbc_extension

Install the extension editable alongside wbc-mjlab:

.. tab-set::

   .. tab-item:: uv

      .. code-block:: bash

         uv pip install -e ../my-robot-wbc
         uv run wbc-mjlab-list-envs          # should list Wbc-<Robot>
         uv run wbc-mjlab-data-to-npz --robot <id> --dataset samples --batch-size 8
         uv run wbc-mjlab-train --task Wbc-<Robot> --dataset samples

   .. tab-item:: pip

      .. code-block:: bash

         pip install -e ../my-robot-wbc
         wbc-mjlab-list-envs                   # should list Wbc-<Robot>
         wbc-mjlab-data-to-npz --robot <id> --dataset samples --batch-size 8
         wbc-mjlab-train --task Wbc-<Robot> --dataset samples

.. toctree::
   :maxdepth: 1
   :caption: Guides

   robot_entity
   extensions
   example_extension

API reference: :doc:`../api/extension` (``WbcRobotSpec``, ``register_wbc_extension``,
``WbcTaskConfig``, ``RobotMotionSpec``).

Related: :doc:`../concepts/index`, :doc:`../contributing`, :doc:`../usage`.
