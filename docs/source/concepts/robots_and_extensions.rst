Robots and extensions
=====================

A **robot** in wbc-mjlab is a **registered entity** — not a folder you drop into
``env/``. The core ships one built-in id (``g1``) as a reference; every other
platform must **register** before train, play, or ``wbc-mjlab-data-to-npz`` will work.

Why registration
----------------

Registration connects four things the CLI needs:

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Registered item
     - Used for
   * - ``robot_id``
     - ``--robot`` and ``data/<robot>/`` paths
   * - ``make_env_cfg`` / ``make_rl_cfg``
     - Train and play env builders
   * - ``motion_spec``
     - Forward kinematics for data conversion
   * - ``project_root`` (extensions)
     - Extension-owned ``data/<robot>/`` tree

``wbc-mjlab-list-envs`` merges tasks from **core + all installed extensions**.

Robot entity vs extension package
---------------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Approach
     - When
   * - **In-tree** (``robots/g1/``)
     - Reference platform shipped with core — study, don't copy-paste for new hardware
   * - **Extension package**
     - New robot — separate repo, ``register_wbc_extension``, ``mjlab.tasks`` entry point

Extensions typically ship:

- ``<robot>_base_cfg()`` — MJCF, sensors, body tuples
- One or more ``WbcTaskConfig`` entries (often ``apply_wbc`` only)
- ``data/<robot_id>/`` motion libraries under ``project_root``

Tasks belong to robots
----------------------

Every ``WbcTaskConfig`` sets ``robot_id`` so the CLI knows which entity and data paths
to use. Task ids (``Wbc-H2``, ``Wbc-G1-Zest``, …) are **names for preset stacks** on
a **specific registered robot** — not global env forks.

Build your own
--------------

Step-by-step extension layout, ``WbcRobotSpec``, and H2 reference walkthrough:

- :doc:`../extensions/index`
- :doc:`../extensions/extensions`
- :doc:`../extensions/example_extension`

Robot entity wiring checklist: :doc:`../extensions/robot_entity`.
