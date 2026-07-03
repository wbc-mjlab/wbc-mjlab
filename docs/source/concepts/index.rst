.. _concepts:

Concepts
========

These pages explain **how wbc-mjlab is put together** — the mental model before you
dive into CLI commands, MDP field reference, or extension how-tos.

Suggested reading order
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Topic
     - Page
   * - **Layer stack**
     - :doc:`modularity` — shared MDP, robot entity, preset, task
   * - **Config assembly**
     - :doc:`../architecture` — pipeline from template → ``--task``
   * - **Paper recipes**
     - :doc:`presets_and_tasks` — ``apply_*`` vs registered task ids
   * - **Curriculum sampling**
     - :doc:`adaptive_sampling` — RSI bins and failure-driven resets
   * - **New hardware**
     - :doc:`robots_and_extensions` — why robots must register

Where to go next
----------------

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Need
     - Section
   * - Field-level MDP reference
     - :doc:`../mdp/index`
   * - Train, resume, multi-GPU
     - :doc:`../workflows/training`
   * - Task catalog + ``apply_zest`` code
     - :doc:`../tasks/index`
   * - Build an extension package
     - :doc:`../extensions/index`
   * - Viser RSI / reference overlays
     - :doc:`../visualization`

.. toctree::
   :maxdepth: 1
   :hidden:

   modularity
   presets_and_tasks
   adaptive_sampling
   robots_and_extensions
