Modularity
==========

wbc-mjlab separates **one shared MDP** from **robot wiring** and **paper recipes**.
Nothing in ``env/mdp/`` should hard-code a platform or a specific paper ablation.

Layer stack
-----------

.. code-block:: text

   ┌─────────────────────────────────────────────────────────┐
   │  Shared WBC MDP (env/) — robot-agnostic term callables  │
   │  Actions · Commands · Observations · Rewards · …        │
   └───────────────────────────┬─────────────────────────────┘
                               │ make_base_wbc_env_cfg()
   ┌───────────────────────────▼─────────────────────────────┐
   │  Robot entity (<robot>_base_cfg) — must be registered   │
   │  MJCF · actuators · sensors · body name tuples          │
   └───────────────────────────┬─────────────────────────────┘
                               │
   ┌───────────────────────────▼─────────────────────────────┐
   │  Presets (presets/) — cfg mutators only                 │
   │  apply_wbc · apply_zest · apply_binary_failure · …      │
   └───────────────────────────┬─────────────────────────────┘
                               │
   ┌───────────────────────────▼─────────────────────────────┐
   │  Tasks (WbcTaskConfig) — CLI ``--task`` ids             │
   │  preset stack + experiment_name + build_env_cfg         │
   └─────────────────────────────────────────────────────────┘

What each layer owns
--------------------

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Layer
     - Responsibility
   * - **MDP**
     - Term *implementations* — same code for every robot and task
   * - **Robot entity**
     - *Which* MJCF, bodies, sensors, and action scales attach to the template
   * - **Preset**
     - *How* rewards, RSI, obs, and terminations are weighted for a method
   * - **Task**
     - A **named, registered** preset stack + log directory for the CLI

Core principle: **compose cfg, don't fork envs.** Paper differences are preset +
task builders; new robots are extension packages — not copies of ``env/``.

What not to duplicate
---------------------

- New reward/RSI **callables** → shared ``env/mdp/`` (needs review)
- New paper **weights / flags** → ``presets/<method>.py``
- New platform → registered **robot entity** + tasks
- New ``--task`` id → ``WbcTaskConfig`` + builder

See :doc:`../architecture` for the assembly pipeline and :doc:`presets_and_tasks` for
how presets relate to tasks.
