.. _adding-task:

Add a task or preset
====================

Turn a new method or setup into a ``--task`` id that trains, plays and exports like the
built-in ones. For one-off tuning without code, a :doc:`Studio preset <../studio>` is usually
enough.

Decide where the change goes
----------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - You want to…
     - Add
   * - Change weights, sampling flags, thresholds, observation terms
     - A **preset** — ``presets/<method>.py`` (or stack existing ones)
   * - Train an existing preset on another robot
     - A **task** — a builder passing that robot's body names; the preset stays as is
   * - A new reward, observation or sampling **function**
     - A term in ``env/mdp/`` — shared code, open an issue first (:doc:`../contributing`)
   * - Just try different values
     - A Studio preset — no code (:doc:`../studio`)

Steps
-----

1. **Write the preset** — ``src/wbc_mjlab/presets/<method>.py``:

   .. code-block:: python

      def apply_mymethod(cfg, *, body_names: tuple[str, ...]) -> None:
        """One line on the method, with the paper."""
        cfg.rewards["motion_body_pos"].params["body_names"] = body_names
        cfg.commands["motion"].rsi.strategy = "binary_failure"

   Export it from ``presets/__init__.py``. Robot names are arguments, never constants.

2. **Add the task** — a builder and a ``WbcTaskConfig`` in ``robots/<robot>/tasks.py`` (or
   your extension package):

   .. code-block:: python

      def g1_mymethod_env_cfg():
        cfg = g1_base_cfg()
        apply_mymethod(cfg, body_names=G1_MOTION_BODY_NAMES)
        return cfg

      WbcTaskConfig(
        task_id="Wbc-G1-MyMethod",
        robot_id="g1",
        description="MyMethod (arXiv:XXXX): …",
        experiment_name="wbc_g1_mymethod",
        build_env_cfg=g1_mymethod_env_cfg,
      )

3. **Check it** — the task registers, builds and trains:

   .. code-block:: bash

      uv run wbc-mjlab-list-envs | grep MyMethod
      uv run wbc-mjlab-train --task Wbc-G1-MyMethod --dataset samples --agent.max-iterations 10

4. **Document it** — a page under ``docs/source/tasks/`` (what it changes, train & play), a
   card in :doc:`index`, and the paper in :doc:`../research`.

.. seealso::

   - :doc:`../concepts/presets_and_tasks` — how presets compose
   - :doc:`../extensions/index` — tasks for your own robot
   - :doc:`../api/presets` · :doc:`../api/registry` — signatures
