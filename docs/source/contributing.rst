.. _contributing:

Contributing
============

Bug fixes and documentation improvements are welcome straight away. For new tasks, MDP
changes or paper reproductions, `open an issue <https://github.com/wbc-mjlab/wbc-mjlab/issues>`_
first: the MDP is shared, and methods are expressed as presets and tasks rather than forks
(:doc:`concepts/modularity`). Planned work: :doc:`roadmap`.

Development setup
-----------------

.. code-block:: bash

   git clone https://github.com/wbc-mjlab/wbc-mjlab.git && cd wbc-mjlab
   make sync                  # CUDA env + dev tools (make sync-cpu without a GPU)
   uvx pre-commit install     # optional: format and lint on commit

.. list-table::
   :widths: 26 74

   * - ``make format``
     - ``ruff format`` + autofix
   * - ``make lint``
     - ``ruff format --check`` + ``ruff check``
   * - ``make smoke``
     - ``wbc-mjlab-list-envs`` — every task registers
   * - ``make test``
     - ``pytest``
   * - ``make check``
     - lint + smoke
   * - ``make docs`` · ``make docs-watch``
     - Build the docs · live-reload server (``uv sync --group docs`` first; see
       ``docs/BUILDING.md``)

Repository layout
-----------------

.. code-block:: text

   src/wbc_mjlab/
     env/                  shared MDP: template (wbc_env_cfg.py) and terms (mdp/)
     presets/              method recipes: apply_wbc, apply_zest, …
     robots/g1/            the in-tree robot: model, actuators, tasks
     robots/custom.py      robots added from model files (robot.yaml)
     tasks/                WbcTaskConfig and registration
     motion/               conversion, formats, mirroring
     export/               policy.onnx + config.yaml
     studio/               WBC Studio: services, web UI (web/), viser UI
     extension.py          register_wbc_extension
   data/<robot>/           motion libraries (samples/ versioned)
   docs/                   this site
   tests/                  pytest suite

Pull requests
-------------

1. Branch from ``main``; keep each PR to one change.
2. Run ``make format`` and ``make check``; ``make test`` for code changes.
3. Smoke-test what you touched — a short train / play on ``--dataset samples``.
4. Update the docs page that describes the behaviour.
5. Link the issue (``Closes #N``).

Where changes go
----------------

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - Change
     - Guide
   * - A paper's setup as a task
     - :doc:`tasks/adding` — neutral names in code, the paper in the task description and
       :doc:`research`
   * - A new robot
     - :doc:`extensions/index` — an extension package, not a fork
   * - A new reward / observation / sampling term
     - ``env/mdp/`` — after an issue
   * - Docs
     - ``docs/source/`` — :doc:`how-to guides <recipes>` show tasks, :doc:`concepts
       <concepts/index>` explain, reference pages list facts

Help and license
----------------

Questions and bugs: `issues <https://github.com/wbc-mjlab/wbc-mjlab/issues>`_ (simulation
stack: `mjlab issues <https://github.com/mujocolab/mjlab/issues>`_). Contributions are
licensed Apache 2.0 (`LICENSE <https://github.com/wbc-mjlab/wbc-mjlab/blob/main/LICENSE>`_).
