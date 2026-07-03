.. _contributing:

Contributing
============

Bug fixes and documentation improvements are welcome. For new **tasks**, MDP
changes, or paper reproductions, `open an issue <https://github.com/wbc-mjlab/wbc-mjlab/issues>`_
first — we keep a shared MDP in ``env/`` and express paper differences as
registered tasks.

Planned themes: :doc:`roadmap`. Layers: :doc:`architecture`, :doc:`mdp/index`, :doc:`extensions/index`.

Setup
-----

.. code-block:: bash

   git clone https://github.com/wbc-mjlab/wbc-mjlab.git && cd wbc-mjlab
   make sync
   uv run wbc-mjlab-list-envs

Uses `uv <https://docs.astral.sh/uv/>`_ with a locked environment (``uv.lock``),
matching the mjlab workflow:

- ``make sync`` — ``uv sync --extra cu128 --group dev`` (CUDA PyTorch)
- ``make sync-cpu`` — CPU-only / macOS evaluation

See :doc:`installation` for pip, PyPI, and local mjlab checkout options.

Optional: ``uvx pre-commit install`` after ``make sync``.

Build the documentation site: ``make docs`` (requires ``uv sync --group docs``).

Pull requests
-------------

1. Fork, branch from ``main``, keep PRs focused.
2. Run ``make check`` (lint + smoke) or at minimum ``make format`` and ``make smoke``.
3. Smoke-test what you touched (short train/play on ``--dataset samples`` when relevant).
4. Link the issue: ``Closes #N``.

Adding a paper task
-------------------

1. Preset: ``presets/<method>.py`` with an ``apply_*`` cfg mutator (or stack existing presets)
2. Task builder in ``robots/<id>/tasks.py``; register ``WbcTaskConfig``
3. Neutral names in code; cite the paper in the task description / docstring
4. Document under :doc:`tasks/index`
5. Example: ``uv run wbc-mjlab-train --task Wbc-... --dataset ...``

**New robot:** use an extension package — :doc:`extensions/extensions`.

Help
----

- **Issues:** https://github.com/wbc-mjlab/wbc-mjlab/issues
- **mjlab:** https://github.com/mujocolab/mjlab/issues

License
-------

Contributions are Apache 2.0 (see `LICENSE <https://github.com/wbc-mjlab/wbc-mjlab/blob/main/LICENSE>`_).
