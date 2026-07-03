.. _installation:

Installation
============

System requirements
-------------------

- **Training:** Linux + NVIDIA GPU (CUDA 12.8+ recommended, same as `mjlab <https://github.com/mujocolab/mjlab>`_)
- **Evaluation:** Linux, macOS, or Windows (WSL) with ``make sync-cpu``
- **Python:** 3.10+

`wbc-mjlab <https://github.com/wbc-mjlab/wbc-mjlab>`_ is an extension of
`mjlab <https://github.com/mujocolab/mjlab>`_. Install mjlab's stack via one of
the paths below.

Method 1 — Develop from source (uv, recommended)
------------------------------------------------

For hacking on ``wbc_mjlab`` or running the bundled samples.

Install uv
^^^^^^^^^^

.. code-block:: bash

   curl -LsSf https://astral.sh/uv/install.sh | sh

Clone and sync
^^^^^^^^^^^^^^

.. code-block:: bash

   git clone https://github.com/wbc-mjlab/wbc-mjlab.git && cd wbc-mjlab
   make sync          # uv sync --extra cu128 --group dev

CPU-only / macOS evaluation:

.. code-block:: bash

   make sync-cpu      # uv sync --extra cpu --group dev

Verify
^^^^^^

.. code-block:: bash

   uv run wbc-mjlab-list-envs

``uv run`` uses the locked environment in ``uv.lock`` (same workflow as mjlab).

Method 2 — Use as a dependency in your own uv project
-----------------------------------------------------

Add ``wbc-mjlab`` to an existing `uv <https://docs.astral.sh/uv/>`_ project:

.. code-block:: bash

   uv add wbc-mjlab mjlab

From GitHub:

.. code-block:: bash

   uv add "wbc-mjlab @ git+https://github.com/wbc-mjlab/wbc-mjlab"

Editable local checkout:

.. code-block:: bash

   uv add --editable /path/to/wbc_mjlab

Ensure your project selects a CUDA or CPU extra for PyTorch (see mjlab
`installation guide <https://mujocolab.github.io/mjlab/main/source/installation.html>`_).

Method 3 — Classic pip
----------------------

.. code-block:: bash

   pip install mjlab wbc-mjlab

Editable from a clone:

.. code-block:: bash

   git clone https://github.com/wbc-mjlab/wbc-mjlab.git && cd wbc-mjlab
   pip install -e .

You are responsible for a CUDA-capable PyTorch build when training on GPU.

Local mjlab checkout (optional)
-------------------------------

When developing alongside a sibling ``mjlab`` repo, pin mjlab in ``pyproject.toml``:

.. code-block:: toml

   [tool.uv.sources]
   mjlab = { path = "../../mjlab", editable = true }

.. code-block:: bash

   uv lock && make sync

Remove the override before publishing the lockfile for PyPI-only users.

After install — quickstart
--------------------------

.. code-block:: bash

   uv run wbc-mjlab-data-to-npz --robot g1 --dataset samples
   uv run wbc-mjlab-train --task Wbc-G1 --dataset samples
   uv run wbc-mjlab-play --task Wbc-G1 --dataset samples

Bundled motion credits: ``data/g1/samples/README.md`` in the repository.

See also :doc:`workflows/quickstart` for the full end-to-end workflow.
