.. _installation:

Installation
============

System requirements
-------------------

- **Training:** Linux + NVIDIA GPU (CUDA 12.8+ recommended, same as `mjlab <https://github.com/mujocolab/mjlab>`_)
- **Evaluation:** Linux, macOS, or Windows (WSL) with CPU PyTorch
- **Python:** 3.10–3.13

`wbc-mjlab <https://github.com/wbc-mjlab/wbc-mjlab>`_ extends
`mjlab <https://github.com/mujocolab/mjlab>`_. Install mjlab's stack via one of
the paths below.

From source (development)
---------------------------

For hacking on ``wbc_mjlab`` or running the bundled samples.

.. tab-set::

   .. tab-item:: uv

      Install `uv <https://docs.astral.sh/uv/getting-started/installation/>`_ if needed, then:

      .. code-block:: bash

         git clone https://github.com/wbc-mjlab/wbc-mjlab.git && cd wbc-mjlab
         make sync

      ``make sync`` runs ``uv sync --extra cu128 --group dev`` (CUDA PyTorch + dev
      tools). CPU-only / macOS evaluation:

      .. code-block:: bash

         make sync-cpu

      ``uv run`` uses the locked environment in ``uv.lock`` (same workflow as mjlab).

   .. tab-item:: pip

      .. code-block:: bash

         git clone https://github.com/wbc-mjlab/wbc-mjlab.git && cd wbc-mjlab
         pip install -e ".[cu128]"

      CPU-only / macOS:

      .. code-block:: bash

         pip install -e ".[cpu]"

      You are responsible for a CUDA-capable PyTorch build when training on GPU.

Verification
------------

.. tab-set::

   .. tab-item:: uv

      .. code-block:: bash

         uv run wbc-mjlab-list-envs

   .. tab-item:: pip

      Activate your virtualenv, then:

      .. code-block:: bash

         wbc-mjlab-list-envs

Optional extras
---------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Extra / group
     - Purpose
   * - ``cu128``
     - CUDA 12.8 PyTorch (Linux training; default for ``make sync``)
   * - ``cpu``
     - CPU PyTorch (evaluation, macOS, WSL smoke tests; ``make sync-cpu``)
   * - ``dev`` (uv group)
     - pytest, ruff, pre-commit
   * - ``docs`` (uv group)
     - Sphinx site (``make docs``)

Build the documentation locally: ``docs/BUILDING.md``.

Use as a dependency
-------------------

Add ``wbc-mjlab`` to an existing project (PyPI or local checkout). Ensure your
environment has a CUDA or CPU PyTorch build when training on GPU (see mjlab
`installation guide <https://mujocolab.github.io/mjlab/main/source/installation.html>`_).

.. tab-set::

   .. tab-item:: uv

      PyPI:

      .. code-block:: bash

         uv add wbc-mjlab mjlab

      From GitHub:

      .. code-block:: bash

         uv add "wbc-mjlab @ git+https://github.com/wbc-mjlab/wbc-mjlab"

      Editable local checkout:

      .. code-block:: bash

         uv add --editable /path/to/wbc-mjlab

   .. tab-item:: pip

      PyPI:

      .. code-block:: bash

         pip install mjlab wbc-mjlab

      From GitHub:

      .. code-block:: bash

         pip install "wbc-mjlab @ git+https://github.com/wbc-mjlab/wbc-mjlab"

      Editable local checkout:

      .. code-block:: bash

         pip install -e /path/to/wbc-mjlab

Local mjlab checkout (optional)
-------------------------------

When developing alongside a sibling ``mjlab`` repo, pin mjlab in ``pyproject.toml``:

.. code-block:: toml

   [tool.uv.sources]
   mjlab = { path = "../../mjlab", editable = true }

.. code-block:: bash

   uv lock && make sync

Remove the override before publishing the lockfile for PyPI-only users.

After install
-------------

.. tab-set::

   .. tab-item:: uv

      .. code-block:: bash

         uv run wbc-mjlab-data-to-npz --robot g1 --dataset samples
         uv run wbc-mjlab-train --task Wbc-G1 --dataset samples
         uv run wbc-mjlab-play --task Wbc-G1 --dataset samples

   .. tab-item:: pip

      With your virtualenv active:

      .. code-block:: bash

         wbc-mjlab-data-to-npz --robot g1 --dataset samples
         wbc-mjlab-train --task Wbc-G1 --dataset samples
         wbc-mjlab-play --task Wbc-G1 --dataset samples

Bundled motion credits: ``data/g1/samples/README.md`` in the repository.

See :doc:`workflows/quickstart` for the full end-to-end workflow.

Related projects
----------------

- `mjlab <https://github.com/mujocolab/mjlab>`_ — manager-based RL on MuJoCo Warp
- `wbc-mjlab-extension-h2 <https://github.com/wbc-mjlab/wbc-mjlab-extension-h2>`_ —
  reference robot extension (Unitree H2)
- `wbc-g1-deploy <https://github.com/wbc-mjlab/wbc-g1-deploy>`_ — optional G1 deploy runtime
