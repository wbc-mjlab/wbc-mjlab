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

Docker from a terminal
----------------------

The repository's ``compose.yaml`` defines one GPU development workspace, shared
with VS Code/Cursor. Install GNU Make, Docker with **Compose 2.30+**, an NVIDIA driver
compatible with CUDA 12.8 and the
`NVIDIA Container Toolkit <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>`_
on the Linux GPU host. Your user must be able to run Docker without ``sudo``.
Compose requests GPU access using its
`GPU service configuration <https://docs.docker.com/compose/how-tos/gpu-support/>`_.

From the server checkout:

.. code-block:: bash

   make docker-shell      # build, install locked dependencies, wait for GPU checks, open bash
   make docker-studio     # start Studio at http://localhost:8765 (Ctrl+C stops Studio)
   make docker-run CMD='wbc-mjlab-train --task Wbc-G1 --dataset samples --recipe wm-blink'
   make docker-run CMD='make check'
   make docker-stop       # stop the workspace; keep source, logs, configs and volumes

Run these as separate commands. ``docker-shell`` opens an interactive shell where
normal ``uv run`` and ``make`` commands work. ``docker-run`` runs a foreground
command in the same workspace. ``docker-up`` starts it without opening a shell.
The first setup downloads several GB; later starts reuse dependencies and caches.
Use ``docker compose logs -f dev`` to inspect setup progress or errors.
Foreground commands use a terminal so Ctrl+C reaches training/Studio. For batch
execution without a terminal, add ``DOCKER_EXEC_FLAGS=-T`` to ``make docker-run``.

The Make targets match the container UID/GID to the host user, wait for setup to
finish, and use Compose's init process for signal handling. They start one persistent
container; exiting a shell or closing the editor leaves it running. ``docker-stop``
stops **all** its jobs, including training. It never deletes named volumes.
Source, datasets and logs live in the checkout. The Docker socket is not mounted,
and no privileged container or host networking is needed.

Studio, TensorBoard and Viser publish only on the server's loopback interface.
For a remote machine, open a tunnel **from your laptop** and browse locally:

.. code-block:: bash

   ssh -N -L 8765:127.0.0.1:8765 -L 6006:127.0.0.1:6006 -L 8080:127.0.0.1:8080 user@server

The mapped host ports can be changed without modifying tracked files:

.. code-block:: bash

   WBC_STUDIO_PORT=18765 WBC_TENSORBOARD_PORT=16006 WBC_VISER_PORT=18080 make docker-studio

Use the same port variables on subsequent Make commands, or put them in an ignored
``.env`` file. Use the chosen server ports as the tunnel destinations. For direct
Compose commands on a host whose UID/GID differ from 1000, export ``WBC_UID=$(id -u)``
and ``WBC_GID=$(id -g)`` first. The Make targets do this automatically.

After changing dependencies in an already running workspace, sync them explicitly:

.. code-block:: bash

   make docker-run CMD='uv sync --locked --extra cu128 --group dev --group docs'

After changing the Dockerfile, run ``make docker-stop``, ``make docker-build`` and
``make docker-up``. Ordinary shell/run commands reuse the image so they do not
rebuild or recreate a workspace that is already training.
After a branch switch, the container installs the checked-out locked environment on
its next start. ``docker compose down -v`` explicitly deletes the dependency/cache
and Studio-config volumes; it is not part of any Make target.

Choosing GPUs
~~~~~~~~~~~~~

.. code-block:: bash

   make docker-gpu-list           # load and free memory, without changing anything
   make docker-gpus               # live checkbox menu: arrows, Space, Enter; Q cancels
   make docker-gpus GPU_IDS=0,2   # non-interactive selection; GPU_IDS=all selects all
   make docker-studio

The host picker needs Python 3 and ``nvidia-smi``, without installing the project.
Green means load below 30% and at least 50% memory free; amber means moderate load;
red means load at least 80% or less than 20% memory free. Unknown metrics remain
explicitly unknown. The menu refreshes every two seconds. These are advisory load
indicators, not reservations or guarantees that a training run will fit.

The picker stores GPU UUIDs in ignored ``.devcontainer/gpus.local.yaml`` and adds
``WBC_GPU_CONFIG`` to your ignored ``.env``, preserving other settings. Compose's
shared GPU service uses these IDs for both terminal and editor containers. Without
a selection it exposes all GPUs. Stop an existing workspace before changing its
GPU set; the picker refuses to change the hardware under a running job. Then start
it again normally. Reopening the picker with the same selection is harmless.

In **Studio → Train → GPUs**, open the dropdown and toggle training GPUs. Each row
shows its name, utilization, free/total GiB and a colored status. Values refresh
every three seconds while open, and keyboard focus stays on the checkbox. Studio
only offers devices visible to CUDA inside its container. Changing this per-job
selection leaves existing jobs alone; playback continues on Studio's own device.

GPU UUIDs are resolved to CUDA ordinals at launch, so a host GPU numbered 3 can
correctly become ``cuda:0`` in a restricted container. Built-in presets can train
on multiple GPUs. Custom Studio presets currently support one; world-model
curriculum probes must be disabled for multi-GPU training. Unsupported combinations
are rejected at launch. Per-job UUIDs are saved with the job metadata.
For direct training commands, pass a quoted list: ``--gpu-ids "[2]"`` for one
visible GPU, or ``--gpu-ids "[0, 2]"`` for two. This training flag differs from the
Docker picker's host-index shortcut ``make docker-gpus GPU_IDS=0,2``.

Remote GPU development in VS Code or Cursor
-------------------------------------------

The GPU ``.devcontainer`` uses the same Compose service as the terminal commands.
Your editor runs locally; Python,
simulation and training run inside Docker on the SSH server. Studio and Viser render
in your local browser. This follows the editors' documented
`Remote SSH + Dev Containers workflow <https://code.visualstudio.com/docs/remote/ssh#_open-a-folder-on-a-remote-ssh-host-in-a-container>`_
and `Cursor container support <https://cursor.com/for/data-science>`_.

Before connecting, the **remote Linux server** needs an NVIDIA GPU/driver compatible
with CUDA 12.8, Docker accessible to your SSH user, and
`NVIDIA Container Toolkit configured for Docker <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>`_.
Verify GPU passthrough on the server:

.. code-block:: bash

   nvidia-smi
   docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi

Then:

1. Clone the repository **on the server**, including the branch you want to develop.
2. In VS Code, install **Remote - SSH** and **Dev Containers**. In Cursor, install
   its **Remote SSH** and **Dev Containers** extensions.
3. Connect to the server through Remote SSH and open its repository folder.
4. Run **Dev Containers: Reopen in Container** from the command palette. Wait for
   the image build and dependency installation; the first setup downloads several GB.
5. The editor selects ``/opt/venv/bin/python``. Open a terminal and run:

   .. code-block:: bash

      make check
      uv run wbc-mjlab-studio --host 127.0.0.1 --no-open-browser

If the terminal workspace already has a running training job, use **Dev Containers:
Attach to Running Container** to attach to it. Reopening/rebuilding can recreate
the Compose service and interrupt its processes.

Open forwarded port **8765** from the editor's **Ports** panel. TensorBoard uses
**6006**, standalone Viser playback uses **8080**; these ports are also configured
for forwarding. If a service chooses another port because one is busy, forward that
port in the same panel. No public Docker ports or X11 sockets are required. Native
MuJoCo windows are outside this headless setup; use Studio or ``--viewer viser``.

The container supplies CUDA development tools, Python 3.12, EGL/video libraries,
``uv`` and the locked CUDA, dev and docs dependencies. GPU access is checked during
setup. Dependencies are installed in editable mode, so source changes are immediate.
After changing ``pyproject.toml`` or switching branches with a different lockfile, run:

.. code-block:: bash

   uv sync --locked --extra cu128 --group dev --group docs

``make`` commands also use this container environment through
``UV_PROJECT_ENVIRONMENT``. The checkout's host ``.venv`` is left alone. Python/Ruff
editor extensions, pytest discovery and large-data file-watcher exclusions are included.

Persistence and external data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Source, ``data/`` and ``logs/`` are bind-mounted from the server checkout. Named
volumes preserve the container virtualenv, download/compiler caches and Studio
configs across rebuilds. Studio's container configs start separately from your
server user's configs; copy any existing presets/recipes into
``/home/developer/.config/wbc-mjlab`` if needed. Deleting those named volumes removes
their contents. The editor leaves the workspace running when closed. Stopping or
rebuilding the container stops training processes; save full checkpoints first.

Datasets or robot assets outside the checkout need an explicit bind mount. Create
an ignored ``compose.override.yaml`` alongside ``compose.yaml``, for example:

.. code-block:: yaml

   services:
     dev:
       volumes:
         - /srv/motions:/datasets:ro

Restart the workspace (or rebuild the editor container) to apply the mount.
For the editor, also set ``dockerComposeFile`` in ``.devcontainer/devcontainer.json``
to ``["../compose.yaml", "../compose.override.yaml"]``: explicit Compose file lists
do not automatically load the override file.
Use ``/datasets`` in Studio. Omit ``:ro`` if conversion should write there.
The source path belongs to the **server**, not your laptop. A local sibling mjlab
checkout similarly needs mounting and the editable override described in
:doc:`installation`. Keep machine-specific mount changes local.
Existing custom robot configs or run snapshots may contain absolute server paths;
mount those assets at the same paths inside the container or update their paths.

If GPU setup fails, check the two host commands above and the NVIDIA runtime
configuration before rebuilding. For a CPU-only machine, use the existing
``make sync-cpu`` workflow outside this GPU container.

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
