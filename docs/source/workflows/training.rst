.. _training:

Training
========

WBC training runs can take a long time (large motion libraries, adaptive RSI
curriculum, high ``max_iterations``).

Built on mjlab
--------------

WBC-MJLab is **inspired by and built on** `mjlab <https://github.com/mujocolab/mjlab>`_ —
the lightweight GPU-accelerated RL framework created by
`Kevin Zakka <https://github.com/kevinzakka>`_. Training reuses mjlab's manager-based
env stack, RSL-RL runner, and ``scripts.train`` entry point (including multi-GPU via
``torchrunx``).

``wbc-mjlab-train`` is a thin wrapper that adds WBC **task** and **dataset** flags on
top; **all mjlab train flags still pass through**. For upstream train/env options, see
`mjlab documentation <https://mujocolab.github.io/mjlab/main/>`_ and cite mjlab in
papers — :doc:`../research`.

WBC-MJLab also draws on open whole-body tracking stacks such as
`HybridRobotics/whole_body_tracking <https://github.com/HybridRobotics/whole_body_tracking>`_
(BeyondMimic on Isaac Lab) and
`NVlabs/GR00T-WholeBodyControl <https://github.com/NVlabs/GR00T-WholeBodyControl>`_,
reimplemented as **lightweight, modular mjlab presets** rather than monolithic
frameworks — see :doc:`../research` (Lineage & inspiration).

mjlab CLI (everything passes through)
-------------------------------------

After wbc-mjlab strips its own flags, the remaining argv is forwarded to mjlab's
tyro-based ``TrainConfig``. That means the full mjlab surface is available:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Flag group
     - Examples
   * - **Agent (PPO / runner)**
     - ``--agent.max-iterations 200000``, ``--agent.save-interval 250``, ``--agent.seed 42``
   * - **Environment**
     - ``--env.scene.num-envs 4096``, ``--env.sim.dt 0.005``, nested ``--env.*`` overrides
   * - **Logging**
     - ``--agent.logger wandb``, ``--agent.run-name my_ablation``, ``--log-root logs/rsl_rl``
   * - **Video**
     - ``--video``, ``--video-interval 2000``
   * - **GPUs**
     - ``--gpu-ids 0 1 2 3``, ``--gpu-ids all``, ``--gpu-ids`` omitted with ``CUDA_VISIBLE_DEVICES=""`` for CPU

Discover the full schema:

.. code-block:: bash

   uv run wbc-mjlab-train --help
   uv run wbc-mjlab-train Wbc-G1 --help    # defaults filled from task config

WBC-specific flags (parsed first)
---------------------------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Flag
     - Effect
   * - ``--task Wbc-G1``
     - Registered task id (selects env + RL cfg)
   * - ``--dataset lafan``
     - Resolve ``data/<robot>/lafan/`` → ``--motion-file``
   * - ``--dataset-path /path``
     - Explicit NPZ file or folder of clips
   * - ``--cache-motion-bundle``
     - Use/write stacked ``<dataset>.npz`` on disk
   * - ``--robot g1``
     - Override robot id (usually inferred from task)
   * - ``--use_wandb``
     - Opt in to wandb (default logger is **tensorboard** in wbc-mjlab)

Shorthand: ``--robot g1 --dataset samples`` maps to the default task for that robot.

Log layout
----------

.. code-block:: text

   logs/rsl_rl/<experiment_name>/<timestamp>_<run_name>/
     model_<iteration>.pt
     params/
       env.yaml
       agent.yaml
       policy.onnx          # policy-only export on save
       config.yaml          # deploy tracking params
       rsi_bin_stats.npz    # when RSI persistence enabled

``experiment_name`` comes from the task's ``WbcTaskConfig`` (e.g. ``wbc_g1`` for
``Wbc-G1``).

Resume training
---------------

WBC defaults to **200k** PPO iterations — runs are meant to be **stopped and
continued**, not restarted from scratch each time.

.. code-block:: bash

   # first launch
   uv run wbc-mjlab-train --task Wbc-G1 --dataset lafan --cache-motion-bundle

   # continue (same task + dataset)
   uv run wbc-mjlab-train --task Wbc-G1 --dataset lafan --cache-motion-bundle \
     --agent.resume True

What ``--agent.resume True`` does
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **New run folder** — each launch creates a fresh timestamped directory under
   ``logs/rsl_rl/<experiment_name>/`` (logs/checkpoints for the continued session).
2. **Load latest checkpoint** — mjlab searches ``logs/rsl_rl/<experiment_name>/`` for
   the most recent run matching ``--agent.load-run`` (default ``.*``) and checkpoint
   matching ``--agent.load-checkpoint`` (default ``model_.*.pt``).
3. **Restore training state** — policy (+ optimizer) weights via RSL-RL ``runner.load``.
4. **Restore RSI curriculum** — wbc-mjlab's runner also loads ``rsi_bin_stats.npz`` from
   the checkpoint run's ``params/`` when ``persist_failure_levels=True`` (WBC/Zest
   presets), so adaptive bin failure EMA continues where you left off.

Pick a specific prior run:

.. code-block:: bash

   uv run wbc-mjlab-train --task Wbc-G1 --dataset lafan \
     --agent.resume True \
     --agent.load-run 2026-07-03_14-30-00 \
     --agent.load-checkpoint model_5000.pt

Use the **same ``--task`` and motion data** as the original run so env dimensions and
motion library match the checkpoint.

Multi-GPU training
------------------

mjlab launches **one process per GPU** via ``torchrunx`` when more than one GPU is
selected. Each worker gets a subset of parallel envs on its device; gradients are
synchronized by RSL-RL.

.. code-block:: bash

   # four GPUs on one machine
   uv run wbc-mjlab-train --task Wbc-G1 --dataset lafan --gpu-ids 0 1 2 3

   # all visible GPUs (respects CUDA_VISIBLE_DEVICES)
   uv run wbc-mjlab-train --task Wbc-G1 --dataset lafan --gpu-ids all

   # restrict visible devices first
   CUDA_VISIBLE_DEVICES=2,3 uv run wbc-mjlab-train --task Wbc-G1 --dataset lafan --gpu-ids 0 1

``--gpu-ids`` indices are **into** ``CUDA_VISIBLE_DEVICES``, not necessarily physical
GPU ids. Single-GPU (default ``--gpu-ids 0``) runs in-process without torchrunx.

Worker logs go to ``<run_dir>/torchrunx/`` unless overridden with ``--torchrunx-log-dir``.

Tips
----

- **TensorBoard:** ``tensorboard --logdir logs/rsl_rl``
- **Fewer envs for debugging:** ``--env.scene.num-envs 512``
- **Extend horizon:** ``--agent.max-iterations 300000`` (works with resume)
- **Play while training:** use ``wbc-mjlab-play`` on the latest ``model_*.pt`` in any
  run folder — see :doc:`../visualization`

Related: :doc:`../usage`, :doc:`quickstart`, :doc:`../tasks/index`, :doc:`../mdp/rsi`.
