.. _training:

Train a policy
==============

Launch a training run, follow it, and continue it later. Runs are long — WBC tasks default
to 200k PPO iterations and are meant to be stopped and resumed, not restarted.

.. admonition:: You need
   :class: tip

   A converted motion library (:doc:`../data`) and a task (:doc:`../tasks/index`).

Launch a run
------------

.. tab-set::
   :sync-group: ui

   .. tab-item:: Studio
      :sync: studio

      *Train*: the preset and dataset come from the previous steps. Set the environments and
      iterations, a run name and the logger, then **Launch**. The job runs detached — it
      survives closing the browser or restarting the Studio.

   .. tab-item:: Command line
      :sync: cli

      .. code-block:: bash

         uv run wbc-mjlab-train --task Wbc-G1 --dataset lafan \
           --env.scene.num-envs 8192 --agent.max-iterations 30000 --agent.run-name first

``wbc-mjlab-train`` wraps mjlab's training script: it adds ``--task`` and the motion-data
options, and passes every other option through (``--agent.*``, ``--env.*``, ``--gpu-ids`` …).
All options: :ref:`cli-train` · ``uv run wbc-mjlab-train Wbc-G1 --help`` shows the task's
defaults.

Choose the algorithm
--------------------

Tasks train with PPO. ``--algo wm`` trains the same task — preset, data, sampling, export —
through a learned world model instead (SuperMimic-style, one GPU): :doc:`../concepts/world_model`.

.. code-block:: bash

   uv run wbc-mjlab-train --task Wbc-G1 --dataset lafan --algo wm

Play and resume pick up the algorithm a run was trained with.

Follow progress
---------------

- **Studio** — *Train → Jobs*: progress and ETA, curves (reward, tracking errors, per-term
  rewards, sampling, losses), the log, and a TensorBoard button.
- **TensorBoard** — ``tensorboard --logdir logs/rsl_rl``; ``--agent.logger wandb`` for W&B.
- **Watch the policy** — load any checkpoint mid-training in the Studio's *Play* step or with
  ``wbc-mjlab-play`` (:doc:`../visualization`).

Each run writes ``logs/rsl_rl/<experiment>/<timestamp>_<run_name>/``: checkpoints, TensorBoard
events and ``params/`` (configs, ``policy.onnx``, ``config.yaml``, sampling state) —
:ref:`reference-configuration`.

Train a Studio preset
---------------------

Presets edited in the Studio train like a task — from the Studio's *Train* step, or from the
command line:

.. code-block:: bash

   uv run python -m wbc_mjlab.studio.train_preset --preset my_preset --dataset samples \
     --env.scene.num-envs 4096 --agent.max-iterations 30000

All ``wbc-mjlab-train`` flags pass through. Single GPU only: ``torchrunx`` workers would not
see the preset task. See :doc:`../studio`.

Resume a run
------------

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

Train on several GPUs
---------------------

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

- **Fewer envs for debugging:** ``--env.scene.num-envs 512``
- **Extend horizon:** ``--agent.max-iterations 300000`` (works with resume)
- **Faster restarts on big libraries:** ``--cache-motion-bundle``

.. seealso::

   - :doc:`deploy` — export the trained policy for a robot
   - :doc:`../concepts/adaptive_sampling` — what the sampling curves show
   - :ref:`cli-train` — every option
