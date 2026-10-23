.. _world-model:

World-model training
====================

``--algo wm`` trains the same task — preset, motion library, sampling, export — with a
different algorithm: a **learned, differentiable model of the robot's dynamics**, fit to the
simulator's own steps, through which the policy is improved by backpropagation instead of
PPO. It follows SuperMimic (Liao et al., 2026, arXiv:2610.28480), which reports reaching a
cluster-trained PPO tracker's return on one GPU with orders of magnitude less compute.

.. code-block:: bash

   uv run wbc-mjlab-train --task Wbc-G1-Lookahead --dataset lafan_seed --debias-z --algo wm \
     --agent.algorithm.imagined-rollouts 2048

In the :doc:`../studio`, pick **World model** under *Train → How to train*: it checks that the
world model can train the preset, offers its recipes (a 24 GB GPU one included) and every
setting, and runs the same command. Presets don't hold training settings, so any preset
trains with either method; the policy network is the same for both. Every built-in G1 task
trains with it except ``Wbc-G1-Terrain-Scan``, whose height scan is not part of the imagined
state.

.. admonition:: Experimental
   :class: warning

   On one RTX 4090, ``Wbc-G1-Lookahead`` on ``lafan_seed`` (4096 envs, 2048 imagined
   rollouts), each algorithm trained for 60 minutes and was scored with the paper's protocol
   on 1024 fixed clips (each clip once from its first frame, friction 1.0, no randomization,
   no noise):

   .. list-table::
      :header-rows: 1

      * - Training time
        - ``wm`` failed clips
        - PPO failed clips
        - ``wm`` / PPO joint error (rad)
      * - 20 min
        - 69
        - 296
        - 1.11 / 1.05
      * - 40 min
        - 36
        - 148
        - 0.85 / 1.01
      * - 60 min
        - 29
        - 125
        - 0.76 / 1.06

   On the ``Wbc-G1`` task with its domain randomization and pushes, on the 10 ``samples``
   clips, PPO was ahead after 32 minutes. Only one run of each has been compared, and only
   on these tasks; treat ``wm`` as a research option. (These runs trained the task
   under its earlier name, ``Wbc-G1-SuperMimic``, with the reference window as one
   interleaved term; they no longer replay.)

Play, export and deploy are unchanged: the policy is the same network as PPO's
(``policy.onnx``: observations → actions), and Play builds the algorithm a run was trained
with (``params/agent.yaml``).

How it trains
-------------

One loop, three phases per iteration:

1. **Collect.** The policy, with fixed Gaussian action noise, steps the simulator. Every step
   — state, action, PD targets, what the step produced — goes into a replay of the last
   ``replay_steps`` steps of each env.
2. **Fit the world model.** The dynamics network predicts the next state from the state, the
   PD targets and the domain-randomization values of the env (friction, center-of-mass
   offset, encoder bias). It is fit on ``window``-step windows, feeding its own predictions
   back, so it learns to stay on track over many steps. Along the same windows, on the
   predicted states (where imagination uses them), two small heads learn whether a step
   ends the episode and the reward terms that are not functions of the state (below).
3. **Improve the policy.** From replay states — with the observation history the env held
   there — ``horizon`` steps are *imagined* through the frozen model. The return — rewards
   weighted by a soft survival probability, plus the critic's value where the rollout
   stops — is differentiated through the model into the policy. The critic, which sees
   the state, the tracking error, the domain-randomization values and the policy's
   observation (so the reference ahead), is fit to GAE(λ) targets of the same rollouts.

The model is invariant to where the robot stands and which way it faces: inputs and
predicted changes are expressed in the robot's heading and body frames.

How it uses your preset
-----------------------

Imagined states are scored by **your task's own terms**. The observation and reward
functions read the robot through ``robot.data`` and the motion command (:ref:`observations`);
for an imagined state, a stand-in env provides exactly those objects, computed from the
state with forward kinematics (bodies, IMU sites). So a preset's observation layout, reward
weights and kernels apply unchanged — no second copy of the reward code. Observation
history is carried through the imagined steps (each term with its own history length), and
the reference frames ahead (``frames`` / ``stride`` of the reference terms,
:ref:`observations-ahead`) are read from the motion library. :doc:`../tasks/wbc-g1-lookahead` is the paper's own
setup.

At start, each nonzero reward term is classified and logged:

- **exact** — a function of the state, the reference and the actions: every tracking term,
  action rate, joint limits, …
- **learned** — needs something an imagined state does not have (actuator forces, contacts,
  the simulator's accelerations): ``actuator_torque_soft_limit``, ``joint_acc``,
  ``foot_slip``, … A head of the world model predicts it, trained on the env's own values.

Terminations are always learned (a probability per step), so any termination set works.

What changes, what is refused
-----------------------------

- **Off for training:** velocity pushes, pulls and the assistive wrench — forces from
  outside the robot that a deterministic model cannot predict (as in Play).
- **Refused, with the reason:** an actor observation that is not a function of the robot
  state (``height_scan``: the terrain is not in the model), observation delay, domain
  randomization other than startup friction / center of mass / encoder bias.
- **One GPU** (no ``--gpu-ids`` with several).

Cost
----

An update is about 20 TFLOP at the paper's sizes (3×2048 dynamics network, 2048 windows of 8
steps ×10, 4096 imagined rollouts of 32 steps): ~0.1 s on the paper's H100, at least ~0.5 s
on an RTX 4090. Each update runs as one CUDA graph of compiled kernels, so the launch cost
of the many small kinematics and reward kernels stays off the CPU.

Measured on an RTX 4090 (``Wbc-G1-Lookahead``, 4096 envs, 2048 imagined rollouts):
world-model update 0.24 s, policy update 0.42 s, ~0.95 s per iteration over a training
run, peak memory ~7 GB for the updates plus the env and the replay. 4096 imagined
rollouts with the Lookahead observation (history and reference window) do not fit in
24 GB; use ``--agent.algorithm.imagined-rollouts 2048``. The first update compiles
(``torch.compile``, several minutes; cached for later runs). Smaller networks trade
accuracy for speed: ``--agent.algorithm.model-hidden-dims "(1024, 1024, 1024)"``.

Settings
--------

``--agent.algorithm.<name>`` (defaults: the paper's):

.. list-table::
   :header-rows: 1
   :widths: 30 14 56

   * - Setting
     - Default
     - Meaning
   * - ``horizon``
     - 32
     - Imagined steps per rollout; the critic bootstraps beyond
   * - ``imagined-rollouts``
     - 4096
     - Rollouts per policy update
   * - ``window``
     - 8
     - Steps of the windows the world model is fit on
   * - ``model-hidden-dims``
     - 2048 ×3
     - Dynamics network
   * - ``model-batch`` · ``model-steps``
     - 2048 · 10
     - Windows per world-model step · steps per update
   * - ``updates-per-iteration``
     - 2
     - Updates per iteration of ``num-steps-per-env`` (5) env steps
   * - ``action-noise``
     - 0.1
     - Exploration noise on actions in the simulator
   * - ``learning-rate``
     - 2e-4
     - World model, actor, critic; cosine decay to 3 % over 250k updates
   * - ``replay-steps``
     - 1000
     - Steps of every env kept for fitting and imagination
   * - ``compile`` · ``cuda-graphs``
     - True · True
     - ``torch.compile`` the imagination step and the losses · run each update as one
       captured CUDA graph (with ``compile``)

The critic's size is ``--agent.critic.hidden-dims`` (default 256 128), the policy's
``--agent.actor.hidden-dims`` (the robot's PPO actor).

Logged during training: ``wm_dynamics`` (window prediction error, in units of a typical
one-step change), ``wm_error_8_step`` (its last step), ``wm_heads``, ``imagined_return``,
``imagined_reward``, ``imagined_termination``, ``critic``.

.. seealso::

   - :doc:`../workflows/training` — launching and resuming runs
   - :doc:`presets_and_tasks` — what a preset configures
   - :doc:`../research` — the SuperMimic paper
