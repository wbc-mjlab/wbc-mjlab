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
trains with either method; the default policy network is the same for both. An optional
motion encoder keeps the same observation-to-action interface. Every built-in G1 task
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
   — state, PD targets, previous action, successor state, what the step produced — goes into a replay of the last
   ``replay_steps`` steps of each env. A pre-reset recorder retains the terminal state
   before auto-reset overwrites it, so the model also fits transitions into failure.
   Windows stop after that transition; they never fit a reset or clip-resample teleport.
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
- **One or several GPUs:** the existing mjlab launcher gives each rank its own simulator
  and replay. Gradients are averaged before clipping, and observation, dynamics, reward
  and critic normalization statistics are pooled across ranks.

Cost
----

An update is about 20 TFLOP at the paper's sizes (3×2048 dynamics network, 2048 windows of 8
steps ×10, 4096 imagined rollouts of 32 steps): ~0.1 s on the paper's H100, at least ~0.5 s
on an RTX 4090. Each update runs as one CUDA graph of compiled kernels, so the launch cost
of the many small kinematics and reward kernels stays off the CPU.

Historical measurements on an RTX 4090 (before terminal-successor replay storage;
``Wbc-G1-Lookahead``, 4096 envs, 2048 imagined rollouts):
world-model update 0.24 s, policy update 0.42 s, ~0.95 s per iteration over a training
run, peak memory ~7 GB for the updates plus the env and the replay. 4096 imagined
rollouts with the Lookahead observation (history and reference window) do not fit in
24 GB; use ``--agent.algorithm.imagined-rollouts 2048``. The first update compiles
(``torch.compile``, several minutes; cached for later runs). Smaller networks trade
accuracy for speed: ``--agent.algorithm.model-hidden-dims "(1024, 1024, 1024)"``.
Successor storage now adds about 1.1 GiB for 1000 replay steps × 4096 envs × a
71-dimensional state in float32. Replay keeps the executed PD target and previous
action; it omits the unused current-action buffer, saving about 0.44 GiB at those
sizes. Model windows contain exactly ``window`` transitions. Optional encoder widths
also change memory use.

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

Optional model and policy experiments
-------------------------------------

The existing increment model and flat MLP actor remain the defaults. These options
adapt ideas from the independent `SuperMimic-Reproduced implementation
<https://github.com/CuboiLeo/SuperMimic-Reproduced/tree/ba6105349069bf040f1fdad60873d13138d505a2>`_
:cite:p:`supermimic_reproduced2026`; they do not change a preset's observations,
rewards, terminations or motion-data interface.

- ``--agent.algorithm.dynamics-mode integrated`` predicts velocity changes and pose
  corrections, with semi-implicit integration of the predicted next velocities
  (root position, exponential-map orientation and joint positions). Its final
  dynamics layer starts at zero, giving a constant-velocity prior. State ordering, heading/body
  frames, running normalization and analytic reward evaluation remain this repo's.
  ``increment`` selects the original model. Changing mode requires a new world model;
  checkpoint metadata rejects loading one mode as the other.
- ``--agent.actor.architecture motion_encoder`` implements the paper's
  bounded motion latent and decoder :cite:p:`supermimic2026` (Appendix C/Table A.4).
  The encoder reads whole actor term blocks; by default these are ``ref_*`` terms
  and terms looking ahead through ``frames``. ``goal-terms`` selects explicit names.
  For example, ``--agent.actor.goal-terms "('ref_joint_pos', 'ref_joint_vel')"``;
  the Studio accepts comma-separated term names, with an empty list for inference.
  ``actor.encoder-hidden-dims`` sizes the encoder, ``actor.latent-dim`` defaults to 32 with tanh,
  and ``agent.actor.hidden-dims`` continues to size the decoder. History and term
  order come from the preset, including mixed history lengths and robots with
  different joint counts. ONNX still takes one flat observation and returns actions.
  PPO uses the same actor option. Studio shows encoder fields only when selected,
  previews inferred goal names and validates explicit names before launch. Resolved
  names are also logged. Old ``algorithm.policy-architecture``, ``goal-terms``,
  ``encoder-hidden-dims`` and ``latent-dim`` settings in saved configs, recipes and
  CLI arguments migrate to ``actor.*``.
- ``--agent.algorithm.separate-action-noise True`` adds exploration only while processing
  the executed PD target. Previous-action observations and action-rate rewards use
  clean actions; replay stores the exact executed setpoints. Without the flag the
  original sampled-action convention applies. Policy learning is deterministic in
  imagination in both cases.

Compare these options one at a time before combining them. The Studio's
``wm-encoder`` recipe is a combined experiment (encoder, integrated residuals,
execution-only noise, 2048 imagined rollouts), not a measured improvement over
``wm`` or ``wm-24gb``. Play and evaluation reconstruct actor sizes and these settings
from the run's ``params/agent.yaml``. Existing flat-actor checkpoints remain loadable.

Distributed training
--------------------

.. code-block:: bash

   uv run wbc-mjlab-train --task Wbc-G1-Lookahead --dataset lafan_seed --algo wm \
     --gpu-ids 0 1 --agent.algorithm.imagined-rollouts 2048

Batch sizes and environment counts are per GPU. The configured learning rate is
multiplied by the square root of the number of ranks, following Appendix D of
SuperMimic; ``--agent.algorithm.scale-lr-by-world-size False`` disables that scaling.
The launcher and runner own process-group initialization, device placement, seeds
and rank-zero logging/checkpoints. Full-update CUDA graphs are disabled for
distributed updates; step/loss ``torch.compile`` remains enabled. This keeps
normalizer collectives and gradient averaging outside captured CUDA graphs.

Each rank maintains its own sampler and replay. A checkpoint stores rank zero's
failure levels and sampling memory alongside model/optimizer state. Resume restores
those statistics when the motion segments match, then refills the replay before
updating. The checkpoint's sampler takes precedence over a newer run sidecar;
legacy checkpoints can still read ``rsi_bin_stats.npz``. The sidecar now also saves
sampling memory, with backward compatibility for files that omit it. PPO and WM
use the same runner-owned sampler schema; segment names, frame counts and bin width
are checked before restoration.

Whole-clip evaluation and drift
-------------------------------

.. code-block:: bash

   uv run wbc-mjlab-evaluate \
     --checkpoint-file logs/rsl_rl/wbc_g1_lookahead/RUN/model_1000.pt \
     --num-envs 128 --out /tmp/benchmark.json

In Studio, select a checkpoint in *Play* and click **Evaluate**. The detached job
appears in *Train → Jobs*, with clip progress, logs and a **Download JSON** action.
It uses the selected motion library and environment count; the run's own library
is selected initially when available.
Both interfaces call ``wbc_mjlab.rl.evaluation.evaluate_checkpoint``; evaluation
loads only the actor, so it allocates no world-model replay or training model.

The task and motion path are inferred from the run; ``--task`` supports legacy runs
whose task identity cannot be inferred. ``--dataset`` can select another library.
The module entry point ``uv run python -m wbc_mjlab.scripts.evaluate`` accepts the
same options. ``--dataset-path`` accepts the existing source-directory or converted
motion input; when neither data flag is supplied, evaluation uses the run's saved
motion path. ``--clips NAME NAME`` selects specific motion segments (also useful
for separately evaluating shards). Saved Studio presets and motion mirror/debias
options are reused, and actor layout changes are rejected.

Play, evaluation and resume use the same run reconstruction: registered robot and
sensor code plus the saved preset snapshot, saved observations/actions/rewards/
terminations, command settings, simulation and terrain parameters, and agent config.
Explicit Play or benchmark overrides are applied afterward. A changed registered
term function is rejected instead of silently using different behavior. Legacy
checkpoints use the same actor/distribution key migration for training and inference.

The simulator and imagination share the joint-target transform. Execution noise is
configured explicitly on an action term, so rebuilding a trainer cannot stack noise
wrappers. Differentiable Torch FK for imagination and Warp FK for motion conversion
remain separate because their gradient and throughput requirements differ.

The protocol follows SuperMimic Appendix F :cite:p:`supermimic2026`: each clip once
from its first frame, deterministic policy, no observation/action/reset noise,
no domain randomization or pushes, friction 1.0. The preset's failure terminations
and rewards stay in force; timeouts are replaced by clip completion. Consequently,
different presets are different benchmarks. In particular, Lookahead currently
uses the torso anchor and reference-relative orientation features; its task is not
identical to the reproduction's pelvis termination and reference-only default goal.

JSON contains per-clip steps, return, failure step and triggering terms, reward-term
means, and a summary of clip-mean and step-weighted normalized return. For the known
Lookahead tracking/penalty terms the upper bound is computed from their weights
and reward timestep scaling. For other presets, supply a positive
``--max-step-reward`` bound to normalize returns; without a known bound normalized
returns are null and raw returns remain available. Failure returns are measured
over executed steps, with failures reported separately.

Drift measurements are taken before the action, before auto-reset can contaminate
the state: mean/final anchor XY error (m), absolute yaw error (degrees), first times
above 0.3 m / 15 degrees, and body velocity mean-square errors in the world frame
and after yaw alignment. Velocity metrics are MSEs, not exponential rewards.
This decomposition adapts ``scripts/diag_drift.py`` in SuperMimic-Reproduced; it
helps distinguish heading drift from articulation/velocity errors. Lookahead
already observes relative reference orientation, so the reproduction's default
policy drift results should not be assumed to apply here.

Numerical validation and sources
--------------------------------

Every ``diagnostic-interval`` updates (default 100), a separate replay sample
reports ``wm_eval_nominal_1_step`` / ``wm_eval_nominal_<window>_step``, physical-unit
RMSEs for root/joint pose and velocity, and the valid-window fraction. Set the
interval to zero to disable; ``diagnostic-batch`` defaults to 256. These diagnostics
do not update normalization. They measure prediction on replay data, not held-out
generalization. The nominal scales follow the reproduction's documented choices:
root position 0.05 m, rotation 0.1 rad, linear velocity 0.5 m/s, angular velocity
1 rad/s, joint position 0.1 rad and joint velocity 2 rad/s. They differ from the
training loss's running one-step-change scales; neither is guaranteed to match
the paper's unspecified nominal error constants.

``tests/test_wm.py`` covers simulator/FK and term parity, terminal successors,
reset/history boundaries, translation/yaw invariance, integration initialization,
fixed simulator-window overfitting, gradients through several steps, TD(lambda)
failure/clip bootstrapping, encoder training/load/export, sampler resume and the
whole-clip evaluation queue. ``tests/test_wm_distributed.py`` checks two CPU ranks
against pooled normalization and averaged optimizer updates. This is not a
multi-GPU throughput or convergence benchmark. Both motion resamplers share the
online library's small-angle SLERP, checked against SciPy.

The algorithm is based on Liao et al.'s `SuperMimic paper
<https://arxiv.org/abs/2610.28480>`_ :cite:p:`supermimic2026`. The independent
`SuperMimic-Reproduced code <https://github.com/CuboiLeo/SuperMimic-Reproduced>`_
:cite:p:`supermimic_reproduced2026` informed terminal recording
(``data/recorder.py``), execution noise (``envs/actions.py``), the integration prior
(``world_model/state_spec.py``), encoder/decoder (``policy/nets.py``), distributed
averaging (``training/distributed.py``), evaluation/drift and numerical tests.
Implementation comments link to reference commit ``ba61053``. Its MIT copyright
and license are retained in the repository's ``THIRD_PARTY_NOTICES``; it is not
official code from the paper authors. The generic preset adapter, reward fallbacks,
online motion library, clip-end bootstrapping and runner/export integration remain
wbc-mjlab's own implementation.

.. seealso::

   - :doc:`../workflows/training` — launching and resuming runs
   - :doc:`presets_and_tasks` — what a preset configures
   - :doc:`../research` — the SuperMimic paper
