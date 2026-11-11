Wbc-G1-Blink
============

``Wbc-G1-Blink`` combines the shared Lookahead task with BlinkMimic-inspired
pelvis tracking and training-only contact shaping. Select the ``wm-blink`` recipe
to use its world-model training features. The task and recipe remain independent:
PPO can train the task, and all flat-ground motion presets can use the recipe,
including WBC, Zest, BinaryFailure, EE and their SE variants.

.. code-block:: bash

   uv run wbc-mjlab-train --task Wbc-G1-Blink --dataset samples --recipe wm-blink

In Studio, choose **Wbc-G1-Blink** and **World model · BlinkMimic**. CLI flags
following the recipe override individual settings, for example:

.. code-block:: bash

   uv run wbc-mjlab-train --task Wbc-G1-Blink --dataset samples --recipe wm-blink \
     --agent.algorithm.probe-interval 0

This is an experimental adaptation to wbc-mjlab's robot, MDP, quaternion state,
observation history, and shared environment. It is not a claim of reproducing
BlinkMimic's three-minute convergence result. That result uses its own G1 asset,
retargeting, evaluator and execution path. Compare task rewards and physical
tracking errors under identical settings before interpreting training times.

What the recipe changes
-----------------------

* A 1024 × 1024 × 1024 dynamics network, five model fitting steps per update,
  and 2048 imagined rollouts. The existing replay, terminal-successor capture,
  clip-end bootstrapping, step heads, and differentiable task terms are retained.
* Integrated dynamics with a substep-dependent velocity blend: ``(n+1)/(2n)``
  for ``n`` physics substeps. Separate normalizers govern residual outputs and
  physical increment loss scales, including per-component error floors.
* Muon for hidden policy/model weight matrices; Adam for first/last layers and
  other parameters; fused Adam for the critic. The optional optimizer adapter
  requires PyTorch with ``torch.optim.Muon`` and its capturable helpers. Older
  versions can continue using the default Adam recipes. Nonfinite gradients skip
  parameters, optimizer moments and counters through a device-side fused gate.
* An EMA deployment policy (decay 0.995), a base learning rate of 0.0004 and
  a 4× actor multiplier. The multiplier decays to 1 over 300 actor updates,
  starting when the running normalized simulator step reward reaches 0.75,
  or at update 1400. This trigger uses the task's known reward upper bound.
* A motion encoder using the selected preset's reference observations and
  proprioception. Joint lookahead is not required; EE references work too.
* Recent-failure resets (share 0.3) and a probe-driven reset mixture (share 0.15).
  Mixtures apply sequentially and fall back to base RSI if no failures are known.
  Failure accumulation handles duplicate bins and never spreads across clip
  boundaries. Sampler state is saved with full checkpoints.
* Compiled collection inference and critic loss, policy-inference CUDA graphs,
  fixed-shape masked critic batches, and one metric transfer per runner update.
  The existing complete model/policy-update graphs remain in use. Model fitting
  compiles a reusable step instead of unrolling the entire window into the compiler.
* Full resumable checkpoints every 250 runner iterations. Lightweight policy-only
  snapshots remain an opt-in for runs where checkpoint I/O matters more than recovery.
* Actions clipped to ±20 in collection and imagination, and a critic normalization
  floor of 0.01, following upstream's collection clamp and normalizer scale.
  Critic targets are recomputed after updating normalization. An additional repository
  safeguard caps bootstrap values at the task's known maximum discounted return
  (``r_max / (1 - gamma)``); this cap also stops actor gradients chasing impossible
  values. Raw critic predictions remain available as ``imagined_value_max``.
  ``bound_critic`` requires known task reward bounds; it is optional for other recipes.
  The Blink task reserves 512 constraint rows and 128 contact matches.

To try the upstream 3500-update budget, add ``--agent.max-iterations 1750``.
Choose a longer budget for different or larger motion libraries; the recipe does
not impose a universal convergence deadline.

The default runner performs two actor updates per five simulator steps. Schedules
and probes use **actor updates**, so their counters do not depend on this batching.
The default SuperMimic recipe is unchanged; Blink options are opt-in.

Task-dependent options
----------------------

The reference-action shortcut and sagittal augmentation are disabled in the
portable recipe. Enable them explicitly on a compatible preset, for example:

.. code-block:: bash

   uv run wbc-mjlab-train --task Wbc-G1-Blink --dataset samples --recipe wm-blink \
     --agent.actor.reference-residual True \
     --agent.algorithm.mirror-probability 0.5

The shortcut adds the next reference joint pose inside the policy. ONNX includes
the shortcut and normalization; deployed actions remain default-relative. It
requires a clean, unscaled joint-lookahead observation in action order.
Reference-relative action terms are rejected to prevent adding the reference twice.

Augmentation reflects dynamics windows (state, executed PD targets and supported
physical context). Reward and termination heads retain the original windows.
It requires robot symmetry metadata and shared friction; missing maps or unsupported
context are rejected explicitly. These options do not alter preset observations.

Existing runs resume with their saved settings, including either option if it
was enabled. Custom recipes also retain their explicit values. Terrain scans
remain outside the shared world model's supported observation and dynamics inputs.

Contact shaping and evaluation
------------------------------

The task adds weight 0.25 times contact agreement plus a stance-slip kernel. The
simulated contact indicator is differentiable in foot height. Reference contact
uses each clip's fifth-percentile foot height plus 1.5 cm, and horizontal speed
below 0.5 m/s. This avoids treating retargeting height offsets as flight. Body names
are explicit task parameters, with no hard-coded tensor indices.

The shared deterministic evaluator and curriculum probes remove training-only
shaping. Training-library probes guide sampling; held-out benchmark results never
feed the curriculum. Use the existing ``wbc-mjlab-evaluate`` command for independent
whole-clip evaluation, including failure rate and drift.

Probe execution and memory
--------------------------

Probes start at update 1000 and run in an independent deterministic environment
using a frozen deployment-policy snapshot. They cover six-second tiles with a
random four-to-nine-second lead-in, stopping on first failure or tile/clip end.
The default queue uses 128 environments and advances 20 steps per runner iteration.
Tiles are shuffled deterministically. Completed batches immediately update failure
counts, then pause for 125 actor updates before the next batch, which takes a fresh
policy snapshot. Counts decay by 0.7 once per sweep. A large archive therefore does
not require a complete sweep before contributing curriculum feedback. ``probe_steps_per_iteration=0`` completes a sweep in one
call. Lookback windows and mixture decays live in the shared ``RsiCfg``.

``probe_async=True`` optionally submits probe chunks on a CUDA side stream. Explicit
events guard policy snapshots and result ownership. The probe simulator is
constructed on its owning stream because mjlab caches stream identity in its
Torch/Warp buffers. It defaults off so other GPU
workloads do not unexpectedly contend with a second simulation. Policy inference
can be captured using ``probe_cuda_graphs``. Physics graphs remain owned by mjlab;
the full manager-driven reset/collection loop is not captured or monkeypatched.
This preserves reset events, observation histories and terminal recorders for
other tasks. The additional world-model head rollout for mirroring also has a
cost absent from Blink's dynamics-only model.

The recipe targets one CUDA GPU. Curriculum probes currently reject distributed
runs; disable probes and collection graphs for multi-GPU experiments. Unsupported
task/action/symmetry combinations fail at construction rather than changing the
MDP silently. Larger models, longer horizons, and other task heads can exceed a
24 GB memory budget even with 2048 imagined rollouts.

Checkpoints and deployment
--------------------------

``actor_state_dict`` always stores the deployable policy. With EMA enabled,
``actor_live_state_dict`` retains training weights; optimizer state and schedule
counters resume from a full checkpoint. Normalization buffers are copied together
with their counters, while policy parameters are exponentially averaged.

Intermediate lightweight checkpoints contain policy weights only. They work with
Play, actor-only loading and evaluation, but cannot resume an optimizer. Full
checkpoints contain world model, critic, optimizers, both policies, schedule,
sampler and probe progress. An in-flight probe batch restarts on resume; completed
batches are retained. Checkpoints predating batch-wise feedback restart their probe
sweep while retaining the saved sampler. Replay is refilled, as with ordinary world-model training.

Resume using the saved run configuration and omit ``--recipe``. Full periodic
checkpoints are now the default. ``--agent.light-checkpoints True`` opts into
policy-only intermediate snapshots, which cannot recover optimizer/world-model state
following a crash. Lightweight
writes use CPU snapshots, a bounded background writer, and atomic replacement;
the final save waits for pending writes and remains a full checkpoint.

Startup and failed-run diagnosis
--------------------------------

Source packing, source-frame FK, reference-contact preprocessing and Torch compilation
are distinct phases. Like ``main``, raw source packing uses a fingerprinted NPZ cache;
first use still parses all source clips. The source FK pass remains necessary for
acceleration and optional height debias. Its duration is now logged separately.
Contact floors retain the exact per-clip quantile on the resampled grid, but compute
FK in fixed-width batches across clip boundaries. Only selected foot heights are
kept on the CPU. No full resampled body-motion tensor is materialized.

Curriculum environments reuse the training source arrays, acceleration table and
height offsets, validated against source options and kinematic model structure.
Their FK buffers, graphs and simulator state remain independent, including on an
asynchronous probe stream. This removes a second dataset read and source FK pass.

In the reported ``lafan_seed`` failure, the original log explicitly recorded first-use
packing of 14,493 clips and **456 seconds** compiling the first update. Rising imagined
returns alongside declining real rewards and exploding critic loss are a warning of
value/model exploitation; a low replay world-model error does not validate policy
rollouts outside the replay distribution. Nonfinite-gradient skipping alone cannot
prevent finite, excessive actions or values. The numerical fixes above address known
integration weaknesses, but do not establish long-run convergence or isolate one
cause for that failed run. Keep earlier full checkpoints and compare real tracking
errors and episode length, not just imagined return.

Validation and provenance
-------------------------

``tests/test_blink.py`` checks inverse dynamics/gradients, independent loss scales,
reference skip normalization, duplicate and boundary-safe failure mixtures,
Muon graph learning-rate changes, EMA/live checkpoint separation, ONNX equivalence,
symmetry involutions, probe isolation and short eager/compiled training. These
checks establish implementation behavior, not long-run tracking convergence.
For performance comparisons use multiple seeds, identical assets/motions/rewards,
first and sustained tracking targets, and report compilation/probe/checkpoint time,
simulator transitions, actor updates and peak memory separately.

A local RTX 4090 regression (seed 42, 4096 environments, 14,493 ``lafan_seed``
source clips with mirroring, the recipe's model sizes) completed 650 runner
iterations / 1288 actor updates without NaNs or capacity overflows. Critic loss
stayed below 0.063 over the last 100 iterations; final mean episode reward was
19.26. The full checkpoint's floating tensors were finite. This checks the reported
failure window, not convergence or held-out tracking quality.

On that run, cached loading took 0.22 s, source FK 0.71 s, and contact floors for
all 28,986 segments 2.05 s. A separate warmed comparison on 1024 segments
(657,544 frames) took 0.555 s with the original per-segment loop and 0.101 s with
batched preprocessing (5.49×, maximum floor difference 3.73e-9 m). These timings
measure preprocessing, not a training speedup. First-update compilation took
280 s; compiler-cache state and configuration affect this cost independently of
dataset loading.

* `SuperMimic (Liao et al., 2026) <https://arxiv.org/abs/2610.28480>`_: the underlying
  differentiable world-model training method.
* `BlinkMimic (Fuyi Chen, 2026), d264658 <https://github.com/fuyichen0725/blinkmimic/tree/d264658d6c9ce61fb52f9f0f8985a369f3ccc409>`_:
  optimizer implementation, structured residuals, normalization, skip, symmetry,
  probes and execution design.
* `InstantTrack <https://github.com/Huy-Quang-Dao/instantTrack>`_ and
  `supermimic_vibed <https://github.com/zita-ch/supermimic_vibed>`_: Blink's inherited
  code and earlier efficiency/contact/EMA ideas.
* `SuperMimic-Reproduced, ba61053 <https://github.com/CuboiLeo/SuperMimic-Reproduced/tree/ba6105349069bf040f1fdad60873d13138d505a2>`_:
  previously integrated encoder, residual, recording and diagnostic ideas.
* `Muon (Jordan et al., 2024) <https://github.com/KellerJordan/Muon>`_: hidden-matrix
  optimizer, accessed through PyTorch.

File-level comments identify adapted implementations. ``THIRD_PARTY_NOTICES``
preserves the applicable upstream MIT notices; data licensing remains separate.

Long-run stability and disturbances
-----------------------------------

For a fresh run, select ``wm-blink`` by name. Studio now passes an unedited recipe
by name to the worker instead of flattening its cached copy into CLI overrides.
The worker resolves current defaults. Explicit edits still override them; resume
continues to restore the saved configuration. Restart an already-running Studio
and reload the Train page after updating this code, then reselect the recipe.
Check ``params/agent.yaml``: ``clip_actions: 20.0``, ``critic_norm_eps: 0.01``,
``bound_critic: true``, ``termination_mode: hybrid`` and ``light_checkpoints: false``.
Old policy-only checkpoints cannot restore the world model or its optimizers.

The hybrid termination option combines the learned head with differentiable
anchor position/height and projected-gravity errors using the task's own
thresholds. This follows the soft-termination approach in
`BlinkMimic mdp.py <https://github.com/fuyichen0725/blinkmimic/blob/d264658d6c9ce61fb52f9f0f8985a369f3ccc409/src/blinkmimic/mdp.py>`_
and `xMimic mdp.py <https://github.com/nico-bohlinger/xmimic/blob/54860e100251fa80143627565568d5e6b0ff2c2b/xmimic/mdp.py>`_.
Unknown failure terms still use the learned head. Soft survival weights critic
fitting; saturated termination stops further state extrapolation. These are
repository safeguards, not evidence of matching upstream training quality.
``imagined_value_min`` and ``imagined_value_max`` expose raw critic drift.
The upper bootstrap bound does not impose an arbitrary floor on negative returns.

Optional between-step velocity kicks reuse the user-supplied ``draft_wbc``
controller and BlinkMimic's pre-kick capture / replay-cut convention:

.. code-block:: bash

   uv run wbc-mjlab-train --task Wbc-G1-Lookahead --dataset lafan_seed \
     --recipe wm-blink --agent.light-checkpoints False \
     --env.events.wm-velocity-kick.params.enabled True

In Studio, use the same ``--env.events.wm-velocity-kick.params.enabled True``
flag in **Extra arguments**. Kicks are off by default. Starting settings are
5–10 simulated seconds between kicks, up to 0.5 m/s horizontally and 0.25 rad/s
in roll/pitch, a 60 simulated-second ramp from 20% strength, and 20% clean
episodes. These are conservative repository defaults, not paper hyperparameters.
The ramp measures each environment's simulation time, not aggregate frames or
wall time, and its elapsed time survives full-checkpoint resume. Timers and
clean-episode assignments restart with the newly reset environments.

The model target is captured before the kick. Autoregressive model windows stop
at that boundary, while the episode, reference clock and policy history continue.
The next state in replay therefore teaches recovery from the disturbed state
without treating an unpredictable impulse as nominal dynamics. Deterministic
benchmarking and curriculum probes remove these kicks. Increase their strength
only after establishing an undisturbed baseline; disturbances are not a cure
for an unstable critic.

Ordinary ground contact is already part of the physical transition the model
learns. Measured contact forces are not automatically available as future inputs
during imagination. A velocity kick is not a sustained external force. Persistent pushes, payloads
or body-specific contacts require a defined force/torque frame and application
point, per-transition conditioning (or an identifiable latent estimate), and
matching force evolution during imagination. Adding an unobserved random force
to the simulator alone would make the current deterministic model's target
ambiguous. The existing rejection of unsupported ongoing force events remains.

xMimic's larger speedups rely on its JAX/XLA execution pipeline, mixed precision,
state-independent goal batching and asynchronous GPU roles. Importing its second
trainer would bypass this repository's manager/task contracts. In particular,
WBC's body-relative reference observations depend on the imagined state, so its
complete goal encoder cannot simply be hoisted out of the rollout. The reusable
termination idea above stays within the existing PyTorch trainer. See the pinned
`xMimic implementation <https://github.com/nico-bohlinger/xmimic/tree/54860e100251fa80143627565568d5e6b0ff2c2b>`_
for the execution design and benchmark qualifications.
