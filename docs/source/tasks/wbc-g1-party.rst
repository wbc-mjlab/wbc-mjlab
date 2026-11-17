Wbc-G1-Party and Wbc-G1-Sparse5
===============================

``Wbc-G1-Party`` and ``Wbc-G1-Sparse5`` adapt choices of
`PartyMimic <https://github.com/Roboparty/PartyMimic>`_ (MIT, itself derived from
BlinkMimic) to this repository's shared MDP and world-model trainer. Pair them with the
``wm-party`` recipe. Task and recipe stay independent: ``wm-party`` trains every flat-ground
preset ``wm-blink`` trains, and PPO trains both tasks.

.. code-block:: bash

   uv run wbc-mjlab-train --task Wbc-G1-Party --dataset samples --recipe wm-party
   uv run wbc-mjlab-train --task Wbc-G1-Sparse5 --dataset samples --recipe wm-party \
     --agent.actor.keypoint-skip True

As with Blink, this is an adaptation, not a reproduction of PartyMimic's reported
numbers: robot asset, retargeting, datasets and evaluator differ. Compare under
identical settings with the held-out evaluation below.

The tasks
---------

``Wbc-G1-Party`` is ``Wbc-G1-Blink`` with

* pose kernels (body position / orientation, global root position / orientation)
  narrowed to 0.6 of their width (PartyMimic ``--std_scale``); velocity kernels keep
  theirs. ``apply_party(kernel_scale=...)`` changes it: 0.35 tracks tighter but slips
  more under soft contact;
* diversity-balanced uniform resets (PartyMimic ``--div_k``): the uniform share of
  episode starts is spread evenly over 24 k-means clusters of 1 s reference bins instead
  of by duration, so a library dominated by walking still starts often in its rare
  motions. ``RsiCfg.diversity_clusters`` / ``diversity_tau`` set it on any task.

``Wbc-G1-Sparse5`` is ``Wbc-G1-Party`` following **keypoint poses** instead of joint
references (PartyMimic ``--interface sparse5``), as a teleoperation suit or video pose
estimator provides them:

* ``ref_keypoints``: position and 6D orientation of the pelvis, torso, wrists and ankles
  over the lookahead frames, each relative to the *current reference anchor's heading
  frame*, so the input does not depend on where the motion is or which way it faces;
* ``keypoint_error``: robot against reference on the keypoints after the first
  (position, then axis-angle orientation), with ±0.02 noise.

``ref_joint_pos``, ``ref_joint_vel`` and ``ref_anchor_shift_b`` leave the actor; the critic
keeps its privileged references and the rewards are unchanged. ``apply_sparse_keypoints``
builds the same interface on other presets and keypoint sets. The deploy ``config.yaml``
lists both terms.

The recipe
----------

``wm-party`` starts from ``wm-blink`` without the curriculum probes and adds:

* **Muon with decoupled weight decay** 0.1 on the hidden matrices
  (``algorithm.muon_weight_decay``) and Muon on the actor's first and last layers too
  (``algorithm.actor_muon_io``).
* **Self-imitation** (``algorithm.self_imitation``): replayed simulator windows give a
  recorded return; where it beats the critic, the actor regresses toward the recorded
  action, weighted by the advantage. ``self_imitation_share`` / ``self_imitation_loss``
  are logged.
* **Value-gradient consistency** (``algorithm.value_gradient``): the critic's state
  gradient is fit to the imagined return's gradient through the model, on
  ``value_gradient_states`` states for ``value_gradient_epochs`` critic epochs.
  ``value_gradient_error`` and ``value_gradient_cosine`` are logged.
* **bf16 autocast** (``algorithm.bf16``) of the network matmuls — world-model dynamics
  and heads, critic, the policy head — while integration, kinematics, rewards and
  normalization stay float32. CUDA only; exported ONNX stays float32. Set it to
  ``False`` on GPUs without bfloat16.
* 64 warm-up steps before the first update.

The reference-action shortcut and mirroring stay opt-ins, as in ``wm-blink``.

Policy options
--------------

``--agent.actor.architecture xattn``
   The cross-attention reference encoder (PartyMimic's RGMT ``XAttnEncoder``): a causal
   self-attention block over the history tokens, then a query attending to one token per
   goal frame. Terms without history or frames are repeated in every token. Exports to
   ONNX. ``xattn_dim`` / ``xattn_heads`` size it; ``goal_terms`` overrides which terms
   are goals.

``--agent.actor.latent-aux True`` with ``--agent.algorithm.latent-aux 0.1``
   A linear head predicts the goal frames' joint positions from the reference latent
   (motion encoder or xattn), so it keeps the joint information a keypoint input only
   implies. Logged as ``latent_aux_rms``. Training-only; not exported.

``--agent.actor.keypoint-skip True``
   For keypoint interfaces: a frozen MLP fit before training maps the next reference
   keypoints to the reference action (PartyMimic's kp2q), added inside the policy like
   the reference residual. The fit runs ``keypoint_skip_steps`` Adam steps on a table
   of every reference step and prints its held-out RMS. It is part of the ONNX. It
   excludes ``reference_residual``.

``--agent.init-checkpoint PATH``
   Warm start (PartyMimic ``--init``): networks from a full checkpoint, fresh optimizers,
   schedules, iteration count and sampling statistics — fine-tuning a general tracker on
   a few hard clips. Use resume to continue a run instead.

Held-out evaluation
-------------------

Evaluate a checkpoint on clips it never trained on, one ``--split`` per set; the report
has each split and their mean, including ``local_body_pos_mm`` (the sim2real
leaderboard's MPJPE), ``global_body_pos_mm`` and ``joint_pos_deg``:

.. code-block:: bash

   uv run wbc-mjlab-evaluate --checkpoint-file logs/rsl_rl/wbc_g1_party/<run>/model_<n>.pt \
     --split loco=data/g1/heldout_loco --split dance=data/g1/heldout_dance --out eval.json

Held-out results never feed the training curriculum. A baseline for comparison is the
same command on a ``wm-blink`` run of ``Wbc-G1-Blink`` with the same dataset and budget.

Motion data
-----------

``wbc-mjlab-motion-data`` (:ref:`cli-motion-data`) ports PartyMimic's data tooling:
clip QA, diversity analysis and a diversity-based subset of a large library, written as
a dataset folder of links for ``--dataset-path``.

Not ported
----------

* PartyMimic's two-GPU actor/learner pipeline (``--pipeline``): it copies the world
  model, replay, motion library and MDP to a second GPU and overlaps collection with the
  policy update. Here the imagination is bound to the mjlab environment's managers on
  one device, so this needs a second imagination environment; multi-GPU training uses
  data parallelism (``torchrun``) instead.
* The xattn encoder broadcasts the keypoint error block into every token rather than
  excluding it, and the sparse5 goal byte layout differs (every keypoint carries position
  and 6D orientation; the current frame comes first).
* PartyMimic's checkpoints and datasets carry non-commercial licenses and are not
  included.
