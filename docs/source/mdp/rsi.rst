.. _rsi:

RSI configuration
=================

**Reference-state initialization (RSI)** chooses *where* in a motion clip each
episode begins. wbc-mjlab uses **adaptive bin sampling**: clips are split into time
bins; bins that correlate with failure get sampled more often on later resets.

Configuration: ``RsiCfg`` (``env/mdp/sampling.py``), attached at
``cfg.commands["motion"].rsi``.

Core fields
-----------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Field
     - Role
   * - ``sampling_mode``
     - ``adaptive`` (default), ``uniform``, or ``start`` (always clip start)
   * - ``strategy``
     - ``similarity_ema`` or ``binary_failure`` (hardest first), or ``success_target``
       (SGS: target success rate)
   * - ``bin_width_s``
     - Bin duration in seconds (default **4.0** in WBC presets)
   * - ``uniform_ratio``
     - Floor mass on uniform bin sampling (exploration)
   * - ``alpha``
     - EMA rate for per-bin failure levels
   * - ``temperature_base``
     - Softmax temperature when converting failure levels → sample weights
   * - ``similarity_terms``
     - Hand-tuned exp-kernels for ``similarity_ema`` (when not reward-aligned)
   * - ``similarity_from_rewards``
     - If True, per-step similarity = weighted active ``motion_*`` reward values
   * - ``similarity_norm_by_remaining_clip``
     - Normalize similarity EMA by remaining clip length (WBC/Zest presets)
   * - ``min_bin_span_ratio``
     - Drop bins shorter than this fraction of ``bin_width`` (WBC/Zest presets)
   * - ``persist_failure_levels`` / ``failure_levels_filename``
     - Save/load ``rsi_bin_stats.npz`` across runs
   * - ``target_success``
     - ``success_target``: the bin success rate sampling aims for (0.66)
   * - ``target_concentration``
     - ``success_target``: κ, how sharply the kernel peaks at the target (5)
   * - ``target_temperature``
     - ``success_target``: T, softmax temperature over the kernel's log weights (2)
   * - ``target_epsilon``
     - ``success_target``: ε, floor inside the kernel (1e-4)

Adaptive loop
-------------

.. code-block:: text

   Episode starts at bin B of trajectory T
        ↓
   Rollout until termination or timeout
        ↓
   Compute failure signal (strategy-dependent)
        ↓
   EMA-update bin_failure_levels[T, B]
        ↓
   Next reset: sample (T', B') ∝ softmax(logits) + uniform_ratio
               logits = failure / temperature          (similarity_ema, binary_failure)
               logits = log w(1 − failure) / T          (success_target)

Implementation: ``sample_adaptive_bins``, ``update_failure_ema`` in
``sampling.py``; bin updates from ``MotionCommand`` during rollouts.

Failure strategies
------------------

``similarity_ema`` (ZEST-style)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Failure = **1 − mean per-step tracking similarity** over the episode.

Two similarity sources:

1. **Reward-aligned** (``similarity_from_rewards=True``) — used by ``apply_wbc`` /
   ``apply_zest``. Per-step similarity matches weighted ``motion_*`` **reward**
   terms, so RSI optimizes the same objective as training.
2. **Hand-tuned** — ``similarity_terms`` exp-kernels on joint / anchor / body errors
   (see ``keybody_similarity_preset()``).

WBC/Zest presets also set ``similarity_norm_by_remaining_clip=True``,
``min_bin_span_ratio=0.5``, ``persist_failure_levels=True``.

``binary_failure`` (BeyondMimic-style)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Failure = **1** if the episode **terminated** before its clip ended, else **0** (timeout or
clip end); every episode updates its start bin, so a bin's level is its recent failure rate.
Used by ``apply_binary_failure`` with ``similarity_terms = keybody_similarity_preset()``.

``success_target`` (SGS)
~~~~~~~~~~~~~~~~~~~~~~~~

Success-guided sampling (`arXiv:2610.12465 <https://arxiv.org/abs/2610.12465>`_). Every
episode updates its start bin: failure **1** if it terminated before its clip ended, **0**
if it timed out or reached the clip end. ``1 − failure`` is the bin's success rate *p*;
bins are sampled with

.. code-block:: text

   w = (p + ε)^(κ·t) · (1 − p + ε)^(κ·(1 − t)),   P(bin) ∝ exp(log w / T)

mixed with ``uniform_ratio``. Unvisited bins start at *p = t*, the kernel's peak, so each
gets tried early. ``apply_sgs`` (``presets/sgs.py``) sets the strategy with
``alpha=0.01`` — about the paper's window of the last 100 episodes per bin — and the paper's
locomotion values *t* = 0.66, κ = 5, T = 2. Its manipulation setting, *t* = 0.5, κ = 1,
is broader. The paper's sweeps over *t*, κ, T and the window change results only mildly.

The assistive wrench keeps working: it reads the same failure levels
(``1 − success rate`` here).

Hand-tuned similarity presets
-----------------------------

.. list-table::
   :header-rows: 1

   * - Preset function
     - Terms
   * - ``joint_pos_similarity_preset()``
     - joint position only (base template default)
   * - ``keybody_similarity_preset()``
     - joint + anchor pos/ori + body pos/ori

Per-step kernel: ``exp(−error² / σ²)`` with defaults in ``DEFAULT_SIMILARITY_STDS``.

Persistence
-----------

When ``persist_failure_levels=True``, training writes ``params/rsi_bin_stats.npz`` with
every checkpoint (``save_rsi_bin_stats``) and ``--agent.resume`` restores it
(``load_rsi_bin_stats``). ``wbc-mjlab-play`` only reads it — it never writes the file, so
playing a run on another dataset can't reset its curriculum.

Inspect bins live in Viser during play or plot offline with ``wbc-mjlab-plot-rsi-bins``
— see :doc:`../visualization`.

API: :doc:`../api/mdp` (RSI / adaptive sampling).

.. seealso::

   - :doc:`motion_command` — resampling hook
   - :doc:`../tasks/index` — which tasks use which RSI strategy
   - :doc:`../research` — paper citations (ZEST, BeyondMimic, SGS)
