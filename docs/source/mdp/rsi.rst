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
     - Failure signal: ``similarity_ema``, ``binary_failure`` or ``discounted_return``
   * - ``bin_width_s``
     - Bin duration in seconds (default **4.0** in WBC presets)
   * - ``uniform_ratio``
     - Floor mass on uniform bin sampling (exploration)
   * - ``alpha``
     - EMA rate for per-bin failure levels
   * - ``temperature_base`` · ``temperature_scale_by_bins``
     - Softmax temperature when converting failure levels → sample weights; divided by
       ``log(1 + bins)`` unless ``temperature_scale_by_bins=False``
   * - ``memory_ratio`` · ``memory_alpha``
     - Share of starts drawn in proportion to each bin's running-average softmax weight
       (and that average's rate), so a hard bin keeps some practice once learned
   * - ``return_gamma``
     - Discount of the episode return (``discounted_return``)
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
   Next reset: sample (T', B') ∝ uniform_ratio · uniform
                                + memory_ratio · memory
                                + the rest · softmax(failure / temperature)

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

Failure = **1** if the episode **terminated early** (before timeout), else **0**.
Used by ``apply_binary_failure`` with ``similarity_terms = keybody_similarity_preset()``.

``discounted_return`` (SuperMimic-style)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Failure = **1 − G / G_max**: ``G`` the discounted return earned from the start bin,
``G_max`` the most it could have earned (every positive reward term at its best until the
clip's end). With ``temperature_base = 1/25`` (unscaled), ``softmax(failure / T)`` is the
paper's ``softmax(−25 G / G_max)``. Used by ``apply_lookahead`` with 1 s bins, 50 %
uniform and 10 % memory.

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

When ``persist_failure_levels=True``, training can write/read
``rsi_bin_stats.npz`` via ``save_rsi_bin_stats`` / ``load_rsi_bin_stats``.

Inspect bins live in Viser during play or plot offline with ``wbc-mjlab-plot-rsi-bins``
— see :doc:`../visualization`.

API: :doc:`../api/mdp` (RSI / adaptive sampling).

.. seealso::

   - :doc:`motion_command` — resampling hook
   - :doc:`../tasks/index` — which tasks use which RSI strategy
   - :doc:`../research` — paper citations (ZEST, BeyondMimic, SuperMimic)
