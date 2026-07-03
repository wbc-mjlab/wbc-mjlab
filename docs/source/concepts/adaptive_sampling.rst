Adaptive sampling (RSI)
=======================

**Reference-state initialization (RSI)** decides *where in a motion clip* each
training episode begins. wbc-mjlab uses **adaptive sampling**: the clip timeline is
split into bins; bins that correlate with **failure** are sampled more often on later
resets, so the policy spends more time on hard segments.

Why it matters
--------------

Whole-body tracking on **motion libraries** spans many skills and contact phases.
Uniform random starts waste steps on easy frames; adaptive RSI concentrates training
on regions where the policy falls or terminates early.

At a high level:

.. code-block:: text

   Episode starts at bin B of clip T
        ↓
   Rollout until termination or horizon
        ↓
   Measure failure signal (strategy-specific)
        ↓
   Update bin failure EMA for (T, B)
        ↓
   Next reset: sample (T', B') ∝ failure weight + exploration floor

Failure signals (strategies)
----------------------------

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Strategy
     - Failure means…
   * - ``similarity_ema``
     - Low **tracking similarity** over the episode (reward-aligned in WBC/Zest presets)
   * - ``binary_failure``
     - Episode **ended early** before timeout (BeyondMimic-style preset)

Presets choose the strategy and whether similarity follows **hand-tuned kernels** or
the **same ``motion_*`` rewards** as training (``similarity_from_rewards=True``).

Connection to training
----------------------

- **Assistive wrench** — bin failure levels can scale curriculum assist on the anchor
  body (harder bins → more help early in training).
- **Persistence** — WBC/Zest presets can save ``rsi_bin_stats.npz``; **resume**
  (``--agent.resume True``) restores bin EMA so curriculum continues across runs.
- **Visualization** — Viser play shows per-clip bin strips during eval — see
  :doc:`../visualization`.

Multi-clip training
-------------------

Each parallel env can be on a different clip. RSI maintains **per-(trajectory, bin)**
statistics, so adaptive sampling works across an entire library, not one motion file.

Implementation reference
------------------------

``RsiCfg`` fields, motion-command hook, and kernel details: :doc:`../mdp/rsi`.
Motion playback and resampling: :doc:`../mdp/motion_command`.
