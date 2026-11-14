Wbc-G1-SGS — success-guided sampling
====================================

Task id: ``Wbc-G1-SGS`` · In-tree reference on robot ``g1``.

**``Wbc-G1`` with success-guided sampling (SGS)** — the same rewards, observations,
terminations and assistive wrench; only *where episodes start* changes. Instead of the
hardest clip bins, adaptive sampling favours bins the policy survives about two thirds of
the time, so bins it can't do yet (or ever: retargeting glitches) stop eating starts.

- **Preset:** ``apply_wbc`` + ``apply_sgs`` (``presets/sgs.py``)
- **Builder:** ``g1_wbc_sgs_env_cfg()``
- **Paper:** `arXiv:2610.12465 <https://arxiv.org/abs/2610.12465>`_

What this task changes
----------------------

RSI only:

.. list-table::
   :header-rows: 1
   :widths: 34 33 33

   * - Field
     - ``Wbc-G1``
     - ``Wbc-G1-SGS``
   * - ``strategy``
     - ``similarity_ema`` (hardest first)
     - ``success_target``
   * - Bin signal
     - 1 − mean reward-aligned similarity
     - 1 if the episode terminated before its clip ended
   * - ``alpha``
     - 0.005
     - 0.01 (≈ last 100 episodes per bin)
   * - Kernel
     - softmax(failure / T_eff)
     - target success *t* = 0.66, κ = 5, T = 2

How the kernel works: :doc:`../concepts/adaptive_sampling`; every field:
:doc:`../mdp/rsi`.

Train & play
------------

.. code-block:: bash

   uv run wbc-mjlab-train --task Wbc-G1-SGS --dataset lafan
   uv run wbc-mjlab-play --task Wbc-G1-SGS --dataset lafan --viewer viser

Logs: ``logs/rsl_rl/wbc_g1_sgs/<run>/``

On other tasks and robots, turn on the ``sgs`` module in a Studio preset
(:doc:`../studio`), or call ``apply_sgs(cfg)`` after a task's presets in code.

Comparing with ``Wbc-G1``
-------------------------

Train both on the same dataset, envs and seeds, then compare tracking evaluated over the
whole library (uniform starts), not the training reward — the two tasks start episodes in
different places, so their training curves aren't directly comparable. The Studio's
*Sampling* chart shows how concentrated each sampler is (``sampling_entropy``,
``sampling_top1_prob``).

Tips
----

- Lower ``rsi.target_success`` (for example 0.5) pushes harder; ``target_concentration=1``
  gives a much broader kernel (the paper's manipulation setting).
- Gains are expected mostly on large, mixed libraries (LAFAN and up); on a handful of clips
  every bin is learnable and the two samplers behave alike.
