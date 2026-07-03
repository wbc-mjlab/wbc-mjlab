.. _visualization:

Visualization (Viser)
=====================

wbc-mjlab extends mjlab's **Viser** stack with WBC-specific overlays for motion
tracking: reference alignment, tracking-error coloring, and adaptive RSI bin panels.

Code lives under ``src/wbc_mjlab/viewer/`` and hooks into ``MotionCommand`` in
``env/mdp/commands.py``.

Commands
--------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Command
     - Purpose
   * - ``wbc-mjlab-play --viewer viser``
     - Policy eval in the browser with WBC GUI overlays (tracking tasks)
   * - ``wbc-mjlab-data-vis --robot <id> --dataset <name>``
     - Preview motion NPZ clips (no policy)
   * - ``wbc-mjlab-plot-rsi-bins <path/to/rsi_bin_stats.npz>``
     - Offline matplotlib plot of saved RSI failure levels

Play viewer
-----------

For WBC tracking tasks, ``wbc-mjlab-play`` selects ``WbcViserPlayViewer``
(``viewer/viser_play.py``) instead of mjlab's default ``ViserPlayViewer``.

Each frame it calls ``MotionCommand.update_viewer_gui`` so the sidebar stays in sync
with the selected parallel env: clip name, frame, RSI mode/strategy, and bin heatmaps.

.. code-block:: bash

   uv run wbc-mjlab-play --task <TaskId> --dataset samples --viewer viser

Native MuJoCo viewer (``--viewer native``) is also available when a display is present;
Viser is the default when headless.

Reference visualization
-----------------------

``MotionCommand`` debug vis supports two modes (``MotionCommandCfg.viz.mode``):

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Mode
     - What you see
   * - ``ghost``
     - Semi-transparent duplicate mesh at the **reference** root + joint pose
   * - ``frames``
     - Per-keybody coordinate frames: **desired** (reference) vs **current** (robot)

**Align xy/yaw to reference** (Viser checkbox, default off) switches the reference
display to **anchor-relative** poses (``body_pos_relative_w`` /
``body_quat_relative_w``): horizontal translation and yaw match the robot anchor so
ghost/frames overlay in a common frame. Turn off to show world-frame reference motion.

**Color bodies by tracking error** tints keybody markers green → red using
``viewer/motion_vis.error_to_rgba`` (combined pos + rot error).

When playback is **paused**, the motion-command panel exposes a **Frame** scrubber and
**Start Here** button to reset the selected env(s) to a reference frame — useful for
probing hard RSI bins without retraining.

RSI visualization
-----------------

The **Adaptive sampling (RSI)** folder in the Viser sidebar shows per-clip bin strips
(rendered by ``viewer/motion_vis.format_rsi_panel_html``):

.. list-table::
   :header-rows: 1
   :widths: 16 84

   * - Row
     - Meaning
   * - **Failure**
     - EMA failure level per bin (green = easy, red = hard)
   * - **Sample p**
     - Renormalized sampling weight within the active clip
   * - **Visits**
     - How often each bin was used as an episode start
   * - **Assist β**
     - Assistive wrench gain from bin failure (when enabled)

The **white border** marks the bin for the current playback frame. Colors are scaled
**relative within the clip** so short and long motions remain readable.

Failure signal label reflects ``RsiCfg.strategy`` (e.g. reward-aligned
``similarity_ema`` vs ``binary_failure``) — see :doc:`mdp/rsi`.

During training, presets with ``persist_failure_levels=True`` write
``params/rsi_bin_stats.npz``; play can load these so bins reflect trained curriculum
state. Plot offline:

.. code-block:: bash

   uv run wbc-mjlab-plot-rsi-bins logs/rsl_rl/<experiment>/<run>/params/rsi_bin_stats.npz

Motion NPZ preview
------------------

``wbc-mjlab-data-vis`` (``scripts/vis_data.py``) uses mjlab's ``MjlabViserScene`` with
a timeline GUI to scrub converted clips before training — same Viser stack, no policy.

Module map
----------

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Module
     - Role
   * - ``viewer/viser_play.py``
     - ``WbcViserPlayViewer`` — syncs motion-command GUI each frame
   * - ``viewer/motion_vis.py``
     - HTML panels, RSI color scales, tracking-error RGBA
   * - ``env/mdp/commands.py``
     - Ghost/frames debug vis, Viser GUI (align, color, scrubber, RSI panel)
   * - ``scripts/vis_data.py``
     - Standalone NPZ playback in Viser
   * - ``scripts/plot_rsi_bins.py``
     - Matplotlib RSI export

Related: :doc:`mdp/motion_command`, :doc:`mdp/rsi`, :doc:`workflows/demo`, :doc:`usage`.
