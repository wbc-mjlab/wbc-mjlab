Motion command
==============

``MotionCommand`` (``env/mdp/commands.py``) is the central command term. It loads a
**multi-clip motion bundle** (NPZ), plays reference kinematics each step, and drives
RSI resampling at episode boundaries.

Configuration
-------------

``MotionCommandCfg`` fields (set per robot entity in ``<robot>_base_cfg``):

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Field
     - Meaning
   * - ``motion_file``
     - Path to converted NPZ (from ``wbc-mjlab-data-to-npz``)
   * - ``anchor_body_name``
     - Body used for anchor-frame errors and assistive wrench (set in robot entity)
   * - ``body_names``
     - Keybodies tracked in rewards / RSI / critic obs
   * - ``actuated_joint_names``
     - Optional subset for joint tracking metrics and RSI
   * - ``pose_range`` / ``velocity_range`` / ``joint_position_range``
     - DR on reference root pose and joint offsets at resample
   * - ``rsi``
     - :doc:`rsi` settings (``RsiCfg``)
   * - ``assistive_wrench_enabled`` / ``assistive_beta_max`` / ``assistive_eta``
     - Bin-conditioned assist on the anchor (Zest-style curriculum)

Runtime behavior
----------------

Each environment step:

1. **Advance** ``time_steps`` along the active clip segment.
2. **Expose** reference body/joint states to reward and observation terms via
   ``command_name="motion"`` params.
3. **Apply assistive wrench** (when enabled) scaled by the current bin's failure level.

At **episode reset** (``_resample_command``):

1. Pick **(trajectory, bin, frame)** from RSI policy (adaptive / uniform / start).
2. Write reference root + joint state into sim (with pose/joint DR).
3. Initialize episode RSI bookkeeping (start bin, similarity denominator, assist gain).

Multi-clip playback
-------------------

``MotionLoader`` indexes a bundle with multiple segments. Each parallel env can be on
a different clip. ``resampling_time_range`` in the base template is effectively
infinite — episodes end via terminations, not command timeout.

Reference vs robot state
------------------------

Tracking rewards compare **robot** bodies/joints to **reference** values from the
command. The action space adds **residual joint deltas** on top of reference joint
positions (``ReferenceJointPositionActionCfg``), so the policy tracks the command
rather than absolute pose targets.

Assistive wrench
----------------

When ``assistive_wrench_enabled=True``, ``compute_assist_gain_matrix`` maps bin
failure levels to a gain β ∈ [0, ``assistive_beta_max``]. Hard bins get more assist
during early training. Play mode disables assist (see :doc:`../architecture`).

Visualization
-------------

``MotionCommandCfg.viz`` selects ghost vs frame debug (see :doc:`../visualization`).
WBC extends mjlab's Viser play viewer with **reference alignment** (anchor-relative
xy/yaw overlay), **tracking-error coloring**, and **RSI bin panels** during
``wbc-mjlab-play --viewer viser``.

Related
-------

- :doc:`rsi` — how start frames are sampled
- :doc:`observations`, :doc:`rewards` — terms that consume the command
- :doc:`../data` — motion NPZ format
