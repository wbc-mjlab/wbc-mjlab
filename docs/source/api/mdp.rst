.. _api_mdp:

MDP terms
=========

Autodoc for shared manager terms under ``wbc_mjlab.env.mdp``. Design rules and
term catalogs: :doc:`../mdp/index` (Shared MDP).

**Jump to:** `Actions`_ · `Motion command`_ · `RSI / adaptive sampling`_ ·
`Observations`_ · `Rewards`_ · `Terminations`_ · `Events (assistive wrench)`_

Actions
-------

.. autoclass:: wbc_mjlab.env.mdp.actions.ReferenceJointPositionActionCfg
   :members:
   :undoc-members: False

.. autoclass:: wbc_mjlab.env.mdp.actions.ReferenceJointPositionAction
   :members:
   :undoc-members: False

.. autoclass:: wbc_mjlab.env.mdp.actions.DefaultOffsetJointPositionActionCfg
   :members:
   :undoc-members: False

.. autoclass:: wbc_mjlab.env.mdp.actions.DefaultOffsetJointPositionAction
   :members:
   :undoc-members: False

Motion command
--------------

.. autoclass:: wbc_mjlab.env.mdp.commands.MotionLoader
   :members:
   :undoc-members: False

.. autoclass:: wbc_mjlab.env.mdp.commands.MotionCommand
   :members:
   :undoc-members: False

.. autoclass:: wbc_mjlab.env.mdp.commands.MotionCommandCfg
   :members:
   :undoc-members: False

RSI / adaptive sampling
-----------------------

.. autoclass:: wbc_mjlab.env.mdp.sampling.RsiCfg
   :members:
   :undoc-members: False

.. autoclass:: wbc_mjlab.env.mdp.sampling.AdaptiveSimilarityTermCfg
   :members:
   :undoc-members: False

.. autoclass:: wbc_mjlab.env.mdp.sampling.TrackingSimilarityState
   :members:
   :undoc-members: False

.. autofunction:: wbc_mjlab.env.mdp.sampling.joint_pos_similarity_preset
.. autofunction:: wbc_mjlab.env.mdp.sampling.keybody_similarity_preset
.. autofunction:: wbc_mjlab.env.mdp.sampling.sample_adaptive_bins
.. autofunction:: wbc_mjlab.env.mdp.sampling.save_rsi_bin_stats
.. autofunction:: wbc_mjlab.env.mdp.sampling.load_rsi_bin_stats

Observations
------------

.. autofunction:: wbc_mjlab.env.mdp.observations.ref_base_height
.. autofunction:: wbc_mjlab.env.mdp.observations.ref_anchor_pos_w
.. autofunction:: wbc_mjlab.env.mdp.observations.ref_base_lin_vel_b
.. autofunction:: wbc_mjlab.env.mdp.observations.ref_base_ang_vel_b
.. autofunction:: wbc_mjlab.env.mdp.observations.ref_gravity_b
.. autofunction:: wbc_mjlab.env.mdp.observations.ref_joint_pos
.. autofunction:: wbc_mjlab.env.mdp.observations.ref_joint_vel
.. autofunction:: wbc_mjlab.env.mdp.observations.ref_anchor_ori_6d
.. autofunction:: wbc_mjlab.env.mdp.observations.motion_anchor_pos_b
.. autofunction:: wbc_mjlab.env.mdp.observations.motion_anchor_ori_b
.. autofunction:: wbc_mjlab.env.mdp.observations.motion_anchor_pos_error_w
.. autofunction:: wbc_mjlab.env.mdp.observations.motion_anchor_ori_error
.. autofunction:: wbc_mjlab.env.mdp.observations.robot_body_pos_b
.. autofunction:: wbc_mjlab.env.mdp.observations.robot_body_ori_b
.. autofunction:: wbc_mjlab.env.mdp.observations.ref_body_pos_b
.. autofunction:: wbc_mjlab.env.mdp.observations.ref_body_ori_b
.. autofunction:: wbc_mjlab.env.mdp.observations.motion_segment_phase
.. autofunction:: wbc_mjlab.env.mdp.observations.motion_tracking_step_rewards

Rewards
-------

Tracking kernels and regularizers. Shared optional params are documented on the
module docstring of ``wbc_mjlab.env.mdp.rewards``.

.. autofunction:: wbc_mjlab.env.mdp.rewards.tracking_std_from_sigma
.. autofunction:: wbc_mjlab.env.mdp.rewards.motion_global_anchor_position_error_exp
.. autofunction:: wbc_mjlab.env.mdp.rewards.motion_global_anchor_orientation_error_exp
.. autofunction:: wbc_mjlab.env.mdp.rewards.motion_anchor_linear_velocity_error_exp
.. autofunction:: wbc_mjlab.env.mdp.rewards.motion_anchor_angular_velocity_error_exp
.. autofunction:: wbc_mjlab.env.mdp.rewards.motion_joint_position_error_exp
.. autofunction:: wbc_mjlab.env.mdp.rewards.motion_joint_velocity_error_exp
.. autofunction:: wbc_mjlab.env.mdp.rewards.motion_relative_body_position_error_exp
.. autofunction:: wbc_mjlab.env.mdp.rewards.motion_relative_body_orientation_error_exp
.. autofunction:: wbc_mjlab.env.mdp.rewards.motion_global_body_linear_velocity_error_exp
.. autofunction:: wbc_mjlab.env.mdp.rewards.motion_global_body_angular_velocity_error_exp
.. autofunction:: wbc_mjlab.env.mdp.rewards.action_rate_l1
.. autofunction:: wbc_mjlab.env.mdp.rewards.actuator_torque_soft_limit
.. autofunction:: wbc_mjlab.env.mdp.rewards.feet_slip
.. autofunction:: wbc_mjlab.env.mdp.rewards.angular_momentum_penalty

Terminations
------------

.. autofunction:: wbc_mjlab.env.mdp.terminations.bad_anchor_pos
.. autofunction:: wbc_mjlab.env.mdp.terminations.bad_anchor_pos_z_only
.. autofunction:: wbc_mjlab.env.mdp.terminations.bad_anchor_ori
.. autofunction:: wbc_mjlab.env.mdp.terminations.bad_motion_body_pos
.. autofunction:: wbc_mjlab.env.mdp.terminations.bad_motion_body_pos_z_only
.. autofunction:: wbc_mjlab.env.mdp.terminations.excessive_contact_force
.. autofunction:: wbc_mjlab.env.mdp.terminations.excessive_keybody_ground_contact_force

Events (assistive wrench)
-------------------------

.. autoclass:: wbc_mjlab.env.mdp.assistive_wrench.AssistiveWrenchEvent
   :members:
   :undoc-members: False

.. autofunction:: wbc_mjlab.env.mdp.assistive_wrench.assistive_wrench_force
.. autofunction:: wbc_mjlab.env.mdp.assistive_wrench.assistive_wrench_torque
.. autofunction:: wbc_mjlab.env.mdp.assistive_wrench.assistive_wrench_gain
