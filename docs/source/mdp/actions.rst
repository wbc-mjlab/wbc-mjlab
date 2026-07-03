Actions
=======

Manager: **ActionManager** · Module: ``env/mdp/actions.py``

Default action
--------------

``ReferenceJointPositionActionCfg`` — the policy outputs **residual joint deltas**
Δq that are **added to reference joint positions** from the motion command:

.. code-block:: text

   q_target = q_ref(motion command) + scale · action

This keeps the action space centered on the current clip frame rather than absolute
joint targets.

Per-robot wiring
----------------

Each robot entity sets ``actions["joint_pos"].scale`` to a per-DoF or grouped scale
tensor (e.g. ``<ROBOT>_ACTION_SCALE`` in the robot package).

Alternative
-----------

``make_base_wbc_env_cfg(use_reference_residual_action=False)`` switches to mjlab's
absolute ``JointPositionActionCfg`` with default joint offsets.

Related: :doc:`motion_command`, :doc:`../extensions/robot_entity`.
