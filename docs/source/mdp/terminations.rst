Terminations
============

Manager: **TerminationManager** · Module: ``env/mdp/terminations.py``

Terms
-----

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Term
     - Condition
   * - ``time_out``
     - Episode horizon exceeded
   * - ``anchor_pos`` / ``anchor_ori``
     - Anchor body drift from reference (thresholds set in preset / robot cfg)
   * - ``ee_body_pos``
     - z-only error on end-effector bodies (body tuple from preset)
   * - ``keybody_ground_contact_force``
     - Catastrophic ground contact on motion keybodies

Presets enable thresholds and body sets; the callables are shared. For example,
``apply_wbc`` adds EE height termination; ``apply_zest`` removes it.

Related: :doc:`rewards`, :doc:`../extensions/robot_entity`.
