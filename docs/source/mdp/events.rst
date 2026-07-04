Events
======

Manager: **EventManager** · Modules: ``env/mdp/assistive_wrench.py``, mjlab DR helpers

Domain randomization and curriculum events in the base WBC template:

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Event
     - Effect
   * - ``assistive_wrench``
     - External wrench on anchor body; gain β from RSI bin failure levels
   * - ``push_robot``
     - Impulsive base velocity perturbation
   * - ``randomize_rigid_body_com``
     - COM offset on selected bodies
   * - ``randomize_foot_friction``
     - Foot friction scaling
   * - ``randomize_encoder_bias``
     - Joint encoder bias

Play mode (``play=True`` on env builders) typically disables push, motion pose/velocity
DR, and assistive wrench — same pattern for every registered robot.

API: :doc:`../api/mdp` (Events / assistive wrench).

Related: :doc:`rsi`, :doc:`motion_command`, :doc:`../architecture`.
