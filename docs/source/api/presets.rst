.. _api_presets:

Presets
=======

Paper / deploy stacks as in-place env mutators. Task builders call one or more
``apply_*`` functions on the base template — no MDP forks.

Design: :doc:`../concepts/presets_and_tasks`. Task catalog: :doc:`../tasks/index`.

.. autofunction:: wbc_mjlab.presets.apply_wbc
.. autofunction:: wbc_mjlab.presets.apply_zest
.. autofunction:: wbc_mjlab.presets.apply_end_effector
.. autofunction:: wbc_mjlab.presets.apply_se_actor
.. autofunction:: wbc_mjlab.presets.apply_binary_failure
