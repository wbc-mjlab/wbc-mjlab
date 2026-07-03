Presets and tasks
=================

Two names, two roles:

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Idea
     - Meaning
   * - **Preset**
     - Reusable **recipe** — ``apply_*(cfg)`` mutates an existing env cfg in place
   * - **Task**
     - **Registered CLI name** for one robot + preset stack (``--task Wbc-…``)

Presets are **not** separate environments. They reweight ``motion_*`` rewards, swap
``RsiCfg`` strategy, drop observation terms, or toggle termination thresholds on slots
already defined in ``make_base_wbc_env_cfg()``.

Tasks add **registration**: ``task_id``, ``robot_id``, ``experiment_name``, and a
``build_env_cfg`` callable that wires robot + preset(s).

.. code-block:: text

   preset  = apply_zest(cfg, reward_body_names=..., ...)
   task    = WbcTaskConfig(task_id="Wbc-…", build_env_cfg=builder_that_calls_preset)

Same preset, many robots
------------------------

``apply_wbc`` is robot-agnostic — the task builder passes **body name tuples** from
the robot entity. The in-tree ``g1`` reference and extension robots (e.g. H2) can share
the same preset with different constants.

Same robot, many tasks
----------------------

One robot entity can expose multiple tasks by stacking different presets:

- ``apply_wbc`` → default WBC
- ``apply_zest`` → paper reproduction
- ``apply_wbc`` + ``apply_se_actor`` → state-estimation actor obs

Compose presets in order: base cfg → primary preset → optional obs/IMU tweaks.

Neutral code, cited methods
---------------------------

Presets use generic RSI names (``similarity_ema``, ``binary_failure``). Paper labels
(ZEST, BeyondMimic, …) live in docs and task descriptions — see :doc:`../research`.

Reference
---------

Preset list, ``apply_zest`` walkthrough, in-tree task catalog, and comparison tables:
:doc:`../tasks/index`.
