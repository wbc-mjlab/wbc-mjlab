.. _api_registry:

Env and task registry
=====================

Builders and registration helpers used by CLIs and extension packages.

Env template and robot builders
-------------------------------

.. autofunction:: wbc_mjlab.env.wbc_env_cfg.make_base_wbc_env_cfg

.. autofunction:: wbc_mjlab.robots.env.register_robot_builders
.. autofunction:: wbc_mjlab.robots.env.make_wbc_env_cfg
.. autofunction:: wbc_mjlab.robots.env.make_wbc_rl_cfg

Task table
----------

.. autofunction:: wbc_mjlab.tasks.get_task_config
.. autofunction:: wbc_mjlab.tasks.list_wbc_task_ids
.. autofunction:: wbc_mjlab.tasks.all_wbc_tasks
.. autofunction:: wbc_mjlab.tasks.register_wbc_task
.. autofunction:: wbc_mjlab.tasks.register_wbc_tasks
.. autofunction:: wbc_mjlab.tasks.register_all_wbc_tasks
.. autofunction:: wbc_mjlab.tasks.resolve_task_id
.. autofunction:: wbc_mjlab.tasks.prepare_wbc_run

One-shot robot + tasks: :func:`wbc_mjlab.extension.register_wbc_extension`
(:doc:`extension`).
