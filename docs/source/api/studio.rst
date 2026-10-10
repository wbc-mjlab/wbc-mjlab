.. _api_studio:

Studio presets and robots
=========================

The Studio's files are plain YAML handled by two modules you can use from scripts too.
Guides: :doc:`../studio` and :doc:`../extensions/custom_robots`.

.. _api-studio-presets:

Presets and modules
-------------------

YAML diffs on top of a registered task (``~/.config/wbc-mjlab/presets``, ``…/modules``),
registered as ``<base>--<name>``.

.. autofunction:: wbc_mjlab.studio.presets.build
.. autofunction:: wbc_mjlab.studio.presets.apply
.. autofunction:: wbc_mjlab.studio.presets.register
.. autofunction:: wbc_mjlab.studio.presets.resolve
.. autofunction:: wbc_mjlab.studio.presets.save
.. autofunction:: wbc_mjlab.studio.presets.save_module
.. autofunction:: wbc_mjlab.studio.presets.list_modules
.. autofunction:: wbc_mjlab.studio.presets.describe
.. autofunction:: wbc_mjlab.studio.presets.library
.. autofunction:: wbc_mjlab.studio.presets.observation_diff
.. autofunction:: wbc_mjlab.studio.presets.deployable
.. autofunction:: wbc_mjlab.user_config.config_dir

.. _api-custom-robots:

Robots without code
-------------------

.. automodule:: wbc_mjlab.robots.custom
   :no-members:

.. autoclass:: wbc_mjlab.robots.custom.CustomRobotSpec
   :members: folder, source, prepared, task_id
.. autoclass:: wbc_mjlab.robots.custom.JointGains
.. autofunction:: wbc_mjlab.robots.custom.prepare
.. autofunction:: wbc_mjlab.robots.custom.inspect_model
.. autofunction:: wbc_mjlab.robots.custom.suggest
.. autofunction:: wbc_mjlab.robots.custom.settle_test
.. autofunction:: wbc_mjlab.robots.custom.save_spec
.. autofunction:: wbc_mjlab.robots.custom.load_spec
.. autofunction:: wbc_mjlab.robots.custom.env_cfg
.. autofunction:: wbc_mjlab.robots.custom.register
.. autofunction:: wbc_mjlab.robots.custom.register_all
.. autofunction:: wbc_mjlab.robots.custom.task_id_for
