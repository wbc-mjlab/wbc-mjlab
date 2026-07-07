.. _api_extension:

Extension API
=============

Public entry points for external robot packages. Register a robot and its tasks
once at import time; train / play / data CLIs then discover them like in-tree G1.

How-to guides: :doc:`../extensions/index`. Reference extension:
`wbc-mjlab-extension-h2 <https://github.com/wbc-mjlab/wbc-mjlab-extension-h2>`_.

Minimal registration
--------------------

Call once from your package entry point (``mjlab.tasks``):

.. code-block:: python

   from wbc_mjlab.extension import WbcRobotSpec, register_wbc_extension
   from wbc_mjlab.motion.robot_assets import RobotMotionSpec
   from wbc_mjlab.tasks.config import WbcTaskConfig

   register_wbc_extension(
     WbcRobotSpec(
       robot_id="my_robot",
       make_env_cfg=make_my_robot_wbc_env_cfg,
       make_rl_cfg=my_robot_wbc_rl_cfg,
       project_root=PROJECT_ROOT,
       motion_spec=RobotMotionSpec(
         scene_cfg_fn=lambda: make_my_robot_wbc_env_cfg().scene,
         # foot_body_names=..., foot_sole_z=..., mjcf_path=...
       ),
     ),
     WbcTaskConfig(
       task_id="Wbc-MyRobot",
       robot_id="my_robot",
       description="Default WBC stack",
       experiment_name="wbc_my_robot",
       build_env_cfg=make_my_robot_wbc_env_cfg,
     ),
   )

Then install the extension editable alongside wbc-mjlab (see :doc:`../installation`)
and confirm ``wbc-mjlab-list-envs`` lists ``Wbc-MyRobot``. Optional
``symmetry_config`` on ``WbcRobotSpec`` enables ``wbc-mjlab-data-to-npz --mirror``
(see :doc:`../data` and :doc:`../extensions/robot_entity`). Full walkthrough:
:doc:`../extensions/example_extension`.

Registration
------------

.. automodule:: wbc_mjlab.extension
   :members:
   :undoc-members: False
   :member-order: bysource

Task config
-----------

.. autoclass:: wbc_mjlab.tasks.config.WbcTaskConfig
   :members:
   :undoc-members: False

Motion conversion metadata
--------------------------

Used by ``wbc-mjlab-data-to-npz --robot <id>`` when the extension provides FK /
debias metadata.

.. autoclass:: wbc_mjlab.motion.robot_assets.RobotMotionSpec
   :members:
   :undoc-members: False
