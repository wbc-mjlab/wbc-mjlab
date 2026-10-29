.. _api_training:

Training algorithms
===================

``--algo ppo|wm`` — :doc:`../concepts/world_model`.

.. automodule:: wbc_mjlab.rl.algos
   :no-members:

.. autodata:: wbc_mjlab.rl.algos.ALGORITHMS
   :no-value:

.. autoclass:: wbc_mjlab.rl.algos.AlgoInfo
   :members:

.. autofunction:: wbc_mjlab.rl.algos.rl_cfg

.. autofunction:: wbc_mjlab.rl.algos.runner_cfg

.. autofunction:: wbc_mjlab.rl.algos.env_cfg

.. autofunction:: wbc_mjlab.rl.algos.check_env

.. autofunction:: wbc_mjlab.rl.algos.run_algo

.. autofunction:: wbc_mjlab.rl.algos.run_rl_cfg

Recipes (Studio)
----------------

.. automodule:: wbc_mjlab.studio.recipes
   :no-members:

.. autofunction:: wbc_mjlab.studio.recipes.fields

.. autofunction:: wbc_mjlab.studio.recipes.list_recipes

.. autofunction:: wbc_mjlab.studio.recipes.save

.. autofunction:: wbc_mjlab.studio.recipes.architecture_of

Shared policy, saved runs and evaluation
----------------------------------------

.. autoclass:: wbc_mjlab.rl.policy.PolicyCfg
   :members:
   :undoc-members:

.. autofunction:: wbc_mjlab.run_params.load_run_config

.. autofunction:: wbc_mjlab.env.mdp.sampling.sampling_state

.. autofunction:: wbc_mjlab.env.mdp.sampling.restore_sampling_state

.. autoclass:: wbc_mjlab.rl.evaluation.EvaluationConfig
   :members:

.. autofunction:: wbc_mjlab.rl.evaluation.evaluate_checkpoint

.. autofunction:: wbc_mjlab.rl.evaluation.deterministic_cfg

.. autofunction:: wbc_mjlab.rl.evaluation.run_benchmark

World model (``wm``)
--------------------

.. autoclass:: wbc_mjlab.rl.wm.config.WmAlgorithmCfg
   :members:
   :undoc-members:

.. autofunction:: wbc_mjlab.rl.wm.config.wm_env_cfg

.. autoclass:: wbc_mjlab.rl.wm.algorithm.WorldModelRL

.. autoclass:: wbc_mjlab.rl.wm.imagined.Imagination
   :members: actor_obs, analytic_reward, tracking_features, joint_setpoint, context

.. autoclass:: wbc_mjlab.rl.wm.model.WorldModel
   :members: step, step_heads

.. autoclass:: wbc_mjlab.rl.wm.kinematics.RobotKinematics
   :members: bodies
