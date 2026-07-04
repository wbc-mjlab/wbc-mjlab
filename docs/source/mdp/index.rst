.. _mdp:

Shared MDP
==========

Design and term catalogs for the shared manager-based MDP (how terms compose,
presets, and robot wiring). For live signatures and docstrings, see
:doc:`../api/mdp`.

wbc-mjlab builds on mjlab's composable RL environment API: each MDP facet (actions,
observations, rewards, …) is a set of named **terms** wired on ``ManagerBasedRlEnvCfg``.
Term callables live in ``env/mdp/``; managers invoke them each step and aggregate
batched tensors.

For mjlab's upstream environment docs, see
`mjlab documentation <https://mujocolab.github.io/mjlab/main/>`_.

Design rules
------------

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Rule
     - Implication
   * - **Terms live in** ``env/mdp/``
     - Robot-agnostic; parameterized by ``command_name``, body lists, weights
   * - **Template in** ``wbc_env_cfg.py``
     - Default term slots; robots fill entity-specific params
   * - **Presets mutate cfg**
     - Change weights, thresholds, or remove terms — not Python forks
   * - **Robot entity wires names**
     - Anchor body, keybody tuple, action scale, sensors — see :doc:`../extensions/robot_entity`

Environment configuration
-------------------------

``make_base_wbc_env_cfg()`` returns a ``ManagerBasedRlEnvCfg`` with all manager
groups pre-populated. Robot builders (``<robot>_base_cfg``) attach the entity and
set motion-command body names; presets finish the recipe.

Term modules (``env/mdp/``)
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Module
     - Manager / topic
   * - ``actions.py``
     - :doc:`actions`
   * - ``commands.py``
     - :doc:`motion_command`
   * - ``observations.py``
     - :doc:`observations`
   * - ``rewards.py``
     - :doc:`rewards`
   * - ``terminations.py``
     - :doc:`terminations`
   * - ``assistive_wrench.py``, ``sampling.py``
     - :doc:`events`, :doc:`rsi`
   * - ``metrics.py``
     - Logging / evaluation helpers

Import surface: ``wbc_mjlab.env.mdp`` (re-exports mjlab MDP helpers plus wbc terms).

MDP components
--------------

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Manager
     - wbc-mjlab contents
   * - **Actions**
     - Reference-residual joint position (default)
   * - **Commands**
     - ``MotionCommandCfg`` — multi-clip playback + RSI
   * - **Observations**
     - Actor: reference + proprio. Critic: actor + privileged features
   * - **Rewards**
     - ``motion_*`` tracking exponentials + regularizers
   * - **Terminations**
     - Timeout, anchor drift, EE height, contact force
   * - **Events**
     - Assistive wrench, push, COM, friction, encoder bias

.. toctree::
   :maxdepth: 1
   :caption: Components

   actions
   motion_command
   observations
   rewards
   terminations
   events
   rsi
