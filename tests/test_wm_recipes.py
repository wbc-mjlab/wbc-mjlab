"""Learner recipes compose with motion presets without replacing their task layout."""

import copy
from dataclasses import asdict

import mjlab  # noqa: F401
import pytest
import torch
from test_wm import _env, _small_agent

from wbc_mjlab.robots.g1.tasks import G1_WBC_TASKS
from wbc_mjlab.studio.recipes import BUILTIN, check_settings

TASKS = [t.task_id for t in G1_WBC_TASKS if t.task_id != "Wbc-G1-Terrain-Scan"]
RECIPES = [name for name, recipe in BUILTIN.items() if recipe["algo"] == "wm"]


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("recipe", RECIPES)
def test_recipe_settings_and_reward_bounds(task, recipe):
  from mjlab.tasks.registry import load_env_cfg

  from wbc_mjlab.rl.evaluation import reward_upper_bound
  from wbc_mjlab.tasks import prepare_wbc_run

  prepare_wbc_run(task_id=task)
  settings = BUILTIN[recipe]["settings"]
  assert check_settings(task, "wm", settings) == settings
  cfg = load_env_cfg(task)
  assert reward_upper_bound(cfg, cfg.sim.mujoco.timestep * cfg.decimation) > 0


def test_reference_shortcut_stays_explicit_and_validated():
  from wbc_mjlab.tasks import prepare_wbc_run

  settings = BUILTIN["wm-blink"]["settings"]
  assert not settings["actor.reference_residual"]
  assert settings["algorithm.mirror_probability"] == 0
  for task in ("Wbc-G1", "Wbc-G1-EE", "Wbc-G1-Lookahead"):
    prepare_wbc_run(task_id=task)
    explicit = settings | {"actor.reference_residual": True}
    if task.endswith("Lookahead"):
      assert check_settings(task, "wm", explicit) == explicit
    else:
      with pytest.raises(ValueError, match="joint lookahead"):
        check_settings(task, "wm", explicit)


@pytest.mark.parametrize("task", TASKS)
def test_positive_regularizers_do_not_get_an_unsafe_bound(task):
  from mjlab.tasks.registry import load_env_cfg

  from wbc_mjlab.rl.evaluation import reward_upper_bound
  from wbc_mjlab.tasks import prepare_wbc_run

  prepare_wbc_run(task_id=task)
  cfg = load_env_cfg(task)
  for term in cfg.rewards.values():
    if term.weight < 0:
      weight = term.weight
      term.weight = -weight
      assert reward_upper_bound(cfg, 0.02) is None
      term.weight = weight


@pytest.fixture(params=TASKS)
def motion_env(request):
  if not torch.cuda.is_available():
    pytest.skip("needs CUDA")
  cfg, env = _env(request.param, num_envs=4, noise=False)
  yield request.param, cfg, env
  env.close()


@pytest.mark.parametrize("recipe", RECIPES)
def test_recipe_trains_preset(motion_env, recipe, tmp_path):
  from mjlab.rl import RslRlVecEnvWrapper

  from wbc_mjlab.rl.runner import PolicyOnlyMotionTrackingRunner
  from wbc_mjlab.studio.presets import _set_path

  task, cfg, env = motion_env
  agent = _small_agent(task)
  for key, value in check_settings(task, "wm", BUILTIN[recipe]["settings"]).items():
    _set_path(agent, key, value)
  # Keep each recipe's algorithms/architectures; only reduce compute and disable
  # compilation here. Capture, symmetry and residual opt-ins have dedicated tests.
  agent.actor.hidden_dims = (64, 32)
  agent.algorithm.model_hidden_dims = (64, 64)
  agent.algorithm.model_steps = 1
  agent.algorithm.imagined_rollouts = 16
  agent.algorithm.compile = agent.algorithm.compile_collection = False
  agent.algorithm.cuda_graphs = agent.algorithm.collection_cuda_graphs = False
  agent.algorithm.probe_cuda_graphs = agent.algorithm.probe_async = False
  agent.algorithm.diagnostic_interval = 0
  agent.num_steps_per_env = 16
  agent.save_interval = 1000
  if recipe == "wm-blink" and task == "Wbc-G1-EE":
    # Exercise probes with an actor that has no joint-reference observations.
    agent.algorithm.probe_start = 0
    agent.algorithm.probe_num_envs = 2
    agent.algorithm.probe_tile_s = 0.04
    agent.algorithm.probe_lead_s = (0.0, 0.0)
    agent.algorithm.probe_steps_per_iteration = 2
  layout = copy.deepcopy((cfg.observations, cfg.actions, cfg.rewards))
  runner = None
  try:
    runner = PolicyOnlyMotionTrackingRunner(
      RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions),
      asdict(agent),
      str(tmp_path),
      "cuda:0",
    )
    alg = runner.alg
    before = [p.detach().clone() for p in alg.actor.parameters()]
    # Reconstruct the selected preset's actual actor input, including EE and IMU.
    im = alg.imagination
    real = env.observation_manager.compute_group("actor")
    imagined, _ = im.actor_obs(im.view(*im._probe()), im.env_history()[:, :-1])
    torch.testing.assert_close(imagined, real, atol=1e-4, rtol=1e-4)
    runner.learn(2)
    assert alg.num_updates > 0
    assert any(
      not torch.equal(p, old)
      for p, old in zip(alg.actor.parameters(), before, strict=True)
    )
    assert all(torch.isfinite(p).all() for p in alg.actor.parameters())
    assert (cfg.observations, cfg.actions, cfg.rewards) == layout
    if recipe == "wm-blink" and task == "Wbc-G1-EE":
      # learn() closes the probe simulator; completed tiles prove it ran.
      assert alg.probes.cursor > 0 or alg.probes.sweeps > 0
  finally:
    if runner is not None:
      runner.alg.close()
