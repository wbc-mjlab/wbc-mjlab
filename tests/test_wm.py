"""World-model training (``--algo wm``): kinematics and imagined terms against the simulator,
state changes, replay windows, a run's algorithm, and a short training + Play load."""

from __future__ import annotations

from dataclasses import asdict

import mjlab  # noqa: F401  (registers the tasks before wbc_mjlab.rl is imported)
import pytest
import torch

from wbc_mjlab.data_paths import repo_root
from wbc_mjlab.rl.wm.kinematics import RobotState
from wbc_mjlab.rl.wm.model import apply_change, state_change
from wbc_mjlab.rl.wm.replay import Replay

SAMPLES = repo_root() / "data" / "g1" / "samples"
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _random_state(n: int, joints: int) -> RobotState:
  q = torch.randn(n, 4)
  return RobotState(
    torch.randn(n, 3),
    q / q.norm(dim=-1, keepdim=True),
    torch.randn(n, 3),
    torch.randn(n, 3),
    torch.randn(n, joints),
    torch.randn(n, joints),
  )


def test_state_change_round_trip():
  s0 = _random_state(512, 29)
  d = torch.randn(512, 12 + 2 * 29) * 0.3
  assert torch.allclose(state_change(s0, apply_change(s0, d)), d, atol=1e-4)
  # Gradients stay finite through a zero change (the world model starts near it).
  d0 = torch.zeros(4, 12 + 2 * 29, requires_grad=True)
  s = _random_state(4, 29)
  state_change(s, apply_change(s, d0)).square().sum().backward()
  assert torch.isfinite(d0.grad).all()


def _replay(capacity: int, num_envs: int, observed_dim: int = 0) -> Replay:
  return Replay(capacity, num_envs, 1, 1, 0, observed_dim, "cpu")


def test_replay_windows_follow_time():
  r = _replay(capacity=5, num_envs=3)
  for t in range(8):  # wraps around the ring
    r.begin(state=torch.full((3, 1), float(t)), time_step=torch.full((3,), t))
    r.end(continues=torch.ones(3, dtype=torch.bool))
  assert r.size == 5
  w = r.sample_windows(64, 3)
  steps = w.time_step  # [3, 64]: consecutive, all from the last 5 steps
  assert ((steps[1:] - steps[:-1]) == 1).all()
  assert steps.min() >= 3 and steps.max() <= 7
  assert torch.equal(w.state[..., 0].long(), steps)


def test_replay_history_restarts_with_the_episode():
  """Past observations as the env's history holds them: backfilled after a reset."""
  r = _replay(capacity=16, num_envs=1, observed_dim=1)
  for t in range(10):
    r.begin(time_step=torch.tensor([t]), observed=torch.tensor([[float(t)]]))
    r.end(done=torch.tensor([t == 5]))  # step 5 ends the episode; step 6 starts one
  b = r.sample_states(256, past_steps=3)
  expect = {
    t: [max(t - k, 6) if t >= 6 else max(t - k, 0) for k in (3, 2, 1)]
    for t in range(3, 10)
  }
  for t, past in zip(b.time_step.tolist(), b.past[..., 0].tolist(), strict=True):
    assert past == expect[t], (t, past)


@cuda
def test_cuda_graphed_step_replays_with_new_inputs():
  """The captured step computes on each call's inputs and keeps optimizer state."""
  from wbc_mjlab.rl.wm.algorithm import CudaGraphed
  from wbc_mjlab.rl.wm.imagined import Goal

  w = torch.nn.Parameter(torch.zeros(3, device="cuda:0"))
  opt = torch.optim.Adam([w], lr=torch.tensor(0.1, device="cuda:0"), capturable=True)

  def step(x: torch.Tensor, goal: Goal) -> torch.Tensor:
    opt.zero_grad(set_to_none=False)
    loss = ((w - x) ** 2).sum() + goal.time_step.float().sum()
    loss.backward()
    opt.step()
    return loss.detach()

  graphed = CudaGraphed(step, warmup=1)
  results = []
  for i in range(4):  # warm-up, capture, two replays
    x = torch.full((3,), float(i), device="cuda:0")
    goal = Goal(None, None, torch.tensor([i], device="cuda:0"), None, None)  # type: ignore[arg-type]
    expected = ((w.detach() - x) ** 2).sum() + i
    results.append((graphed(x, goal).item(), expected.item()))
  for got, expected in results:
    assert abs(got - expected) < 1e-4, results
  assert w.detach().abs().sum() > 0  # the optimizer stepped inside the graph


def test_run_algo(tmp_path):
  from wbc_mjlab.rl.algos import run_algo
  from wbc_mjlab.rl.wm.config import WmAlgorithmCfg

  assert run_algo(tmp_path) == "ppo"  # no params: a run of the default algorithm
  (tmp_path / "params").mkdir()
  (tmp_path / "params" / "agent.yaml").write_text(
    f"algorithm:\n  class_name: {WmAlgorithmCfg.class_name}\n  horizon: !!python/tuple [32]\n"
  )
  assert run_algo(tmp_path) == "wm"


def test_play_env_has_no_external_forces():
  """Play builds a wm run's algorithm, which refuses forces it cannot predict."""
  from mjlab.tasks.registry import load_env_cfg

  from wbc_mjlab.rl.algos import env_cfg
  from wbc_mjlab.tasks import prepare_wbc_run

  prepare_wbc_run(task_id="Wbc-G1")
  for play in (False, True):
    cfg = env_cfg(load_env_cfg("Wbc-G1", play=play), "wm")
    active = [
      name
      for name, ev in cfg.events.items()
      if ev.mode in ("interval", "step") and ev.params.get("enabled", True)
    ]
    assert active == [], active


def _env(task: str, num_envs: int = 32, noise: bool = True, edit=None):
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  from wbc_mjlab.rl.wm.config import wm_env_cfg
  from wbc_mjlab.tasks import prepare_wbc_run

  prepare_wbc_run(task_id=task)
  cfg = wm_env_cfg(load_env_cfg(task))
  cfg.commands["motion"].motion_file = str(SAMPLES)
  cfg.scene.num_envs = num_envs
  cfg.observations["actor"].enable_corruption = noise
  if edit is not None:
    edit(cfg)
  return cfg, ManagerBasedRlEnv(cfg=cfg, device="cuda:0")


@pytest.fixture(scope="module")
def env():
  cfg, env = _env("Wbc-G1")
  env.reset()
  for _ in range(30):  # away from the reference: non-trivial errors, joint limits, …
    env.step(
      torch.randn(env.num_envs, env.action_manager.total_action_dim, device="cuda:0")
    )
  yield env
  env.close()


@cuda
def test_kinematics_match_simulator(env):
  from wbc_mjlab.rl.wm.imagined import Imagination

  im = Imagination(env)
  d = im.robot.data
  b = im.kinematics.bodies(im.robot_state())
  origins = env.scene.env_origins[:, None]
  assert torch.allclose(b.pos + origins, d.body_link_pos_w, atol=1e-4)
  same = (b.quat - d.body_link_quat_w).abs().amax(-1) < 1e-4
  flipped = (b.quat + d.body_link_quat_w).abs().amax(-1) < 1e-4
  assert (same | flipped).all()
  assert torch.allclose(b.lin_vel, d.body_link_lin_vel_w, atol=1e-3)
  assert torch.allclose(b.ang_vel, d.body_link_ang_vel_w, atol=1e-3)


@cuda
def test_imagined_terms_match_env(env):
  """At the simulator's state, imagination's observation and rewards are the env's."""
  from wbc_mjlab.rl.wm.imagined import Imagination, call_term

  im = Imagination(env)
  view = im.view(*im._probe())
  om = env.observation_manager
  for name, cfg in zip(
    om._group_obs_term_names["actor"], om._group_obs_term_cfgs["actor"], strict=True
  ):
    real, mine = cfg.func(env, **cfg.params), call_term(cfg, view)
    assert torch.allclose(real, mine, atol=1e-4), name
  analytic = [n for n, _ in im.analytic_rewards]
  for name, cfg in im.analytic_rewards:
    if name == "survival":  # imagined states are alive (termination: the step heads)
      continue
    real, mine = cfg.func(env, **cfg.params), call_term(cfg, view)
    assert torch.allclose(real, mine, atol=1e-4, rtol=1e-4), name
  assert "motion_body_pos" in analytic and "action_rate_l1" in analytic
  assert "actuator_torque_soft_limit" in [n for _, n, _ in im.learned_rewards]


@cuda
def test_lookahead_observation_matches_env():
  """History terms and the reference window, rebuilt in imagination, equal the env's."""
  from wbc_mjlab.rl.wm.imagined import Imagination

  _, env = _env("Wbc-G1-Lookahead", num_envs=16, noise=False)
  try:
    env.reset()
    for _ in range(15):  # past the history length, so it holds distinct steps
      env.step(torch.randn(16, env.action_manager.total_action_dim, device="cuda:0"))
    im = Imagination(env)
    assert im.history_length == 10 and im.window_frames == 10
    real = env.observation_manager.compute_group("actor")
    mine, history = im.actor_obs(im.view(*im._probe()), im.env_history()[:, :-1])
    assert torch.allclose(mine, real, atol=1e-4)
    assert torch.allclose(history, im.env_history(), atol=1e-4)
  finally:
    env.close()


@cuda
def test_reference_ahead_and_mixed_history_match_env():
  """Reference terms looking ahead (``frames`` / ``stride``) and terms with different
  history lengths, rebuilt in imagination, equal the env's observation."""
  from wbc_mjlab.rl.wm.imagined import Imagination

  def edit(cfg):
    actor = cfg.observations["actor"]
    actor.history_length = None
    actor.terms["ref_joint_pos"].params = {
      "command_name": "motion",
      "frames": 4,
      "stride": 3,
    }
    actor.terms["ref_base_lin_vel_b"].params = {"command_name": "motion", "frames": 2}
    actor.terms["joint_pos"].history_length = 5
    actor.terms["base_ang_vel"].history_length = 3

  _, env = _env("Wbc-G1", num_envs=16, noise=False, edit=edit)
  try:
    env.reset()
    for _ in range(8):
      env.step(torch.randn(16, env.action_manager.total_action_dim, device="cuda:0"))
    im = Imagination(env)
    assert im.history_length == 5 and im.window_frames == 10
    real = env.observation_manager.compute_group("actor")
    mine, _ = im.actor_obs(im.view(*im._probe()), im.env_history()[:, :-1])
    assert torch.allclose(mine, real, atol=1e-4)
    # the last frame of ref_joint_pos is the reference 9 steps ahead (or the clip's end)
    c = env.command_manager.get_term("motion")
    end = c.motion.segment_end_idx[c.trajectory_ids] - 1
    later = c.motion.frame(torch.minimum(c.time_steps + 9, end)).joint_pos
    ids = c._tracked_joint_ids
    later = later if ids is None else later[:, ids]
    assert torch.allclose(c.ahead(4, 3)[3].tracked_joint_pos, later)
  finally:
    env.close()


@cuda
def test_terrain_scan_is_rejected():
  from wbc_mjlab.rl.wm.imagined import Imagination

  _, env = _env("Wbc-G1-Terrain-Scan", num_envs=4)
  try:
    with pytest.raises(ValueError, match="height_scan"):
      Imagination(env)
  finally:
    env.close()


@cuda
def test_train_and_play(env, tmp_path):
  """A few iterations end to end (collect, fit, imagine), then Play's load and export."""
  from mjlab.rl import RslRlVecEnvWrapper

  from wbc_mjlab.rl.algos import rl_cfg
  from wbc_mjlab.rl.runner import PolicyOnlyMotionTrackingRunner

  agent = rl_cfg("Wbc-G1", "wm")
  agent.logger = "tensorboard"
  agent.algorithm.compile = False
  agent.algorithm.model_batch = 64
  agent.algorithm.model_steps = 2
  agent.algorithm.imagined_rollouts = 64
  agent.algorithm.horizon = 4
  agent.algorithm.replay_steps = 32
  agent.algorithm.warmup_steps = 10
  agent.save_interval = 1000
  wrapped = RslRlVecEnvWrapper(env)
  runner = PolicyOnlyMotionTrackingRunner(
    wrapped, asdict(agent), str(tmp_path), "cuda:0"
  )
  runner.learn(num_learning_iterations=4)
  alg = runner.alg
  assert alg.num_updates > 0
  stats = alg._update_world_model() | alg._update_policy()
  assert all(torch.isfinite(torch.tensor(v)) for v in stats.values()), stats
  checkpoint = sorted(tmp_path.glob("model_*.pt"))[-1]

  # Play: a fresh runner of the run's algorithm loads the actor and exports it.
  play = PolicyOnlyMotionTrackingRunner(
    wrapped, asdict(rl_cfg("Wbc-G1", "wm")), None, "cuda:0"
  )
  play.load(
    str(checkpoint), load_cfg={"actor": True}, strict=True, map_location="cuda:0"
  )
  policy = play.get_inference_policy(device="cuda:0")
  actions = policy(wrapped.get_observations())
  assert actions.shape == (env.num_envs, env.action_manager.total_action_dim)
  play.export_policy_to_onnx(str(tmp_path), "policy.onnx")
  assert (tmp_path / "policy.onnx").stat().st_size > 0
