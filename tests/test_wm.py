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


def test_heading_invariance_and_integrated_prior():
  from mjlab.utils.lab_api.math import quat_apply, quat_mul

  from wbc_mjlab.rl.wm.model import (
    WorldModel,
    apply_integrated_change,
    integrate_velocity,
    state_features,
  )

  s = _random_state(16, 23)  # the model must remain robot/DoF independent
  angle = torch.linspace(-2, 2, 16)
  yaw = torch.stack(
    [torch.cos(angle / 2), angle * 0, angle * 0, torch.sin(angle / 2)], -1
  )
  shift = torch.randn(16, 3)
  shift[:, 2] = 0
  transformed = RobotState(
    quat_apply(yaw, s.root_pos) + shift,
    quat_mul(yaw, s.root_quat),
    quat_apply(yaw, s.root_lin_vel),
    quat_apply(yaw, s.root_ang_vel),
    s.joint_pos,
    s.joint_vel,
  )
  assert torch.allclose(state_features(s), state_features(transformed), atol=2e-5)
  wm = WorldModel(23, 0, 2, 0, (32,), (16,), dynamics_mode="integrated", step_dt=0.03)
  predicted = wm.step(s, s.joint_pos, torch.zeros(16, 0))
  prior = integrate_velocity(s, 0.03)
  assert torch.allclose(predicted.vector(), prior.vector(), atol=1e-6)
  assert torch.allclose(prior.joint_pos, s.joint_pos + 0.03 * s.joint_vel)
  assert torch.allclose(prior.root_pos, s.root_pos + 0.03 * s.root_lin_vel)
  correction = torch.randn(16, 12 + 2 * 23) * 0.01
  successor = apply_integrated_change(s, correction, 0.03)
  assert torch.allclose(wm.target_change(s, successor), correction, atol=2e-5)
  # Equivariance of the whole predicted transition, not just input features.
  rotated = wm.step(transformed, s.joint_pos, torch.zeros(16, 0))
  assert torch.allclose(
    rotated.root_pos, quat_apply(yaw, predicted.root_pos) + shift, atol=2e-5
  )


def test_lambda_returns_terminal_and_clip_bootstrap():
  from wbc_mjlab.rl.wm.algorithm import lambda_returns

  r = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
  values = torch.tensor([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0], [40.0, 40.0]])
  ends = torch.zeros_like(r)
  inside = torch.ones_like(r, dtype=torch.bool)
  ends[1, 0] = 1  # failure: no bootstrap from V_2
  inside[2, 1] = False  # clip ends: bootstrap from V_2, not V_3
  out = lambda_returns(r, ends, values, inside, 0.9, 1.0)
  assert torch.allclose(out[0], torch.tensor([1 + 0.9 * 2, 1 + 0.9 * (2 + 0.9 * 30)]))


def test_reward_bound_uses_functions_and_penalty_signs():
  from types import SimpleNamespace

  from mjlab.envs import mdp

  from wbc_mjlab.env.mdp.rewards import motion_global_anchor_position_error_exp
  from wbc_mjlab.rl.evaluation import reward_upper_bound

  tracking = SimpleNamespace(
    func=motion_global_anchor_position_error_exp, weight=1.0, params={}
  )
  penalty = SimpleNamespace(func=mdp.joint_pos_limits, weight=-2.0, params={})
  cfg = SimpleNamespace(
    rewards={"arbitrary_name": tracking, "joint_limit": penalty},
    scale_rewards_by_dt=True,
  )
  assert reward_upper_bound(cfg, 0.02) == 0.02
  penalty.weight = 2.0
  assert reward_upper_bound(cfg, 0.02) is None
  penalty.weight = -2.0
  tracking.func = lambda env: 1000.0
  assert reward_upper_bound(cfg, 0.02) is None


def test_drift_uses_heading_difference_without_tilt_contamination():
  from types import SimpleNamespace

  from mjlab.utils.lab_api.math import quat_from_euler_xyz

  from wbc_mjlab.rl.evaluation import DriftMeter

  zero = torch.zeros(1)
  velocity = torch.tensor([[[1.0, 0.0, 0.0]]])
  command = SimpleNamespace(
    anchor_pos_w=torch.zeros(1, 3),
    robot_anchor_pos_w=torch.zeros(1, 3),
    anchor_quat_w=quat_from_euler_xyz(zero + 0.7, zero, zero),
    robot_anchor_quat_w=quat_from_euler_xyz(zero, zero + 0.8, zero),
    body_lin_vel_w=velocity,
    robot_body_lin_vel_w=velocity,
    body_ang_vel_w=torch.zeros(1, 1, 3),
    robot_body_ang_vel_w=torch.zeros(1, 1, 3),
  )
  meter = DriftMeter(1, "cpu")
  meter.update(command, torch.tensor([True]), torch.tensor([0]), 0.02)
  result = meter.result(0, 1)
  assert result["mean_yaw_deg"] < 1e-5
  assert result["mean_linear_velocity_aligned_mse"] < 1e-10


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

  from wbc_mjlab.rl.evaluation import deterministic_cfg

  cfg = load_env_cfg("Wbc-G1")
  cfg.events.pop("foot_friction")
  deterministic_cfg(cfg)
  assert cfg.events["benchmark_friction"].params["ranges"] == (1.0, 1.0)
  assert cfg.events["benchmark_friction"].params["operation"] == "abs"


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


def _small_agent(task="Wbc-G1-Lookahead"):
  from wbc_mjlab.rl.algos import rl_cfg

  agent = rl_cfg(task, "wm")
  agent.logger = "tensorboard"
  agent.actor.hidden_dims = (64, 32)
  agent.critic.hidden_dims = (32, 16)
  agent.algorithm.model_hidden_dims = (64, 64)
  agent.algorithm.heads_hidden_dims = (32,)
  agent.actor.encoder_hidden_dims = (64, 32)
  agent.algorithm.compile = False
  agent.algorithm.model_batch = 32
  agent.algorithm.model_steps = 1
  agent.algorithm.imagined_rollouts = 16
  agent.algorithm.horizon = 4
  agent.algorithm.window = 4
  agent.algorithm.replay_steps = 64
  agent.algorithm.warmup_steps = 12
  agent.algorithm.critic_epochs = 1
  agent.algorithm.critic_mini_batches = 2
  agent.algorithm.diagnostic_interval = 1
  agent.algorithm.diagnostic_batch = 16
  return agent


@cuda
def test_encoder_noise_terminal_resume_and_export(tmp_path):
  import yaml
  from mjlab.rl import RslRlVecEnvWrapper

  from wbc_mjlab.env.mdp.sampling import load_rsi_bin_stats, save_rsi_bin_stats
  from wbc_mjlab.rl.algos import run_rl_cfg
  from wbc_mjlab.rl.policy import load_actor
  from wbc_mjlab.rl.runner import PolicyOnlyMotionTrackingRunner

  cfg, env = _env("Wbc-G1-Lookahead", num_envs=16, noise=False)
  try:
    agent = _small_agent()
    agent.actor.architecture = "motion_encoder"
    agent.algorithm.dynamics_mode = "integrated"
    agent.algorithm.separate_action_noise = True
    wrapped = RslRlVecEnvWrapper(env)
    runner = PolicyOnlyMotionTrackingRunner(
      wrapped, asdict(agent), str(tmp_path), "cuda:0"
    )
    alg = runner.alg
    # Force one terminal transition and prove replay retains its pre-reset state.
    with torch.inference_mode():
      clean = alg.act(wrapped.get_observations())
      reference = alg.imagination.command.reference()
      env.episode_length_buf[:] = env.max_episode_length - 1
      obs, rew, done, extra = wrapped.step(clean)
      expected_setpoint = alg.imagination.joint_setpoint(
        alg.imagination._action_term.executed_action,
        reference,
        torch.arange(16, device="cuda:0"),
      ).clone()
      terminal = alg.terminal_recorder.state.clone()
      alg.process_env_step(obs, rew, done, extra)
    replay = alg.replay._fields
    assert done.all() and replay["valid"][0].all() and not replay["continues"][0].any()
    assert torch.equal(replay["next_state"][0], terminal)
    assert torch.allclose(replay["setpoint"][0], expected_setpoint)
    assert not torch.allclose(terminal, alg.imagination.robot_state().vector())
    runner.learn(num_learning_iterations=4)
    # A nonterminal step preserves clean history/action rate, but executes noisy targets.
    with torch.inference_mode():
      clean = alg.act(wrapped.get_observations())
      obs, rew, done, extra = wrapped.step(clean)
      assert torch.allclose(
        env.action_manager.action[~done.bool()], clean[~done.bool()]
      )
      assert torch.allclose(
        alg.imagination._action_term.raw_action[~done.bool()], clean[~done.bool()]
      )
      alg.process_env_step(obs, rew, done, extra)
    assert alg.num_updates > 0
    diagnostics = alg.rollout_diagnostics(16)
    assert all(torch.isfinite(torch.tensor(v)) for v in diagnostics.values())
    command = alg.imagination.command
    command._bin_memory.fill_(0.123)
    command.bin_failure_levels.fill_(0.456)
    checkpoint = tmp_path / "encoder.pt"
    runner.save(str(checkpoint))
    command._bin_memory.zero_()
    runner.load(str(checkpoint))
    assert torch.allclose(
      command._bin_memory, torch.full_like(command._bin_memory, 0.123)
    )
    stats = save_rsi_bin_stats(tmp_path / "rsi.npz", command)
    command._bin_memory.zero_()
    assert load_rsi_bin_stats(stats, command)
    assert torch.allclose(
      command._bin_memory, torch.full_like(command._bin_memory, 0.123)
    )
    # Saved run config reconstructs both actor branches for CLI and Studio Play.
    (tmp_path / "params").mkdir(exist_ok=True)
    (tmp_path / "params" / "agent.yaml").write_text(yaml.dump(asdict(agent)))
    restored = run_rl_cfg("Wbc-G1-Lookahead", tmp_path)
    loaded = load_actor(env, asdict(restored), checkpoint, "cuda:0")
    with torch.no_grad():
      observations = wrapped.get_observations()
      expected = runner.get_inference_policy()(observations)
      assert torch.allclose(loaded(observations), expected)
      exported = loaded.as_onnx(verbose=False).to("cuda:0")
      assert torch.allclose(exported(observations["actor"]), expected)
    runner.export_policy_to_onnx(str(tmp_path), "encoder.onnx")
    assert (tmp_path / "encoder.onnx").stat().st_size > 0
  finally:
    env.close()


@pytest.fixture
def overfit_env():
  _, env = _env("Wbc-G1-Lookahead", num_envs=32, noise=False)
  env.reset()
  yield env
  env.close()


@cuda
def test_world_model_overfits_fixed_simulator_windows(overfit_env):
  """Adapted validation idea: SuperMimic-Reproduced tests/test_world_model.py (ba61053)."""
  from wbc_mjlab.rl.wm.imagined import Imagination
  from wbc_mjlab.rl.wm.model import WorldModel

  env = overfit_env
  im = Imagination(env)
  states, setpoints = [], []
  ids = torch.arange(env.num_envs, device=env.device)
  valid = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
  with torch.no_grad():
    states.append(
      RobotState.from_vector(im.robot_state().vector().clone(), len(im.joint_names))
    )
    for _ in range(8):
      actions = 0.2 * torch.randn(
        env.num_envs, env.action_manager.total_action_dim, device=env.device
      )
      setpoints.append(im.joint_setpoint(actions, im.command.reference(), ids))
      _, _, term, timeout, _ = env.step(actions)
      valid &= ~(term | timeout)
      states.append(
        RobotState.from_vector(im.robot_state().vector().clone(), len(im.joint_names))
      )
  assert valid.any()
  context = im.context(ids)
  wm = WorldModel(
    len(im.joint_names),
    context.shape[-1],
    1,
    0,
    (128, 128),
    (16,),
    dynamics_mode="integrated",
  ).to(env.device)
  wm.update_dynamics_normalization(
    states[0], setpoints[0], context, wm.target_change(states[0], states[1]), valid
  )
  opt = torch.optim.Adam(wm.parameters(), lr=0.002)

  def errors():
    s, error = states[0], []
    for t in range(8):
      s = wm.step(s, setpoints[t], context)
      error.append(wm.change_error(s, states[t + 1])[valid].mean())
    return torch.stack(error)

  before = errors().detach()
  for _ in range(150):
    opt.zero_grad()
    errors().mean().backward()
    torch.nn.utils.clip_grad_norm_(wm.parameters(), 1.0)
    opt.step()
  after = errors().detach()
  assert after[0] < 0.5 * before[0] and after[-1] < 0.5 * before[-1], (before, after)
  action = setpoints[0].clone().requires_grad_()
  s = states[0]
  for _ in range(4):
    s = wm.step(s, action, context)
  s.vector().square().mean().backward()
  assert torch.isfinite(action.grad).all() and action.grad.abs().sum() > 0


@cuda
def test_benchmark_queue_and_drift(tmp_path):
  from wbc_mjlab.rl.evaluation import deterministic_cfg, run_benchmark

  _, env = _env("Wbc-G1-Lookahead", num_envs=2, noise=False, edit=deterministic_cfg)
  try:
    # Short synthetic clip boundaries keep queue tests quick while exercising real
    # command resets, rewards, terminations and online reference frames.
    command = env.command_manager.get_term("motion")
    command.motion.segment_end_idx = command.motion.segment_start_idx + 3
    command.motion.segment_length = torch.full_like(command.motion.segment_length, 3)

    class Zero(torch.nn.Module):
      def forward(self, obs):
        return torch.zeros(
          env.num_envs, env.action_manager.total_action_dim, device=env.device
        )

    selected = [0, 1, 2]
    result = run_benchmark(env, Zero(), trajectories=selected, max_step_reward=0.1)
    assert [r["trajectory"] for r in result["clips"]] == selected
    assert all(0 < r["steps"] <= 3 for r in result["clips"])
    assert result["summary"]["clips"] == 3
    assert all("mean_yaw_deg" in r["drift"] for r in result["clips"])
    assert all(r["normalized_return"] is not None for r in result["clips"])
  finally:
    env.close()


@cuda
def test_ppo_encoder_saved_run_and_evaluation(tmp_path):
  """PPO and WM share actor construction, sampler persistence and run restoration."""
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg

  from wbc_mjlab.rl.algos import rl_cfg
  from wbc_mjlab.rl.evaluation import EvaluationConfig, evaluate_checkpoint
  from wbc_mjlab.rl.policy import MotionEncoderModel
  from wbc_mjlab.rl.runner import PolicyOnlyMotionTrackingRunner
  from wbc_mjlab.scripts.train import _safe_dump_yaml
  from wbc_mjlab.tasks import prepare_wbc_run

  prepare_wbc_run(task_id="Wbc-G1-Lookahead")
  cfg = load_env_cfg("Wbc-G1-Lookahead")
  cfg.commands["motion"].motion_file = str(SAMPLES)
  cfg.scene.num_envs = 8
  # A saved scalar override must survive the full evaluator, not just config parsing.
  reward_name = next(iter(cfg.rewards))
  cfg.rewards[reward_name].weight = 1.234
  agent = rl_cfg("Wbc-G1-Lookahead")
  agent.actor.architecture = "motion_encoder"
  agent.actor.hidden_dims = agent.actor.encoder_hidden_dims = (32, 16)
  agent.critic.hidden_dims = (32, 16)
  agent.num_steps_per_env = 4
  agent.algorithm.num_learning_epochs = 1
  agent.algorithm.num_mini_batches = 1
  agent.logger = "tensorboard"
  for name, value in (("env", asdict(cfg)), ("agent", asdict(agent))):
    _safe_dump_yaml(tmp_path / "params" / name, value)
  env = ManagerBasedRlEnv(cfg, device="cuda:0")
  try:
    runner = PolicyOnlyMotionTrackingRunner(
      RslRlVecEnvWrapper(env), asdict(agent), str(tmp_path), "cuda:0"
    )
    assert isinstance(runner.alg.actor, MotionEncoderModel)
    runner.learn(num_learning_iterations=1)
    command = env.command_manager.get_term("motion")
    command._bin_memory.fill_(0.234)
    checkpoint = tmp_path / "model_test.pt"
    runner.save(str(checkpoint))
    command._bin_memory.zero_()
    runner.load(str(checkpoint))
    assert torch.allclose(
      command._bin_memory, torch.full_like(command._bin_memory, 0.234)
    )
    runner.learn(num_learning_iterations=1)
    clip = command.motion.segment_names[0]
  finally:
    env.close()
  result, out = evaluate_checkpoint(
    EvaluationConfig(checkpoint, num_envs=2, clips=[clip])
  )
  assert out.is_file() and len(result["clips"]) == 1
  assert result["protocol"]["task"] == "Wbc-G1-Lookahead"
  assert reward_name in result["clips"][0]["reward_terms"]
