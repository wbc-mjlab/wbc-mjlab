"""Official-inspired options: physical contracts, curriculum, export and resume."""

import copy
from dataclasses import asdict

import mjlab  # noqa: F401
import pytest
import torch
from test_wm import _env, _random_state, _small_agent

from wbc_mjlab.rl.wm.model import WorldModel, apply_velocity_change


@pytest.mark.parametrize("joints", [7, 29])
@pytest.mark.parametrize("blend", [0.5, 0.625, 1.0])
def test_velocity_only_integration(joints, blend):
  from mjlab.utils.lab_api.math import quat_apply

  s = _random_state(8, joints)
  delta = (torch.randn(8, 6 + joints) * 0.1).requires_grad_()
  model = WorldModel(
    joints, 0, 2, 0, (16,), (16,), dynamics_mode="velocity", loss_scale="physical"
  )
  after = apply_velocity_change(s, delta, 0.02, blend)
  torch.testing.assert_close(model.target_change(s, after), delta, atol=1e-6, rtol=1e-5)
  torch.testing.assert_close(
    after.root_pos - s.root_pos,
    0.02 * (s.root_lin_vel + blend * quat_apply(s.root_quat, delta[:, :3])),
    atol=1e-7,
    rtol=1e-4,
  )
  torch.testing.assert_close(
    after.joint_pos,
    s.joint_pos + 0.02 * torch.lerp(s.joint_vel, after.joint_vel, blend),
  )
  # Zero change preserves WORLD angular velocity even though orientation advances.
  zero = apply_velocity_change(s, torch.zeros_like(delta), 0.02, blend)
  torch.testing.assert_close(zero.root_ang_vel, s.root_ang_vel)
  after.vector().square().sum().backward()
  assert torch.isfinite(delta.grad).all()
  assert model.dynamics[-1].out_features == 6 + joints


def test_fixed_loss_independent_of_statistics():
  s = _random_state(4, 7)
  model = WorldModel(7, 0, 2, 0, (16,), (16,), loss_scale="physical")
  other = copy.deepcopy(s)
  other.root_pos = s.root_pos + torch.tensor([0.01, 0, 0])
  expected = model.change_error(other, s)
  torch.testing.assert_close(
    expected[:, :3].sum(-1), torch.ones(4), atol=1e-5, rtol=1e-5
  )
  model.change_norm.std.fill_(1000)
  torch.testing.assert_close(model.change_error(other, s), expected)
  with pytest.raises(ValueError, match="six finite positive"):
    WorldModel(
      7, 0, 2, 0, (16,), (16,), physical_error_scales=(1, 1, 1, 1, 1, float("nan"))
    )


def test_graph_storage_guard_accepts_values_rejects_replacement():
  from wbc_mjlab.rl.wm.algorithm import GraphStorageGuard, graph_tensors

  net = torch.nn.Linear(2, 1)
  optimizer = torch.optim.Adam(net.parameters())
  net(torch.ones(1, 2)).sum().backward()
  optimizer.step()
  guard = GraphStorageGuard(lambda: graph_tensors([net], [optimizer]))
  guard.capture()
  with torch.no_grad():
    net.weight.add_(1)
  guard.check()
  state = optimizer.state[net.weight]
  state["exp_avg"] = state["exp_avg"].clone()
  with pytest.raises(RuntimeError, match="storage changed"):
    guard.check()


def test_visited_returns_boundaries_and_per_episode_votes():
  from wbc_mjlab.env.mdp.return_sampling import VisitedReturns

  sampler = VisitedReturns(
    (2, 2), torch.ones(2, 2, dtype=torch.bool), 3, 2, 256, 0.5, "cpu"
  )
  bins = torch.tensor([0, 2, 3])
  no = torch.zeros(3, dtype=torch.bool)
  sampler.record(bins, bins, torch.ones(3) * 0.25, no, no, no)
  # All boundaries: a fall, timeout, clip end. Clip end cannot read next clip's bin.
  sampler.record(
    bins,
    torch.tensor([1, 2, 0]),
    torch.ones(3) * 0.25,
    torch.tensor([True, False, False]),
    ~no,
    torch.tensor([False, False, True]),
  )
  torch.testing.assert_close(
    sampler.value.flatten(), torch.tensor([0.3125, 1.0, 0.6875, 0.6875])
  )
  torch.testing.assert_close(
    sampler.count.flatten(), torch.tensor([1.0, 0.0, 1.0, 1.0])
  )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_visited_returns_observes_terminal_step_before_reset():
  from mjlab.managers.termination_manager import TerminationTermCfg

  from wbc_mjlab.rl.evaluation import reward_upper_bound

  def always_terminate(env):
    return torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

  def edit(cfg):
    cfg.commands["motion"].rsi.strategy = "visited_return"
    cfg.commands["motion"].rsi.return_horizon = 1
    cfg.terminations = {"forced": TerminationTermCfg(func=always_terminate)}

  cfg, env = _env("Wbc-G1-Lookahead", num_envs=8, noise=False, edit=edit)
  try:
    env.reset()
    command = env.command_manager.get_term("motion")
    traj = command.trajectory_ids.clone()
    local = command.time_steps - command.motion.segment_start_idx[traj]
    bins = traj * command.bins_per_trajectory + local // command.bin_width_frames
    _, rewards, terminated, _, _ = env.step(torch.zeros(8, 29, device=env.device))
    assert terminated.all()
    sampler = command._visited_returns
    counts = torch.zeros_like(sampler.count.flatten()).scatter_add_(
      0, bins, torch.ones_like(rewards)
    )
    returns = rewards * (1 - sampler.gamma) / reward_upper_bound(cfg, env.step_dt)
    expected = torch.zeros_like(counts).scatter_add_(0, bins, returns)
    visited = counts > 0
    torch.testing.assert_close(sampler.count.flatten(), counts)
    torch.testing.assert_close(
      sampler.value.flatten()[visited],
      (expected / counts.clamp_min(1))[visited],
      atol=1e-7,
      rtol=1e-4,
    )
  finally:
    env.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("graphs", [False, True])
def test_modular_training_resume_and_export(tmp_path, graphs):
  from mjlab.rl import RslRlVecEnvWrapper

  from wbc_mjlab.env.mdp.sampling import restore_sampling_state, sampling_state
  from wbc_mjlab.rl.runner import PolicyOnlyMotionTrackingRunner

  def edit(cfg):
    cfg.commands["motion"].rsi.strategy = "visited_return"
    cfg.commands["motion"].rsi.return_horizon = 2

  _, env = _env("Wbc-G1-Lookahead", num_envs=8, noise=False, edit=edit)
  runner = None
  try:
    agent = _small_agent("Wbc-G1-Lookahead")
    agent.actor.architecture = "motion_encoder"
    a = agent.algorithm
    a.dynamics_mode, a.dynamics_loss_scale, a.velocity_blend = (
      "velocity",
      "physical",
      None,
    )
    a.critic_inputs, a.critic_layer_norm, a.critic_activation = (
      "structured",
      True,
      "gelu",
    )
    a.termination_mode, a.bound_critic = "hybrid", True
    a.skip_nonfinite_updates = True
    a.compile = a.compile_collection = graphs
    a.collection_cuda_graphs = graphs
    a.diagnostic_interval = 0
    agent.clip_actions = 20.0
    runner = PolicyOnlyMotionTrackingRunner(
      RslRlVecEnvWrapper(env), asdict(agent), str(tmp_path), "cuda:0"
    )
    runner.learn(5)
    alg = runner.alg
    saved = copy.deepcopy(alg.save())
    assert alg.num_updates > 0
    assert all(torch.isfinite(t).all() for t in alg.world_model.state_dict().values())
    command = env.command_manager.get_term("motion")
    assert command._visited_returns.count.sum() > 0
    stats = sampling_state(command)
    command._visited_returns.value.zero_()
    assert restore_sampling_state(command, stats, strict=True)
    torch.testing.assert_close(command._visited_returns.value, stats["return_values"])
    wrong = dict(stats, return_config=(0.5, 2, 256))
    with pytest.raises(ValueError, match="discount, horizon or memory differs"):
      restore_sampling_state(command, wrong, strict=True)
    assert (tmp_path / "params" / "policy.onnx").exists()
    alg.load(saved, None, True)
    if graphs:
      assert alg._policy_step._graph is None
    obs = runner.env.get_observations()
    for _ in range(5):
      with torch.inference_mode():
        actions = alg.act(obs)
        obs, rewards, dones, extras = runner.env.step(actions)
      alg.process_env_step(obs, rewards, dones, extras)
      losses = alg.update()
      assert all(torch.isfinite(torch.tensor(v)) for v in losses.values())
    incompatible = copy.deepcopy(saved)
    incompatible["wm_metadata"]["critic_layout"] = ("actor", False, "elu")
    with pytest.raises(ValueError, match="critic layout"):
      alg.load(incompatible, None, True)
  finally:
    if runner is not None:
      runner.alg.close()
    env.close()
