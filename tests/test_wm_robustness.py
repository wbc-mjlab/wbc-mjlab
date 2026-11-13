"""Disturbance boundaries and task-derived imagination survival, independent of PPO."""

from types import SimpleNamespace

import mjlab  # noqa: F401
import pytest
import torch
from test_wm import _env, _small_agent

from wbc_mjlab.rl.wm.model import RunningNorm
from wbc_mjlab.rl.wm.perturbations import EVENT, InterStepVelocityKicks, VelocityKickCfg
from wbc_mjlab.rl.wm.replay import Replay, transition_mask


def test_kick_captures_successor_and_skips_resets():
  cfg = VelocityKickCfg(
    (0.02, 0.02), ((1, 1),) * 6, initial_scale=0.5, ramp_duration_s=0.04
  )
  controller = InterStepVelocityKicks(cfg, 3, "cpu")
  state = torch.arange(18, dtype=torch.float).view(3, 6)
  before = state.clone()

  def write(velocity, *, env_ids):
    state[env_ids] = velocity

  robot = SimpleNamespace(
    data=SimpleNamespace(root_link_vel_w=state), write_root_link_velocity_to_sim=write
  )
  result = controller.after_step(
    done=torch.tensor([False, True, False]),
    dt=0.02,
    robot=robot,
    capture_successor=lambda: state,
  )
  torch.testing.assert_close(result.successor, before)
  torch.testing.assert_close(
    state - before, torch.tensor([[0.75] * 6, [0.0] * 6, [0.75] * 6])
  )
  assert result.pushed.tolist() == [True, False, True]
  assert result.model_cut.all()
  controller.clean_episode.fill_(True)
  result = controller.after_step(
    done=torch.zeros(3, dtype=torch.bool),
    dt=0.02,
    robot=robot,
    capture_successor=lambda: state,
  )
  assert not result.pushed.any()


def test_kick_cuts_dynamics_but_not_actor_history():
  replay = Replay(8, 1, 1, 1, 0, 1, "cpu")
  for t in range(4):
    replay.begin(observed=torch.tensor([[float(t)]]))
    replay.end(
      done=torch.tensor([False]),
      continues=torch.tensor([t != 1]),
      valid=torch.tensor([True]),
    )
  batch = replay.sample_states(1, past_steps=3)
  torch.testing.assert_close(batch.past.flatten(), torch.tensor([0.0, 1.0, 2.0]))
  mask = transition_mask(replay._fields["continues"][:4], replay._fields["valid"][:4])
  assert mask.flatten().tolist() == [True, True, False, False]


def test_excluded_critic_rows_cannot_poison_statistics():
  norm = RunningNorm(2, 0.01)
  norm.update(
    torch.tensor([[2.0, 4.0], [float("inf"), float("nan")]]), torch.tensor([0.25, 0.0])
  )
  torch.testing.assert_close(norm.mean, torch.tensor([2.0, 4.0]))
  torch.testing.assert_close(norm.var, torch.zeros(2))
  torch.testing.assert_close(norm.count, torch.tensor(0.25))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_real_kicks_preserve_replay_and_hybrid_thresholds(tmp_path):
  from dataclasses import asdict

  from mjlab.rl import RslRlVecEnvWrapper

  from wbc_mjlab.env.mdp import terminations
  from wbc_mjlab.rl.evaluation import deterministic_cfg
  from wbc_mjlab.rl.runner import PolicyOnlyMotionTrackingRunner
  from wbc_mjlab.rl.wm.imagined import Goal
  from wbc_mjlab.rl.wm.kinematics import RobotState
  from wbc_mjlab.rl.wm.recording import simulator_state

  def enable(cfg):
    cfg.events[EVENT].params.update(
      enabled=True,
      interval_range_s=(0.02, 0.02),
      initial_scale=1.0,
      ramp_duration_s=0.0,
      clean_episode_fraction=0.0,
      velocity_ranges=tuple((v, v) for v in (0.5, 0.0, 0.0, 0.2, -0.3, 0.4)),
    )

  cfg, env = _env("Wbc-G1-Lookahead", num_envs=8, noise=False, edit=enable)
  try:
    agent = _small_agent()
    agent.algorithm.termination_mode = "hybrid"
    agent.clip_actions = 20.0
    runner = PolicyOnlyMotionTrackingRunner(
      RslRlVecEnvWrapper(env), asdict(agent), str(tmp_path), "cuda:0"
    )
    alg, im = runner.alg, runner.alg.imagination
    obs, _ = env.reset()
    with torch.inference_mode():
      action = alg.act(obs)
      obs, reward, terminated, timeout, extras = env.step(action)
      done = terminated | timeout
      alg.process_env_step(obs, reward, done, extras)
    kick = env._wm_velocity_kick_result
    assert kick.pushed.any()
    row = alg.replay._fields
    torch.testing.assert_close(
      row["next_state"][0, kick.pushed], kick.successor[kick.pushed]
    )
    after = simulator_state(env, im.robot).vector()
    torch.testing.assert_close(
      after[kick.pushed, 7:13] - kick.successor[kick.pushed, 7:13],
      after.new_tensor([0.5, 0.0, 0.0, 0.2, -0.3, 0.4]).expand(
        int(kick.pushed.sum()), -1
      ),
      atol=1e-5,
      rtol=1e-4,
    )
    assert not row["continues"][0, kick.pushed].any()
    assert not row["done"][0, kick.pushed].any()
    assert row["valid"][0, kick.pushed].all()
    checkpoint = alg.save()
    assert checkpoint["perturbation_elapsed_s"] == pytest.approx(0.02)
    env._wm_kick_elapsed_s = env._wm_velocity_kicks.elapsed_s = 99.0
    alg.load(checkpoint, None, strict=True)
    assert env._wm_kick_elapsed_s == pytest.approx(0.02)
    assert env._wm_velocity_kicks.elapsed_s == pytest.approx(0.02)

    # Compare each relaxed term against exactly the configured hard predicate.
    s = im.robot_state()
    ids = torch.arange(env.num_envs, device=env.device)
    goal = Goal(
      im.command.reference(),
      None,
      im.command.time_steps,
      im.command.trajectory_ids,
      im.command.reference_offset_w,
    )
    actions = torch.zeros(env.num_envs, im.num_actions, device=env.device)
    bad = s.vector().clone()
    bad[:, 2] += 2.0
    shifted = RobotState.from_vector(bad, im.num_actions)
    view = im.view(shifted, im.kinematics.bodies(shifted), ids, goal, actions, actions)
    assert alg.tracking_termination(view).min() > 0.999
    pos = cfg.terminations["anchor_pos"]
    assert pos.func(view, **pos.params).all()
    assert any(
      f is terminations.bad_anchor_ori for f, _ in alg.tracking_termination.terms
    )
    deterministic_cfg(cfg)
    assert EVENT not in cfg.events
  finally:
    env.close()


def test_overflowed_gradient_norm_does_not_step_momentum():
  from wbc_mjlab.rl.optim import step_if_finite

  p = torch.nn.Parameter(torch.ones(2))
  opt = torch.optim.Adam([p], lr=0.01)
  p.grad = torch.ones_like(p)
  opt.step()
  before = p.detach().clone()
  moment = opt.state[p]["exp_avg"].clone()
  count = opt.state[p]["step"].clone()
  p.grad.fill_(1e30)  # individually finite; float32 L2 accumulation overflows
  norm = torch.nn.utils.clip_grad_norm_([p], 1.0)
  assert not torch.isfinite(norm) and torch.isfinite(p.grad).all()
  step_if_finite(opt, [p], enabled=True, valid=torch.isfinite(norm))
  torch.testing.assert_close(p, before, rtol=0, atol=0)
  torch.testing.assert_close(opt.state[p]["exp_avg"], moment, rtol=0, atol=0)
  torch.testing.assert_close(opt.state[p]["step"], count, rtol=0, atol=0)


def test_soft_termination_tracks_the_task_threshold_and_has_gradients():
  from mjlab.managers.termination_manager import TerminationTermCfg

  from wbc_mjlab.env.mdp.terminations import bad_anchor_pos_z_only
  from wbc_mjlab.rl.wm.termination import TrackingTermination

  term = TerminationTermCfg(
    func=bad_anchor_pos_z_only, params={"command_name": "motion", "threshold": 0.35}
  )
  soft = TrackingTermination({"anchor": term})
  height = torch.tensor([0.0, 0.33, 0.35, 0.37, 1.0], requires_grad=True)
  positions = torch.stack([height * 0, height * 0, height], dim=-1)
  command = SimpleNamespace(
    anchor_pos_w=positions, robot_anchor_pos_w=torch.zeros_like(positions)
  )
  env = SimpleNamespace(
    num_envs=5,
    device="cpu",
    command_manager=SimpleNamespace(get_term=lambda _: command),
  )
  probability = soft(env)
  torch.testing.assert_close(probability, torch.sigmoid((height - 0.35) / 0.02))
  assert bad_anchor_pos_z_only(env, "motion", 0.35).tolist() == [
    False,
    False,
    False,
    True,
    True,
  ]
  probability.sum().backward()
  assert torch.isfinite(height.grad).all() and height.grad[2] > 0


def test_dead_imagination_rows_do_not_enter_critic_targets():
  from wbc_mjlab.rl.wm.algorithm import WorldModelRL

  alg = object.__new__(WorldModelRL)
  alg.device, alg.gamma, alg.lam = "cpu", 0.99, 0.95
  alg.termination_mode = "hybrid"
  alg.value_upper_bound = 10.0
  alg.critic = torch.nn.Linear(1, 1)
  alg.critic_norm = RunningNorm(1, 0.01)
  alg.critic_optimizer = torch.optim.SGD(alg.critic.parameters(), lr=0.0)
  alg.critic_epochs = alg.critic_mini_batches = 1
  alg.max_grad_norm, alg.skip_nonfinite_updates = 1.0, True
  inputs = torch.tensor([[[2.0]], [[float("inf")]], [[float("nan")]]])
  loss = alg._update_critic(
    inputs,
    torch.zeros(3, 1),
    torch.tensor([[0.1], [float("inf")]]),
    torch.ones(2, 1),
    torch.ones(2, 1, dtype=torch.bool),
  )
  assert torch.isfinite(loss)
  torch.testing.assert_close(alg.critic_norm.mean, torch.tensor([2.0]))
  torch.testing.assert_close(alg.critic_norm.count, torch.tensor(1.0))
