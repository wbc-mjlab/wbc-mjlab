"""Nabla-inspired actor updates: numerical contracts and real simulator integration."""

import copy
from dataclasses import asdict

import mjlab  # noqa: F401
import pytest
import torch

from wbc_mjlab.rl.gradients import clip_action_gradient, guard_action_gradient


def test_action_gradient_outliers_and_invalid_rows():
  x = torch.ones(6, 2, requires_grad=True)
  g = torch.tensor(
    [
      [3.0, 4.0],
      [0.0, 5.0],
      [300.0, 400.0],
      [0.0, 0.0],
      [float("nan"), 1.0],
      [float("inf"), 0.0],
    ]
  )
  y = guard_action_gradient(x, 3)
  assert torch.equal(x, y)
  y.backward(g)
  torch.testing.assert_close(
    x.grad[:3], torch.tensor([[3.0, 4.0], [0.0, 5.0], [9.0, 12.0]])
  )
  assert torch.equal(x.grad[3:], torch.zeros(3, 2))
  assert torch.equal(
    clip_action_gradient(torch.zeros_like(g), 3)[0], torch.zeros_like(g)
  )
  assert guard_action_gradient(x, 0) is x


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_action_gradient_compiled_capture():
  x = torch.ones(4, 2, device="cuda", requires_grad=True)
  weights = torch.tensor(
    [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [100.0, 0.0]], device="cuda"
  )
  fn = torch.compile(
    lambda a, w: (guard_action_gradient(a, 3) * w).sum(), fullgraph=True
  )
  for _ in range(3):
    fn(x, weights).backward()
    x.grad.zero_()
  graph = torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph):
    fn(x, weights).backward()
  x.grad.zero_()
  graph.replay()
  torch.testing.assert_close(
    x.grad[:, 0], torch.tensor([1.0, 1.0, 1.0, 3.0], device="cuda")
  )
  weights.mul_(2)
  x.grad.zero_()
  graph.replay()
  torch.testing.assert_close(
    x.grad[:, 0], torch.tensor([2.0, 2.0, 2.0, 6.0], device="cuda")
  )


def test_targets_use_physical_units_and_ignore_invalid_rows():
  from wbc_mjlab.rl.action_targets import ActionTargetReplay, action_targets

  a = torch.zeros(2, 3, 2)
  g = torch.ones_like(a)
  g[0, 0] = float("nan")
  valid = torch.ones(2, 3, dtype=torch.bool)
  valid[1, 2] = False
  scale = torch.tensor([0.1, 2.0])
  target, weight, stats = action_targets(a, g, valid, scale, 0.03, 3)
  assert weight.sum() == 4 and torch.isfinite(target).all()
  torch.testing.assert_close(stats["action_target_rms_rad"], torch.tensor(0.03))
  assert (target[weight.bool()] > a[weight.bool()]).all()
  zero, w, _ = action_targets(a, g, valid & False, scale, 0.03, 3)
  assert not zero.any() and not w.any()
  buf = ActionTargetReplay(2)
  obs = torch.ones(2, 3, 4)
  for _ in range(3):
    buf.append(obs, target, weight)
  assert len(buf.rows) == 2
  saved = buf.state_dict()
  other = ActionTargetReplay(2)
  other.load_state_dict(saved, "cpu")
  for x, y in zip(buf.rows[-1], other.rows[-1], strict=True):
    torch.testing.assert_close(x, y)
  with pytest.raises(ValueError, match="capacity"):
    ActionTargetReplay(1).load_state_dict(saved, "cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("compiled", [False, True])
def test_target_training_resume_and_export(tmp_path, compiled):
  from mjlab.rl import RslRlVecEnvWrapper
  from test_wm import _env, _small_agent

  from wbc_mjlab.rl.runner import PolicyOnlyMotionTrackingRunner

  _, env = _env("Wbc-G1-Lookahead", num_envs=8, noise=False)
  try:
    agent = _small_agent()
    a = agent.algorithm
    a.horizon = 1  # same-step reward derivatives must survive torch.compile
    a.actor_update = "action_targets"
    a.action_gradient_clip_ratio = 3
    a.action_target_steps = 2
    a.action_target_windows = 2
    a.action_target_batch = 32
    if not compiled:
      a.termination_margin_weight = 1.0
      a.termination_margin_start = 0.0
    a.compile = compiled
    a.cuda_graphs = compiled
    wrapped = RslRlVecEnvWrapper(env)
    runner = PolicyOnlyMotionTrackingRunner(
      wrapped, asdict(agent), str(tmp_path), "cuda:0"
    )
    alg = runner.alg
    before = [p.detach().clone() for p in alg.actor.parameters()]
    runner.learn(4)
    assert alg.num_updates > 0 and len(alg.action_target_replay.rows) == 2
    assert any(
      not torch.equal(x, y) for x, y in zip(before, alg.actor.parameters(), strict=True)
    )
    stats = alg._update_policy()
    assert all(torch.isfinite(v).all() for v in stats.values())
    assert stats["action_target_valid_fraction"] > 0
    # Eager and compiled gradients must include direct reward contributions:
    # horizon=1 provides no later policy/history action to hide missing gradients.
    assert alg.action_target_replay.rows[-1][2].sum() > 0
    saved = copy.deepcopy(alg.save())
    alg.action_target_replay.rows.clear()
    alg.load(saved, None, True)
    assert len(alg.action_target_replay.rows) == 2
    assert alg.num_updates == saved["num_updates"]
    assert all(torch.isfinite(v).all() for v in alg._update_policy().values())
    # Empty targets must not advance an optimizer with existing momentum.
    alg.action_target_replay.rows.clear()
    alg.action_target_replay._flat = None
    obs = saved["action_target_state"]["rows"][0][0].to("cuda")
    probe = torch.zeros(len(obs), alg.num_joints, device="cuda", requires_grad=True)
    probe.grad = torch.zeros_like(probe)
    before_empty = [p.detach().clone() for p in alg.actor.parameters()]
    steps_before = [v["step"].clone() for v in alg.actor_optimizer.state.values()]
    alg._fit_action_targets(
      [
        (
          obs,
          probe.detach(),
          probe,
          torch.zeros(len(obs), device="cuda", dtype=torch.bool),
        )
      ]
    )
    assert all(
      torch.equal(x, y)
      for x, y in zip(before_empty, alg.actor.parameters(), strict=True)
    )
    assert all(
      torch.equal(x, v["step"])
      for x, v in zip(steps_before, alg.actor_optimizer.state.values(), strict=True)
    )
    wrong = dict(saved, actor_update="pathwise")
    with pytest.raises(ValueError, match="actor update"):
      alg.load(wrong, None, True)
    runner.export_policy_to_onnx(str(tmp_path), "nabla.onnx")
    assert (tmp_path / "nabla.onnx").stat().st_size > 0
  finally:
    env.close()


@pytest.mark.parametrize(
  "overrides, message",
  [
    ({"action_gradient_clip_ratio": -1}, "clip_ratio"),
    ({"actor_update": "missing"}, "actor_update"),
    ({"action_target_steps": 0}, "positive"),
    ({"action_target_delta": float("nan")}, "finite"),
    ({"actor_update": "action_targets", "value_gradient": 0.5}, "disabled"),
    ({"actor_update": "action_targets", "self_imitation": 1.0}, "disabled"),
    ({"actor_update": "action_targets", "latent_aux": 1.0}, "disabled"),
  ],
)
def test_invalid_update_settings_fail_before_environment_setup(overrides, message):
  from wbc_mjlab.rl.wm.algorithm import WorldModelRL
  from wbc_mjlab.rl.wm.config import WmAlgorithmCfg

  settings = asdict(WmAlgorithmCfg()) | overrides
  settings.pop("class_name")
  with pytest.raises(ValueError, match=message):
    WorldModelRL(None, None, (8,), "elu", **settings)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("compiled", [False, True])
def test_gradient_guard_alone_keeps_pathwise_and_old_checkpoints(tmp_path, compiled):
  from mjlab.rl import RslRlVecEnvWrapper
  from test_wm import _env, _small_agent

  from wbc_mjlab.rl.runner import PolicyOnlyMotionTrackingRunner

  _, env = _env("Wbc-G1-EE", num_envs=4, noise=False)
  try:
    agent = _small_agent("Wbc-G1-EE")
    agent.algorithm.action_gradient_clip_ratio = 3
    agent.algorithm.compile = compiled
    agent.algorithm.cuda_graphs = compiled
    agent.num_steps_per_env = 16
    runner = PolicyOnlyMotionTrackingRunner(
      RslRlVecEnvWrapper(env), asdict(agent), str(tmp_path), "cuda:0"
    )
    alg = runner.alg
    before = [p.detach().clone() for p in alg.actor.parameters()]
    runner.learn(4)
    assert alg.actor_update == "pathwise" and alg.num_updates > 0
    if compiled:
      assert alg._policy_step._graph is not None
    assert any(
      not torch.equal(x, y) for x, y in zip(before, alg.actor.parameters(), strict=True)
    )
    saved = copy.deepcopy(alg.save())
    saved.pop("actor_update")
    saved.pop("action_target_state")
    alg.load(saved, None, True)
    assert not alg.action_target_replay.rows
  finally:
    env.close()
