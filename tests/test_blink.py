"""Blink recipe regression contracts: physics frames, export, resume and curriculum."""

import copy
from dataclasses import asdict

import mjlab  # noqa: F401
import pytest
import torch
from test_wm import _env, _random_state, _small_agent
from torch import nn

from wbc_mjlab.rl.wm.model import WorldModel, apply_integrated_change, state_change

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.parametrize("blend", [0.5, 0.625, 1.0])
def test_structured_transition_inverse_and_scale(blend):
  s = _random_state(16, 9)
  wm = WorldModel(
    9,
    0,
    2,
    0,
    (16, 16),
    (16,),
    dynamics_mode="integrated",
    velocity_blend=blend,
    loss_scale="increment",
  )
  change = torch.randn(16, 30) * 0.01
  after = apply_integrated_change(s, change, 0.02, blend)
  torch.testing.assert_close(wm.target_change(s, after), change, atol=1e-5, rtol=1e-4)
  wm.change_norm.std.fill_(1e-8)
  wm.increment_norm.std.fill_(1)
  torch.testing.assert_close(wm.change_error(after, s), state_change(s, after).square())
  zero = torch.zeros_like(change, requires_grad=True)
  apply_integrated_change(s, zero, 0.02, blend).vector().sum().backward()
  assert torch.isfinite(zero.grad).all()


def test_failure_mixture_duplicates_empty_and_clip_boundaries():
  from wbc_mjlab.env.mdp.sampling import failure_counts, failure_mixture

  valid = torch.tensor([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1]])
  base = torch.ones(5) / 5
  counts = failure_counts(
    (2, 3), torch.tensor([0, 0, 1]), torch.tensor([2, 2, 0]), torch.ones(3)
  )
  assert counts.tolist() == [[0, 0, 2], [1, 0, 0]]
  torch.testing.assert_close(failure_mixture(base, counts * 0, valid, 0.3, 10), base)
  focus = failure_mixture(base, counts, valid, 1, 1)
  torch.testing.assert_close(focus, torch.tensor([0.0, 2 / 5, 2 / 5, 1 / 5, 0.0]))


def test_reference_residual_undoes_normalization_and_exports():
  from rsl_rl.modules import EmpiricalNormalization

  from wbc_mjlab.rl.policy import ReferenceResidualHead

  norm = EmpiricalNormalization(6)
  norm._mean.fill_(3)
  norm._std.fill_(2)
  head = nn.Linear(6, 2)
  nn.init.zeros_(head.weight)
  nn.init.zeros_(head.bias)
  residual = ReferenceResidualHead(head, norm, [2, 3], [1, 2], [0.5, 2])
  x = torch.randn(8, 6)
  expected = (x[:, 2:4] - torch.tensor([1, 2])) / torch.tensor([0.5, 2])
  torch.testing.assert_close(residual(norm(x)), expected)
  exported = copy.deepcopy(residual)
  torch.testing.assert_close(exported(norm(x)), expected)


@cuda
def test_muon_graph_learning_rate_and_resume():
  from wbc_mjlab.rl.optim import MuonAdam

  torch.manual_seed(7)
  model = nn.Sequential(
    nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 8), nn.Tanh(), nn.Linear(8, 3)
  ).cuda()
  other = copy.deepcopy(model)
  a = MuonAdam(model, [model], 0.001, capturable=True)
  b = MuonAdam(other, [other], 0.001, capturable=True)
  for net in (model, other):
    for p in net.parameters():
      p.grad = torch.ones_like(p) * 0.125
  a.step()
  b.step()  # initialize all optimizer state before capture
  graph = torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph):
    b.step()
  for lr in (0.0007, 0.0001, 0.00003):
    for opt in (a, b):
      for group in opt.param_groups:
        group["lr"].fill_(lr)
    a.step()
    graph.replay()
  for x, y in zip(model.parameters(), other.parameters(), strict=True):
    torch.testing.assert_close(x, y)
  saved = copy.deepcopy(b.state_dict())
  b.load_state_dict(saved)
  for group in b.param_groups:
    assert isinstance(group["lr"], torch.Tensor) and group["lr"].is_cuda
  assert len(b.muon.param_groups[0]["params"]) == 1


@cuda
@pytest.mark.parametrize("graphs", [False, True])
def test_blink_training_ema_mirror_probes_and_export(tmp_path, graphs):
  from mjlab.rl import RslRlVecEnvWrapper
  from tensordict import TensorDict

  from wbc_mjlab.env.mdp.sampling import restore_sampling_state, sampling_state
  from wbc_mjlab.rl.evaluation import deterministic_cfg
  from wbc_mjlab.rl.policy import load_actor
  from wbc_mjlab.rl.runner import PolicyOnlyMotionTrackingRunner

  cfg, env = _env("Wbc-G1-Blink", num_envs=8, noise=False)
  runner = None
  try:
    agent = _small_agent("Wbc-G1-Blink")
    agent.light_checkpoints = True
    agent.save_interval = 1
    agent.actor.architecture = "motion_encoder"
    agent.actor.reference_residual = True
    a = agent.algorithm
    a.dynamics_mode, a.dynamics_loss_scale, a.velocity_blend = (
      "integrated",
      "increment",
      None,
    )
    a.model_optimizer = a.actor_optimizer = "muon"
    a.fused_optimizer, a.policy_ema, a.mirror_probability = True, 0.995, 0.5
    a.skip_nonfinite_updates = True
    a.critic_norm_eps, a.bound_critic = 0.01, True
    a.termination_mode = "hybrid"
    agent.clip_actions = 20.0
    a.actor_lr_multiplier, a.actor_lr_decay_start, a.actor_lr_decay_updates = 4.0, 1, 3
    a.actor_lr_return_threshold = 0.75
    a.recent_failure_share = 0.3
    a.probe_interval, a.probe_start, a.probe_num_envs = 1, 0, 2
    a.probe_tile_s, a.probe_lead_s, a.probe_steps_per_iteration = 0.04, (0.0, 0.0), 2
    a.diagnostic_interval = 0
    a.compile = a.compile_collection = graphs
    a.collection_cuda_graphs = a.probe_cuda_graphs = graphs
    a.probe_async = graphs
    wrapped = RslRlVecEnvWrapper(env)
    runner = PolicyOnlyMotionTrackingRunner(
      wrapped, asdict(agent), str(tmp_path), "cuda:0"
    )
    alg = runner.alg
    before = copy.deepcopy(alg.actor.state_dict())
    runner.learn(4)
    assert alg.num_updates > 0
    lightweight = torch.load(tmp_path / "model_0.pt", weights_only=False)
    assert lightweight["checkpoint_kind"] == "policy"
    with pytest.raises(ValueError, match="lightweight"):
      alg.load(lightweight, None, True)
    final = torch.load(tmp_path / "model_3.pt", weights_only=False)
    assert "world_model_state_dict" in final
    if graphs:
      assert alg._model_step._graph is not None
      assert alg._policy_step._graph is not None
      assert alg._collection_graph._graph is not None
    assert any(not torch.equal(v, before[k]) for k, v in alg.actor.state_dict().items())
    assert any(
      not torch.equal(x, y)
      for x, y in zip(alg.actor.parameters(), alg.ema_actor.parameters(), strict=True)
    )
    for x, y in zip(alg.actor.buffers(), alg.ema_actor.buffers(), strict=True):
      torch.testing.assert_close(x, y)
    assert torch.isfinite(alg._reward_ema)
    # Reflection is an involution in quaternion and joint coordinates and in DR context.
    state = alg.imagination.robot_state().vector()
    torch.testing.assert_close(alg.mirror.state(alg.mirror.state(state)), state)
    context = alg.imagination._context
    mirrored = context[:, alg.mirror.context_perm] * alg.mirror.context_sign
    torch.testing.assert_close(
      mirrored[:, alg.mirror.context_perm] * alg.mirror.context_sign, context
    )
    assert "reference_contact" in [n for n, _ in alg.imagination.analytic_rewards]
    assert "reference_contact" not in deterministic_cfg(copy.deepcopy(cfg)).rewards
    # Save/load live optimizer state separately from deployable EMA weights.
    saved = copy.deepcopy(alg.save())
    path = tmp_path / "resume.pt"
    torch.save(saved, path)
    actor = load_actor(env, asdict(agent), path, "cuda:0")
    obs = TensorDict(env.observation_manager.compute(), batch_size=[8])
    torch.testing.assert_close(actor(obs), alg.get_policy()(obs))
    alg.load(saved, None, True)
    for k, v in alg.actor.state_dict().items():
      torch.testing.assert_close(v, saved["actor_live_state_dict"][k])
    exported = alg.get_policy().as_onnx(verbose=False).cuda()
    torch.testing.assert_close(exported(obs["actor"]), actor(obs))
    assert (tmp_path / "params" / "policy.onnx").exists()
    c = alg.imagination.command
    c._probe_failures.fill_(2)
    snapshot = sampling_state(c)
    c._probe_failures.zero_()
    assert restore_sampling_state(c, snapshot, strict=True)
    assert c._probe_failures.sum() > 0
    # Probes actually run separately and preserve the training state.
    training_state = alg.imagination.robot_state().vector().clone()
    probes = alg.probes
    probes.next_update = 0
    probes.tick()
    probe_motion = probes.env.command_manager.get_term("motion").motion
    assert probe_motion.library is alg.imagination.command.motion.library
    assert probe_motion.kin is not alg.imagination.command.motion.kin
    torch.testing.assert_close(alg.imagination.robot_state().vector(), training_state)
    probes.state_dict()
  finally:
    if runner is not None:
      runner.alg.close()
    env.close()


@cuda
def test_nonfinite_gate_preserves_optimizer_state():
  from wbc_mjlab.rl.optim import MuonAdam, step_if_finite

  model = nn.Sequential(
    nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 8), nn.Tanh(), nn.Linear(8, 2)
  ).cuda()
  opt = MuonAdam(model, [model], 0.001, capturable=True)
  for p in model.parameters():
    p.grad = torch.ones_like(p)
  step_if_finite(opt, model.parameters(), enabled=True)
  params = [p.detach().clone() for p in model.parameters()]
  states = [
    v.clone()
    for sub in (opt.muon, opt.adam)
    for st in sub.state.values()
    for v in st.values()
    if isinstance(v, torch.Tensor)
  ]
  next(model.parameters()).grad.fill_(float("nan"))
  step_if_finite(opt, model.parameters(), enabled=True)
  for p, before in zip(model.parameters(), params, strict=True):
    torch.testing.assert_close(p, before, rtol=0, atol=0)
  after = [
    v
    for sub in (opt.muon, opt.adam)
    for st in sub.state.values()
    for v in st.values()
    if isinstance(v, torch.Tensor)
  ]
  for a, b in zip(after, states, strict=True):
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_blink_recipe_cli_and_studio_schema():
  from wbc_mjlab.rl.algos import rl_cfg
  from wbc_mjlab.scripts.train import _pop_recipe
  from wbc_mjlab.studio import recipes
  from wbc_mjlab.studio.presets import _set_path
  from wbc_mjlab.tasks import prepare_wbc_run

  prepare_wbc_run(task_id="Wbc-G1-Blink")
  settings = recipes.check_settings(
    "Wbc-G1-Blink", "wm", recipes.BUILTIN["wm-blink"]["settings"]
  )
  cfg = rl_cfg("Wbc-G1-Blink", "wm")
  for key, value in settings.items():
    _set_path(cfg, key, value)
  assert cfg.algorithm.velocity_blend is None
  assert cfg.algorithm.policy_ema == 0.995 and not cfg.light_checkpoints
  assert cfg.clip_actions == 20.0 and cfg.algorithm.bound_critic
  assert _pop_recipe(["--recipe=wm-blink", "--help"]) == (["--help"], "wm-blink")


def test_critic_targets_use_updated_normalizer_and_bounded_bootstrap():
  from wbc_mjlab.rl.wm.algorithm import WorldModelRL, lambda_returns
  from wbc_mjlab.rl.wm.model import RunningNorm

  alg = object.__new__(WorldModelRL)
  alg.device = "cpu"
  alg.gamma, alg.lam = 0.99, 0.95
  alg.value_upper_bound = 10.0
  alg.critic = nn.Linear(1, 1)
  with torch.no_grad():
    alg.critic.weight.fill_(2)
    alg.critic.bias.fill_(0.5)
  alg.critic_norm = RunningNorm(1, eps=0.01)
  alg.critic_optimizer = torch.optim.SGD(alg.critic.parameters(), lr=0)
  alg.critic_epochs = alg.critic_mini_batches = 1
  alg.max_grad_norm = 1.0
  alg.skip_nonfinite_updates = False
  x = torch.tensor([[[100.0], [120.0]], [[140.0], [160.0]], [[180.0], [200.0]]])
  rewards, ends, inside = (
    torch.ones(2, 2),
    torch.zeros(2, 2),
    torch.ones(2, 2, dtype=torch.bool),
  )
  stale = alg.critic(alg.critic_norm(x)).squeeze(-1).detach()
  loss = alg._update_critic(x, stale, rewards, ends, inside)
  fresh = alg._bootstrap_value(alg.critic(alg.critic_norm(x)).squeeze(-1)).detach()
  targets = lambda_returns(rewards, ends, fresh, inside, alg.gamma, alg.lam)
  expected = (alg.critic(alg.critic_norm(x[:-1])).squeeze(-1) - targets).square().mean()
  torch.testing.assert_close(loss, expected)
  stale_targets = lambda_returns(rewards, ends, stale, inside, alg.gamma, alg.lam)
  assert not torch.allclose(targets, stale_targets)
  value = torch.tensor([-2.0, 5.0, 1000.0], requires_grad=True)
  bounded = alg._bootstrap_value(value)
  torch.testing.assert_close(bounded, torch.tensor([-2.0, 5.0, 10.0]))
  bounded.sum().backward()
  torch.testing.assert_close(value.grad, torch.tensor([1.0, 1.0, 0.0]))


def test_contact_floors_batch_across_segments():
  from types import SimpleNamespace

  from wbc_mjlab.motion.library import contact_floors

  torch.manual_seed(1)
  heights = torch.randn(29, 3, 3)
  starts, ends = torch.tensor([0, 3, 20, 21]), torch.tensor([3, 20, 21, 29])
  calls = []

  def frame(ids):
    calls.append(ids.clone())
    return SimpleNamespace(body_pos_w=heights[ids])

  motion = SimpleNamespace(
    time_step_total=29, segment_start_idx=starts, segment_end_idx=ends, frame=frame
  )
  actual = contact_floors(motion, [0, 2], 0.05, batch_size=8)
  expected = torch.stack(
    [
      heights[a:b, [0, 2], 2].quantile(0.05, dim=0)
      for a, b in zip(starts, ends, strict=True)
    ]
  )
  torch.testing.assert_close(actual, expected)
  assert len(calls) == 4 and max(len(x) for x in calls) <= 8


def test_probe_batches_publish_without_waiting_for_library_sweep():
  from types import SimpleNamespace

  from wbc_mjlab.rl.wm.probes import CurriculumProbes

  probes = object.__new__(CurriculumProbes)
  probes.event = None
  probes.alg = SimpleNamespace(num_updates=100, device="cpu")
  probes.command = SimpleNamespace(
    cfg=SimpleNamespace(rsi=SimpleNamespace(probe_decay=0.7)),
    _probe_failures=torch.tensor([[10.0, 0.0]]),
  )
  probes.tiles = torch.arange(4)
  probes.cursor, probes.sweeps, probes.num_envs = 0, 0, 2
  probes.counts = torch.zeros(1, 2)
  probes.steps_per_iteration, probes.interval = 20, 125
  probes.active = probes.batch_active = True
  probes.left = 0
  probes.batch_counts = torch.tensor([[0.0, 2.0]])
  probes._tick()
  torch.testing.assert_close(probes.command._probe_failures, torch.tensor([[7.0, 2.0]]))
  assert probes.cursor == 2 and not probes.active and probes.next_update == 225
  probes.active = probes.batch_active = True
  probes.batch_counts = torch.tensor([[1.0, 0.0]])
  probes._tick()
  torch.testing.assert_close(probes.command._probe_failures, torch.tensor([[8.0, 2.0]]))
  assert probes.sweeps == 1 and probes.cursor == 0
