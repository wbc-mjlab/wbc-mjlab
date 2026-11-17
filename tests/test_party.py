"""PartyMimic additions: diversity-balanced resets, Muon weight decay, recorded replay
windows, the Party task, warm start, self-imitation and value-gradient consistency."""

from __future__ import annotations

import math
from dataclasses import asdict

import mjlab  # noqa: F401  (registers the tasks before wbc_mjlab.rl is imported)
import numpy as np
import pytest
import torch
from torch import nn

from wbc_mjlab.env.mdp.sampling import (
  bin_probabilities,
  diversity_weights,
  kmeans_labels,
)
from wbc_mjlab.rl.optim import make_optimizer
from wbc_mjlab.rl.wm.replay import Replay

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def test_kmeans_finds_separated_groups_and_weights_balance_them():
  torch.manual_seed(0)
  centers = torch.tensor([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
  sizes = (60, 30, 10)
  x = torch.cat(
    [c + 0.1 * torch.randn(n, 2) for c, n in zip(centers, sizes, strict=True)]
  )
  truth = torch.cat([torch.full((n,), i) for i, n in enumerate(sizes)])
  labels = kmeans_labels(x, 3)
  for i in range(3):  # each true group is one cluster, distinct from the others
    assert labels[truth == i].unique().numel() == 1
  assert labels.unique().numel() == 3

  w = diversity_weights(labels, tau=0.0)  # every cluster the same mass
  assert torch.allclose(w.sum(), torch.tensor(1.0))
  for i in range(3):
    assert torch.allclose(w[truth == i].sum(), torch.tensor(1 / 3))
  # tau = 1: mass by size, i.e. uniform over bins (the duration-weighted default)
  assert torch.allclose(diversity_weights(labels, tau=1.0), torch.full((100,), 0.01))


def test_uniform_share_follows_the_given_distribution():
  levels = torch.zeros(4)
  uniform = torch.tensor([0.7, 0.1, 0.1, 0.1])
  probs, softmax = bin_probabilities(
    levels, temperature_base=1.0, uniform_ratio=0.5, uniform=uniform
  )
  assert torch.allclose(probs, 0.5 * softmax + 0.5 * uniform)
  default, _ = bin_probabilities(levels, temperature_base=1.0, uniform_ratio=0.5)
  assert torch.allclose(default, torch.full((4,), 0.25))


def _mlp():
  return nn.Sequential(
    nn.Linear(3, 8), nn.ELU(), nn.Linear(8, 8), nn.ELU(), nn.Linear(8, 2)
  )


@pytest.mark.parametrize(
  "device,capturable",
  [("cpu", False)] + ([("cuda:0", True)] if torch.cuda.is_available() else []),
)
def test_muon_weight_decay_shrinks_matrices_only(device, capturable):
  """With zero gradients only the decay acts: Muon matrices shrink by 1 - lr·wd."""
  if not hasattr(torch.optim, "Muon"):
    pytest.skip("needs torch.optim.Muon")
  torch.manual_seed(0)
  mlp = _mlp().to(device)
  before = [p.detach().clone() for p in mlp.parameters()]
  lr, wd = 0.1, 0.5
  opt = make_optimizer(
    mlp,
    [mlp],
    kind="muon",
    lr=torch.tensor(lr, device=device) if capturable else lr,
    fused=capturable,
    capturable=capturable,
    weight_decay=wd,
  )
  for p in mlp.parameters():
    p.grad = torch.zeros_like(p)
  opt.step()
  hidden = mlp[2].weight
  assert torch.allclose(hidden, before[2] * (1 - lr * wd), atol=1e-6)
  # the first and last matrices and all biases are Adam's: zero gradient, no change
  for p, b in zip(mlp.parameters(), before, strict=True):
    if p is not hidden:
      assert torch.allclose(p, b, atol=1e-6)


def test_muon_io_puts_first_and_last_layers_on_muon():
  if not hasattr(torch.optim, "Muon"):
    pytest.skip("needs torch.optim.Muon")
  mlp = _mlp()
  muon = lambda opt: {id(p) for g in opt.muon.param_groups for p in g["params"]}  # noqa: E731
  hidden = muon(make_optimizer(mlp, [mlp], kind="muon", lr=0.1))
  every = muon(make_optimizer(mlp, [mlp], kind="muon", lr=0.1, muon_io=True))
  matrices = {id(m.weight) for m in mlp if isinstance(m, nn.Linear)}
  assert hidden == {id(mlp[2].weight)} and every == matrices
  with pytest.raises(ValueError, match="Muon only"):
    make_optimizer(mlp, [mlp], kind="adam", lr=torch.tensor(0.1), weight_decay=0.1)


def test_replay_forward_reads_recorded_continuations():
  r = Replay(8, 2, 1, 1, 0, 0, "cpu")
  for t in range(6):
    r.begin(state=torch.full((2, 1), float(t)), time_step=torch.full((2,), t))
    r.end(continues=torch.ones(2, dtype=torch.bool))
  w, held = r.forward(torch.tensor([1, 3]), torch.tensor([0, 1]), 3)
  assert w.time_step[:, 0].tolist() == [1, 2, 3]
  assert w.time_step[:, 1].tolist() == [3, 4, 5]
  assert held.tolist() == [True, True]
  _, held = r.forward(torch.tensor([4]), torch.tensor([0]), 3)
  assert held.tolist() == [False]  # would read past the newest row


def test_party_task_narrows_pose_kernels_only():
  from mjlab.tasks.registry import load_env_cfg

  from wbc_mjlab.presets.party import POSE_KERNELS
  from wbc_mjlab.tasks import prepare_wbc_run

  prepare_wbc_run(task_id="Wbc-G1-Party")
  party = load_env_cfg("Wbc-G1-Party")
  blink = load_env_cfg("Wbc-G1-Blink")
  for name, term in blink.rewards.items():
    std = term.params.get("std")
    if std is None:
      continue
    expected = std * 0.6 if name in POSE_KERNELS else std
    assert party.rewards[name].params["std"] == pytest.approx(expected), name
  assert party.commands["motion"].rsi.diversity_clusters == 24
  assert blink.commands["motion"].rsi.diversity_clusters == 0


def test_keypoint_features_ignore_where_the_motion_is_and_faces():
  from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul

  from wbc_mjlab.env.mdp.observations import keypoint_features

  torch.manual_seed(0)
  pos, quat = (
    torch.randn(4, 6, 3),
    torch.nn.functional.normalize(torch.randn(4, 6, 4), dim=-1),
  )
  anchor_pos = torch.randn(4, 3)
  anchor_quat = torch.nn.functional.normalize(torch.randn(4, 4), dim=-1)
  yaw = quat_from_euler_xyz(torch.zeros(4), torch.zeros(4), torch.full((4,), 1.3))
  shift = torch.tensor([2.0, -1.0, 0.5])

  def moved(p, q):  # yaw the whole scene about the origin, then shift it
    from mjlab.utils.lab_api.math import quat_apply

    y = yaw.unsqueeze(1).expand(q.shape).contiguous() if q.dim() == 3 else yaw
    return quat_apply(y, p) + shift, quat_mul(y, q.contiguous())

  a = keypoint_features(pos, quat, anchor_pos, anchor_quat)
  b = keypoint_features(*moved(pos, quat), *moved(anchor_pos, anchor_quat))
  assert a.shape == (4, 6 * 9)
  assert torch.allclose(a, b, atol=1e-5)


@cuda
def test_sparse5_terms_imagined_and_keypoint_skip(tmp_path):
  from mjlab.rl import RslRlVecEnvWrapper
  from test_wm import _env, _small_agent

  from wbc_mjlab.rl.policy import (
    KeypointSkipHead,
    _keypoint_skip_data,
    keypoint_skip_table,
  )
  from wbc_mjlab.rl.runner import PolicyOnlyMotionTrackingRunner
  from wbc_mjlab.rl.wm.imagined import Imagination, call_term

  _, env = _env("Wbc-G1-Sparse5", num_envs=8, noise=False)
  try:
    env.reset()
    om = env.observation_manager
    names = om._group_obs_term_names["actor"]
    assert "ref_keypoints" in names and "keypoint_error" in names
    assert "ref_joint_pos" not in names and "ref_joint_vel" not in names
    # position errors: robot minus reference keypoints, each in its own heading frame
    from wbc_mjlab.env.mdp.observations import heading_frame_poses

    terms = dict(zip(names, om._group_obs_term_cfgs["actor"], strict=True))
    params = terms["keypoint_error"].params
    error = terms["keypoint_error"].func(env, **params)
    c = env.command_manager.get_term("motion")
    ids = [list(c.cfg.body_names).index(n) for n in params["body_names"]]
    robot, _ = heading_frame_poses(
      c.robot_body_pos_w[:, ids],
      c.robot_body_quat_w[:, ids],
      c.robot_anchor_pos_w,
      c.robot_anchor_quat_w,
    )
    ref, _ = heading_frame_poses(
      c.body_pos_w[:, ids], c.body_quat_w[:, ids], c.anchor_pos_w, c.anchor_quat_w
    )
    n = len(ids)
    assert error.shape[1] == 6 * n
    assert torch.allclose(error[:, : 3 * n], (robot - ref).flatten(1), atol=1e-5)
    # deploy params describe every actor term's width
    from wbc_mjlab.export.tracking_params_yaml import build_wbc_tracking_params

    deploy = build_wbc_tracking_params(
      env.cfg, robot_id="g1", has_state_estimation=False
    )["actor_observations"]
    for name, shape, cfg in zip(
      names,
      om._group_obs_term_dim["actor"],
      om._group_obs_term_cfgs["actor"],
      strict=True,
    ):
      width = int(torch.tensor(shape).prod())
      assert deploy[name]["dim"] * max(cfg.history_length, 1) == width, name
    for _ in range(12):
      env.step(torch.randn(8, env.action_manager.total_action_dim, device="cuda:0"))
    im = Imagination(env)
    view = im.view(*im._probe())
    for name in ("ref_keypoints", "keypoint_error"):
      real = terms[name].func(env, **terms[name].params)
      assert torch.allclose(real, call_term(terms[name], view), atol=1e-4), name

    agent = _small_agent("Wbc-G1-Sparse5")
    agent.actor.keypoint_skip = True
    agent.actor.keypoint_skip_hidden = (64, 64)
    agent.actor.keypoint_skip_steps = 300
    agent.algorithm.horizon = 4
    wrapped = RslRlVecEnvWrapper(env)
    runner = PolicyOnlyMotionTrackingRunner(
      wrapped, asdict(agent), str(tmp_path), "cuda:0"
    )
    skip = runner.alg.actor.mlp
    assert isinstance(skip, KeypointSkipHead) and not bool(skip.kp2q.fitted)
    # the fit's goal is the observation the policy gets
    command = env.command_manager.get_term("motion")
    table = keypoint_skip_table(skip, command.motion)
    x, _ = _keypoint_skip_data(skip, table, command.time_steps)
    obs = env.observation_manager.compute_group("actor")
    assert torch.allclose(obs[:, skip.indices], x, atol=1e-4)
    runner.learn(num_learning_iterations=4)
    assert bool(skip.kp2q.fitted) and runner.alg.num_updates > 0
    x, y = _keypoint_skip_data(skip, table, command.time_steps)
    with torch.no_grad():
      fit = (skip.kp2q(x) - y).square().mean()
    assert fit < y.square().mean()  # better than the default pose
    with torch.no_grad():
      observations = wrapped.get_observations()
      expected = runner.get_inference_policy()(observations)
      exported = runner.alg.get_policy().as_onnx(verbose=False).to("cuda:0")
      assert torch.allclose(exported(observations["actor"]), expected, atol=1e-5)
  finally:
    env.close()


@cuda
def test_cross_attention_tokens_train_and_export(tmp_path):
  from mjlab.rl import RslRlVecEnvWrapper
  from test_wm import _env, _small_agent

  from wbc_mjlab.rl.policy import CrossAttentionDecoder, cross_attention_indices
  from wbc_mjlab.rl.runner import PolicyOnlyMotionTrackingRunner

  _, env = _env("Wbc-G1-Lookahead", num_envs=8, noise=False)
  try:
    env.reset()
    for _ in range(12):  # past the history length, so it holds distinct steps
      env.step(torch.randn(8, env.action_manager.total_action_dim, device="cuda:0"))
    history, frames, goal = cross_attention_indices(env)
    assert len(history) == 10 and len(frames) == 10
    # history tokens are the history observation terms step by step, oldest first
    obs = env.observation_manager.compute_group("actor")
    from wbc_mjlab.rl.wm.imagined import Imagination

    im = Imagination(env)
    assert torch.allclose(obs[:, torch.tensor(history)], im.env_history(), atol=1e-5)
    # frame f holds frame f of the joint lookahead (current frame first)
    om = env.observation_manager
    names = om._group_obs_term_names["actor"]
    widths = [int(torch.tensor(s).prod()) for s in om._group_obs_term_dim["actor"]]
    start = sum(widths[: names.index("ref_joint_pos")])
    joints = widths[names.index("ref_joint_pos")] // 10
    for f, row in enumerate(frames):
      block = range(start + f * joints, start + (f + 1) * joints)
      assert set(block) <= set(row) and set(row) <= set(goal)

    agent = _small_agent()
    agent.actor.architecture = "xattn"
    agent.actor.xattn_dim = 32
    agent.actor.xattn_heads = 4
    agent.actor.latent_aux = True
    agent.algorithm.latent_aux = 1.0
    agent.algorithm.actor_optimizer = "muon"
    agent.algorithm.actor_muon_io = True
    wrapped = RslRlVecEnvWrapper(env)
    runner = PolicyOnlyMotionTrackingRunner(
      wrapped, asdict(agent), str(tmp_path), "cuda:0"
    )
    alg = runner.alg
    assert isinstance(alg.actor.mlp, CrossAttentionDecoder)
    attention = alg.actor.mlp.encoder.cross_attention.q.weight
    assert any(
      p is attention for g in alg.actor_optimizer.muon.param_groups for p in g["params"]
    )
    before = [p.detach().clone() for p in alg.actor.parameters()]
    runner.learn(num_learning_iterations=4)
    assert alg.num_updates > 0
    assert any(
      not torch.equal(p, b) for p, b in zip(alg.actor.parameters(), before, strict=True)
    )
    assert alg.actor.mlp.aux.out_features == 10 * 29
    first = alg._update_policy()["latent_aux_rms"]
    for _ in range(20):
      last = alg._update_policy()["latent_aux_rms"]
    assert torch.isfinite(last) and last < first  # the head learns the goal's joints
    with torch.no_grad():
      observations = wrapped.get_observations()
      expected = runner.get_inference_policy()(observations)
      exported = alg.actor.as_onnx(verbose=False).to("cuda:0")
      assert torch.allclose(exported(observations["actor"]), expected, atol=1e-5)
    runner.export_policy_to_onnx(str(tmp_path), "xattn.onnx")
    assert (tmp_path / "xattn.onnx").stat().st_size > 0
  finally:
    env.close()


def _clip(seconds, joint, yaw=0.0, shift=(0.0, 0.0), fps=50.0, speed=0.0):
  t = torch.arange(int(seconds * fps), dtype=torch.float64) / fps
  q = torch.zeros(len(t), 7 + 3, dtype=torch.float64)
  c, s = math.cos(yaw), math.sin(yaw)
  q[:, 0] = shift[0] + c * speed * t
  q[:, 1] = shift[1] + s * speed * t
  q[:, 2] = 0.75
  q[:, 3], q[:, 6] = math.cos(yaw / 2), math.sin(yaw / 2)
  q[:, 7:] = joint + 0.1 * torch.sin(2 * t)[:, None]
  return q.numpy()


def test_motion_windows_ignore_heading_and_position():
  from wbc_mjlab.motion.diversity import WINDOW, clip_windows

  a = clip_windows(_clip(3.0, 0.2, speed=1.0), 50.0)
  b = clip_windows(_clip(3.0, 0.2, yaw=2.0, shift=(4.0, -1.0), speed=1.0), 50.0)
  assert a.shape == (5, WINDOW * (3 + 1 + 3 + 3 + 1))  # 3 s at 10 Hz, 1 s / 0.5 s
  assert np.allclose(a, b, atol=1e-5)
  assert not np.allclose(a, clip_windows(_clip(3.0, -0.5, speed=1.0), 50.0))
  assert clip_windows(_clip(0.5, 0.2), 50.0).shape[0] == 0


def test_diversity_selection_fills_rare_motions_first():
  from wbc_mjlab.motion.diversity import analyze_windows, clip_windows, select_clips

  common = [_clip(4.0, 0.2 + 0.001 * i) for i in range(8)]
  rare = [_clip(4.0, -1.0), _clip(4.0, 1.5)]
  windows = [clip_windows(q, 50.0) for q in common + rare]
  count = np.array([len(w) for w in windows])
  x = np.concatenate(windows)
  seconds = np.full(len(count), 4.0)
  order = select_clips(x, count, seconds, num_clips=3, clusters=6, device="cpu")
  assert len(order) == 3 and len(set(order)) == 3
  assert {8, 9} <= set(order)  # both rare motions, one of the common
  assert (
    len(select_clips(x, count, seconds, hours=8.0 / 3600, clusters=6, device="cpu"))
    == 2
  )
  report = analyze_windows(x, count, clusters=6, heldout=windows[8], device="cpu")
  assert report["heldout_coverage"] >= 0.8
  assert report["rarity"][8] > report["rarity"][0]
  unseen = clip_windows(_clip(4.0, 4.0), 50.0)
  far = analyze_windows(x, count, clusters=6, heldout=unseen, device="cpu")
  assert far["heldout_coverage"] == 0.0 and far["heldout_median_distance_ratio"] > 1
  assert 1.0 <= report["effective_clusters"] <= report["occupied_clusters"]


def test_bf16_autocast_keeps_parameters_copies_and_cpu_float32():
  import copy

  from wbc_mjlab.rl.precision import autocast_bf16

  torch.manual_seed(0)
  net = nn.Sequential(nn.Linear(8, 64), nn.SiLU(), nn.Linear(64, 3))
  keys = list(net.state_dict())
  x = torch.randn(5, 8)
  plain = net(x)
  autocast_bf16(autocast_bf16(net))
  assert list(net.state_dict()) == keys and isinstance(net, nn.Sequential)
  assert torch.equal(net(x), plain)  # CPU inputs (ONNX export) stay float32
  copied = copy.deepcopy(net)
  assert type(copied) is type(net)
  if torch.cuda.is_available():
    net, copied, x = net.cuda(), copied.cuda(), x.cuda()
    out = net(x)
    assert out.dtype == torch.float32
    with torch.autocast("cuda", dtype=torch.bfloat16):
      reference = nn.Sequential.forward(net, x).float()
    assert torch.equal(out, reference)
    assert not torch.equal(out, nn.Sequential.forward(net, x))
    copied[0].weight.data.zero_()
    assert torch.equal(net(x), out)  # the copy runs its own weights


@cuda
@pytest.mark.parametrize("compile", [False, pytest.param(True, marks=pytest.mark.slow)])
def test_self_imitation_value_gradient_and_warm_start(tmp_path, compile):
  from mjlab.rl import RslRlVecEnvWrapper
  from test_wm import _env, _small_agent

  from wbc_mjlab.rl.runner import PolicyOnlyMotionTrackingRunner

  _, env = _env("Wbc-G1-Lookahead", num_envs=16, noise=False)
  try:
    agent = _small_agent()
    agent.algorithm.compile = compile  # with CUDA graphs of the updates
    agent.algorithm.cuda_graphs = compile
    agent.algorithm.horizon = 10  # covers the 10-step observation history
    agent.algorithm.self_imitation = 1.0
    agent.algorithm.value_gradient = 0.5
    agent.algorithm.value_gradient_states = 64
    agent.algorithm.muon_weight_decay = 0.1
    agent.algorithm.actor_muon_io = True
    agent.algorithm.replay_steps = 64
    agent.algorithm.warmup_steps = 24
    agent.algorithm.bf16 = True
    wrapped = RslRlVecEnvWrapper(env)
    runner = PolicyOnlyMotionTrackingRunner(
      wrapped, asdict(agent), str(tmp_path), "cuda:0"
    )
    runner.learn(num_learning_iterations=6)  # 5 env steps each: past the warmup
    alg = runner.alg
    assert alg.num_updates > 0
    stats = alg._update_world_model() | alg._update_policy()
    for key in (
      "self_imitation_share",
      "self_imitation_loss",
      "value_gradient_error",
      "value_gradient_cosine",
    ):
      assert key in stats and torch.isfinite(torch.as_tensor(stats[key])), key
    assert 0 <= float(stats["self_imitation_share"]) <= 1
    assert -1 <= float(stats["value_gradient_cosine"]) <= 1
    checkpoint = tmp_path / "trained.pt"
    runner.save(str(checkpoint))

    fresh = PolicyOnlyMotionTrackingRunner(wrapped, asdict(agent), None, "cuda:0")
    fresh.warm_start(str(checkpoint))
    for a, b in zip(fresh.alg.actor.parameters(), alg.actor.parameters(), strict=True):
      assert torch.equal(a, b)
    for a, b in zip(
      fresh.alg.critic.parameters(), alg.critic.parameters(), strict=True
    ):
      assert torch.equal(a, b)
    assert fresh.alg.num_updates == 0 and fresh.current_learning_iteration == 0
    fresh.current_learning_iteration = 5
    with pytest.raises(ValueError, match="resume"):
      fresh.warm_start(str(checkpoint))
  finally:
    env.close()
