"""Optional goal encoder while retaining rsl_rl's observation/export interface.

SuperMimic, Appendix C/Table A.4: https://arxiv.org/abs/2610.28480.
The bounded motion latent and separate decoder follow the design in
CuboiLeo/SuperMimic-Reproduced (ba61053), src/supermimic/policy/nets.py:
https://github.com/CuboiLeo/SuperMimic-Reproduced/blob/ba6105349069bf040f1fdad60873d13138d505a2/src/supermimic/policy/nets.py
Observation indices come from the preset, rather than a fixed 680/930-d layout.
"""

from dataclasses import dataclass, field, fields
from typing import Literal

import torch
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg
from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP
from torch import nn


@dataclass
class PolicyCfg(RslRlModelCfg):
  """Policy architecture, independent of the optimizer used to train it."""

  activation: Literal[
    "elu",
    "selu",
    "relu",
    "crelu",
    "lrelu",
    "tanh",
    "sigmoid",
    "softplus",
    "gelu",
    "swish",
    "silu",
    "mish",
    "identity",
  ] = "elu"

  reference_residual: bool = field(
    default=False,
    metadata={
      "setting": True,
      "checkpoint": True,
      "help": "add the next reference joint target inside the policy; requires unscaled lookahead joint observations and default-relative actions",
    },
  )

  keypoint_skip: bool = field(
    default=False,
    metadata={
      "setting": True,
      "checkpoint": True,
      "help": "add a joint target estimated from the keypoint reference (ref_keypoints) inside the policy; fit to the motions before training",
    },
  )
  keypoint_skip_hidden: tuple[int, ...] = field(
    default=(1024, 1024, 512),
    metadata={
      "setting": True,
      "checkpoint": True,
      "visible_if": {"actor.keypoint_skip": True},
    },
  )
  keypoint_skip_steps: int = field(
    default=20000,
    metadata={
      "setting": True,
      "visible_if": {"actor.keypoint_skip": True},
      "help": "Adam steps (batch 8192) of the keypoint skip's fit",
    },
  )

  architecture: Literal["mlp", "motion_encoder", "xattn"] = field(
    default="mlp",
    metadata={
      "setting": True,
      "checkpoint": True,
      "help": "flat MLP, a bounded motion latent with a proprioceptive decoder, or the latent of the observation history attending to the reference frames (xattn)",
    },
  )
  xattn_dim: int = field(
    default=128,
    metadata={
      "setting": True,
      "checkpoint": True,
      "visible_if": {"actor.architecture": "xattn"},
      "help": "width of the cross-attention encoder's tokens and latent",
    },
  )
  xattn_heads: int = field(
    default=4,
    metadata={
      "setting": True,
      "checkpoint": True,
      "visible_if": {"actor.architecture": "xattn"},
    },
  )
  encoder_hidden_dims: tuple[int, ...] = field(
    default=(2048, 1024, 512, 256),
    metadata={
      "setting": True,
      "checkpoint": True,
      "visible_if": {"actor.architecture": "motion_encoder"},
    },
  )
  latent_dim: int = field(
    default=32,
    metadata={
      "setting": True,
      "checkpoint": True,
      "visible_if": {"actor.architecture": "motion_encoder"},
    },
  )
  latent_aux: bool = field(
    default=False,
    metadata={
      "setting": True,
      "checkpoint": True,
      "visible_if": {"actor.architecture": ["motion_encoder", "xattn"]},
      "help": "a linear head predicting the goal frames' joint positions from the reference latent (trained with algorithm.latent_aux)",
    },
  )
  goal_terms: tuple[str, ...] = field(
    default=(),
    metadata={
      "setting": True,
      "checkpoint": True,
      "visible_if": {"actor.architecture": ["motion_encoder", "xattn"]},
      "help": "actor term names to encode; empty selects reference/lookahead terms",
    },
  )


@dataclass
class TrackingRunnerCfg(RslRlOnPolicyRunnerCfg):
  clip_actions: float | None = field(
    default=None, metadata={"nullable": True, "kind": "number"}
  )
  light_checkpoints: bool = False
  """Periodic actor-only checkpoints; the final checkpoint always includes training state."""
  init_checkpoint: str | None = None
  """Warm start (fine-tuning, e.g. on a few hard clips): networks from this full checkpoint,
  with fresh optimizers, schedules, iteration count and RSI statistics (PartyMimic ``--init``)."""
  actor: PolicyCfg = field(default_factory=PolicyCfg)


def tracking_runner_cfg(cfg):
  """Extend the upstream runner's actor settings without changing their defaults."""
  values = {f.name: getattr(cfg, f.name) for f in fields(RslRlOnPolicyRunnerCfg)}
  actor = values["actor"]
  values["actor"] = PolicyCfg(**{f.name: getattr(actor, f.name) for f in fields(actor)})
  return TrackingRunnerCfg(**values)


# Read old run configs/recipes without maintaining a second set of public knobs.
OLD_ACTOR_KEYS = {
  "policy_architecture": "architecture",
  "encoder_hidden_dims": "encoder_hidden_dims",
  "latent_dim": "latent_dim",
  "goal_terms": "goal_terms",
}


def migrate_actor_settings(values):
  """Return run/recipe settings with pre-refactor algorithm-owned policy keys moved."""
  import copy

  values = copy.deepcopy(values)
  actor = values.setdefault("actor", {})
  algorithm = values.get("algorithm", {})
  for old, new in OLD_ACTOR_KEYS.items():
    if old in algorithm:
      actor.setdefault(new, algorithm.pop(old))
  return values


class MotionDecoder(nn.Module):
  """An MLPModel's head: normalized flat observation → encoded goal + proprioception."""

  def __init__(
    self,
    obs_dim,
    goal_indices,
    output_dim,
    encoder_hidden,
    decoder_hidden,
    latent_dim=32,
    activation="silu",
    aux_dim=0,
  ):
    super().__init__()
    goal = set(goal_indices)
    if not goal or len(goal) >= obs_dim:
      raise ValueError(
        "motion encoder needs goal terms and non-goal actor observations"
      )
    self.aux = nn.Linear(latent_dim, aux_dim) if aux_dim else None
    self.register_buffer("goal_indices", torch.tensor(goal_indices, dtype=torch.long))
    self.register_buffer(
      "proprio_indices", torch.tensor([i for i in range(obs_dim) if i not in goal])
    )
    self.encoder = MLP(len(goal_indices), latent_dim, encoder_hidden, activation)
    self.decoder = MLP(
      obs_dim - len(goal_indices) + latent_dim, output_dim, decoder_hidden, activation
    )
    # Small initial actions, as in the reproduced policy implementation.
    nn.init.orthogonal_(self.decoder[-1].weight, gain=0.01)
    nn.init.zeros_(self.decoder[-1].bias)

  def latent(self, obs: torch.Tensor) -> torch.Tensor:
    return torch.tanh(self.encoder(obs[..., self.goal_indices]))

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    z = self.latent(obs)
    return self.decoder(torch.cat([z, obs[..., self.proprio_indices]], dim=-1))


def _silu_mlp(inp, hidden, out):
  layers = []
  for width in hidden:
    layers += [nn.Linear(inp, width), nn.SiLU()]
    inp = width
  return nn.Sequential(*layers, nn.Linear(inp, out))


def _sinusoid(n, d):
  pos = torch.arange(n, dtype=torch.float32).unsqueeze(1)
  angle = pos / 10000 ** (torch.arange(0, d, 2, dtype=torch.float32) / d)
  return torch.stack((angle.sin(), angle.cos()), -1).flatten(1)


class _Attention(nn.Module):
  """Multi-head attention of Linear projections (matrices Muon can take)."""

  def __init__(self, d, heads):
    super().__init__()
    self.heads = heads
    self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))

  def forward(self, x, y, causal=False):
    b, n, d = x.shape

    def split(t):
      return t.reshape(b, -1, self.heads, d // self.heads).transpose(1, 2)

    out = nn.functional.scaled_dot_product_attention(
      split(self.q(x)), split(self.k(y)), split(self.v(y)), is_causal=causal
    )
    return self.o(out.transpose(1, 2).reshape(b, n, d))


class CrossAttentionEncoder(nn.Module):
  """RGMT's dynamics-conditioned command encoder, as in PartyMimic (Roboparty, 2026)
  networks.py ``XAttnEncoder``, MIT; see THIRD_PARTY_NOTICES.

  A causal pre-LN Transformer block over the observation history tokens gives the
  dynamics embedding (its newest token); a query from it attends to the reference frame
  tokens (MLP + sinusoidal position) in one pre-LN cross-attention block.
  """

  def __init__(self, history_dim, frame_dim, frames, history, d=128, heads=4):
    super().__init__()
    if d % heads or d % 2:
      raise ValueError("xattn_dim must be even and divisible by xattn_heads")
    self.history_in = _silu_mlp(history_dim, [d], d)
    self.register_buffer("history_pos", _sinusoid(history, d))
    self.ln_h1, self.self_attention = nn.LayerNorm(d), _Attention(d, heads)
    self.ln_h2, self.ff_h = nn.LayerNorm(d), _silu_mlp(d, [2 * d], d)
    self.query = _silu_mlp(d, [d], d)
    self.frame_in = _silu_mlp(frame_dim, [d], d)
    self.register_buffer("frame_pos", _sinusoid(frames, d))
    self.ln_q, self.cross_attention = nn.LayerNorm(d), _Attention(d, heads)
    self.ln_s, self.ff = nn.LayerNorm(d), _silu_mlp(d, [2 * d], d)
    self.ln_out = nn.LayerNorm(d)

  def forward(self, history, frames):
    x = self.history_in(history) + self.history_pos
    h = self.ln_h1(x)
    x = x + self.self_attention(h, h, causal=True)
    x = x + self.ff_h(self.ln_h2(x))
    q = self.query(x[:, -1:])
    z = self.frame_in(frames) + self.frame_pos
    s = q + self.cross_attention(self.ln_q(q), z)
    s = s + self.ff(self.ln_s(s))
    return self.ln_out(s).squeeze(1)


class CrossAttentionDecoder(nn.Module):
  """An MLPModel's head: normalized flat observation → cross-attention latent of its
  history and reference tokens + proprioception → action."""

  def __init__(
    self,
    obs_dim,
    history_index,
    frame_index,
    goal_indices,
    output_dim,
    decoder_hidden,
    activation,
    d=128,
    heads=4,
    aux_dim=0,
  ):
    super().__init__()
    goal = set(goal_indices)
    self.aux = nn.Linear(d, aux_dim) if aux_dim else None
    self.register_buffer("history_index", torch.tensor(history_index))
    self.register_buffer("frame_index", torch.tensor(frame_index))
    self.register_buffer(
      "proprio_indices", torch.tensor([i for i in range(obs_dim) if i not in goal])
    )
    self.encoder = CrossAttentionEncoder(
      self.history_index.shape[1],
      self.frame_index.shape[1],
      self.frame_index.shape[0],
      self.history_index.shape[0],
      d,
      heads,
    )
    self.decoder = MLP(obs_dim - len(goal) + d, output_dim, decoder_hidden, activation)
    nn.init.orthogonal_(self.decoder[-1].weight, gain=0.01)
    nn.init.zeros_(self.decoder[-1].bias)

  def latent(self, obs: torch.Tensor) -> torch.Tensor:
    flat = obs.reshape(-1, obs.shape[-1])
    u = self.encoder(flat[:, self.history_index], flat[:, self.frame_index])
    return u.reshape(*obs.shape[:-1], -1)

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    flat = obs.reshape(-1, obs.shape[-1])
    u = self.latent(flat)
    out = self.decoder(torch.cat([u, flat[:, self.proprio_indices]], dim=-1))
    return out.reshape(*obs.shape[:-1], -1)


def cross_attention_indices(env, names=()):
  """Observation indices of the history tokens ``[history, d_h]`` (non-goal terms, oldest
  first) and reference frame tokens ``[frames, d_g]`` (goal terms, current frame first),
  and the goal indices. Terms without history (frames) repeat in every token."""
  from wbc_mjlab.env.mdp.observations import takes_frames

  om = env.observation_manager
  term_names = om._group_obs_term_names["actor"]
  configs = om._group_obs_term_cfgs["actor"]
  goal = set(goal_term_names(dict(zip(term_names, configs, strict=True)), names))
  blocks, offset = [], 0
  for name, cfg, shape in zip(
    term_names, configs, om._group_obs_term_dim["actor"], strict=True
  ):
    width = int(torch.tensor(shape).prod())
    if cfg.history_length > 1 and not cfg.flatten_history_dim:
      raise ValueError("xattn needs flattened observation histories")
    steps = max(cfg.history_length, 1)
    frames = int(cfg.params.get("frames", 1)) if takes_frames(cfg) else 1
    if name in goal and steps > 1 and frames > 1:
      raise ValueError(f"xattn goal term {name} has both history and lookahead")
    parts = (frames if steps == 1 else 1) if name in goal else steps
    blocks.append((name in goal, offset, width, parts))
    offset += width

  def tokens(of_goal):
    entries = [(o, w, p) for g, o, w, p in blocks if g == of_goal]
    count = max(p for _, _, p in entries)
    rows = [[] for _ in range(count)]
    for start, width, parts in entries:
      if parts not in (1, count):
        kind = "lookahead frames" if of_goal else "history lengths"
        raise ValueError(f"xattn needs equal {kind} (or none) across terms")
      for i, row in enumerate(rows):
        if parts == 1:
          row.extend(range(start, start + width))
        else:
          size = width // count
          row.extend(range(start + i * size, start + (i + 1) * size))
    return rows

  goal_indices = [i for g, o, w, _ in blocks if g for i in range(o, o + w)]
  return tokens(False), tokens(True), goal_indices


def goal_term_names(terms, names=()) -> list[str]:
  """Resolve goal names from configured terms, for launch validation and the model."""
  active = {name: cfg for name, cfg in terms.items() if cfg is not None}
  if set(names) - set(active):
    raise ValueError(f"unknown encoder goal terms: {set(names) - set(active)}")
  selected = [
    name
    for name, cfg in active.items()
    if name in names
    or (not names and (name.startswith("ref_") or cfg.params.get("frames", 1) > 1))
  ]
  if not selected or len(selected) == len(active):
    raise ValueError("motion encoder needs goal terms and non-goal actor observations")
  return selected


def goal_frames(env, names=()) -> int:
  """Reference frames the goal terms look ahead (the most of any)."""
  from wbc_mjlab.env.mdp.observations import takes_frames

  om = env.observation_manager
  terms = dict(
    zip(
      om._group_obs_term_names["actor"], om._group_obs_term_cfgs["actor"], strict=True
    )
  )
  return max(
    int(terms[n].params.get("frames", 1)) if takes_frames(terms[n]) else 1
    for n in goal_term_names(terms, names)
  )


def goal_observation_indices(env, names=()) -> list[int]:
  """Resolve complete term blocks, including their configured history, in actor order."""
  om = env.observation_manager
  term_names = om._group_obs_term_names["actor"]
  configs = om._group_obs_term_cfgs["actor"]
  selected = goal_term_names(dict(zip(term_names, configs, strict=True)), names)
  print(f"[INFO] Motion encoder goal terms: {', '.join(selected)}")
  indices, offset = [], 0
  for name, shape in zip(term_names, om._group_obs_term_dim["actor"], strict=True):
    width = int(torch.tensor(shape).prod())
    if name in selected:
      indices.extend(range(offset, offset + width))
    offset += width
  return indices


def actor_options(env, cfg):
  """Resolve the actor once for the upstream PPO runner, WM and actor-only inference."""
  options = dict(cfg["actor"])
  if options.get("activation") == "silu":
    options["activation"] = "swish"  # rsl_rl names torch.nn.SiLU swish
  for key in ("cnn_cfg", "distribution_cfg"):
    if options.get(key) is None:
      options.pop(key, None)
  if options.get("rnn_type") is None:
    for key in ("rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
      options.pop(key, None)
  if isinstance(options.get("distribution_cfg"), dict):
    options["distribution_cfg"] = dict(options["distribution_cfg"])
  algorithm = cfg["algorithm"]
  # PPO distribution config is retained by the evaluator; wm uses fixed exploration.
  if algorithm.get("class_name") == "wbc_mjlab.rl.wm.algorithm:WorldModelRL":
    options["distribution_cfg"] = {
      "class_name": "GaussianDistribution",
      "init_std": algorithm.get("action_noise", 0.1),
      "std_type": "scalar",
      "learn_std": False,
    }
  if any(
    k in options for k in ("goal_indices", "reference_skip", "history_index")
  ) or isinstance(options.get("keypoint_skip"), dict):
    return options  # already resolved by the common runner
  residual = options.pop("reference_residual", False)
  keypoint = options.pop("keypoint_skip", False)
  keypoint_hidden = options.pop("keypoint_skip_hidden", (1024, 1024, 512))
  options.pop("keypoint_skip_steps", None)
  if residual and keypoint:
    raise ValueError("choose the reference residual or the keypoint skip, not both")
  if residual:
    options["reference_skip"] = reference_skip_options(env)
  if keypoint:
    options["keypoint_skip"] = keypoint_skip_options(env, keypoint_hidden)
  residual = residual or keypoint
  architecture = options.pop("architecture", "mlp")
  encoder_hidden = options.pop("encoder_hidden_dims", (2048, 1024, 512, 256))
  latent = options.pop("latent_dim", 32)
  names = options.pop("goal_terms", ())
  xattn_dim = options.pop("xattn_dim", 128)
  xattn_heads = options.pop("xattn_heads", 4)
  latent_aux = options.pop("latent_aux", False)
  if latent_aux and architecture == "mlp":
    raise ValueError("latent_aux needs an encoder architecture (motion_encoder, xattn)")
  if latent_aux:
    options["aux_dim"] = goal_frames(env, names) * env.action_manager.total_action_dim
  if architecture == "xattn":
    history_index, frame_index, goal_indices = cross_attention_indices(env, names)
    print(
      f"[INFO] Cross-attention tokens: {len(history_index)} history × "
      f"{len(history_index[0])}, {len(frame_index)} frames × {len(frame_index[0])}"
    )
    options.update(
      class_name="wbc_mjlab.rl.policy:CrossAttentionModel",
      history_index=history_index,
      frame_index=frame_index,
      goal_indices=goal_indices,
      xattn_dim=xattn_dim,
      xattn_heads=xattn_heads,
    )
  elif architecture == "motion_encoder":
    options.update(
      class_name="wbc_mjlab.rl.policy:MotionEncoderModel",
      goal_indices=goal_observation_indices(env, names),
      encoder_hidden_dims=encoder_hidden,
      latent_dim=latent,
    )
  elif architecture == "mlp" and residual:
    options["class_name"] = "wbc_mjlab.rl.policy:ReferenceResidualModel"
  elif architecture != "mlp":
    raise ValueError(f"unknown policy architecture: {architecture}")
  return options


class MotionEncoderModel(MLPModel):
  """A standard rsl_rl actor with an encoded goal head; export remains inherited."""

  def __init__(
    self,
    obs,
    obs_groups,
    obs_set,
    output_dim,
    *,
    goal_indices,
    encoder_hidden_dims=(2048, 1024, 512, 256),
    latent_dim=32,
    reference_skip=None,
    keypoint_skip=None,
    aux_dim=0,
    **options,
  ):
    super().__init__(obs, obs_groups, obs_set, output_dim, **options)
    self.mlp = MotionDecoder(
      self.obs_dim,
      goal_indices,
      output_dim,
      encoder_hidden_dims,
      options.get("hidden_dims", (256, 256, 256)),
      latent_dim,
      options.get("activation", "elu"),
      aux_dim,
    )
    _add_skips(self, output_dim, reference_skip, keypoint_skip)


class CrossAttentionModel(MLPModel):
  """A standard rsl_rl actor with the cross-attention head; export remains inherited."""

  def __init__(
    self,
    obs,
    obs_groups,
    obs_set,
    output_dim,
    *,
    history_index,
    frame_index,
    goal_indices,
    xattn_dim=128,
    xattn_heads=4,
    reference_skip=None,
    keypoint_skip=None,
    aux_dim=0,
    **options,
  ):
    super().__init__(obs, obs_groups, obs_set, output_dim, **options)
    self.mlp = CrossAttentionDecoder(
      self.obs_dim,
      history_index,
      frame_index,
      goal_indices,
      output_dim,
      options.get("hidden_dims", (256, 256, 256)),
      options.get("activation", "elu"),
      xattn_dim,
      xattn_heads,
      aux_dim,
    )
    _add_skips(self, output_dim, reference_skip, keypoint_skip)


def _skip_action_contract(env):
  """Tracked joint ids and the action's per-joint offset and scale, for skips that add a
  reference joint target to the policy's action."""
  action = env.action_manager.get_term("joint_pos")
  command = env.command_manager.get_term("motion")
  if getattr(action, "reference_relative", False) or action.cfg.clip is not None:
    raise ValueError("reference skip needs unclipped default-relative joint actions")
  ids = action._target_ids.tolist()
  if ids != (
    command._tracked_joint_ids.tolist()
    if command._tracked_joint_ids is not None
    else list(range(len(command.robot.joint_names)))
  ):
    raise ValueError("reference skip needs tracked joints in action order")

  def vector(value):
    value = torch.as_tensor(value, device=env.device).expand(env.num_envs, len(ids))
    if not torch.allclose(value, value[:1].expand_as(value)):
      raise ValueError(
        "reference skip needs the same action scale/offset in every environment"
      )
    return value[0].tolist()

  scale = vector(action._scale)
  if any(v == 0 for v in scale):
    raise ValueError("reference skip action scale must be nonzero")
  return ids, vector(action._offset), scale


def reference_skip_options(env):
  """Resolve Blink's next-frame skip from the actual observation/action contract.

  BlinkMimic d264658, networks.Policy; SuperMimic arXiv:2610.28480.
  https://github.com/fuyichen0725/blinkmimic/blob/d264658d6c9ce61fb52f9f0f8985a369f3ccc409/src/blinkmimic/networks.py
  """
  from wbc_mjlab.env.mdp.observations import ref_joint_pos

  ids, offset, scale = _skip_action_contract(env)
  om = env.observation_manager
  start, indices = 0, None
  for cfg, shape in zip(
    om._group_obs_term_cfgs["actor"], om._group_obs_term_dim["actor"], strict=True
  ):
    width = int(torch.tensor(shape).prod())
    if cfg.func is ref_joint_pos:
      if (
        cfg.params.get("frames", 1) < 2
        or cfg.params.get("stride", 1) != 1
        or cfg.noise is not None
        or cfg.scale is not None
        or cfg.clip is not None
        or cfg.history_length > 1
      ):
        raise ValueError(
          "reference skip needs clean, unscaled joint lookahead including frame +1"
        )
      indices = list(range(start + len(ids), start + 2 * len(ids)))
    start += width
  if indices is None:
    raise ValueError("reference skip requires the ref_joint_pos lookahead observation")
  return dict(indices=indices, offset=offset, scale=scale)


def _raw(obs, normalizer, indices):
  """``obs[..., indices]`` with the observation normalization undone."""
  x = obs[..., indices]
  if hasattr(normalizer, "_mean"):
    x = (
      x * (normalizer._std[..., indices] + normalizer.eps)
      + normalizer._mean[..., indices]
    )
  return x


class ReferenceResidualHead(nn.Module):
  """Exportable skip around any shared policy head; undo observation normalization."""

  def __init__(self, head, normalizer, indices, offset, scale):
    super().__init__()
    self.head, self.normalizer = head, normalizer
    self.register_buffer("indices", torch.tensor(indices, dtype=torch.long))
    self.register_buffer("offset", torch.tensor(offset))
    self.register_buffer("scale", torch.tensor(scale))

  def forward(self, obs):
    ref = _raw(obs, self.normalizer, self.indices)
    return self.head(obs) + (ref - self.offset) / self.scale


class FrozenMLP(nn.Module):
  """An ELU MLP with its input standardization, held as buffers: no optimizer, EMA or
  Muon grouping trains it, and it is saved and exported with the module owning it.

  PartyMimic (Roboparty, 2026) networks.py ``FrozenMLP``, MIT; see THIRD_PARTY_NOTICES.
  """

  def __init__(self, inp, hidden, out):
    super().__init__()
    dims = [inp, *hidden, out]
    self.layers = len(dims) - 1
    for i, (a, b) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
      self.register_buffer(f"w{i}", torch.zeros(b, a))
      self.register_buffer(f"b{i}", torch.zeros(b))
    self.register_buffer("mu", torch.zeros(inp))
    self.register_buffer("sd", torch.ones(inp))
    self.register_buffer("fitted", torch.zeros((), dtype=torch.bool))

  def forward(self, x):
    x = (x - self.mu) / self.sd
    for i in range(self.layers):
      x = nn.functional.linear(x, getattr(self, f"w{i}"), getattr(self, f"b{i}"))
      if i < self.layers - 1:
        x = nn.functional.elu(x)
    return x

  @torch.no_grad()
  def load_from(self, net: nn.Sequential, mu, sd):
    linears = [m for m in net if isinstance(m, nn.Linear)]
    assert len(linears) == self.layers
    for i, m in enumerate(linears):
      getattr(self, f"w{i}").copy_(m.weight)
      getattr(self, f"b{i}").copy_(m.bias)
    self.mu.copy_(mu)
    self.sd.copy_(sd)
    self.fitted.fill_(True)


class KeypointSkipHead(nn.Module):
  """Blink's reference skip with an estimated reference: ``a = head(obs) + kp2q(goal)``,
  kp2q a frozen MLP from the raw keypoint goal to the next frame's reference joint target
  (in action units), fit to the motion library before training (:func:`fit_keypoint_skips`).

  PartyMimic (Roboparty, 2026) ``--kp_skip``, networks.py/train.py, MIT; see
  THIRD_PARTY_NOTICES.
  """

  def __init__(
    self,
    head,
    normalizer,
    output_dim,
    indices,
    hidden,
    body_indexes,
    anchor_index,
    frames,
    stride,
    joint_ids,
    offset,
    scale,
  ):
    super().__init__()
    self.head, self.normalizer = head, normalizer
    self.register_buffer("indices", torch.tensor(indices, dtype=torch.long))
    self.kp2q = FrozenMLP(len(indices), hidden, output_dim)
    self.hidden, self.output_dim = tuple(hidden), int(output_dim)
    # How the goal is computed from reference frames: for the fit only.
    self.body_indexes, self.anchor_index = list(body_indexes), int(anchor_index)
    self.frames, self.stride = int(frames), int(stride)
    self.joint_ids = None if joint_ids is None else list(joint_ids)
    self.offset, self.scale = list(offset), list(scale)

  def forward(self, obs):
    return self.head(obs) + self.kp2q(_raw(obs, self.normalizer, self.indices))


def keypoint_skip_options(env, hidden):
  """Resolve the keypoint skip from the ``ref_keypoints`` observation and the action."""
  from wbc_mjlab.env.mdp.observations import ref_keypoints

  ids, offset, scale = _skip_action_contract(env)
  command = env.command_manager.get_term("motion")
  om = env.observation_manager
  offset_obs, found = 0, None
  for cfg, shape in zip(
    om._group_obs_term_cfgs["actor"], om._group_obs_term_dim["actor"], strict=True
  ):
    width = int(torch.tensor(shape).prod())
    if cfg.func is ref_keypoints:
      if cfg.noise is not None or cfg.scale is not None or cfg.history_length > 1:
        raise ValueError("keypoint skip needs a clean ref_keypoints without history")
      found = (list(range(offset_obs, offset_obs + width)), cfg.params)
    offset_obs += width
  if found is None:
    raise ValueError("keypoint skip requires the ref_keypoints observation")
  indices, params = found
  names = list(command.cfg.body_names)
  tracked = command._tracked_joint_ids
  return dict(
    indices=indices,
    hidden=tuple(hidden),
    body_indexes=[names.index(n) for n in params["body_names"]],
    anchor_index=int(command.motion_anchor_body_index),
    frames=int(params.get("frames", 1)),
    stride=int(params.get("stride", 1)),
    joint_ids=None if tracked is None else tracked.tolist(),
    offset=offset,
    scale=scale,
  )


@torch.no_grad()
def keypoint_skip_table(skip, motion, chunk=65536):
  """Keypoint and anchor poses, the reference joint target in action units and the clip
  end of every reference step (interpolating the motions once, not per batch)."""
  total = int(motion.segment_end_idx[-1])
  device = motion.segment_end_idx.device
  offset = torch.tensor(skip.offset, device=device)
  scale = torch.tensor(skip.scale, device=device)
  parts = []
  for steps in torch.arange(total, device=device).split(chunk):
    frame = motion.frame(steps)
    q = (
      frame.joint_pos if skip.joint_ids is None else frame.joint_pos[:, skip.joint_ids]
    )
    parts.append(
      (
        frame.body_pos_w[:, skip.body_indexes],
        frame.body_quat_w[:, skip.body_indexes],
        frame.body_pos_w[:, skip.anchor_index],
        frame.body_quat_w[:, skip.anchor_index],
        (q - offset) / scale,
      )
    )
  pos, quat, anchor_pos, anchor_quat, action = (
    torch.cat(p) for p in zip(*parts, strict=True)
  )
  end = motion.segment_end_idx[
    torch.searchsorted(
      motion.segment_end_idx, torch.arange(total, device=device), right=True
    )
  ]
  return dict(
    pos=pos,
    quat=quat,
    anchor_pos=anchor_pos,
    anchor_quat=anchor_quat,
    action=action,
    end=end,
  )


def _keypoint_skip_data(skip, table, steps):
  """The keypoint goal at reference steps ``steps`` and the next frame's reference joint
  target in action units (frames past a clip's end repeat its last one)."""
  from wbc_mjlab.env.mdp.observations import keypoint_features

  last = table["end"][steps] - 1
  ahead = torch.arange(skip.frames, device=steps.device) * skip.stride
  window = torch.minimum(steps[:, None] + ahead, last[:, None])
  k = skip.frames
  x = keypoint_features(
    table["pos"][window],
    table["quat"][window],
    table["anchor_pos"][steps, None].expand(-1, k, 3),
    table["anchor_quat"][steps, None].expand(-1, k, 4),
  ).flatten(1)
  return x, table["action"][torch.minimum(steps + 1, last)]


def fit_keypoint_skip(skip, motion, steps=20000, batch=8192, seed=0):
  """Fit ``skip.kp2q`` on every reference step but a held-out 5 % (PartyMimic train.py
  ``fit_kp2q``): Adam with a cosine learning rate, then freeze into the buffers."""
  import math
  import time

  device = skip.kp2q.mu.device
  start = time.time()
  table = keypoint_skip_table(skip, motion)
  total = table["end"].shape[0]
  g = torch.Generator(device=device).manual_seed(seed)
  order = torch.randperm(total, device=device, generator=g)
  held, train = order[: max(1, total // 20)], order[max(1, total // 20) :]

  def sample(n):
    return train[torch.randint(len(train), (n,), device=device, generator=g)]

  with torch.no_grad():
    x0, _ = _keypoint_skip_data(skip, table, sample(min(65536, 8 * batch)))
    mu, sd = x0.mean(0), x0.std(0).clamp_min(1e-3)
  dims = [x0.shape[1], *skip.hidden]
  layers = []
  for a, b in zip(dims[:-1], dims[1:], strict=True):
    layers += [nn.Linear(a, b), nn.ELU()]
  net = nn.Sequential(*layers, nn.Linear(dims[-1], skip.output_dim)).to(device)
  optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
  with torch.inference_mode(False), torch.enable_grad():
    for step in range(steps):
      for group in optimizer.param_groups:
        group["lr"] = 1e-3 * 0.5 * (1 + math.cos(math.pi * step / steps))
      with torch.no_grad():
        x, y = _keypoint_skip_data(skip, table, sample(batch))
      loss = (net((x - mu) / sd) - y).square().mean()
      optimizer.zero_grad(set_to_none=True)
      loss.backward()
      optimizer.step()
  with torch.no_grad():
    scale = torch.tensor(skip.scale, device=device)
    error, base = 0.0, 0.0
    for chunk in held.split(65536):
      x, y = _keypoint_skip_data(skip, table, chunk)
      error = error + ((net((x - mu) / sd) - y) * scale).square().sum(0)
      base = base + (y * scale).square().sum(0)
    rms = (error / len(held)).sqrt()
    default = (base / len(held)).sqrt()
  print(
    f"[INFO] Keypoint skip: kp2q fit in {time.time() - start:.0f} s, held-out joint RMS "
    f"{rms.mean():.4f} rad (max {rms.max():.4f}; default pose {default.mean():.4f})"
  )
  skip.kp2q.load_from(net, mu, sd)


def fit_keypoint_skips(modules, motion, steps=20000):
  """Fit the keypoint skip of the first module that has an unfitted one, and copy it into
  every other (EMA copies, live actors). Fitted skips (loaded checkpoints) are kept."""
  skips = [
    m for module in modules for m in module.modules() if isinstance(m, KeypointSkipHead)
  ]
  if not skips or all(bool(s.kp2q.fitted) for s in skips):
    return
  source = next((s for s in skips if bool(s.kp2q.fitted)), None)
  if source is None:
    source = skips[0]
    fit_keypoint_skip(source, motion, steps)
  for s in skips:
    if s is not source:
      s.kp2q.load_state_dict(source.kp2q.state_dict())


class ReferenceResidualModel(MLPModel):
  """The flat MLP with a reference skip (joint lookahead) or keypoint skip."""

  def __init__(
    self,
    obs,
    obs_groups,
    obs_set,
    output_dim,
    *,
    reference_skip=None,
    keypoint_skip=None,
    **options,
  ):
    super().__init__(obs, obs_groups, obs_set, output_dim, **options)
    _add_skips(self, output_dim, reference_skip, keypoint_skip)


def _add_skips(model, output_dim, reference_skip, keypoint_skip):
  if reference_skip is not None:
    model.mlp = ReferenceResidualHead(model.mlp, model.obs_normalizer, **reference_skip)
  if keypoint_skip is not None:
    model.mlp = KeypointSkipHead(
      model.mlp, model.obs_normalizer, output_dim, **keypoint_skip
    )


def policy_head(actor):
  """The actor's head inside its skips (an encoder head has ``latent`` and ``aux``)."""
  head = actor.mlp
  while isinstance(head, (ReferenceResidualHead, KeypointSkipHead)):
    head = head.head
  return head


def policy_mlps(actor):
  """Independent encoder/decoder MLP boundaries for optimizer parameter grouping."""
  head = policy_head(actor)
  if isinstance(head, (MotionDecoder, CrossAttentionDecoder)):
    return [head.encoder, head.decoder]
  return [head]


def build_actor(obs, env, cfg):
  from rsl_rl.utils import resolve_callable

  options = actor_options(env, cfg)
  cls = resolve_callable(options.pop("class_name"))
  return cls(
    obs, cfg["obs_groups"], "actor", env.action_manager.total_action_dim, **options
  )


def load_actor(env, cfg, checkpoint, device):
  """Load only the actor; evaluation neither allocates a replay nor a world model."""
  from rsl_rl.utils import resolve_obs_groups
  from tensordict import TensorDict

  obs = TensorDict(env.observation_manager.compute(), batch_size=[env.num_envs])
  cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], ["actor", "critic"])
  actor = build_actor(obs, env, cfg).to(device)
  from wbc_mjlab.rl.checkpoints import read_checkpoint

  state = read_checkpoint(checkpoint, device)
  actor.load_state_dict(state["actor_state_dict"])
  return actor.eval()
