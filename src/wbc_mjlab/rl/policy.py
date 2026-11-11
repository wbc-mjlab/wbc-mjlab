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

  architecture: Literal["mlp", "motion_encoder"] = field(
    default="mlp",
    metadata={
      "setting": True,
      "checkpoint": True,
      "help": "flat MLP or a bounded motion latent with a proprioceptive decoder",
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
  goal_terms: tuple[str, ...] = field(
    default=(),
    metadata={
      "setting": True,
      "checkpoint": True,
      "visible_if": {"actor.architecture": "motion_encoder"},
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
  ):
    super().__init__()
    goal = set(goal_indices)
    if not goal or len(goal) >= obs_dim:
      raise ValueError(
        "motion encoder needs goal terms and non-goal actor observations"
      )
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

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    z = torch.tanh(self.encoder(obs[..., self.goal_indices]))
    return self.decoder(torch.cat([z, obs[..., self.proprio_indices]], dim=-1))


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
  if "goal_indices" in options or "reference_skip" in options:
    return options  # already resolved by the common runner
  residual = options.pop("reference_residual", False)
  if residual:
    options["reference_skip"] = reference_skip_options(env)
  architecture = options.pop("architecture", "mlp")
  encoder_hidden = options.pop("encoder_hidden_dims", (2048, 1024, 512, 256))
  latent = options.pop("latent_dim", 32)
  names = options.pop("goal_terms", ())
  if architecture == "motion_encoder":
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
    )

    if reference_skip is not None:
      self.mlp = ReferenceResidualHead(self.mlp, self.obs_normalizer, **reference_skip)


def reference_skip_options(env):
  """Resolve Blink's next-frame skip from the actual observation/action contract.

  BlinkMimic d264658, networks.Policy; SuperMimic arXiv:2610.28480.
  https://github.com/fuyichen0725/blinkmimic/blob/d264658d6c9ce61fb52f9f0f8985a369f3ccc409/src/blinkmimic/networks.py
  """
  from wbc_mjlab.env.mdp.observations import ref_joint_pos

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
  om = env.observation_manager
  offset, indices = 0, None
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
      indices = list(range(offset + len(ids), offset + 2 * len(ids)))
    offset += width
  if indices is None:
    raise ValueError("reference skip requires the ref_joint_pos lookahead observation")

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
  return dict(indices=indices, offset=vector(action._offset), scale=scale)


class ReferenceResidualHead(nn.Module):
  """Exportable skip around any shared policy head; undo observation normalization."""

  def __init__(self, head, normalizer, indices, offset, scale):
    super().__init__()
    self.head, self.normalizer = head, normalizer
    self.register_buffer("indices", torch.tensor(indices, dtype=torch.long))
    self.register_buffer("offset", torch.tensor(offset))
    self.register_buffer("scale", torch.tensor(scale))

  def forward(self, obs):
    ref = obs[..., self.indices]
    if hasattr(self.normalizer, "_mean"):
      ref = (
        ref * (self.normalizer._std[..., self.indices] + self.normalizer.eps)
        + self.normalizer._mean[..., self.indices]
      )
    return self.head(obs) + (ref - self.offset) / self.scale


class ReferenceResidualModel(MLPModel):
  def __init__(self, *args, reference_skip, **kwargs):
    super().__init__(*args, **kwargs)
    self.mlp = ReferenceResidualHead(self.mlp, self.obs_normalizer, **reference_skip)


def policy_mlps(actor):
  """Independent encoder/decoder MLP boundaries for optimizer parameter grouping."""
  head = actor.mlp
  if isinstance(head, ReferenceResidualHead):
    head = head.head
  return [head.encoder, head.decoder] if isinstance(head, MotionDecoder) else [head]


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
