"""Configuration of world-model training (``--algo wm``).

Defaults follow SuperMimic (Liao et al., arXiv:2610.28480, Table A.5).
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from typing import Literal

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg

from wbc_mjlab.rl.policy import TrackingRunnerCfg, tracking_runner_cfg


@dataclass
class WmAlgorithmCfg:
  """World model fit to simulator steps; actor trained by backpropagation through it."""

  window: int = 8
  """Steps of the autoregressive windows the world model is fit on (k)."""
  model_hidden_dims: tuple[int, ...] = field(
    default=(2048, 2048, 2048), metadata={"checkpoint": True}
  )
  """Dynamics network."""
  heads_hidden_dims: tuple[int, ...] = field(
    default=(512, 256), metadata={"checkpoint": True}
  )
  """Step heads (termination, learned reward terms)."""
  dynamics_mode: Literal["increment", "integrated", "velocity"] = field(
    default="increment", metadata={"checkpoint": True}
  )
  """Full increments, integrated pose residuals, or velocity-only integration."""
  velocity_blend: float | None = field(
    default=1.0, metadata={"checkpoint": True, "nullable": True, "kind": "number"}
  )
  """Integrated pose prior: new-velocity share; None derives (n+1)/(2n) from decimation."""
  dynamics_loss_scale: Literal["output", "increment", "physical"] = field(
    default="output", metadata={"checkpoint": True}
  )
  """Loss units: output statistics, full increment statistics, or fixed physical tolerances."""
  physical_error_scales: tuple[float, ...] = field(
    default=(0.01, 0.02, 0.5, 1.0, 0.1, 3.14), metadata={"checkpoint": True}
  )
  """Fixed tolerances: root m/rad/m/s/rad/s, joint rad/rad/s; physical loss only.

  Pose/joint tolerances follow official SuperMimic; explicit root velocity supervision
  is our extension. Scales never change with replay statistics.
  """
  critic_inputs: Literal["actor", "structured"] = field(
    default="actor", metadata={"checkpoint": True}
  )
  """Structured: physical state/tracking/context, clean goals and policy history."""
  critic_layer_norm: bool = field(default=False, metadata={"checkpoint": True})
  critic_activation: Literal["task", "elu", "gelu", "silu"] = field(
    default="task", metadata={"checkpoint": True}
  )
  model_optimizer: Literal["adam", "muon"] = "adam"
  actor_optimizer: Literal["adam", "muon"] = "adam"
  fused_optimizer: bool = False
  skip_nonfinite_updates: bool = False
  critic_norm_eps: float = 1e-4
  """Minimum critic input scale; Blink uses 0.01 to limit near-constant-feature gain."""
  bound_critic: bool = False
  """Cap bootstrap values at the task reward upper bound / (1-gamma)."""
  termination_mode: Literal["learned", "hybrid"] = "learned"
  """Hybrid uses task tracking thresholds as a lower bound on imagined termination."""
  policy_ema: float = 0.0
  """Deployment policy exponential average; zero disables. Live weights resume training."""
  mirror_probability: float = 0.0
  """Share of dynamics windows reflected using robot symmetry; step heads remain unmirrored."""
  actor_lr_multiplier: float = 1.0
  actor_lr_decay_start: int = 1400
  actor_lr_decay_updates: int = 300
  actor_lr_return_threshold: float | None = field(
    default=None, metadata={"kind": "number", "nullable": True}
  )
  """Start multiplier decay at this normalized simulator reward, or the update deadline."""
  recent_failure_share: float | None = field(
    default=None, metadata={"kind": "number", "nullable": True}
  )
  """Override the task's recent-failure reset mixture when set."""
  probe_share: float = 0.15
  collection_cuda_graphs: bool = False
  probe_cuda_graphs: bool = False
  """Capture policy inference; physics graphs remain owned by mjlab."""
  probe_interval: int = 0
  """Actor updates between curriculum probe batches; zero disables."""
  probe_start: int = 1000
  probe_num_envs: int = 128
  probe_tile_s: float = 6.0
  probe_lead_s: tuple[float, float] = (4.0, 9.0)
  probe_steps_per_iteration: int = 20
  """Bounded probe work per runner iteration; zero completes a sweep synchronously."""
  probe_async: bool = False
  """Run probe chunks on a CUDA side stream with explicit snapshot/event ownership."""
  compile_collection: bool = False
  """Compile the collection actor and critic loss; simulation uses mjlab's own graphs."""
  separate_action_noise: bool = False
  """Exploration only in execution; observations and action-rate reward use clean actions."""
  scale_lr_by_world_size: bool = True
  """Distributed runs use sqrt(world_size) times the configured learning rate."""
  diagnostic_interval: int = 100
  """Updates between independent replay rollout diagnostics; zero disables them."""
  diagnostic_batch: int = 256
  model_batch: int = 2048
  """Windows per world-model gradient step."""
  model_steps: int = 10
  """World-model gradient steps per update."""
  horizon: int = 32
  """Imagined steps per rollout (H); the critic bootstraps beyond."""
  imagined_rollouts: int = 4096
  """Rollouts per policy update, started from replay states."""
  updates_per_iteration: int = 2
  """Updates per runner iteration (``num_steps_per_env`` env steps): 2 per 5 steps is the
  paper's update-to-data ratio of 0.4."""
  gamma: float = 0.99
  lam: float = 0.95
  """GAE λ of the critic targets."""
  action_noise: float = 0.1
  """Std of the Gaussian noise on actions in the simulator (fixed)."""
  learning_rate: float = 2.0e-4
  """World model, actor and critic."""
  lr_final_ratio: float = 0.03
  """Learning rates decay (cosine) to this fraction..."""
  lr_decay_updates: int = 250_000
  """...over this many updates, then stay constant."""
  max_grad_norm: float = 1.0
  actor_grad_clip: float = 0.1
  """Element-wise clip of actor gradients, after the norm clip."""
  critic_epochs: int = 5
  critic_mini_batches: int = 4
  replay_steps: int = 1000
  """Steps of every env kept in the replay."""
  warmup_steps: int = 32
  """Env steps collected before the first update."""
  compile: bool = True
  """``torch.compile`` the imagination step and the world-model losses (first update
  compiles for a few minutes; off: slower, for debugging)."""
  cuda_graphs: bool = True
  """Run each world-model and policy update as one captured CUDA graph (thousands of
  small kernels launched at once); with ``compile`` only."""
  class_name: str = "wbc_mjlab.rl.wm.algorithm:WorldModelRL"


@dataclass
class WmRunnerCfg(TrackingRunnerCfg):
  """Runner settings for ``--algo wm``: the PPO runner's, with :class:`WmAlgorithmCfg`.

  ``actor`` is the policy (exploration noise: ``algorithm.action_noise``); ``critic`` sizes
  the value network, which sees the world-model state, the tracking features and the
  domain-randomization context instead of an observation group.
  """

  algorithm: WmAlgorithmCfg = field(default_factory=WmAlgorithmCfg)  # type: ignore[assignment]


def wm_runner_cfg(ppo: RslRlOnPolicyRunnerCfg) -> WmRunnerCfg:
  """The ``wm`` runner config of a robot, from its PPO one."""
  # The actor's noise is set by ``algorithm.action_noise`` (fixed, not learned).
  actor = replace(
    tracking_runner_cfg(ppo).actor, obs_normalization=True, distribution_cfg=None
  )
  return WmRunnerCfg(
    seed=ppo.seed,
    num_steps_per_env=5,
    max_iterations=160_000,
    obs_groups=dict(ppo.obs_groups),
    save_interval=1000,
    experiment_name=ppo.experiment_name,
    logger=ppo.logger,
    wandb_project=ppo.wandb_project,
    clip_actions=ppo.clip_actions,
    upload_model=ppo.upload_model,
    actor=actor,
    critic=RslRlModelCfg(hidden_dims=(256, 128)),
    algorithm=WmAlgorithmCfg(),
  )


def wm_env_cfg(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
  """Remove unobserved ongoing forces; offer optional, explicitly recorded velocity kicks.

  Kicks preserve a pre-disturbance target and cut model windows (perturbations.py).
  """
  removed = [
    name for name in ("push_robot", "pull_robot") if cfg.events.pop(name, None)
  ]
  motion = cfg.commands.get("motion")
  if getattr(motion, "assistive_wrench_enabled", False):
    motion.assistive_wrench_enabled = False
    removed.append("assistive wrench")
  if "assistive_wrench" in cfg.events:
    cfg.events["assistive_wrench"].params["enabled"] = False
  if removed:
    print(f"[INFO] wm: off for world-model training: {', '.join(removed)}")
  # Constraint capacity is a method-wide safety margin, including Lookahead + wm-blink.
  cfg.sim.njmax = max(cfg.sim.njmax or 0, 512)
  cfg.sim.contact_sensor_maxmatch = max(cfg.sim.contact_sensor_maxmatch or 0, 128)
  from mjlab.managers.event_manager import EventTermCfg

  from wbc_mjlab.rl.wm.perturbations import EVENT, inter_step_velocity_kick

  cfg.events.setdefault(
    EVENT,
    EventTermCfg(
      func=inter_step_velocity_kick,
      mode="step",
      params={
        "enabled": False,
        "interval_range_s": (5.0, 10.0),
        "velocity_ranges": (
          (-0.5, 0.5),
          (-0.5, 0.5),
          (0.0, 0.0),
          (-0.25, 0.25),
          (-0.25, 0.25),
          (0.0, 0.0),
        ),
        "initial_scale": 0.2,
        "ramp_duration_s": 60.0,
        "clean_episode_fraction": 0.2,
      },
    ),
  )
  # Pre-reset capture retains the state reached by a terminal action; see recording.py
  # for SuperMimic / SuperMimic-Reproduced provenance.
  from mjlab.envs.mdp.actions import JointPositionActionCfg
  from mjlab.managers.recorder_manager import RecorderTermCfg

  from wbc_mjlab.env.mdp.actions import TrackingJointPositionActionCfg
  from wbc_mjlab.rl.wm.recording import RECORDER, TerminalStateRecorder

  for name, action in cfg.actions.items():
    if type(action) is JointPositionActionCfg:
      cfg.actions[name] = TrackingJointPositionActionCfg(
        **{f.name: getattr(action, f.name) for f in fields(action) if f.init}
      )

  cfg.recorders[RECORDER] = RecorderTermCfg(
    func=TerminalStateRecorder, params={"command_name": "motion"}
  )
  return cfg
