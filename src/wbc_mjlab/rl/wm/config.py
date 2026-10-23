"""Configuration of world-model training (``--algo wm``).

Defaults follow SuperMimic (Liao et al., arXiv:2610.28480, Table A.5).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg


@dataclass
class WmAlgorithmCfg:
  """World model fit to simulator steps; actor trained by backpropagation through it."""

  window: int = 8
  """Steps of the autoregressive windows the world model is fit on (k)."""
  model_hidden_dims: tuple[int, ...] = (2048, 2048, 2048)
  """Dynamics network."""
  heads_hidden_dims: tuple[int, ...] = (512, 256)
  """Step heads (termination, learned reward terms)."""
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
class WmRunnerCfg(RslRlOnPolicyRunnerCfg):
  """Runner settings for ``--algo wm``: the PPO runner's, with :class:`WmAlgorithmCfg`.

  ``actor`` is the policy (exploration noise: ``algorithm.action_noise``); ``critic`` sizes
  the value network, which sees the world-model state, the tracking features and the
  domain-randomization context instead of an observation group.
  """

  algorithm: WmAlgorithmCfg = field(default_factory=WmAlgorithmCfg)  # type: ignore[assignment]


def wm_runner_cfg(ppo: RslRlOnPolicyRunnerCfg) -> WmRunnerCfg:
  """The ``wm`` runner config of a robot, from its PPO one."""
  # The actor's noise is set by ``algorithm.action_noise`` (fixed, not learned).
  actor = replace(ppo.actor, obs_normalization=True, distribution_cfg=None)
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
  """The env for world-model training: without the external forces a deterministic model
  of the robot cannot predict (velocity pushes, pulls, the assistive wrench), as in Play."""
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
  return cfg
