"""World-model RL for motion tracking (``--algo wm``), after SuperMimic (arXiv:2610.28480).

Optional BlinkMimic components follow train.py at d264658 (Chen, 2026):
https://github.com/fuyichen0725/blinkmimic/blob/d264658d6c9ce61fb52f9f0f8985a369f3ccc409/src/blinkmimic/train.py
MIT; see THIRD_PARTY_NOTICES. All methods retain the shared task/imagination contract.

One loop, three phases (Algorithm A.1):

1. **Collect** — the actor, with fixed Gaussian noise, steps the simulator; every step goes
   into a replay of the last ``replay_steps`` steps of each env.
2. **Fit the world model** — the dynamics on ``window``-step autoregressive windows of the
   replay, and along them, on the predicted states, the termination and the reward terms
   that are not functions of the state (:mod:`wbc_mjlab.rl.wm.model`).
3. **Improve the policy** — from replay states (with the observation history the env held
   there), imagine ``horizon`` steps through the frozen world model, with the env's own
   observation and reward terms evaluated on the imagined states
   (:mod:`wbc_mjlab.rl.wm.imagined`), and backpropagate the return
   ``Σ γᵗ cₜ rₜ₊₁ + γᴴ c_H V(s_H)`` (``c``: soft survival) into the actor. The critic is fit to
   GAE(λ) targets of the same rollouts.

Each update of phases 2 and 3 is one CUDA graph (forward, backward, optimizer step): its
thousands of small kernels launch at once. The class implements rsl_rl's algorithm
interface, so the PPO runner drives it: logging, checkpoints, resume, ONNX export and Play
are unchanged.
"""

from __future__ import annotations

import copy
import dataclasses
import math
import time
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP
from rsl_rl.utils import resolve_obs_groups
from tensordict import TensorDict
from torch import nn

from wbc_mjlab.motion.library import MotionFrame
from wbc_mjlab.rl.optim import make_optimizer, step_if_finite
from wbc_mjlab.rl.policy import build_actor, policy_mlps
from wbc_mjlab.rl.wm.distributed import average_gradients, broadcast, sum_, world_size
from wbc_mjlab.rl.wm.imagined import Goal, Imagination
from wbc_mjlab.rl.wm.kinematics import RobotState
from wbc_mjlab.rl.wm.model import (
  RunningNorm,
  WorldModel,
  feature_dim,
  nominal_error,
  state_change,
  state_features,
  update_running_stats,
)
from wbc_mjlab.rl.wm.recording import RECORDER
from wbc_mjlab.rl.wm.replay import Replay, transition_mask

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from rsl_rl.env import VecEnv


FIRST_UPDATE = "[INFO] wm: first update:"
"""Log prefix of the first update's compile (the Studio shows it as the job's phase)."""


class WorldModelRL:
  """rsl_rl algorithm: replay → world model → actor by BPTT, critic by GAE."""

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    actor: MLPModel,
    critic_hidden_dims: tuple[int, ...],
    task_critic_activation: str,
    *,
    window: int,
    model_hidden_dims: tuple[int, ...],
    heads_hidden_dims: tuple[int, ...],
    model_batch: int,
    model_steps: int,
    horizon: int,
    imagined_rollouts: int,
    updates_per_iteration: int,
    gamma: float,
    lam: float,
    action_noise: float,
    learning_rate: float,
    lr_final_ratio: float,
    lr_decay_updates: int,
    max_grad_norm: float,
    actor_grad_clip: float,
    critic_epochs: int,
    critic_mini_batches: int,
    replay_steps: int,
    warmup_steps: int,
    compile: bool,
    cuda_graphs: bool,
    dynamics_mode: str = "increment",
    velocity_blend: float | None = 1.0,
    dynamics_loss_scale: str = "output",
    physical_error_scales: tuple[float, ...] = (0.01, 0.02, 0.5, 1.0, 0.1, 3.14),
    critic_inputs: str = "actor",
    critic_layer_norm: bool = False,
    critic_activation: str = "task",
    model_optimizer: str = "adam",
    actor_optimizer: str = "adam",
    fused_optimizer: bool = False,
    skip_nonfinite_updates: bool = False,
    critic_norm_eps: float = 1e-4,
    bound_critic: bool = False,
    termination_mode: str = "learned",
    policy_ema: float = 0.0,
    mirror_probability: float = 0.0,
    actor_lr_multiplier: float = 1.0,
    actor_lr_decay_start: int = 1400,
    actor_lr_decay_updates: int = 300,
    actor_lr_return_threshold: float | None = None,
    recent_failure_share: float | None = None,
    probe_share: float = 0.15,
    collection_cuda_graphs: bool = False,
    probe_cuda_graphs: bool = False,
    probe_interval: int = 0,
    probe_start: int = 1000,
    probe_num_envs: int = 128,
    probe_tile_s: float = 6.0,
    probe_lead_s: tuple[float, float] = (4.0, 9.0),
    probe_steps_per_iteration: int = 20,
    probe_async: bool = False,
    compile_collection: bool = False,
    separate_action_noise: bool = False,
    scale_lr_by_world_size: bool = True,
    diagnostic_interval: int = 100,
    diagnostic_batch: int = 256,
    clip_actions: float | None = None,
    device: str = "cpu",
  ):
    if not 0 <= policy_ema < 1 or not 0 <= mirror_probability <= 1:
      raise ValueError("EMA must be in [0,1); mirror probability in [0,1]")
    if (
      actor_lr_multiplier <= 0
      or actor_lr_decay_updates < 1
      or actor_lr_decay_start < 0
      or lr_decay_updates < 1
    ):
      raise ValueError("invalid learning-rate schedule")
    self.skip_nonfinite_updates = skip_nonfinite_updates
    self.policy_ema = policy_ema
    self.mirror_probability = mirror_probability
    self.actor_lr_multiplier = actor_lr_multiplier
    self.actor_lr_decay_updates = actor_lr_decay_updates
    self.actor_lr_return_threshold = actor_lr_return_threshold
    self._actor_decay_at = torch.tensor(float(actor_lr_decay_start), device=device)
    self._reward_ema = torch.zeros((), device=device)
    from wbc_mjlab.rl.evaluation import reward_upper_bound

    self._reward_bound = reward_upper_bound(env.cfg, env.step_dt)
    if actor_lr_return_threshold is not None and self._reward_bound is None:
      raise ValueError(
        "return-triggered learning rate needs a task with known reward bounds"
      )
    if critic_norm_eps <= 0:
      raise ValueError("critic_norm_eps must be positive")
    if bound_critic and (self._reward_bound is None or not 0 <= gamma < 1):
      raise ValueError("bounded critic needs known task reward bounds and gamma < 1")
    self.value_upper_bound = self._reward_bound / (1 - gamma) if bound_critic else None
    self.action_noise, self.clip_actions = action_noise, clip_actions
    self.env, self.device = env, device
    if termination_mode not in ("learned", "hybrid"):
      raise ValueError("termination_mode must be learned or hybrid")
    self.termination_mode = termination_mode
    self.imagination = im = Imagination(env)
    from wbc_mjlab.rl.wm.termination import TrackingTermination

    self.tracking_termination = (
      TrackingTermination(env.cfg.terminations)
      if termination_mode == "hybrid"
      else None
    )
    print(
      f"[INFO] wm: clip_actions={clip_actions}, critic_norm_eps={critic_norm_eps}, "
      f"value_upper_bound={self.value_upper_bound}, termination_mode={termination_mode}"
    )
    print("[INFO] wm:", im.summary().replace("\n", "\n[INFO] wm: "))
    if recent_failure_share is not None:
      if not 0 <= recent_failure_share <= 1:
        raise ValueError("recent failure share must be in [0,1]")
      im.command.cfg.rsi.recent_failure_share = recent_failure_share
    if probe_interval:
      if not 0 <= probe_share <= 1:
        raise ValueError("probe share must be in [0,1]")
      im.command.cfg.rsi.probe_share = probe_share
    self.num_joints = len(im.joint_names)
    probe = im.view(*im._probe())
    tracking_dim = im.tracking_features(probe).shape[-1]
    context_dim = im.context(torch.arange(1, device=device)).shape[-1]
    obs_dim = im.actor_obs(probe, im.env_history()[:, :-1])[0].shape[-1]
    self.num_learned_rewards = len(im.learned_rewards)

    self.actor = actor.to(device)
    self.ema_actor = (
      copy.deepcopy(self.actor).requires_grad_(False).eval() if policy_ema else None
    )
    self._collect_actor = (
      torch.compile(self.actor.forward, dynamic=False)
      if compile_collection
      else self.actor
    )
    self.mirror = None
    if mirror_probability:
      from wbc_mjlab.rl.wm.symmetry import DynamicsMirror
      from wbc_mjlab.tasks import last_registered_robot_id

      robot_id = im.command.cfg.robot_id or last_registered_robot_id()
      self.mirror = DynamicsMirror(im, robot_id)
    self.world_model = WorldModel(
      self.num_joints,
      context_dim,
      tracking_dim,
      self.num_learned_rewards,
      model_hidden_dims,
      heads_hidden_dims,
      dynamics_mode=dynamics_mode,
      step_dt=env.step_dt,
      velocity_blend=(env.cfg.decimation + 1) / (2 * env.cfg.decimation)
      if velocity_blend is None
      else velocity_blend,
      loss_scale=dynamics_loss_scale,
      physical_error_scales=physical_error_scales,
    ).to(device)
    # The critic sees the state, its tracking features, the context and the actor's
    # observation (the reference ahead): V(s, g, ξ).
    critic_in = feature_dim(self.num_joints) + tracking_dim + context_dim + obs_dim
    self.critic_norm = RunningNorm(critic_in, eps=critic_norm_eps).to(device)
    from wbc_mjlab.rl.wm.critic import StructuredCritic, value_mlp

    if critic_inputs not in ("actor", "structured"):
      raise ValueError("critic_inputs must be actor or structured")
    activation = (
      task_critic_activation if critic_activation == "task" else critic_activation
    )
    self.critic_layout = (critic_inputs, critic_layer_norm, activation)
    self.critic_goal_indices = self.critic_history_indices = None
    if critic_inputs == "structured":
      from wbc_mjlab.rl.policy import goal_observation_indices

      goals = goal_observation_indices(env)
      self.critic_goal_indices = torch.tensor(goals, device=device)
      self.critic_history_indices = torch.tensor(
        [i for i in range(obs_dim) if i not in set(goals)], device=device
      )
      self.critic = StructuredCritic(
        critic_in - obs_dim + len(goals),
        obs_dim - len(goals),
        critic_hidden_dims,
        activation,
        critic_layer_norm,
      ).to(device)
    elif critic_layer_norm:
      self.critic = value_mlp(
        critic_in, critic_hidden_dims, activation, layer_norm=True
      ).to(device)
    else:
      self.critic = MLP(critic_in, 1, critic_hidden_dims, activation).to(device)

    # Learning rates as tensors: the schedule changes them inside captured graphs.
    learning_rate *= math.sqrt(world_size()) if scale_lr_by_world_size else 1.0
    lr = torch.tensor(learning_rate, device=device)
    # Graphs need compiled steps: eager terms copy Python index lists to the GPU.
    graphs = (
      cuda_graphs
      and compile
      and torch.device(device).type == "cuda"
      and world_size() == 1
    )
    if cuda_graphs and world_size() > 1:
      print(
        "[INFO] wm: distributed updates use torch.compile without full-update CUDA graphs"
      )
    # Finite-update gating must stay on device during capture; unfused Adam's
    # conditional step needs a host boolean. Keep independent options composable.
    fused = (fused_optimizer or (graphs and skip_nonfinite_updates)) and (
      torch.device(device).type == "cuda"
    )
    self.model_optimizer = make_optimizer(
      self.world_model,
      [self.world_model.dynamics, self.world_model.heads],
      kind=model_optimizer,
      lr=lr,
      fused=fused,
      capturable=graphs,
    )
    self.actor_optimizer = make_optimizer(
      self.actor,
      policy_mlps(self.actor),
      kind=actor_optimizer,
      lr=lr,
      betas=(0.9, 0.95),
      fused=fused,
      capturable=graphs,
    )
    self.critic_optimizer = make_optimizer(
      self.critic, [self.critic], kind="adam", lr=lr, fused=fused
    )
    self._optimizers = (
      self.model_optimizer,
      self.actor_optimizer,
      self.critic_optimizer,
    )

    self.window, self.model_batch, self.model_steps = window, model_batch, model_steps
    self.horizon, self.imagined_rollouts = horizon, imagined_rollouts
    self.updates_per_iteration = updates_per_iteration
    self.gamma, self.lam = gamma, lam
    self.base_learning_rate = self.learning_rate = learning_rate
    self.lr_final_ratio, self.lr_decay_updates = lr_final_ratio, lr_decay_updates
    self.max_grad_norm, self.actor_grad_clip = max_grad_norm, actor_grad_clip
    self.critic_epochs, self.critic_mini_batches = critic_epochs, critic_mini_batches
    self.replay_steps, self.warmup_steps = replay_steps, warmup_steps
    self.num_updates = 0
    self.diagnostic_interval, self.diagnostic_batch = (
      diagnostic_interval,
      diagnostic_batch,
    )
    self.separate_action_noise = separate_action_noise
    term = im._action_term
    if separate_action_noise and not hasattr(term, "set_execution_noise"):
      raise ValueError(
        "execution-only noise needs a tracking joint-position action term"
      )
    if hasattr(term, "set_execution_noise"):
      term.set_execution_noise(action_noise if separate_action_noise else 0.0)
    self.terminal_recorder = (
      env.recorder_manager.get_term(RECORDER)
      if RECORDER in env.recorder_manager.active_terms
      else None
    )
    self._updated = False  # in this process (the first update compiles)
    self._compile = compile
    self.replay: Replay | None = None  # allocated on the first step (not for Play)
    self._within_clip: torch.Tensor | None = None  # the step does not end its clip
    if compile:  # fuse the many small ops of kinematics, terms and quaternions
      self._imagine_step = torch.compile(self._imagine_step, dynamic=False)
      self._model_loss_step = torch.compile(self._model_loss_step, dynamic=False)
      self._tracking_features = torch.compile(self._tracking_features, dynamic=False)
    self._model_step = (
      CudaGraphed(
        self._model_step,
        state=lambda: graph_tensors([self.world_model], [self.model_optimizer]),
      )
      if graphs
      else self._model_step
    )
    self._policy_step = (
      CudaGraphed(
        self._policy_step,
        state=lambda: graph_tensors(
          [self.actor, self.world_model, self.critic, self.critic_norm],
          [self.actor_optimizer],
        ),
      )
      if graphs
      else self._policy_step
    )
    if compile_collection:
      self._critic_loss = torch.compile(self._critic_loss, dynamic=False)
    self._collection_graph = None
    if collection_cuda_graphs:
      if not graphs:
        raise ValueError(
          "collection policy graphs require compiled single-GPU CUDA training"
        )
      self._collection_graph = CudaGraphed(
        self._collection_forward,
        state=lambda: [*self.actor.parameters(), *self.actor.buffers()],
      )
    self._set_learning_rate()
    self.probes = None
    if probe_interval:
      from wbc_mjlab.rl.wm.probes import CurriculumProbes

      self.probes = CurriculumProbes(
        self,
        interval=probe_interval,
        start=probe_start,
        num_envs=probe_num_envs,
        tile_s=probe_tile_s,
        lead_s=probe_lead_s,
        steps_per_iteration=probe_steps_per_iteration,
        asynchronous=probe_async,
        cuda_graphs=probe_cuda_graphs,
      )

  # --- rsl_rl algorithm interface ---

  @staticmethod
  def construct_algorithm(
    obs: TensorDict, env: VecEnv, cfg: dict, device: str
  ) -> WorldModelRL:
    alg_cfg = dict(cfg["algorithm"])
    alg_cfg.pop("class_name", None)
    cfg["algorithm"]["rnd_cfg"] = None  # the runner reads it
    cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], ["actor", "critic"])
    if list(cfg["obs_groups"]["actor"]) != ["actor"]:
      raise ValueError(
        "world-model training needs the actor to read the 'actor' group only"
      )
    actor = build_actor(obs, env.unwrapped, cfg)
    print(f"Actor Model: {actor}")
    return WorldModelRL(
      env.unwrapped,
      actor,
      tuple(cfg["critic"]["hidden_dims"]),
      cfg["critic"]["activation"],
      device=device,
      clip_actions=cfg.get("clip_actions"),
      **alg_cfg,
    )

  def _collection_forward(self, obs):
    return self._collect_actor(
      TensorDict({"actor": obs}, batch_size=[obs.shape[0]]),
      stochastic_output=not self.separate_action_noise,
    )

  def act(self, obs: TensorDict) -> torch.Tensor:
    if self.replay is None:
      with torch.inference_mode(False):
        self.replay = self._new_replay()
    im, command = self.imagination, self.imagination.command
    if self.terminal_recorder is not None:
      self.terminal_recorder.captured.zero_()
    self.env._wm_velocity_kick_result = None
    actions = (
      self._collection_graph(obs["actor"])
      if self._collection_graph is not None
      else self._collect_actor(obs, stochastic_output=not self.separate_action_noise)
    )
    if self.clip_actions is not None:
      actions = actions.clamp(-self.clip_actions, self.clip_actions)
    if self.separate_action_noise:
      self.actor.distribution.update(actions)  # runner's action-std logging
    if self.ema_actor is not None and self.ema_actor.distribution is not None:
      self.ema_actor.distribution.update(
        actions.detach()
      )  # runner reads deployed policy output_std
    ids = torch.arange(self.env.num_envs, device=self.device)
    self.replay.begin(
      state=im.robot_state().vector(),
      setpoint=im.joint_setpoint(actions, command.reference(), ids),
      last_action=self.env.action_manager.action,
      time_step=command.time_steps,
      trajectory=command.trajectory_ids,
      reference_offset=command.reference_offset_w,
      observed=im.env_history()[:, -1],
    )
    clip_end = command.motion.segment_end_idx[command.trajectory_ids]
    self._within_clip = command.time_steps + 1 < clip_end
    return actions

  def process_env_step(
    self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict
  ) -> None:
    assert self.replay is not None and self._within_clip is not None
    if self.actor_lr_return_threshold is not None:
      self._reward_ema.lerp_(
        sum_(rewards.mean()) / (world_size() * self._reward_bound), 0.01
      )
    # In place, outside inference mode: imagination reads (and captured graphs hold) the
    # normalizer's buffers.
    norm = self.actor.obs_normalizer
    with torch.inference_mode(False):
      update_running_stats(norm._mean, norm._var, norm._std, norm.count, obs["actor"])
    next_state = self.imagination.robot_state().vector()
    valid = ~dones.bool() & self._within_clip
    kick = getattr(self.env, "_wm_velocity_kick_result", None)
    pushed = torch.zeros_like(dones, dtype=torch.bool)
    if kick is not None:
      pushed = kick.pushed
      next_state = torch.where(pushed[:, None], kick.successor, next_state)
    if self.terminal_recorder is not None:
      captured = self.terminal_recorder.captured
      next_state = torch.where(
        captured[:, None], self.terminal_recorder.state, next_state
      )
      valid = valid | captured
    # The next row continues this one unless the env reset or the clip ended (resample).
    self.replay.end(
      **(
        {"setpoint": self.imagination._action_term.executed_setpoint}
        if hasattr(self.imagination._action_term, "executed_setpoint")
        else {}
      ),
      rewards=self.imagination.learned_reward_targets(),
      terminated=self.env.reset_terminated,
      done=dones.bool(),
      continues=~dones.bool() & self._within_clip & ~pushed,
      next_state=next_state,
      valid=valid,
    )

  def compute_returns(self, obs: TensorDict) -> None:
    pass

  def update(self) -> dict[str, float]:
    assert self.replay is not None
    if self.replay.size < max(self.warmup_steps, self.window):
      return {}
    if not self._updated and self._compile:  # minutes of silence otherwise
      print(FIRST_UPDATE, "compiling (minutes; cached for later runs)", flush=True)
      start = time.perf_counter()
    totals: dict[str, torch.Tensor] = {}
    for _ in range(self.updates_per_iteration):
      stats = self._update_world_model() | self._update_policy()
      self._update_ema()
      for k, v in stats.items():
        totals[k] = totals.get(k, 0.0) + v / self.updates_per_iteration
      self.num_updates += 1
      self._set_learning_rate()
    # One host transfer per runner iteration, outside all update graphs.
    totals = dict(zip(totals, torch.stack(list(totals.values())).tolist(), strict=True))
    if self.probes is not None:
      self.probes.tick()
    if (
      self.diagnostic_interval
      and self.num_updates % self.diagnostic_interval < self.updates_per_iteration
    ):
      totals.update(self.rollout_diagnostics(self.diagnostic_batch))
    if not self._updated and self._compile:
      print(
        FIRST_UPDATE, f"compiled in {time.perf_counter() - start:.0f} s", flush=True
      )
    self._updated = True
    return totals

  def close(self):
    if self.probes is not None:
      self.probes.close()

  def train_mode(self) -> None:
    for m in (self.actor, self.world_model, self.critic, self.critic_norm):
      m.train()

  def eval_mode(self) -> None:
    for m in (self.actor, self.world_model, self.critic, self.critic_norm):
      m.eval()

  def get_policy(self) -> MLPModel:
    return self.ema_actor if self.ema_actor is not None else self.actor

  @torch.no_grad()
  def _update_ema(self):
    if self.ema_actor is None:
      return
    torch._foreach_lerp_(
      list(self.ema_actor.parameters()),
      list(self.actor.parameters()),
      1 - self.policy_ema,
    )
    # Normalization is an estimator, not an optimized parameter; copy its current
    # statistics and counters together. Also preserve integer layout buffers.
    for target, source in zip(
      self.ema_actor.buffers(), self.actor.buffers(), strict=True
    ):
      target.copy_(source)

  def save(self) -> dict:
    return {
      "actor_state_dict": self.get_policy().state_dict(),
      "actor_live_state_dict": self.actor.state_dict(),
      "perturbation_elapsed_s": getattr(self.env, "_wm_kick_elapsed_s", 0.0),
      "blink_state": {
        "actor_decay_at": self._actor_decay_at.clone(),
        "reward_ema": self._reward_ema.clone(),
        "probes": self.probes.state_dict() if self.probes is not None else None,
      },
      "critic_state_dict": {
        "norm": self.critic_norm.state_dict(),
        "mlp": self.critic.state_dict(),
      },
      "world_model_state_dict": self.world_model.state_dict(),
      "optimizer_state_dict": [o.state_dict() for o in self._optimizers],
      "num_updates": self.num_updates,
      "wm_metadata": {
        "dynamics_mode": self.world_model.dynamics_mode,
        "step_dt": self.world_model.step_dt,
        "velocity_blend": self.world_model.velocity_blend,
        "loss_scale": self.world_model.loss_scale,
        "physical_error_scales": self.world_model.physical_error_scales,
        "critic_layout": self.critic_layout,
      },
    }

  def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
    load_cfg = load_cfg or {
      "actor": True,
      "critic": True,
      "optimizer": True,
      "iteration": True,
    }
    if loaded_dict.get("checkpoint_kind") == "policy" and load_cfg.get("optimizer"):
      raise ValueError(
        "This is a lightweight policy checkpoint; resume from a full checkpoint or load actor-only"
      )
    # In place (``copy_``): captured graphs hold these tensors.
    if load_cfg.get("actor"):
      live = (
        loaded_dict.get("actor_live_state_dict", loaded_dict["actor_state_dict"])
        if load_cfg.get("optimizer")
        else loaded_dict["actor_state_dict"]
      )
      self.actor.load_state_dict(live, strict=strict)
      if self.ema_actor is not None:
        self.ema_actor.load_state_dict(loaded_dict["actor_state_dict"], strict=strict)
    if "world_model_state_dict" not in loaded_dict:  # e.g. a PPO checkpoint: actor only
      return False
    if load_cfg.get("critic"):
      metadata = loaded_dict.get("wm_metadata", {"dynamics_mode": "increment"})
      if metadata["dynamics_mode"] != self.world_model.dynamics_mode:
        raise ValueError(
          "checkpoint dynamics mode differs from the configured world model"
        )
      if not math.isclose(metadata.get("step_dt", self.env.step_dt), self.env.step_dt):
        raise ValueError(
          "checkpoint world-model control timestep differs from this task"
        )
      if (
        metadata.get("velocity_blend", 1.0) != self.world_model.velocity_blend
        or metadata.get("loss_scale", "output") != self.world_model.loss_scale
      ):
        raise ValueError("checkpoint dynamics integration/loss settings differ")
      if (
        self.world_model.loss_scale == "physical"
        and tuple(metadata.get("physical_error_scales", ()))
        != self.world_model.physical_error_scales
      ):
        raise ValueError("checkpoint physical loss tolerances differ")
      if (
        tuple(metadata.get("critic_layout", ("actor", False, self.critic_layout[2])))
        != self.critic_layout
      ):
        raise ValueError("checkpoint critic layout differs")
      self.world_model.load_state_dict(
        loaded_dict["world_model_state_dict"], strict=strict
      )
      self.critic_norm.load_state_dict(
        loaded_dict["critic_state_dict"]["norm"], strict=strict
      )
      self.critic.load_state_dict(
        loaded_dict["critic_state_dict"]["mlp"], strict=strict
      )
    if load_cfg.get("optimizer"):
      for o, state in zip(
        self._optimizers, loaded_dict["optimizer_state_dict"], strict=True
      ):
        o.load_state_dict(state)
      self.env._wm_kick_elapsed_s = loaded_dict.get("perturbation_elapsed_s", 0.0)
      kicks = getattr(self.env, "_wm_velocity_kicks", None)
      if kicks is not None:
        kicks.elapsed_s = self.env._wm_kick_elapsed_s
      state = loaded_dict.get("blink_state", {})
      self._actor_decay_at.copy_(state.get("actor_decay_at", self._actor_decay_at))
      self._reward_ema.copy_(state.get("reward_ema", self._reward_ema))
      if self.probes is not None and state.get("probes") is not None:
        self.probes.load_state_dict(state["probes"])
      self.num_updates = loaded_dict.get("num_updates", 0)
      self._set_learning_rate()
      # Optimizer.load_state_dict replaces moment tensors. Captured update graphs
      # must not continue stepping the old state after an in-process resume.
      for step in (self._model_step, self._policy_step):
        if isinstance(step, CudaGraphed):
          step.reset()
    return bool(load_cfg.get("iteration"))

  def broadcast_parameters(self) -> None:
    for module in (self.actor, self.world_model, self.critic, self.critic_norm):
      broadcast(module)
    if self.ema_actor is not None:
      broadcast(self.ema_actor)

  # --- phases ---

  def _new_replay(self) -> Replay:
    return Replay(
      self.replay_steps,
      self.env.num_envs,
      13 + 2 * self.num_joints,
      self.imagination.num_actions,
      self.num_learned_rewards,
      self.imagination.history_dim,
      self.device,
    )

  def _state(self, x: torch.Tensor) -> RobotState:
    return RobotState.from_vector(x, self.num_joints)

  @torch.no_grad()
  def rollout_diagnostics(self, batch: int = 256) -> dict[str, float]:
    """Independent replay windows, fixed physical scales; never update normalization.

    After SuperMimic Fig. 3c and SuperMimic-Reproduced's wm_trainer.online_error
    (ba61053). These are in-replay diagnostics, not held-out generalization scores.
    """
    assert self.replay is not None
    w = self.replay.sample_windows(batch, self.window)
    context = self.imagination.context(w.env_ids)
    s = self._state(w.state[0])
    valid_steps = transition_mask(w.continues, w.valid)
    metrics = {}
    for t in range(self.window):
      valid = valid_steps[t]
      s = self.world_model.step(s, w.setpoint[t], context)
      true = self._state(w.next_state[t])
      score = _masked_mean(nominal_error(s, true).mean(-1), valid)
      if t in (0, self.window - 1):
        metrics[f"wm_eval_nominal_{t + 1}_step"] = score.item()
      if t == self.window - 1:
        error = state_change(true, s)
        j = self.num_joints
        for name, chunk in zip(
          (
            "root_pos_m",
            "root_rot_rad",
            "root_vel_m_s",
            "root_omega_rad_s",
            "joint_pos_rad",
            "joint_vel_rad_s",
          ),
          error.split((3, 3, 3, 3, j, j), dim=-1),
          strict=True,
        ):
          metrics[f"wm_eval_{name}_rmse"] = (
            _masked_mean(chunk.square().mean(-1), valid).sqrt().item()
          )
        metrics["wm_eval_valid_fraction"] = valid.float().mean().item()
    return metrics

  def _update_world_model(self) -> dict[str, torch.Tensor]:
    assert self.replay is not None
    k = self.window
    totals = torch.zeros(3, device=self.device)
    for _ in range(self.model_steps):
      w = self.replay.sample_windows(self.model_batch, k)
      with torch.no_grad():
        frames = self.imagination.command.motion.frame(w.time_step[:k].reshape(-1))
      totals += self._model_step(
        w.state,
        w.next_state,
        w.setpoint,
        w.continues,
        w.valid,
        w.terminated,
        w.rewards,
        w.time_step,
        w.trajectory,
        w.reference_offset,
        w.env_ids,
        _unflatten(frames, k),
      )
    dynamics, last_step, heads = (totals / self.model_steps).unbind()
    return {
      "wm_dynamics": dynamics,
      f"wm_error_{k}_step": last_step,
      "wm_heads": heads,
    }

  def _model_step(
    self,
    states: torch.Tensor,
    next_states: torch.Tensor,
    setpoints: torch.Tensor,
    continues: torch.Tensor,
    captured: torch.Tensor,
    terminated: torch.Tensor,
    rewards: torch.Tensor,
    time_steps: torch.Tensor,
    trajectories: torch.Tensor,
    offsets: torch.Tensor,
    env_ids: torch.Tensor,
    frames: MotionFrame,
  ) -> torch.Tensor:
    """One gradient step of the world model on a batch of windows (``[k, B]`` rows,
    reference ``frames`` ``[k, B]``); returns its losses."""
    wm, k = self.world_model, self.window
    context = self.imagination.context(env_ids)
    goals = [
      Goal(_row(frames, j), None, time_steps[j], trajectories[j], offsets[j])
      for j in range(k)
    ]
    # Step j of a window is a logged transition when no reset / resample came before.
    valid = transition_mask(continues, captured)
    dyn = (states, next_states, setpoints, context)
    if self.mirror is not None:
      dyn = self.mirror.windows(*dyn, self.mirror_probability)
    ds, dn, du, dc = dyn
    s0, s1 = self._state(ds[0]), self._state(dn[0])
    wm.update_dynamics_normalization(s0, du[0], dc, wm.target_change(s0, s1), valid[0])
    if wm.increment_norm is not None:
      wm.increment_norm.update(state_change(s0, s1), valid[0])
    s0 = self._state(states[0])
    tracking = self._tracking_features(states[0], env_ids, goals[0])
    wm.update_heads_normalization(s0, setpoints[0], tracking, context, rewards[0])
    dynamics, last_step, heads = self._model_losses(
      states,
      next_states,
      setpoints,
      valid,
      context,
      goals,
      env_ids,
      terminated,
      rewards,
      dyn,
    )
    self.model_optimizer.zero_grad(set_to_none=False)
    (dynamics + heads).backward()
    average_gradients(wm)
    grad_norm = nn.utils.clip_grad_norm_(wm.parameters(), self.max_grad_norm)
    step_if_finite(
      self.model_optimizer,
      wm.parameters(),
      enabled=self.skip_nonfinite_updates,
      valid=torch.isfinite(dynamics + heads) & torch.isfinite(grad_norm),
    )
    return torch.stack([dynamics, last_step, heads]).detach()

  def _model_losses(
    self,
    states: torch.Tensor,
    next_states: torch.Tensor,
    setpoints: torch.Tensor,
    valid: torch.Tensor,
    context: torch.Tensor,
    goals: list[Goal],
    env_ids: torch.Tensor,
    terminated: torch.Tensor,
    rewards: torch.Tensor,
    dynamics_window: tuple[torch.Tensor, ...],
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Roll the model out on a window's logged PD targets from its first logged state:
    (state error over the window, at its last step, step-heads loss on the predicted
    states — where imagination uses them)."""
    ds, dn, du, dc = dynamics_window
    state, dynamic_state = states[0], ds[0]
    errors, heads = [], []
    for j in range(self.window):
      state, dynamic_state, error, head = self._model_loss_step(
        state,
        dynamic_state,
        setpoints[j],
        du[j],
        context,
        dc,
        dn[j],
        valid[j],
        goals[j],
        env_ids,
        terminated[j],
        rewards[j],
      )
      errors.append(error)
      heads.append(head)
    return torch.stack(errors).mean(), errors[-1], torch.stack(heads).mean()

  def _model_loss_step(
    self,
    state,
    dynamic_state,
    setpoint,
    dynamic_setpoint,
    context,
    dynamic_context,
    successor,
    valid,
    goal,
    env_ids,
    terminated,
    rewards,
  ):
    """Compile one recurrent fitting step; the enclosing CUDA graph captures the window.

    Sharing this graph across window positions avoids recompiling an unrolled FK /
    observation graph for every window length. Heads keep the unmirrored trajectory.
    """
    wm, im = self.world_model, self.imagination
    s = self._state(state)
    zeros = torch.zeros(state.shape[0], im.num_actions, device=self.device)
    view = im.view(s, im.kinematics.bodies(s), env_ids, goal, zeros, zeros)
    logit, predicted = wm.step_heads(s, setpoint, im.tracking_features(view), context)
    loss = F.binary_cross_entropy_with_logits(
      logit, terminated.float(), reduction="none"
    )
    if self.num_learned_rewards:
      loss = loss + wm.reward_error(predicted, rewards).mean(-1)
    d = wm.step(self._state(dynamic_state), dynamic_setpoint, dynamic_context)
    s = wm.step(s, setpoint, context) if self.mirror is not None else d
    error = wm.change_error(d, self._state(successor)).mean(-1)
    return s.vector(), d.vector(), _masked_mean(error, valid), _masked_mean(loss, valid)

  def _tracking_features(
    self, state: torch.Tensor, env_ids: torch.Tensor, goal: Goal
  ) -> torch.Tensor:
    im = self.imagination
    s = self._state(state)
    zeros = torch.zeros(state.shape[0], im.num_actions, device=self.device)
    view = im.view(s, im.kinematics.bodies(s), env_ids, goal, zeros, zeros)
    return im.tracking_features(view).detach()

  def _update_policy(self) -> dict[str, torch.Tensor]:
    assert self.replay is not None
    im, n, h = self.imagination, self.imagined_rollouts, self.horizon
    b = self.replay.sample_states(n, past_steps=im.history_length - 1)
    past = b.past if b.past is not None else b.observed.new_zeros(n, 0, 0)
    # References of every imagined step and those it looks ahead to; past its clip's end
    # a rollout stops and bootstraps.
    clip_end = im.command.motion.segment_end_idx[b.trajectory]
    steps = (
      b.time_step + torch.arange(h + im.window_frames, device=self.device)[:, None]
    )
    inside = steps[: h + 1] < clip_end  # [h + 1, n]
    steps = torch.minimum(steps, clip_end - 1)
    with torch.no_grad():
      frames = im.command.motion.frame(steps.reshape(-1))
    # Only the actor learns from the imagined return.
    self.world_model.requires_grad_(False)
    self.critic.requires_grad_(False)
    objective, critic_in, values, rewards, ends = self._policy_step(
      b.state,
      past,
      b.last_action,
      b.env_ids,
      b.trajectory,
      b.reference_offset,
      steps,
      inside,
      _unflatten(frames, steps.shape[0]),
    )
    self.world_model.requires_grad_(True)
    self.critic.requires_grad_(True)
    critic_loss = self._update_critic(critic_in, values, rewards, ends, inside[:h])
    return {
      "imagined_return": objective.detach(),
      "imagined_reward": rewards.mean(),
      "imagined_termination": ends.mean(),
      "critic": critic_loss,
      "imagined_value_max": values.max(),
      "imagined_value_min": values.min(),
    }

  def _policy_step(
    self,
    state: torch.Tensor,
    past: torch.Tensor,
    last_action: torch.Tensor,
    env_ids: torch.Tensor,
    trajectory: torch.Tensor,
    reference_offset: torch.Tensor,
    steps: torch.Tensor,
    inside: torch.Tensor,
    frames: MotionFrame,
  ) -> tuple[torch.Tensor, ...]:
    """Imagine ``horizon`` steps from replay states and take one actor step on the return.

    Returns the mean return and, per step, the critic inputs, values, rewards and
    termination probabilities (for the critic's targets).
    """
    im, h, ahead = self.imagination, self.horizon, self.imagination.window_frames
    context = im.context(env_ids)
    survival = torch.ones(state.shape[0], device=self.device)  # c_t
    objective = torch.zeros_like(survival)
    critic_in, values, rewards, ends = [], [], [], []
    s, prev_action = state, last_action
    for t in range(h + 1):
      window = _window(frames, t, ahead) if ahead > 1 else None
      goal = Goal(_row(frames, t), window, steps[t], trajectory, reference_offset)
      out = self._imagine_step(
        s, past, prev_action, goal, env_ids, context, last=t == h
      )
      value, x = out[0], out[1]
      critic_in.append(x)
      values.append(value.detach())
      # Bootstrap where a rollout stops: its clip ended at this step, or the horizon.
      if t > 0:
        stop = inside[t - 1] & (~inside[t] if t < h else True)
        objective = (
          objective + self.gamma**t * survival * self._bootstrap_value(value) * stop
        )
      if t == h:
        break
      s, past, action, reward, end = out[2:]
      objective = objective + self.gamma**t * survival * reward * inside[t]
      survival = survival * (1 - end)
      prev_action = action
      rewards.append(reward.detach())
      ends.append(end.detach())

    actor_params = [p for p in self.actor.parameters() if p.requires_grad]
    self.actor_optimizer.zero_grad(set_to_none=False)
    (-objective.mean()).backward()
    average_gradients(self.actor)
    grad_norm = nn.utils.clip_grad_norm_(actor_params, self.max_grad_norm)
    for p in actor_params:
      p.grad.clamp_(-self.actor_grad_clip, self.actor_grad_clip)
    step_if_finite(
      self.actor_optimizer,
      actor_params,
      enabled=self.skip_nonfinite_updates,
      valid=torch.isfinite(objective).all() & torch.isfinite(grad_norm),
    )
    return (
      objective.mean().detach(),
      torch.stack(critic_in),
      torch.stack(values),
      torch.stack(rewards),
      torch.stack(ends),
    )

  def _imagine_step(
    self,
    state: torch.Tensor,
    past: torch.Tensor,
    prev_action: torch.Tensor,
    goal: Goal,
    env_ids: torch.Tensor,
    context: torch.Tensor,
    last: bool = False,
  ) -> tuple:
    """One imagined step from ``state``: its value (and critic input) and, unless ``last``,
    the next state, observation history, action, the step's reward and termination
    probability."""
    im, wm = self.imagination, self.world_model
    s = self._state(state)
    at_s = im.view(s, im.kinematics.bodies(s), env_ids, goal, prev_action, prev_action)
    obs, history = im.actor_obs(at_s, past)  # noisy, as the policy sees it
    tracking = im.tracking_features(at_s)
    critic_obs = obs
    if self.critic_goal_indices is not None:
      clean, _ = im.actor_obs(at_s, past, corruption=False)
      critic_obs = torch.cat(
        [clean[:, self.critic_goal_indices], obs[:, self.critic_history_indices]],
        dim=-1,
      )
    x = torch.cat([state_features(s), tracking, context, critic_obs], dim=-1)
    value = self.critic(self.critic_norm(x)).squeeze(-1)
    if last:
      return value, x.detach()
    action = self.actor(TensorDict({"actor": obs}, batch_size=[obs.shape[0]]))
    if self.clip_actions is not None:
      action = action.clamp(-self.clip_actions, self.clip_actions)
    setpoint = im.joint_setpoint(action, goal.frame, env_ids)
    logit, learned = wm.step_heads(s, setpoint, tracking, context)
    s_next = wm.step(s, setpoint, context)
    # Reward of the step: the state it reached against this step's reference.
    at_next = im.view(
      s_next, im.kinematics.bodies(s_next), env_ids, goal, action, prev_action
    )
    reward = im.analytic_reward(at_next) + learned.sum(-1) * im.reward_scale
    end = torch.sigmoid(logit)
    next_state = s_next.vector()
    if self.tracking_termination is not None:
      end = torch.maximum(end, self.tracking_termination(at_next))
      # Once soft survival has saturated, do not extrapolate through a dead robot.
      # Keep a finite predecessor for unused tail rows; their survival weight is zero.
      stopped = end >= 1 - 1e-6
      end = torch.where(stopped, torch.ones_like(end), end)
      next_state = torch.where(stopped[:, None], state, next_state)
    return (
      value,
      x.detach(),
      next_state,
      history[:, 1:],
      action,
      reward,
      end,
    )

  def _update_critic(
    self,
    critic_in: torch.Tensor,
    values: torch.Tensor,
    rewards: torch.Tensor,
    ends: torch.Tensor,
    inside: torch.Tensor,
  ) -> torch.Tensor:
    """Fit V to GAE(λ) targets of the imagined rollouts (Eq. 3)."""
    # BlinkMimic train.py: update normalization BEFORE evaluating target values.
    # Otherwise targets and predictions describe different value functions.
    x = critic_in[:-1].flatten(0, 1)
    survival = torch.ones_like(inside, dtype=rewards.dtype)
    if getattr(self, "termination_mode", "learned") == "hybrid":
      survival = torch.cat([torch.ones_like(ends[:1]), (1 - ends[:-1]).cumprod(0)])
    weights = (inside * survival).flatten().float()
    # Include the terminal bootstrap input, as upstream does. After a clip ends,
    # include its first bootstrap state but exclude unused trailing rollout rows.
    value_weights = torch.cat([inside[:1], inside], dim=0).flatten().float()
    if getattr(self, "termination_mode", "learned") == "hybrid":
      value_weights *= torch.cat(
        [torch.ones_like(ends[:1]), (1 - ends).cumprod(0)]
      ).flatten()
    self.critic_norm.update(critic_in.flatten(0, 1), value_weights)
    with torch.no_grad():
      # Do not evaluate the critic on excluded states: masking an infinite value
      # afterwards would still produce 0 * inf in the lambda-return recursion.
      target_in = torch.where(
        value_weights.reshape(critic_in.shape[:2])[..., None] > 0,
        critic_in,
        torch.zeros_like(critic_in),
      )
      values = self._bootstrap_value(
        self.critic(self.critic_norm(target_in)).squeeze(-1)
      )
      target_rewards = torch.where(
        weights.reshape(rewards.shape) > 0, rewards, torch.zeros_like(rewards)
      )
      targets = lambda_returns(
        target_rewards, ends, values, inside, self.gamma, self.lam
      )
    y = targets.flatten()
    total = torch.zeros((), device=self.device)
    for _ in range(self.critic_epochs):
      # tensor_split keeps the collective count identical even for small or uneven
      # clip-masked batches on different ranks. Empty minibatches yield zero gradients.
      batches = torch.randperm(x.shape[0], device=self.device).tensor_split(
        self.critic_mini_batches
      )
      for idx in batches:
        loss = self._critic_loss(x[idx], y[idx], weights[idx])
        self.critic_optimizer.zero_grad()
        loss.backward()
        average_gradients(self.critic)
        grad_norm = nn.utils.clip_grad_norm_(
          self.critic.parameters(), self.max_grad_norm
        )
        step_if_finite(
          self.critic_optimizer,
          self.critic.parameters(),
          enabled=self.skip_nonfinite_updates,
          valid=torch.isfinite(loss) & torch.isfinite(grad_norm),
        )
        total += loss.detach() / (self.critic_epochs * len(batches))
    return total

  def _bootstrap_value(self, value):
    # Repository safeguard, not an upstream Blink feature: no policy can earn
    # more than r_max/(1-gamma). Keep the raw critic for regression and diagnostics.
    return (
      value
      if self.value_upper_bound is None
      else value.clamp_max(self.value_upper_bound)
    )

  def _critic_loss(self, x, y, weights):
    x = torch.where(weights[:, None] > 0, x, torch.zeros_like(x))
    y = torch.where(weights > 0, y, torch.zeros_like(y))
    error = (self.critic(self.critic_norm(x)).squeeze(-1) - y).square()
    return (error * weights).sum() / weights.sum().clamp_min(1)

  def _set_learning_rate(self) -> None:
    """Cosine decay to ``lr_final_ratio`` over ``lr_decay_updates`` updates, then constant."""
    p = min(self.num_updates / self.lr_decay_updates, 1.0)
    ratio = self.lr_final_ratio + (1 - self.lr_final_ratio) * 0.5 * (
      1 + math.cos(math.pi * p)
    )
    self.learning_rate = self.base_learning_rate * ratio
    if self.actor_lr_return_threshold is not None:
      self._actor_decay_at.copy_(
        torch.where(
          self._reward_ema >= self.actor_lr_return_threshold,
          self._actor_decay_at.clamp_max(self.num_updates),
          self._actor_decay_at,
        )
      )
    progress = (
      (self.num_updates - self._actor_decay_at) / self.actor_lr_decay_updates
    ).clamp(0, 1)
    multiplier = 1 + (self.actor_lr_multiplier - 1) * (1 - progress)
    for o in self._optimizers:
      for group in o.param_groups:
        value = (
          self.learning_rate * multiplier
          if o is self.actor_optimizer
          else torch.tensor(self.learning_rate, device=self.device)
        )
        if isinstance(group["lr"], torch.Tensor):
          group["lr"].copy_(value)
        else:
          group["lr"] = value.clone()


def graph_tensors(modules, optimizers=()):
  """Tensor storage read by a captured update, including nested Muon/Adam state.

  Address validation inspired by official SuperMimic learner_utils.py (Apache-2.0),
  abc124c1bd5976341034668af3d12c087f2f1f55. No device synchronization is required.
  """
  tensors = [t for module in modules for t in (*module.parameters(), *module.buffers())]

  def visit(value):
    if isinstance(value, torch.Tensor):
      tensors.append(value)
    elif isinstance(value, dict):
      for child in value.values():
        visit(child)
    elif isinstance(value, (list, tuple)):
      for child in value:
        visit(child)

  for optimizer in optimizers:
    visit(optimizer.state_dict())
  return tensors


class GraphStorageGuard:
  """Fail before replay if a referenced tensor is replaced, resized or re-strided."""

  def __init__(self, state):
    self.state = state
    self.signature = None

  def current(self):
    return [
      (t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype, t.device)
      for t in self.state()
    ]

  def capture(self):
    self.signature = self.current()

  def check(self):
    if self.signature != self.current():
      raise RuntimeError(
        "CUDA graph storage changed; reset and recapture the graph before replay"
      )


class CudaGraphed:
  """``fn`` as one CUDA graph: after ``warmup`` eager calls (on a side stream, as capture
  wants), it is captured once; each call then copies its tensor arguments into the
  captured ones and replays. Arguments must keep their shapes; the returned tensors are
  overwritten by the next call."""

  def __init__(self, fn, warmup: int = 2, state=None):
    self._fn, self._warmup, self._calls = fn, warmup, 0
    self._graph: torch.cuda.CUDAGraph | None = None
    self._inputs: Any = None
    self._outputs: Any = None
    self._guard = GraphStorageGuard(state) if state is not None else None

  def reset(self):
    self._graph = self._inputs = self._outputs = None
    self._calls = 0

  def __call__(self, *args):
    # Contiguous in every call: compiled callees specialize on input strides.
    if self._graph is None and self._calls < self._warmup:
      self._calls += 1
      stream = torch.cuda.Stream()
      stream.wait_stream(torch.cuda.current_stream())
      with torch.cuda.stream(stream):
        out = self._fn(*_map_tensors(args, torch.Tensor.contiguous))
      torch.cuda.current_stream().wait_stream(stream)
      return out
    if self._graph is None:
      self._inputs = _map_tensors(
        args, lambda x: x.clone(memory_format=torch.contiguous_format)
      )
      self._graph = torch.cuda.CUDAGraph()
      with torch.cuda.graph(self._graph):
        self._outputs = self._fn(*self._inputs)
      if self._guard is not None:
        self._guard.capture()
    else:
      if self._guard is not None:
        self._guard.check()
      _copy_tensors(self._inputs, args)
    self._graph.replay()
    return self._outputs


@torch.no_grad()
def lambda_returns(rewards, ends, values, inside, gamma, lam):
  """SuperMimic Eq. 3, with this repo's clip-end bootstrap preserved.

  Numerical reference: SuperMimic-Reproduced (ba61053), imagination/objective.py,
  https://github.com/CuboiLeo/SuperMimic-Reproduced/blob/ba6105349069bf040f1fdad60873d13138d505a2/src/supermimic/imagination/objective.py
  """
  target, targets = values[-1], []
  for t in reversed(range(rewards.shape[0])):
    blend = (1 - lam) * values[t + 1] + lam * target
    target = torch.where(
      inside[t], rewards[t] + gamma * (1 - ends[t]) * blend, values[t]
    )
    targets.append(target)
  return torch.stack(targets[::-1])


def _map_tensors(x, fn):
  if isinstance(x, torch.Tensor):
    return fn(x)
  if dataclasses.is_dataclass(x):
    return type(x)(
      *(_map_tensors(getattr(x, f.name), fn) for f in dataclasses.fields(x))
    )
  if isinstance(x, (list, tuple)):
    return type(x)(_map_tensors(v, fn) for v in x)
  return x


def _copy_tensors(dst, src) -> None:
  if isinstance(dst, torch.Tensor):
    dst.copy_(src)
  elif dataclasses.is_dataclass(dst):
    for f in dataclasses.fields(dst):
      _copy_tensors(getattr(dst, f.name), getattr(src, f.name))
  elif isinstance(dst, (list, tuple)):
    for d, s in zip(dst, src, strict=True):
      _copy_tensors(d, s)


def _unflatten(frame: MotionFrame, k: int) -> MotionFrame:
  """A frame of ``k * n`` rows (step-major) as fields ``(k, n, …)``."""
  return _map_tensors(frame, lambda x: x.unflatten(0, (k, -1)))


def _row(frames: MotionFrame, t: int) -> MotionFrame:
  """Step ``t`` of frames shaped ``(steps, n, …)``."""
  return _map_tensors(frames, lambda x: x[t])


def _window(frames: MotionFrame, t: int, count: int) -> MotionFrame:
  """Steps ``t … t + count - 1`` of frames shaped ``(steps, n, …)``, as ``(n, count, …)``."""
  return _map_tensors(frames, lambda x: x[t : t + count].transpose(0, 1))


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
  return torch.where(mask, x, 0.0).sum() / mask.sum().clamp(min=1)
