"""World-model RL for motion tracking (``--algo wm``), after SuperMimic (arXiv:2610.28480).

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

import dataclasses
import math
import time
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP
from rsl_rl.utils import resolve_callable, resolve_obs_groups
from tensordict import TensorDict
from torch import nn

from wbc_mjlab.motion.library import MotionFrame
from wbc_mjlab.rl.wm.imagined import Goal, Imagination
from wbc_mjlab.rl.wm.kinematics import RobotState
from wbc_mjlab.rl.wm.model import (
  RunningNorm,
  WorldModel,
  feature_dim,
  state_change,
  state_features,
  update_running_stats,
)
from wbc_mjlab.rl.wm.replay import Replay

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
    critic_activation: str,
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
    device: str = "cpu",
  ):
    del action_noise  # set on the actor's distribution by ``construct_algorithm``
    self.env, self.device = env, device
    self.imagination = im = Imagination(env)
    print("[INFO] wm:", im.summary().replace("\n", "\n[INFO] wm: "))
    self.num_joints = len(im.joint_names)
    probe = im.view(*im._probe())
    tracking_dim = im.tracking_features(probe).shape[-1]
    context_dim = im.context(torch.arange(1, device=device)).shape[-1]
    obs_dim = im.actor_obs(probe, im.env_history()[:, :-1])[0].shape[-1]
    self.num_learned_rewards = len(im.learned_rewards)

    self.actor = actor.to(device)
    self.world_model = WorldModel(
      self.num_joints,
      context_dim,
      tracking_dim,
      self.num_learned_rewards,
      model_hidden_dims,
      heads_hidden_dims,
    ).to(device)
    # The critic sees the state, its tracking features, the context and the actor's
    # observation (the reference ahead): V(s, g, ξ).
    critic_in = feature_dim(self.num_joints) + tracking_dim + context_dim + obs_dim
    self.critic_norm = RunningNorm(critic_in, eps=1e-4).to(device)
    self.critic = MLP(critic_in, 1, critic_hidden_dims, critic_activation).to(device)

    # Learning rates as tensors: the schedule changes them inside captured graphs.
    lr = torch.tensor(learning_rate, device=device)
    # Graphs need compiled steps: eager terms copy Python index lists to the GPU.
    graphs = cuda_graphs and compile and torch.device(device).type == "cuda"
    self.model_optimizer = torch.optim.Adam(
      self.world_model.parameters(), lr=lr.clone(), capturable=graphs
    )
    self.actor_optimizer = torch.optim.Adam(
      [p for p in self.actor.parameters() if p.requires_grad],
      lr=lr.clone(),
      betas=(0.9, 0.95),
      capturable=graphs,
    )
    self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr.clone())
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
    self._updated = False  # in this process (the first update compiles)
    self._compile = compile
    self.replay: Replay | None = None  # allocated on the first step (not for Play)
    self._within_clip: torch.Tensor | None = None  # the step does not end its clip
    if compile:  # fuse the many small ops of kinematics, terms and quaternions
      self._imagine_step = torch.compile(self._imagine_step, dynamic=False)
      self._model_losses = torch.compile(self._model_losses, dynamic=False)
      self._tracking_features = torch.compile(self._tracking_features, dynamic=False)
    self._model_step = CudaGraphed(self._model_step) if graphs else self._model_step
    self._policy_step = CudaGraphed(self._policy_step) if graphs else self._policy_step

  # --- rsl_rl algorithm interface ---

  @staticmethod
  def construct_algorithm(
    obs: TensorDict, env: VecEnv, cfg: dict, device: str
  ) -> WorldModelRL:
    if cfg.get("multi_gpu"):
      raise ValueError("world-model training runs on a single GPU")
    alg_cfg = dict(cfg["algorithm"])
    alg_cfg.pop("class_name", None)
    cfg["algorithm"]["rnd_cfg"] = None  # the runner reads it
    cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], ["actor", "critic"])
    if list(cfg["obs_groups"]["actor"]) != ["actor"]:
      raise ValueError(
        "world-model training needs the actor to read the 'actor' group only"
      )
    actor_cfg = dict(cfg["actor"])
    actor_class: type[MLPModel] = resolve_callable(actor_cfg.pop("class_name"))  # type: ignore
    if actor_class is not MLPModel:
      raise ValueError("world-model training needs an MLPModel actor")
    actor_cfg["distribution_cfg"] = {
      "class_name": "GaussianDistribution",
      "init_std": alg_cfg["action_noise"],
      "std_type": "scalar",
      "learn_std": False,
    }
    actor = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **actor_cfg)
    print(f"Actor Model: {actor}")
    return WorldModelRL(
      env.unwrapped,
      actor,
      tuple(cfg["critic"]["hidden_dims"]),
      cfg["critic"]["activation"],
      device=device,
      **alg_cfg,
    )

  def act(self, obs: TensorDict) -> torch.Tensor:
    if self.replay is None:
      with torch.inference_mode(False):
        self.replay = self._new_replay()
    im, command = self.imagination, self.imagination.command
    actions = self.actor(obs, stochastic_output=True)
    ids = torch.arange(self.env.num_envs, device=self.device)
    self.replay.begin(
      state=im.robot_state().vector(),
      action=actions,
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
    # In place, outside inference mode: imagination reads (and captured graphs hold) the
    # normalizer's buffers.
    norm = self.actor.obs_normalizer
    with torch.inference_mode(False):
      update_running_stats(norm._mean, norm._var, norm._std, norm.count, obs["actor"])
    # The next row continues this one unless the env reset or the clip ended (resample).
    self.replay.end(
      rewards=self.imagination.learned_reward_targets(),
      terminated=self.env.reset_terminated,
      done=dones.bool(),
      continues=~dones.bool() & self._within_clip,
    )

  def compute_returns(self, obs: TensorDict) -> None:
    pass

  def update(self) -> dict[str, float]:
    assert self.replay is not None
    if self.replay.size < max(self.warmup_steps, self.window + 1):
      return {}
    if not self._updated and self._compile:  # minutes of silence otherwise
      print(FIRST_UPDATE, "compiling (minutes; cached for later runs)", flush=True)
      start = time.perf_counter()
    totals: dict[str, float] = {}
    for _ in range(self.updates_per_iteration):
      stats = self._update_world_model() | self._update_policy()
      for k, v in stats.items():
        totals[k] = totals.get(k, 0.0) + v / self.updates_per_iteration
      self.num_updates += 1
      self._set_learning_rate()
    if not self._updated and self._compile:
      print(
        FIRST_UPDATE, f"compiled in {time.perf_counter() - start:.0f} s", flush=True
      )
    self._updated = True
    return totals

  def train_mode(self) -> None:
    for m in (self.actor, self.world_model, self.critic, self.critic_norm):
      m.train()

  def eval_mode(self) -> None:
    for m in (self.actor, self.world_model, self.critic, self.critic_norm):
      m.eval()

  def get_policy(self) -> MLPModel:
    return self.actor

  def save(self) -> dict:
    return {
      "actor_state_dict": self.actor.state_dict(),
      "critic_state_dict": {
        "norm": self.critic_norm.state_dict(),
        "mlp": self.critic.state_dict(),
      },
      "world_model_state_dict": self.world_model.state_dict(),
      "optimizer_state_dict": [o.state_dict() for o in self._optimizers],
      "num_updates": self.num_updates,
    }

  def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
    load_cfg = load_cfg or {
      "actor": True,
      "critic": True,
      "optimizer": True,
      "iteration": True,
    }
    # In place (``copy_``): captured graphs hold these tensors.
    if load_cfg.get("actor"):
      self.actor.load_state_dict(loaded_dict["actor_state_dict"], strict=strict)
    if "world_model_state_dict" not in loaded_dict:  # e.g. a PPO checkpoint: actor only
      return False
    if load_cfg.get("critic"):
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
      self.num_updates = loaded_dict.get("num_updates", 0)
      self._set_learning_rate()
    return bool(load_cfg.get("iteration"))

  def broadcast_parameters(self) -> None:
    raise NotImplementedError("world-model training runs on a single GPU")

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

  def _update_world_model(self) -> dict[str, float]:
    assert self.replay is not None
    k = self.window
    totals = torch.zeros(3, device=self.device)
    for _ in range(self.model_steps):
      w = self.replay.sample_windows(self.model_batch, k + 1)
      with torch.no_grad():
        frames = self.imagination.command.motion.frame(w.time_step[:k].reshape(-1))
      totals += self._model_step(
        w.state,
        w.setpoint,
        w.continues,
        w.terminated,
        w.rewards,
        w.time_step,
        w.trajectory,
        w.reference_offset,
        w.env_ids,
        _unflatten(frames, k),
      )
    dynamics, last_step, heads = (totals / self.model_steps).tolist()
    return {
      "wm_dynamics": dynamics,
      f"wm_error_{k}_step": last_step,
      "wm_heads": heads,
    }

  def _model_step(
    self,
    states: torch.Tensor,
    setpoints: torch.Tensor,
    continues: torch.Tensor,
    terminated: torch.Tensor,
    rewards: torch.Tensor,
    time_steps: torch.Tensor,
    trajectories: torch.Tensor,
    offsets: torch.Tensor,
    env_ids: torch.Tensor,
    frames: MotionFrame,
  ) -> torch.Tensor:
    """One gradient step of the world model on a batch of windows (``[k + 1, B]`` rows,
    reference ``frames`` ``[k, B]``); returns its losses."""
    wm, k = self.world_model, self.window
    context = self.imagination.context(env_ids)
    goals = [
      Goal(_row(frames, j), None, time_steps[j], trajectories[j], offsets[j])
      for j in range(k)
    ]
    # Step j of a window is a logged transition when no reset / resample came before.
    valid = continues[:k].long().cumprod(0).bool()
    s0, s1 = self._state(states[0]), self._state(states[1])
    wm.update_dynamics_normalization(
      s0, setpoints[0], context, state_change(s0, s1), valid[0]
    )
    tracking = self._tracking_features(states[0], env_ids, goals[0])
    wm.update_heads_normalization(s0, setpoints[0], tracking, context, rewards[0])
    dynamics, last_step, heads = self._model_losses(
      states, setpoints, valid, context, goals, env_ids, terminated, rewards
    )
    self.model_optimizer.zero_grad(set_to_none=False)
    (dynamics + heads).backward()
    nn.utils.clip_grad_norm_(wm.parameters(), self.max_grad_norm)
    self.model_optimizer.step()
    return torch.stack([dynamics, last_step, heads]).detach()

  def _model_losses(
    self,
    states: torch.Tensor,
    setpoints: torch.Tensor,
    valid: torch.Tensor,
    context: torch.Tensor,
    goals: list[Goal],
    env_ids: torch.Tensor,
    terminated: torch.Tensor,
    rewards: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Roll the model out on a window's logged PD targets from its first logged state:
    (state error over the window, at its last step, step-heads loss on the predicted
    states — where imagination uses them)."""
    wm, im = self.world_model, self.imagination
    s = self._state(states[0])
    zeros = torch.zeros(states.shape[1], im.num_actions, device=self.device)
    errors, heads = [], []
    for j in range(self.window):
      # Heads on the state reached so far (the logged one at j = 0).
      on = valid[j - 1] if j else torch.ones_like(valid[0])
      view = im.view(s, im.kinematics.bodies(s), env_ids, goals[j], zeros, zeros)
      logit, predicted = wm.step_heads(
        s, setpoints[j], im.tracking_features(view), context
      )
      loss = F.binary_cross_entropy_with_logits(
        logit, terminated[j].float(), reduction="none"
      )
      if self.num_learned_rewards:
        loss = loss + wm.reward_error(predicted, rewards[j]).mean(-1)
      heads.append(_masked_mean(loss, on))
      s = wm.step(s, setpoints[j], context)
      err = wm.change_error(s, self._state(states[j + 1])).mean(-1)
      errors.append(_masked_mean(err, valid[j]))
    return torch.stack(errors).mean(), errors[-1], torch.stack(heads).mean()

  def _tracking_features(
    self, state: torch.Tensor, env_ids: torch.Tensor, goal: Goal
  ) -> torch.Tensor:
    im = self.imagination
    s = self._state(state)
    zeros = torch.zeros(state.shape[0], im.num_actions, device=self.device)
    view = im.view(s, im.kinematics.bodies(s), env_ids, goal, zeros, zeros)
    return im.tracking_features(view).detach()

  def _update_policy(self) -> dict[str, float]:
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
      "imagined_return": objective.item(),
      "imagined_reward": rewards.mean().item(),
      "imagined_termination": ends.mean().item(),
      "critic": critic_loss,
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
        objective = objective + self.gamma**t * survival * value * stop
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
    nn.utils.clip_grad_norm_(actor_params, self.max_grad_norm)
    for p in actor_params:
      p.grad.clamp_(-self.actor_grad_clip, self.actor_grad_clip)
    self.actor_optimizer.step()
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
    x = torch.cat([state_features(s), tracking, context, obs], dim=-1)
    value = self.critic(self.critic_norm(x)).squeeze(-1)
    if last:
      return value, x.detach()
    action = self.actor(TensorDict({"actor": obs}, batch_size=[obs.shape[0]]))
    setpoint = im.joint_setpoint(action, goal.frame, env_ids)
    logit, learned = wm.step_heads(s, setpoint, tracking, context)
    s_next = wm.step(s, setpoint, context)
    # Reward of the step: the state it reached against this step's reference.
    at_next = im.view(
      s_next, im.kinematics.bodies(s_next), env_ids, goal, action, prev_action
    )
    reward = im.analytic_reward(at_next) + learned.sum(-1) * im.reward_scale
    return (
      value,
      x.detach(),
      s_next.vector(),
      history[:, 1:],
      action,
      reward,
      torch.sigmoid(logit),
    )

  def _update_critic(
    self,
    critic_in: torch.Tensor,
    values: torch.Tensor,
    rewards: torch.Tensor,
    ends: torch.Tensor,
    inside: torch.Tensor,
  ) -> float:
    """Fit V to GAE(λ) targets of the imagined rollouts (Eq. 3)."""
    target = values[-1]
    targets = []
    for t in reversed(range(rewards.shape[0])):
      blend = (1 - self.lam) * values[t + 1] + self.lam * target
      target = torch.where(
        inside[t], rewards[t] + self.gamma * (1 - ends[t]) * blend, values[t]
      )
      targets.append(target)
    targets.reverse()
    x, y = critic_in[:-1][inside], torch.stack(targets)[inside]
    self.critic_norm.update(x)
    total = torch.zeros((), device=self.device)
    for _ in range(self.critic_epochs):
      batches = torch.randperm(x.shape[0], device=self.device).chunk(
        self.critic_mini_batches
      )
      for idx in batches:
        loss = (
          (self.critic(self.critic_norm(x[idx])).squeeze(-1) - y[idx]).square().mean()
        )
        self.critic_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()
        total += loss.detach() / (self.critic_epochs * len(batches))
    return total.item()

  def _set_learning_rate(self) -> None:
    """Cosine decay to ``lr_final_ratio`` over ``lr_decay_updates`` updates, then constant."""
    p = min(self.num_updates / self.lr_decay_updates, 1.0)
    ratio = self.lr_final_ratio + (1 - self.lr_final_ratio) * 0.5 * (
      1 + math.cos(math.pi * p)
    )
    self.learning_rate = self.base_learning_rate * ratio
    for o in self._optimizers:
      for group in o.param_groups:
        group["lr"].fill_(self.learning_rate)


class CudaGraphed:
  """``fn`` as one CUDA graph: after ``warmup`` eager calls (on a side stream, as capture
  wants), it is captured once; each call then copies its tensor arguments into the
  captured ones and replays. Arguments must keep their shapes; the returned tensors are
  overwritten by the next call."""

  def __init__(self, fn, warmup: int = 2):
    self._fn, self._warmup, self._calls = fn, warmup, 0
    self._graph: torch.cuda.CUDAGraph | None = None
    self._inputs: Any = None
    self._outputs: Any = None

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
    else:
      _copy_tensors(self._inputs, args)
    self._graph.replay()
    return self._outputs


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
