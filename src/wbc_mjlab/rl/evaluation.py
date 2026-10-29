"""Whole-clip tracking evaluation and drift measurements for any WBC algorithm.

Protocol: SuperMimic (Liao et al., 2026), Appendix F,
https://arxiv.org/abs/2610.28480. Queueing and drift decomposition are adapted from
CuboiLeo/SuperMimic-Reproduced (ba61053), src/supermimic/eval/benchmark.py and
scripts/diag_drift.py:
https://github.com/CuboiLeo/SuperMimic-Reproduced/tree/ba6105349069bf040f1fdad60873d13138d505a2
The implementation uses the repo's presets, motion library and policy interface;
normalization bounds are supplied per task, never assumed to be 0.1.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from mjlab.utils.lab_api.math import quat_apply, quat_conjugate, quat_mul, yaw_quat
from tensordict import TensorDict


def clip_finished(env, command_name: str = "motion") -> torch.Tensor:
  command = env.command_manager.get_term(command_name)
  return (
    command.time_steps >= command.motion.segment_end_idx[command.trajectory_ids] - 1
  )


def deterministic_cfg(cfg):
  """Keep preset terms/weights, disable noise, randomization and external events."""
  cfg.seed = 0
  cfg.episode_length_s = 1e9
  cfg.curriculum = {}
  cfg.recorders = {}
  # Explicit deterministic protocol: startup physical parameters are the nominal model.
  from mjlab.envs.mdp import dr

  events = {}
  for name, event in cfg.events.items():
    if event.func is dr.geom_friction:
      event.mode = "startup"
      event.params["ranges"] = (1.0, 1.0)
      event.params["operation"] = "abs"
      events[name] = event
    elif event.mode == "reset" and not getattr(event.func, "__module__", "").startswith(
      dr.__name__
    ):
      events[name] = event
  if not any(event.func is dr.geom_friction for event in events.values()):
    from mjlab.managers.event_manager import EventTermCfg
    from mjlab.managers.scene_entity_config import SceneEntityCfg

    events["benchmark_friction"] = EventTermCfg(
      func=dr.geom_friction,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg(
          cfg.commands["motion"].entity_name, geom_names=(".*",)
        ),
        "operation": "abs",
        "ranges": (1.0, 1.0),
        "shared_random": True,
      },
    )
  cfg.events = events
  for group in cfg.observations.values():
    group.enable_corruption = False
    for term in group.terms.values():
      if term is not None:
        term.noise = None
        term.delay_min_lag = term.delay_max_lag = 0
  command = cfg.commands["motion"]
  command.pose_range = {}
  command.velocity_range = {}
  command.joint_position_range = (0.0, 0.0)
  command.assistive_wrench_enabled = False
  command.rsi.sampling_mode = "start"
  from mjlab.managers.termination_manager import TerminationTermCfg

  cfg.terminations = {n: t for n, t in cfg.terminations.items() if not t.time_out}
  cfg.terminations["benchmark_clip_end"] = TerminationTermCfg(
    func=clip_finished, time_out=True
  )
  return cfg


def reward_upper_bound(cfg, step_dt: float) -> float | None:
  """Infer only known kernel bounds and nonpositive penalties, by function identity."""
  from mjlab.envs import mdp

  from wbc_mjlab.env.mdp import rewards

  kernels = {
    func
    for name, func in vars(rewards).items()
    if name.startswith("motion_") and name.endswith("_exp") and callable(func)
  }
  penalties = {mdp.action_rate_l2, mdp.joint_pos_limits, rewards.action_rate_l1}
  bound = 0.0
  for term in cfg.rewards.values():
    if not term.weight:
      continue
    if term.func in kernels and term.params.get("kappa", 1.0) >= 0:
      bound += max(term.weight, 0.0)
    elif term.func is mdp.is_alive:
      bound += max(term.weight, 0.0)
    elif term.func not in penalties or term.weight > 0:
      return None
  bound *= step_dt if cfg.scale_rewards_by_dt else 1.0
  return bound if bound > 0 else None


@dataclass
class ClipResult:
  trajectory: int
  name: str
  frames: int
  steps: int
  return_sum: float
  normalized_return: float | None
  failed: bool
  failure_step: int | None
  failure_terms: list[str]
  reward_terms: dict[str, float]
  drift: dict[str, float | None]


def summarize(results: list[ClipResult]) -> dict:
  n = len(results)
  steps = sum(r.steps for r in results)
  normalized = [r for r in results if r.normalized_return is not None]
  return {
    "clips": n,
    "failed": sum(r.failed for r in results),
    "failure_rate": sum(r.failed for r in results) / n if n else 0.0,
    "steps": steps,
    "mean_normalized_return": sum(r.normalized_return for r in normalized)
    / len(normalized)
    if normalized
    else None,
    "step_weighted_normalized_return": sum(
      r.normalized_return * r.steps for r in normalized
    )
    / sum(r.steps for r in normalized)
    if normalized
    else None,
    "mean_reward_per_step": sum(r.return_sum for r in results) / steps
    if steps
    else None,
  }


class DriftMeter:
  """Measurements before each action, aligned with that action's reference target.

  Reports velocity MSE directly rather than exponentiating a time-averaged MSE
  (which would not equal the mean exponential tracking reward).
  """

  names = (
    "xy_m",
    "yaw_deg",
    "linear_velocity_world_mse",
    "linear_velocity_aligned_mse",
    "angular_velocity_world_mse",
    "angular_velocity_aligned_mse",
  )

  def __init__(self, num_envs, device):
    self.sums = torch.zeros(num_envs, len(self.names), device=device)
    self.last = torch.zeros(num_envs, 2, device=device)
    self.crossing = torch.full((num_envs, 2), -1.0, device=device)

  def update(self, command, active, steps, dt):
    xy = (command.anchor_pos_w[:, :2] - command.robot_anchor_pos_w[:, :2]).norm(dim=-1)
    # Difference of headings, not heading of the relative 3D rotation: the latter
    # can report spurious yaw error when the two bodies have different roll/tilt.
    q = quat_mul(
      yaw_quat(command.anchor_quat_w),
      quat_conjugate(yaw_quat(command.robot_anchor_quat_w)),
    )
    yaw = 2 * torch.atan2(q[:, 3], q[:, 0])
    yaw = torch.atan2(yaw.sin(), yaw.cos()).abs() * (180 / math.pi)
    body_q = q[:, None, :].expand(-1, command.robot_body_lin_vel_w.shape[1], -1)
    errors = []
    for desired, actual in (
      (command.body_lin_vel_w, command.robot_body_lin_vel_w),
      (command.body_ang_vel_w, command.robot_body_ang_vel_w),
    ):
      errors.extend(
        [
          (desired - actual).square().sum(-1).mean(-1),
          (desired - quat_apply(body_q, actual)).square().sum(-1).mean(-1),
        ]
      )
    values = torch.stack([xy, yaw, *errors], dim=-1)
    self.sums += values * active[:, None]
    self.last = torch.where(active[:, None], values[:, :2], self.last)
    over = torch.stack([xy > 0.3, yaw > 15], dim=-1)
    self.crossing = torch.where(
      active[:, None] & (self.crossing < 0) & over, (steps * dt)[:, None], self.crossing
    )

  def result(self, env_id, steps):
    values = (self.sums[env_id] / max(steps, 1)).tolist()
    result = {f"mean_{n}": v for n, v in zip(self.names, values, strict=True)}
    result.update(
      final_xy_m=self.last[env_id, 0].item(), final_yaw_deg=self.last[env_id, 1].item()
    )
    for name, value in zip(
      ("time_xy_gt_0p3_m_s", "time_yaw_gt_15_deg_s"),
      self.crossing[env_id].tolist(),
      strict=True,
    ):
      result[name] = value if value >= 0 else None
    return result

  def reset(self, ids):
    self.sums[ids] = 0
    self.last[ids] = 0
    self.crossing[ids] = -1


@torch.no_grad()
def run_benchmark(
  env, policy, *, trajectories=None, max_step_reward=None, clip_actions=None
):
  """Play each requested trajectory once. Returns serializable summary and clip rows.

  ``env`` must use deterministic_cfg; the caller owns and closes it. Reward means
  include the terminal step. Drift is measured before stepping to avoid auto-reset
  contamination. A failure is any configured non-timeout termination.
  """
  if max_step_reward is not None and (
    not math.isfinite(max_step_reward) or max_step_reward <= 0
  ):
    raise ValueError("max_step_reward must be finite and positive")
  command = env.command_manager.get_term("motion")
  motion, device, n = command.motion, env.device, env.num_envs
  selected = (
    list(range(motion.num_trajectories)) if trajectories is None else list(trajectories)
  )
  if not selected or len(set(selected)) != len(selected):
    raise ValueError("benchmark needs a nonempty list of unique trajectories")
  if min(selected) < 0 or max(selected) >= motion.num_trajectories:
    raise ValueError("benchmark trajectory is outside the motion library")
  queue = torch.tensor(selected, device=device, dtype=torch.long)
  slots = torch.arange(n, device=device)

  def assignment(indices):
    return queue[indices.clamp(max=len(selected) - 1)]

  command.set_start_trajectories(assignment(slots))
  obs, _ = env.reset()
  command.set_start_trajectories(assignment(slots + n))
  active = slots < len(selected)
  returns = torch.zeros(n, device=device)
  steps = torch.zeros(n, dtype=torch.long, device=device)
  reward_names = env.reward_manager.active_terms
  terms = torch.zeros(n, len(reward_names), device=device)
  drift = DriftMeter(n, device)
  results = []
  policy.eval()
  while len(results) < len(selected):
    drift.update(command, active, steps, env.step_dt)
    action = policy(TensorDict(obs, batch_size=[n]))
    if clip_actions is not None:
      action = action.clamp(-clip_actions, clip_actions)
    obs, reward, terminated, truncated, _ = env.step(action)
    returns += reward * active
    terms += env.reward_manager._step_reward * active[:, None]
    steps += active.long()
    done = terminated | truncated
    ids = (done & active).nonzero().flatten()
    for e in ids.tolist():
      trajectory, count = selected[int(slots[e])], int(steps[e])
      failed = bool(terminated[e])
      result = ClipResult(
        trajectory,
        motion.segment_names[trajectory],
        int(motion.segment_length[trajectory]),
        count,
        returns[e].item(),
        returns[e].item() / (max_step_reward * count)
        if max_step_reward is not None
        else None,
        failed,
        count if failed else None,
        [
          name
          for name in env.termination_manager.active_terms
          if not env.termination_manager.get_term_cfg(name).time_out
          and env.termination_manager.get_term(name)[e]
        ],
        {name: terms[e, i].item() / count for i, name in enumerate(reward_names)},
        drift.result(e, count),
      )
      results.append(result)
    # Inactive envs still auto-reset. Advance only active queue slots; their reset
    # already used the next assignment installed before the step.
    slots[ids] += n
    active[ids] = slots[ids] < len(selected)
    if ids.numel():
      command.set_start_trajectories(assignment(slots[ids] + n), env_ids=ids)
      print(f"[EVAL] clips {len(results)}/{len(selected)}", flush=True)
    returns[ids], steps[ids], terms[ids] = 0, 0, 0
    drift.reset(ids)
  order = {trajectory: i for i, trajectory in enumerate(selected)}
  results.sort(key=lambda r: order[r.trajectory])
  return {"summary": summarize(results), "clips": [asdict(r) for r in results]}


@dataclass
class EvaluationConfig:
  checkpoint_file: Path | str
  task: str | None = None
  dataset: str | None = None
  dataset_path: str | None = None
  num_envs: int = 128
  device: str = "cuda:0"
  clips: list[str] | None = None
  max_step_reward: float | None = None
  out: Path | str | None = None


def evaluate_checkpoint(config: EvaluationConfig):
  """Shared CLI/Studio backend: restore a run, apply the protocol, evaluate, write JSON."""
  import json
  from pathlib import Path

  from mjlab.envs import ManagerBasedRlEnv

  from wbc_mjlab.data_paths import resolve_training_motion_file
  from wbc_mjlab.rl.policy import load_actor
  from wbc_mjlab.run_params import load_run_config
  from wbc_mjlab.tasks import get_task_config

  if config.num_envs < 1:
    raise ValueError("num_envs must be positive")
  checkpoint = Path(config.checkpoint_file).resolve()
  run = load_run_config(checkpoint.parent, config.task)
  cfg, agent = run.env, run.agent
  motion = cfg.commands["motion"]
  if config.dataset is not None or config.dataset_path is not None:
    motion.motion_file = str(
      resolve_training_motion_file(
        get_task_config(run.task_id).robot_id,
        dataset=config.dataset,
        dataset_path=config.dataset_path,
      )
    )
  cfg.scene.num_envs = config.num_envs
  deterministic_cfg(cfg)
  env = ManagerBasedRlEnv(cfg, device=config.device)
  try:
    env.reset()
    actor = load_actor(env, asdict(agent), checkpoint, config.device)
    names = env.command_manager.get_term("motion").motion.segment_names
    if config.clips is not None and set(config.clips) - set(names):
      raise ValueError(f"unknown clips: {set(config.clips) - set(names)}")
    trajectories = (
      [names.index(name) for name in config.clips] if config.clips else None
    )
    bound = config.max_step_reward
    if bound is None:
      bound = reward_upper_bound(cfg, env.step_dt)
    result = run_benchmark(
      env,
      actor,
      trajectories=trajectories,
      max_step_reward=bound,
      clip_actions=agent.clip_actions,
    )
    result["protocol"] = {
      "task": run.task_id,
      "checkpoint": str(checkpoint),
      "motion_file": str(motion.motion_file),
      "step_dt_s": env.step_dt,
      "max_step_reward": bound,
      "friction": 1.0,
      "seed": 0,
      "start": "first frame",
      "deterministic_policy": True,
      "noise": False,
      "domain_randomization": False,
      "reward_terms": "weighted mean per step, before dt scaling",
      "sources": [
        "https://arxiv.org/abs/2610.28480",
        "https://github.com/CuboiLeo/SuperMimic-Reproduced/tree/ba6105349069bf040f1fdad60873d13138d505a2",
      ],
    }
    out = (
      Path(config.out)
      if config.out
      else checkpoint.parent / f"benchmark_{checkpoint.stem}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result, out
  finally:
    env.close()
