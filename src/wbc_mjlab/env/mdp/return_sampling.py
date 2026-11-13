"""Visited-bin return curriculum, adapted from official SuperMimic (Apache-2.0).

Source: runners/reset_sampling.py at abc124c1bd5976341034668af3d12c087f2f1f55,
https://github.com/HybridRobotics/supermimic ; Liao et al., arXiv:2610.28480.
Use the existing segment-local bins, reset mixture and checkpoint machinery.
Values are normalized by r_max/(1-gamma), so initial optimism is one.
"""

import torch


class VisitedReturns:
  def __init__(self, shape, valid, num_envs, horizon, memory, gamma, device):
    if horizon < 1 or memory < 1 or not 0 < gamma < 1:
      raise ValueError(
        "visited returns needs positive horizon/memory and gamma in (0,1)"
      )
    self.gamma, self.memory, self.horizon = gamma, memory, horizon
    self.value = torch.ones(shape, device=device)
    self.count = torch.zeros(shape, device=device)
    self.valid = valid.flatten()
    self.bins = torch.zeros((horizon, num_envs), dtype=torch.long, device=device)
    self.next_bins = torch.zeros_like(self.bins)
    self.rewards = torch.zeros((horizon, num_envs), device=device)
    self.terminated = torch.zeros_like(self.bins, dtype=torch.bool)
    self.boundary = torch.zeros_like(self.terminated)
    self.clip_end = torch.zeros_like(self.terminated)
    self.step = 0

  @torch.no_grad()
  def record(self, bins, next_bins, rewards, terminated, boundary, clip_end):
    t = self.step
    for target, source in (
      (self.bins, bins),
      (self.next_bins, next_bins),
      (self.rewards, rewards),
      (self.terminated, terminated),
      (self.boundary, boundary),
      (self.clip_end, clip_end),
    ):
      target[t].copy_(source)
    self.step += 1
    if self.step < self.horizon:
      return False
    self.step = 0
    values = self.value.flatten()
    bootstrap = values[self.next_bins]
    bootstrap = torch.where(self.clip_end, values[self.valid].mean(), bootstrap)
    bootstrap = torch.where(self.terminated, 0.0, bootstrap)
    ret = bootstrap[-1]
    returns = torch.empty_like(self.rewards)
    for t in reversed(range(self.horizon)):
      ret = self.rewards[t] + self.gamma * torch.where(
        self.boundary[t], bootstrap[t], ret
      )
      returns[t] = ret
    # One sample per visited (env, episode, bin), not one vote per repeated frame.
    episode = self.boundary.long().cumsum(0) - self.boundary.long()
    envs = torch.arange(self.bins.shape[1], device=values.device)
    key = (envs * self.horizon + episode) * values.numel() + self.bins
    keys, inverse, visits = torch.unique(
      key.flatten(), return_inverse=True, return_counts=True
    )
    samples = (
      torch.zeros_like(keys, dtype=returns.dtype).scatter_add_(
        0, inverse, returns.flatten()
      )
      / visits
    )
    bins = keys % values.numel()
    new = torch.zeros_like(values).scatter_add_(0, bins, torch.ones_like(samples))
    total = torch.zeros_like(values).scatter_add_(0, bins, samples)
    count = self.count.flatten()
    count.copy_(count.clamp_max(self.memory) + new)
    values.add_((total - new * values) / count.clamp_min(1))
    return True


def record_visited_returns(env, env_ids=None, command_name="motion"):
  """Read-only step event: after reward/termination and BEFORE resets or clip advance."""
  del env_ids
  command = env.command_manager.get_term(command_name)
  rsi = command.cfg.rsi
  sampler = command._visited_returns
  if sampler is None or rsi.sampling_mode != "adaptive":
    return
  from wbc_mjlab.rl.evaluation import reward_upper_bound

  if not hasattr(command, "_visited_reward_bound"):
    bound = reward_upper_bound(env.cfg, env.step_dt)
    if bound is None or bound <= 0:
      raise ValueError("visited_return requires a positive, known task reward bound")
    command._visited_reward_bound = bound
  traj, frame = command.trajectory_ids, command.time_steps
  local = frame - command.motion.segment_start_idx[traj]
  bins = (local // command.bin_width_frames).clamp_max(command.bins_per_trajectory - 1)
  next_bins = ((local + 1) // command.bin_width_frames).clamp_max(
    command.bins_per_trajectory - 1
  )
  clip_end = frame + 1 >= command.motion.segment_end_idx[traj]
  offset = traj * command.bins_per_trajectory
  updated = sampler.record(
    offset + bins,
    offset + next_bins,
    env.reward_buf * (1 - rsi.return_gamma) / command._visited_reward_bound,
    env.reset_terminated.bool(),
    env.reset_buf.bool() | clip_end,
    clip_end,
  )
  if updated:
    command.bin_failure_levels.copy_(1 - sampler.value)
    valid = command._valid_bin_indices
    focus = torch.softmax(
      -sampler.value[valid[:, 0], valid[:, 1]] / rsi.temperature_base, 0
    )
    command._bin_memory.add_(focus.double().square())
