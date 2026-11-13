"""Between-step root velocity kicks, without asking dynamics to predict a random hit.

Adapted from the user-supplied draft_wbc implementation.
Precedent (pinned): BlinkMimic d264658d6c9ce61fb52f9f0f8985a369f3ccc409,
src/blinkmimic/env.py:380-392,425-439 and buffer.py:85-97:
https://github.com/fuyichen0725/blinkmimic/blob/d264658d6c9ce61fb52f9f0f8985a369f3ccc409/src/blinkmimic/env.py

Record the physical successor before a kick. Cut autoregressive model windows at
the kick; retain the episode, reference clock and actor observation history. The
next real observation contains the resulting deviation. This models instantaneous
velocity perturbations, not a sustained force or a particular body contact.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass
class VelocityKickCfg:
  """Explicit seconds and world-frame velocity increments (m/s, then rad/s).

  ``velocity_ranges`` orders linear x/y/z, then angular x/y/z. All bounds are supplied
  by the caller. The optional training event supplies mild starting values.
  """

  interval_range_s: tuple[float, float]
  velocity_ranges: tuple[tuple[float, float], ...]
  initial_scale: float = 1.0
  ramp_duration_s: float = 0.0
  clean_episode_fraction: float = 0.0

  def __post_init__(self):
    pairs = (self.interval_range_s, *self.velocity_ranges)
    if len(self.velocity_ranges) != 6:
      raise ValueError("velocity_ranges must contain six (lower, upper) pairs")
    for pair in pairs:
      if len(pair) != 2 or not all(math.isfinite(v) for v in pair):
        raise ValueError("kick ranges must contain finite (lower, upper) pairs")
      if pair[0] > pair[1]:
        raise ValueError("kick range lower bound exceeds upper bound")
    if self.interval_range_s[0] <= 0:
      raise ValueError("kick intervals must be positive")
    if not math.isfinite(self.initial_scale) or not 0 <= self.initial_scale <= 1:
      raise ValueError("initial_scale must lie in [0, 1]")
    if not math.isfinite(self.ramp_duration_s) or self.ramp_duration_s < 0:
      raise ValueError("ramp_duration_s must be finite and nonnegative")
    if (
      not math.isfinite(self.clean_episode_fraction)
      or not 0 <= self.clean_episode_fraction <= 1
    ):
      raise ValueError("clean_episode_fraction must lie in [0, 1]")


class _VelocityData(Protocol):
  root_link_vel_w: torch.Tensor


class _VelocityRobot(Protocol):
  data: _VelocityData

  def write_root_link_velocity_to_sim(
    self, velocity: torch.Tensor, *, env_ids: torch.Tensor
  ) -> None: ...


@dataclass(frozen=True)
class KickResult:
  successor: torch.Tensor
  pushed: torch.Tensor
  model_cut: torch.Tensor


class InterStepVelocityKicks:
  """Own per-environment timers; apply only at an explicit capture boundary.

  The robot interface is the same as Studio's interactive push path. Call once
  after physics/reward/termination capture, before constructing the next actor
  observation. ``capture_successor`` must read the physical pre-reset successor;
  this helper clones it before writing simulator velocities. Existing terminal
  recorders must still handle environments whose reset already happened. A
  post-physics event must supply current integrated velocities through the robot
  interface, not an entity cache that is stale by a physics substep.
  """

  def __init__(
    self,
    cfg: VelocityKickCfg,
    num_envs: int,
    device: str | torch.device,
    *,
    generator: torch.Generator | None = None,
  ):
    if num_envs <= 0:
      raise ValueError("num_envs must be positive")
    self.cfg, self.num_envs = cfg, num_envs
    self.device = torch.device(device)
    self.generator = generator
    bounds = torch.tensor(cfg.velocity_ranges, device=self.device)
    self._lower, self._width = bounds[:, 0], bounds[:, 1] - bounds[:, 0]
    self.remaining_s = self._sample_intervals(num_envs)
    self.elapsed_s = 0.0
    self.clean_episode = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
    if cfg.clean_episode_fraction:
      self.clean_episode = (
        torch.rand(num_envs, device=self.device, generator=generator)
        < cfg.clean_episode_fraction
      )

  def _sample_intervals(self, count: int) -> torch.Tensor:
    low, high = self.cfg.interval_range_s
    return low + (high - low) * torch.rand(
      count, device=self.device, generator=self.generator
    )

  def after_step(
    self,
    *,
    done: torch.Tensor,
    dt: float,
    robot: _VelocityRobot,
    capture_successor: Callable[[], torch.Tensor],
  ) -> KickResult:
    """Capture, then kick live environments; never turn a kick into ``done``.

    ``model_cut`` is separate from the caller's episode flags. It may be combined
    with other model-continuity boundaries; it must not reset actor histories or
    be passed off as a terminal label. Terminal envs get a fresh timer, no kick.
    """
    if not math.isfinite(dt) or dt <= 0:
      raise ValueError("dt must be finite and positive")
    if done.shape != (self.num_envs,) or done.dtype != torch.bool:
      raise ValueError("done must be a boolean vector with one entry per environment")
    if done.device != self.remaining_s.device:
      raise ValueError("done and kick timers must be on the same device")
    if robot.data.root_link_vel_w.shape != (self.num_envs, 6):
      raise ValueError("root_link_vel_w must have shape (num_envs, 6)")

    # Clone even when the simulator reader returns a view into mutable storage.
    successor = capture_successor().clone()
    if successor.shape[0] != self.num_envs:
      raise ValueError("successor must have one row per environment")
    remaining = self.remaining_s - dt
    self.elapsed_s += dt
    fraction = (
      min(self.elapsed_s / self.cfg.ramp_duration_s, 1.0)
      if self.cfg.ramp_duration_s
      else 1.0
    )
    scale = self.cfg.initial_scale + (1.0 - self.cfg.initial_scale) * fraction
    due = remaining <= 0
    pushed = due & ~done & ~self.clean_episode
    ids = pushed.nonzero(as_tuple=False).squeeze(-1)
    if ids.numel():
      increments = self._lower + self._width * torch.rand(
        ids.numel(), 6, device=self.device, generator=self.generator
      )
      increments = increments * scale
      velocity = robot.data.root_link_vel_w[ids].clone() + increments
      robot.write_root_link_velocity_to_sim(velocity, env_ids=ids)
    renew = due | done
    self.remaining_s = torch.where(
      renew, self._sample_intervals(self.num_envs), remaining
    )
    if self.cfg.clean_episode_fraction:
      fresh_clean = (
        torch.rand(self.num_envs, device=self.device, generator=self.generator)
        < self.cfg.clean_episode_fraction
      )
      self.clean_episode = torch.where(done, fresh_clean, self.clean_episode)
    return KickResult(successor, pushed, done | pushed)


EVENT = "wm_velocity_kick"
RESULT = "_wm_velocity_kick_result"


def inter_step_velocity_kick(
  env,
  env_ids,
  *,
  enabled: bool,
  interval_range_s: tuple[float, float],
  velocity_ranges: tuple[tuple[float, float], ...],
  initial_scale: float,
  ramp_duration_s: float,
  clean_episode_fraction: float,
) -> None:
  """Capture a real successor and cut model windows, without ending an episode.

  This must be the only ongoing physical perturbation in the task: other events
  could change the transition before it is captured. The environment's single
  forward() after events exposes the kicked state to the next actor observation.
  """
  if not enabled:
    return
  from types import SimpleNamespace

  from wbc_mjlab.rl.wm.recording import simulator_state

  cfg = VelocityKickCfg(
    interval_range_s,
    velocity_ranges,
    initial_scale,
    ramp_duration_s,
    clean_episode_fraction,
  )
  if env_ids is not None:
    raise ValueError("world-model kicks must run once per step for all environments")
  command = env.command_manager.get_term("motion")
  robot = command.robot
  controller = getattr(env, "_wm_velocity_kicks", None)
  if controller is None:
    controller = InterStepVelocityKicks(cfg, env.num_envs, env.device)
    controller.elapsed_s = getattr(env, "_wm_kick_elapsed_s", 0.0)
    env._wm_velocity_kicks = controller
  elif controller.cfg != cfg:
    raise ValueError("kick configuration cannot change after construction")

  state = simulator_state(env, robot)
  # Entity link caches lag integrated qvel until forward(). Do not add a kick to
  # that stale velocity: use the same integrated world-frame state as replay.
  integrated_robot = SimpleNamespace(
    data=SimpleNamespace(
      root_link_vel_w=torch.cat((state.root_lin_vel, state.root_ang_vel), dim=-1)
    ),
    write_root_link_velocity_to_sim=robot.write_root_link_velocity_to_sim,
  )
  result = controller.after_step(
    done=env.reset_buf.bool(),
    dt=env.step_dt,
    robot=integrated_robot,
    capture_successor=state.vector,
  )
  env._wm_kick_elapsed_s = controller.elapsed_s
  setattr(env, RESULT, result)
