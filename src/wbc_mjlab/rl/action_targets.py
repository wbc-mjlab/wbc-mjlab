"""PODS-style action targets adapted from NablaMimic train.py (bfbb2a6).

Copyright 2026 The NablaMimic Authors. Apache-2.0; see THIRD_PARTY_NOTICES.
The gradient source is supplied by the caller: no simulator or policy layout assumptions.
"""

import torch

from wbc_mjlab.rl.gradients import clip_action_gradient
from wbc_mjlab.rl.wm.distributed import sum_


@torch.no_grad()
def action_targets(actions, gradients, valid, scale, delta, clip_ratio=0.0, limit=None):
  """Return detached targets, valid weights and diagnostics, in action units.

  ``gradients`` ascend return. The RMS target displacement and regression metric
  use joint-setpoint radians, including nonuniform action scales. Invalid rows are
  discarded in full, and masked trajectory tails cannot affect the normalization.
  """
  finite = torch.isfinite(gradients).all(-1) & torch.isfinite(actions).all(-1)
  valid = valid & finite
  g = torch.where(valid[..., None], gradients, 0.0)
  if clip_ratio:
    g, clipping = clip_action_gradient(g, clip_ratio)
  else:
    clipping = g.new_zeros(2)
  physical = g.double() * scale.double()
  live = valid & (torch.linalg.vector_norm(physical, dim=-1) > 0)
  totals = sum_(
    torch.stack(
      [
        physical.square().sum(),
        live.sum().to(physical.dtype) * g.shape[-1],
      ]
    )
  )
  rms = (totals[0] / totals[1].clamp_min(1)).sqrt()
  step = (g.double() * (delta / rms.clamp_min(1e-300))).to(g.dtype)
  target = torch.where(live[..., None], actions + step, 0.0)
  if limit is not None:
    target.clamp_(-limit, limit)
  live = live & torch.isfinite(target).all(-1)
  target = torch.where(live[..., None], target, 0.0)
  displacement = torch.where(live[..., None], (target - actions) * scale, 0.0)
  stats = {
    "action_target_rms_rad": (
      displacement.square().sum() / (live.sum() * g.shape[-1]).clamp_min(1)
    ).sqrt(),
    "action_target_valid_fraction": live.float().mean(),
    "action_gradient_clipped_fraction": clipping[0],
    "action_gradient_invalid_fraction": (~finite).float().mean(),
  }
  return target, live.float(), stats


class ActionTargetReplay:
  """Bounded FIFO of detached observation/target windows; checkpointable on CPU."""

  def __init__(self, windows: int):
    self.windows = windows
    self.rows = []
    self._flat = None

  def append(self, observations, targets, weights):
    valid = weights > 0
    # Never let an invalid observation poison a zero-weight regression row.
    valid = valid & torch.isfinite(observations).all(-1)
    self.rows.append(
      (
        torch.where(valid[..., None], observations, 0).flatten(0, 1).detach().clone(),
        torch.where(valid[..., None], targets, 0).flatten(0, 1).detach().clone(),
        valid.flatten().float(),
      )
    )
    self.rows = self.rows[-self.windows :]
    self._flat = None

  def sample(self, batch: int):
    if self._flat is None:
      self._flat = tuple(torch.cat(x) for x in zip(*self.rows, strict=True))
    obs, targets, weights = self._flat
    ids = torch.randint(len(weights), (batch,), device=weights.device)
    return obs[ids], targets[ids], weights[ids]

  def state_dict(self):
    return {
      "windows": self.windows,
      "rows": [tuple(x.detach().cpu().clone() for x in row) for row in self.rows],
    }

  def load_state_dict(self, state, device):
    if state["windows"] != self.windows:
      raise ValueError("checkpoint action-target buffer capacity differs")
    self.rows = [tuple(x.to(device).clone() for x in row) for row in state["rows"]]
    self._flat = None
