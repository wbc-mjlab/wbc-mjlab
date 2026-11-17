"""Robust action cotangents, adapted from NablaMimic nets.py (bfbb2a6).

Copyright 2026 The NablaMimic Authors. Apache-2.0; see THIRD_PARTY_NOTICES.
PyTorch adaptation: reject whole nonfinite rows and use fixed-shape reductions.
"""

import torch


def clip_action_gradient(gradient: torch.Tensor, ratio: float):
  """Clip each finite, nonzero row to ratio × median live row norm.

  Returns clipped gradients and [clipped fraction, invalid fraction]. Empty/live-zero
  batches stay zero. Invalid rows never participate in the threshold calculation.
  """
  finite = torch.isfinite(gradient).all(-1)
  g = torch.where(finite[..., None], gradient, 0.0)
  norm = torch.linalg.vector_norm(g, dim=-1)
  finite = finite & torch.isfinite(norm)
  g = torch.where(finite[..., None], g, 0.0)
  live = finite & (norm > 0)
  median = torch.nanmedian(torch.where(live, norm, float("nan")))
  threshold = torch.nan_to_num(median, nan=0.0) * ratio
  scale = torch.where(live, (threshold / norm.clamp_min(1e-30)).clamp_max(1), 0)
  stats = torch.stack(
    [
      (live & (scale < 1)).float().sum() / live.sum().clamp_min(1),
      (~finite).float().mean(),
    ]
  )
  return g * scale[..., None], stats


class _ClipActionGradient(torch.autograd.Function):
  @staticmethod
  def forward(ctx, action, ratio):
    ctx.ratio = ratio
    return action.clone()

  @staticmethod
  def backward(ctx, gradient):
    return clip_action_gradient(gradient, ctx.ratio)[0], None


def guard_action_gradient(action: torch.Tensor, ratio: float) -> torch.Tensor:
  """Identity forward; clip sample cotangents before they reach earlier steps."""
  return _ClipActionGradient.apply(action, ratio) if ratio else action
