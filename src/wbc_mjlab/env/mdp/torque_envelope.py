"""Torque–speed envelope math (Unitree / OmniXtreme)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

_V_EPS = 1.0e-2


@dataclass(frozen=True)
class TorqueSpeedEnvelope:
  """Trapezoidal torque–speed corners (rad/s, N·m)."""

  x1: float
  x2: float
  y1: float
  y2: float


@dataclass
class TorqueSpeedEnvelopeTensors:
  x1: torch.Tensor
  x2: torch.Tensor
  y1: torch.Tensor
  y2: torch.Tensor


def build_envelope_tensors(
  envelopes: Sequence[TorqueSpeedEnvelope],
  *,
  device: torch.device | str,
  dtype: torch.dtype = torch.float32,
) -> TorqueSpeedEnvelopeTensors:
  return TorqueSpeedEnvelopeTensors(
    x1=torch.tensor([e.x1 for e in envelopes], device=device, dtype=dtype),
    x2=torch.tensor([e.x2 for e in envelopes], device=device, dtype=dtype),
    y1=torch.tensor([e.y1 for e in envelopes], device=device, dtype=dtype),
    y2=torch.tensor([e.y2 for e in envelopes], device=device, dtype=dtype),
  )


def tapered_peak(
  abs_vel: torch.Tensor,
  peak: torch.Tensor,
  x1: torch.Tensor,
  x2: torch.Tensor,
) -> torch.Tensor:
  """Peak torque magnitude after linear taper beyond ``x1``."""
  denom = torch.clamp(x2 - x1, min=1.0e-6)
  over = torch.clamp(abs_vel - x1, min=0.0)
  slope = peak / denom
  tapered = torch.clamp(peak - slope * over, min=0.0)
  return torch.where(abs_vel < x1, peak, tapered)


def clip_unitree_effort(
  effort: torch.Tensor,
  joint_vel: torch.Tensor,
  envelope: TorqueSpeedEnvelopeTensors,
) -> torch.Tensor:
  """Clip PD torque with direction-dependent peak (unitree_rl_lab model)."""
  same_direction = (joint_vel * effort) > 0
  peak = torch.where(same_direction, envelope.y1, envelope.y2)
  limit = tapered_peak(joint_vel.abs(), peak, envelope.x1, envelope.x2)
  return torch.clamp(effort, -limit, limit)


def torque_speed_limits(
  joint_vel: torch.Tensor,
  envelope: TorqueSpeedEnvelopeTensors,
  *,
  v_eps: float = _V_EPS,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Admissible torque bounds ``[tau_low, tau_high]`` at fixed velocity."""
  abs_dq = joint_vel.abs()
  over = torch.clamp(abs_dq - envelope.x1, min=0.0)
  denom = torch.clamp(envelope.x2 - envelope.x1, min=1.0e-6)

  peak_high = torch.where(
    abs_dq <= v_eps,
    envelope.y2,
    torch.where(joint_vel >= 0.0, envelope.y1, envelope.y2),
  )
  slope_high = peak_high / denom
  tau_high = torch.clamp(peak_high - slope_high * over, min=0.0)

  peak_low = torch.where(
    abs_dq <= v_eps,
    envelope.y2,
    torch.where(joint_vel >= 0.0, envelope.y2, envelope.y1),
  )
  slope_low = peak_low / denom
  tau_low = -torch.clamp(peak_low - slope_low * over, min=0.0)
  return tau_low, tau_high
