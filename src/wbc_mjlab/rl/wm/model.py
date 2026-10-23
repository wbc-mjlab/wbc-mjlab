"""World model: robot dynamics, termination and the reward terms imagination cannot compute.

The dynamics predict the change of the state over one control step from the state, the PD
targets and the domain-randomization context. Inputs and the change are expressed in the
robot's heading frame (and body frame for rotations), so the model is invariant to where the
robot stands and which way it faces, as the physics on flat ground is.
"""

from __future__ import annotations

import torch
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  quat_apply_inverse,
  quat_conjugate,
  quat_mul,
  yaw_quat,
)
from torch import nn

from wbc_mjlab.rl.wm.kinematics import RobotState


@torch.no_grad()
def update_running_stats(
  mean: torch.Tensor,
  var: torch.Tensor,
  std: torch.Tensor,
  count: torch.Tensor,
  x: torch.Tensor,
  weight: torch.Tensor | None = None,
) -> None:
  """rsl_rl's running mean / variance update, in place (the tensors keep their storage, as
  CUDA graphs need), over the rows of ``x`` with nonzero ``weight``."""
  w = torch.ones(x.shape[0], device=x.device) if weight is None else weight.float()
  n = w.sum()
  mean_x = (w[:, None] * x).sum(0) / n.clamp(min=1)
  var_x = (w[:, None] * (x - mean_x).square()).sum(0) / n.clamp(min=1)
  count.add_(n.to(count.dtype))
  rate = n / count.clamp(min=1)
  delta = mean_x - mean
  mean.add_(rate * delta)
  var.add_(rate * (var_x - var + delta * (mean_x - mean)))
  std.copy_(var.sqrt())


class RunningNorm(nn.Module):
  """Feature normalization by a running mean / std, updated in place (see
  :func:`update_running_stats`)."""

  def __init__(self, dim: int, eps: float):
    super().__init__()
    self.eps = eps
    self.register_buffer("mean", torch.zeros(dim))
    self.register_buffer("var", torch.ones(dim))
    self.register_buffer("std", torch.ones(dim))
    self.register_buffer("count", torch.zeros(()))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return (x - self.mean) / (self.std + self.eps)

  def inverse(self, y: torch.Tensor) -> torch.Tensor:
    return y * (self.std + self.eps) + self.mean

  def update(self, x: torch.Tensor, weight: torch.Tensor | None = None) -> None:
    if self.training and x.shape[-1]:
      update_running_stats(self.mean, self.var, self.std, self.count, x, weight)


def mlp(in_dim: int, out_dim: int, hidden: tuple[int, ...]) -> nn.Sequential:
  """Linear → LayerNorm → SiLU blocks and a linear output."""
  layers: list[nn.Module] = []
  for h in hidden:
    layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.SiLU()]
    in_dim = h
  layers.append(nn.Linear(in_dim, out_dim))
  return nn.Sequential(*layers)


def state_features(s: RobotState) -> torch.Tensor:
  """Height, tilt (6D, heading removed), body-frame velocities, joint state."""
  heading = yaw_quat(s.root_quat)
  tilt = matrix_from_quat(quat_mul(quat_conjugate(heading), s.root_quat))[..., :2]
  return torch.cat(
    [
      s.root_pos[:, 2:3],
      tilt.reshape(-1, 6),
      quat_apply_inverse(s.root_quat, s.root_lin_vel),
      quat_apply_inverse(s.root_quat, s.root_ang_vel),
      s.joint_pos,
      s.joint_vel,
    ],
    dim=-1,
  )


def feature_dim(num_joints: int) -> int:
  return 13 + 2 * num_joints


def change_dim(num_joints: int) -> int:
  return 12 + 2 * num_joints


def _quat_exp(v: torch.Tensor) -> torch.Tensor:
  """Rotation vector → unit quaternion (wxyz); smooth (and its gradient finite) at zero."""
  half = (v.square().sum(-1, keepdim=True) + 1e-12).sqrt() / 2
  return torch.cat([half.cos(), v * 0.5 * torch.sinc(half / torch.pi)], dim=-1)


def _quat_log(q: torch.Tensor) -> torch.Tensor:
  """Unit quaternion (wxyz) → rotation vector of angle ≤ π; gradient finite at zero."""
  q = q * torch.where(q[..., :1] < 0, -1.0, 1.0)
  sin_half = (q[..., 1:].square().sum(-1, keepdim=True) + 1e-12).sqrt()
  return q[..., 1:] * (2 * torch.atan2(sin_half, q[..., :1]) / sin_half)


def apply_change(s: RobotState, d: torch.Tensor) -> RobotState:
  """The state after a change ``d`` (:func:`state_change`)."""
  j = s.joint_pos.shape[-1]
  heading = yaw_quat(s.root_quat)
  quat = quat_mul(s.root_quat, _quat_exp(d[:, 3:6]))
  quat = quat / quat.norm(dim=-1, keepdim=True)
  ang_b = quat_apply_inverse(s.root_quat, s.root_ang_vel) + d[:, 9:12]
  return RobotState(
    s.root_pos + quat_apply(heading, d[:, 0:3]),
    quat,
    s.root_lin_vel + quat_apply(heading, d[:, 6:9]),
    quat_apply(quat, ang_b),
    s.joint_pos + d[:, 12 : 12 + j],
    s.joint_vel + d[:, 12 + j :],
  )


def state_change(s0: RobotState, s1: RobotState) -> torch.Tensor:
  """Change from ``s0`` to ``s1``: root displacement and velocity change in ``s0``'s heading
  frame, rotation (vector) and angular-velocity change in the body frame, joint changes."""
  heading = yaw_quat(s0.root_quat)
  return torch.cat(
    [
      quat_apply_inverse(heading, s1.root_pos - s0.root_pos),
      _quat_log(quat_mul(quat_conjugate(s0.root_quat), s1.root_quat)),
      quat_apply_inverse(heading, s1.root_lin_vel - s0.root_lin_vel),
      quat_apply_inverse(s1.root_quat, s1.root_ang_vel)
      - quat_apply_inverse(s0.root_quat, s0.root_ang_vel),
      s1.joint_pos - s0.joint_pos,
      s1.joint_vel - s0.joint_vel,
    ],
    dim=-1,
  )


class WorldModel(nn.Module):
  """Dynamics ``f(s, u, ξ) → s'`` plus step heads ``h(s, u, g, ξ) → (termination, rewards)``.

  ``u`` are the PD targets relative to the joint positions, ``g`` the tracking features of
  the state against the reference, ``ξ`` the domain-randomization context. The step heads
  predict whether the step ends the episode and the reward terms that are not functions of
  the state (actuator forces, contacts, …).
  """

  def __init__(
    self,
    num_joints: int,
    context_dim: int,
    tracking_dim: int,
    num_learned_rewards: int,
    dynamics_hidden: tuple[int, ...],
    heads_hidden: tuple[int, ...],
  ):
    super().__init__()
    self.num_joints = num_joints
    feat, change = feature_dim(num_joints), change_dim(num_joints)
    dyn_in = feat + num_joints + context_dim
    head_in = dyn_in + tracking_dim
    self.dynamics_in = RunningNorm(dyn_in, eps=1e-4)
    self.change_norm = RunningNorm(change, eps=1e-6)
    self.dynamics = mlp(dyn_in, change, dynamics_hidden)
    self.heads_in = RunningNorm(head_in, eps=1e-4)
    self.reward_norm = RunningNorm(num_learned_rewards, eps=1e-4)
    self.heads = mlp(head_in, 1 + num_learned_rewards, heads_hidden)

  def _dynamics_input(
    self, s: RobotState, setpoint: torch.Tensor, context: torch.Tensor
  ):
    return torch.cat([state_features(s), setpoint - s.joint_pos, context], dim=-1)

  def step(
    self, s: RobotState, setpoint: torch.Tensor, context: torch.Tensor
  ) -> RobotState:
    x = self.dynamics_in(self._dynamics_input(s, setpoint, context))
    return apply_change(s, self.change_norm.inverse(self.dynamics(x)))

  def step_heads(
    self,
    s: RobotState,
    setpoint: torch.Tensor,
    tracking: torch.Tensor,
    context: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """(termination logit, learned reward terms) of the step from ``s``."""
    x = torch.cat([self._dynamics_input(s, setpoint, context), tracking], dim=-1)
    out = self.heads(self.heads_in(x))
    return out[:, 0], self.reward_norm.inverse(out[:, 1:])

  def update_dynamics_normalization(
    self,
    s: RobotState,
    setpoint: torch.Tensor,
    context: torch.Tensor,
    change: torch.Tensor,
    weight: torch.Tensor,
  ) -> None:
    """From logged steps (``weight``: those that are real transitions)."""
    self.dynamics_in.update(self._dynamics_input(s, setpoint, context), weight)
    self.change_norm.update(change, weight)

  def update_heads_normalization(
    self,
    s: RobotState,
    setpoint: torch.Tensor,
    tracking: torch.Tensor,
    context: torch.Tensor,
    rewards: torch.Tensor,
  ) -> None:
    x = torch.cat([self._dynamics_input(s, setpoint, context), tracking], dim=-1)
    self.heads_in.update(x)
    self.reward_norm.update(rewards)

  def reward_error(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Squared error of the learned reward terms in units of their spread."""
    return ((pred - target) / (self.reward_norm.std + 1e-4)).square()

  def change_error(self, pred: RobotState, true: RobotState) -> torch.Tensor:
    """Squared error of ``pred`` vs ``true`` in units of the typical one-step change."""
    return (state_change(true, pred) / (self.change_norm.std + 1e-6)).square()
