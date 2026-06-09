from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse, quat_error_magnitude

from .commands import MotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


# Zest (Table S4) uses exp(-κ ‖e‖² / σ²) with κ = 1/4. Our kernels use exp(-‖e‖² / std²).
WBC_KERNEL_KAPPA = 0.25


def action_rate_l1(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Penalize action changes with L1 (sum of absolute deltas)."""
  return torch.sum(
    torch.abs(env.action_manager.action - env.action_manager.prev_action),
    dim=1,
  )


def joint_acc_l1(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Penalize joint accelerations with L1 (sum of absolute values)."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(
    torch.abs(asset.data.joint_acc[:, asset_cfg.joint_ids]),
    dim=1,
  )


def wbc_kernel_std(sigma: float, *, dim: int = 1, kappa: float = WBC_KERNEL_KAPPA) -> float:
  """Map Zest σ (optionally scaled by √dim) to ``std`` in ``exp(-error / std²)``."""
  return sigma * math.sqrt(dim) / math.sqrt(kappa)


def _get_body_indexes(
  command: MotionCommand, body_names: tuple[str, ...] | None
) -> list[int]:
  return [
    i
    for i, name in enumerate(command.cfg.body_names)
    if (body_names is None) or (name in body_names)
  ]


def motion_global_anchor_position_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = torch.sum(
    torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1
  )
  return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
  return torch.exp(-error / std**2)


def motion_anchor_linear_velocity_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = torch.sum(
    torch.square(command.anchor_lin_vel_w - command.robot_anchor_lin_vel_w),
    dim=-1,
  )
  return torch.exp(-error / std**2)


def motion_anchor_angular_velocity_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = torch.sum(
    torch.square(command.anchor_ang_vel_w - command.robot_anchor_ang_vel_w),
    dim=-1,
  )
  return torch.exp(-error / std**2)


def motion_anchor_linear_velocity_body_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  """Root linear velocity tracking in the robot anchor (base) frame."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  anchor_quat = command.robot_anchor_quat_w
  ref_lin_b = quat_apply_inverse(anchor_quat, command.anchor_lin_vel_w)
  robot_lin_b = quat_apply_inverse(anchor_quat, command.robot_anchor_lin_vel_w)
  error = torch.sum(torch.square(ref_lin_b - robot_lin_b), dim=-1)
  return torch.exp(-error / std**2)


def motion_anchor_angular_velocity_body_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  """Root angular velocity tracking in the robot anchor (base) frame."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  anchor_quat = command.robot_anchor_quat_w
  ref_ang_b = quat_apply_inverse(anchor_quat, command.anchor_ang_vel_w)
  robot_ang_b = quat_apply_inverse(anchor_quat, command.robot_anchor_ang_vel_w)
  error = torch.sum(torch.square(ref_ang_b - robot_ang_b), dim=-1)
  return torch.exp(-error / std**2)


def motion_joint_position_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float = 1.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Joint position tracking (sum of squared joint errors, same kernel as body terms)."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  jnt_ids = asset_cfg.joint_ids
  error = torch.sum(
    torch.square(command.joint_pos[:, jnt_ids] - command.robot_joint_pos[:, jnt_ids]),
    dim=-1,
  )
  return torch.exp(-error / std**2)


def motion_joint_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float = 1.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Joint velocity tracking (sum of squared joint vel errors, same kernel as joint pos)."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  jnt_ids = asset_cfg.joint_ids
  error = torch.sum(
    torch.square(command.joint_vel[:, jnt_ids] - command.robot_joint_vel[:, jnt_ids]),
    dim=-1,
  )
  return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
  sigma_per_keybody: float | None = None,
  kappa: float = WBC_KERNEL_KAPPA,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  if sigma_per_keybody is not None:
    std = wbc_kernel_std(sigma_per_keybody, dim=len(body_indexes), kappa=kappa)
  error = torch.sum(
    torch.square(
      command.body_pos_relative_w[:, body_indexes]
      - command.robot_body_pos_w[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
  sigma_per_keybody: float | None = None,
  kappa: float = WBC_KERNEL_KAPPA,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  if sigma_per_keybody is not None:
    std = wbc_kernel_std(sigma_per_keybody, dim=len(body_indexes), kappa=kappa)
  error = (
    quat_error_magnitude(
      command.body_quat_relative_w[:, body_indexes],
      command.robot_body_quat_w[:, body_indexes],
    )
    ** 2
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
  sigma_per_keybody: float | None = None,
  kappa: float = WBC_KERNEL_KAPPA,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  if sigma_per_keybody is not None:
    std = wbc_kernel_std(sigma_per_keybody, dim=len(body_indexes), kappa=kappa)
  error = torch.sum(
    torch.square(
      command.body_lin_vel_w[:, body_indexes]
      - command.robot_body_lin_vel_w[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_ang_vel_w[:, body_indexes]
      - command.robot_body_ang_vel_w[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def self_collision_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    force_mag = torch.norm(data.force_history, dim=-1)
    hit = (force_mag > force_threshold).any(dim=1)
    return hit.sum(dim=-1).float()
  assert data.found is not None
  return data.found.squeeze(-1)


def feet_slip(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]
  assert contact_sensor.data.found is not None
  in_contact = (contact_sensor.data.found > 0).float()
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]
  vel_xy_norm = torch.norm(foot_vel_xy, dim=-1)
  vel_xy_norm_sq = torch.square(vel_xy_norm)
  cost = torch.sum(vel_xy_norm_sq * in_contact, dim=1)
  num_in_contact = torch.sum(in_contact)
  mean_slip_vel = torch.sum(vel_xy_norm * in_contact) / torch.clamp(num_in_contact, min=1)
  env.extras["log"]["Metrics/slip_velocity_mean"] = mean_slip_vel
  return cost
