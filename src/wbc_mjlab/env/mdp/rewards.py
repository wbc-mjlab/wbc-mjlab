"""Motion-tracking and regularization rewards for WBC tasks.

All ``motion_*`` terms share optional kernel params:

- ``kappa`` — exponential stiffness (default ``1.0``; Zest uses ``0.25``)
- ``std`` / ``sigma_per_joint`` / ``sigma_per_keybody`` — error scale
- ``per_joint`` / ``per_keybody`` — mean of per-DoF/per-body exponentials
- ``body_error_aggregate`` — ``mean`` (default) or ``sum`` over keybodies
- ``body_names`` — subset of tracked bodies (default: all command bodies)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse, quat_error_magnitude

from wbc_mjlab.actuation.envelope import torque_speed_limits

from .commands import MotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def tracking_std_from_sigma(sigma: float, *, dim: int = 1) -> float:
  """Map per-DoF σ to aggregate ``std`` with ``std = 2σ√dim`` (Table S4 convention)."""
  return 2.0 * sigma * math.sqrt(dim)


def dim_scaled_std(per_dim: float, *, dim: int) -> float:
  """``std`` when σ scales as ``per_dim · √dim`` over joints or keybodies."""
  return tracking_std_from_sigma(per_dim, dim=dim)


def _resolve_tracking_std(
  std: float | None,
  *,
  sigma_per: float | None,
  dim: int,
) -> float:
  if sigma_per is not None:
    return dim_scaled_std(sigma_per, dim=dim)
  if std is not None:
    return std
  raise ValueError("Either ``std`` or ``sigma_per`` must be set.")


def _tracking_exp(
  error: torch.Tensor,
  *,
  std: float,
  kappa: float = 1.0,
) -> torch.Tensor:
  std = max(std, 1.0e-6)
  return torch.exp(-kappa * error / std**2)


def _body_sq_error_per_keybody(
  command: MotionCommand,
  body_indexes: list[int],
  *,
  relative_pos: bool,
  relative_ori: bool,
  lin_vel: bool,
  ang_vel: bool,
) -> torch.Tensor:
  if relative_pos:
    diff = (
      command.body_pos_relative_w[:, body_indexes]
      - command.robot_body_pos_w[:, body_indexes]
    )
    return torch.sum(torch.square(diff), dim=-1)
  if relative_ori:
    error = quat_error_magnitude(
      command.body_quat_relative_w[:, body_indexes],
      command.robot_body_quat_w[:, body_indexes],
    )
    return error**2
  if lin_vel:
    diff = (
      command.body_lin_vel_w[:, body_indexes]
      - command.robot_body_lin_vel_w[:, body_indexes]
    )
    return torch.sum(torch.square(diff), dim=-1)
  assert ang_vel
  diff = (
    command.body_ang_vel_w[:, body_indexes]
    - command.robot_body_ang_vel_w[:, body_indexes]
  )
  return torch.sum(torch.square(diff), dim=-1)


def _motion_keybody_tracking_exp(
  sq_error_per_body: torch.Tensor,
  *,
  std: float | None,
  sigma_per_keybody: float | None,
  per_keybody: bool,
  body_error_aggregate: str,
  kappa: float,
  num_bodies: int,
) -> torch.Tensor:
  if per_keybody:
    if sigma_per_keybody is None:
      raise ValueError("``sigma_per_keybody`` is required when ``per_keybody`` is True.")
    std_b = tracking_std_from_sigma(sigma_per_keybody, dim=1)
    return _tracking_exp(sq_error_per_body, std=std_b, kappa=kappa).mean(dim=-1)

  std_eff = _resolve_tracking_std(
    std, sigma_per=sigma_per_keybody, dim=num_bodies
  )
  if body_error_aggregate == "sum":
    error = sq_error_per_body.sum(dim=-1)
  elif body_error_aggregate == "mean":
    error = sq_error_per_body.mean(dim=-1)
  else:
    raise ValueError(
      f"Unsupported body_error_aggregate {body_error_aggregate!r}; use 'mean' or 'sum'."
    )
  return _tracking_exp(error, std=std_eff, kappa=kappa)


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


def _get_body_indexes(
  command: MotionCommand, body_names: tuple[str, ...] | None
) -> list[int]:
  return [
    i
    for i, name in enumerate(command.cfg.body_names)
    if (body_names is None) or (name in body_names)
  ]


def anti_shake_ang_vel_l2(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  """Penalize high-frequency wrist spin above a deadzone (SONIC-style anti-shake)."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  ang_vel = command.robot_body_ang_vel_w[:, body_indexes]
  speed = torch.linalg.norm(ang_vel, dim=-1)
  excess = torch.relu(speed - threshold)
  return (excess * excess).mean(dim=-1)


def motion_global_anchor_position_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float, *, kappa: float = 1.0
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = torch.sum(
    torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1
  )
  return _tracking_exp(error, std=std, kappa=kappa)


def motion_global_anchor_orientation_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float, *, kappa: float = 1.0
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
  return _tracking_exp(error, std=std, kappa=kappa)


def motion_anchor_linear_velocity_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float, *, kappa: float = 1.0
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = torch.sum(
    torch.square(command.anchor_lin_vel_w - command.robot_anchor_lin_vel_w),
    dim=-1,
  )
  return _tracking_exp(error, std=std, kappa=kappa)


def motion_anchor_angular_velocity_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float, *, kappa: float = 1.0
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = torch.sum(
    torch.square(command.anchor_ang_vel_w - command.robot_anchor_ang_vel_w),
    dim=-1,
  )
  return _tracking_exp(error, std=std, kappa=kappa)


def motion_anchor_linear_velocity_body_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float, *, kappa: float = 1.0
) -> torch.Tensor:
  """Root linear velocity tracking in the robot anchor (base) frame."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  anchor_quat = command.robot_anchor_quat_w
  ref_lin_b = quat_apply_inverse(anchor_quat, command.anchor_lin_vel_w)
  robot_lin_b = quat_apply_inverse(anchor_quat, command.robot_anchor_lin_vel_w)
  error = torch.sum(torch.square(ref_lin_b - robot_lin_b), dim=-1)
  return _tracking_exp(error, std=std, kappa=kappa)


def motion_anchor_angular_velocity_body_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float, *, kappa: float = 1.0
) -> torch.Tensor:
  """Root angular velocity tracking in the robot anchor (base) frame."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  anchor_quat = command.robot_anchor_quat_w
  ref_ang_b = quat_apply_inverse(anchor_quat, command.anchor_ang_vel_w)
  robot_ang_b = quat_apply_inverse(anchor_quat, command.robot_anchor_ang_vel_w)
  error = torch.sum(torch.square(ref_ang_b - robot_ang_b), dim=-1)
  return _tracking_exp(error, std=std, kappa=kappa)


def motion_joint_position_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float | None = None,
  sigma_per_joint: float | None = None,
  per_joint: bool = False,
  *,
  kappa: float = 1.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Joint position tracking with optional per-joint exponential kernels.

  Default: ``exp(-κ Σ e_i² / std²)`` with ``std = 2σ√n`` when ``sigma_per_joint`` is set.
  ``per_joint=True``: mean over joints of ``exp(-κ e_i² / (2σ)²)``.
  """
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  jnt_ids = asset_cfg.joint_ids
  ref_joint = command.joint_pos[:, jnt_ids]
  robot_joint = command.robot_joint_pos[:, jnt_ids]
  sq_err = torch.square(ref_joint - robot_joint)
  if per_joint:
    if sigma_per_joint is None:
      raise ValueError("``sigma_per_joint`` is required when ``per_joint`` is True.")
    std_j = tracking_std_from_sigma(sigma_per_joint, dim=1)
    return _tracking_exp(sq_err, std=std_j, kappa=kappa).mean(dim=-1)
  std_eff = _resolve_tracking_std(
    std, sigma_per=sigma_per_joint, dim=ref_joint.shape[-1]
  )
  return _tracking_exp(torch.sum(sq_err, dim=-1), std=std_eff, kappa=kappa)


def motion_joint_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float | None = 1.0,
  sigma_per_joint: float | None = None,
  per_joint: bool = False,
  *,
  kappa: float = 1.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Joint velocity tracking; same kernel options as ``motion_joint_position_error_exp``."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  jnt_ids = asset_cfg.joint_ids
  sq_err = torch.square(
    command.joint_vel[:, jnt_ids] - command.robot_joint_vel[:, jnt_ids]
  )
  if per_joint:
    if sigma_per_joint is None:
      raise ValueError("``sigma_per_joint`` is required when ``per_joint`` is True.")
    std_j = tracking_std_from_sigma(sigma_per_joint, dim=1)
    return _tracking_exp(sq_err, std=std_j, kappa=kappa).mean(dim=-1)
  std_eff = _resolve_tracking_std(
    std, sigma_per=sigma_per_joint, dim=sq_err.shape[-1]
  )
  return _tracking_exp(torch.sum(sq_err, dim=-1), std=std_eff, kappa=kappa)


def motion_relative_body_position_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float | None = None,
  body_names: tuple[str, ...] | None = None,
  sigma_per_keybody: float | None = None,
  per_keybody: bool = False,
  body_error_aggregate: str = "mean",
  *,
  kappa: float = 1.0,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  sq_err = _body_sq_error_per_keybody(
    command, body_indexes, relative_pos=True, relative_ori=False, lin_vel=False, ang_vel=False
  )
  return _motion_keybody_tracking_exp(
    sq_err,
    std=std,
    sigma_per_keybody=sigma_per_keybody,
    per_keybody=per_keybody,
    body_error_aggregate=body_error_aggregate,
    kappa=kappa,
    num_bodies=len(body_indexes),
  )


def motion_relative_body_orientation_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float | None = None,
  body_names: tuple[str, ...] | None = None,
  sigma_per_keybody: float | None = None,
  per_keybody: bool = False,
  body_error_aggregate: str = "mean",
  *,
  kappa: float = 1.0,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  sq_err = _body_sq_error_per_keybody(
    command, body_indexes, relative_pos=False, relative_ori=True, lin_vel=False, ang_vel=False
  )
  return _motion_keybody_tracking_exp(
    sq_err,
    std=std,
    sigma_per_keybody=sigma_per_keybody,
    per_keybody=per_keybody,
    body_error_aggregate=body_error_aggregate,
    kappa=kappa,
    num_bodies=len(body_indexes),
  )


def motion_global_body_linear_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float | None = None,
  body_names: tuple[str, ...] | None = None,
  sigma_per_keybody: float | None = None,
  per_keybody: bool = False,
  body_error_aggregate: str = "mean",
  *,
  kappa: float = 1.0,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  sq_err = _body_sq_error_per_keybody(
    command, body_indexes, relative_pos=False, relative_ori=False, lin_vel=True, ang_vel=False
  )
  return _motion_keybody_tracking_exp(
    sq_err,
    std=std,
    sigma_per_keybody=sigma_per_keybody,
    per_keybody=per_keybody,
    body_error_aggregate=body_error_aggregate,
    kappa=kappa,
    num_bodies=len(body_indexes),
  )


def motion_global_body_angular_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float | None = None,
  body_names: tuple[str, ...] | None = None,
  sigma_per_keybody: float | None = None,
  per_keybody: bool = False,
  body_error_aggregate: str = "mean",
  *,
  kappa: float = 1.0,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  sq_err = _body_sq_error_per_keybody(
    command, body_indexes, relative_pos=False, relative_ori=False, lin_vel=False, ang_vel=True
  )
  return _motion_keybody_tracking_exp(
    sq_err,
    std=std,
    sigma_per_keybody=sigma_per_keybody,
    per_keybody=per_keybody,
    body_error_aggregate=body_error_aggregate,
    kappa=kappa,
    num_bodies=len(body_indexes),
  )


def actuator_torque_soft_limit(
  env: ManagerBasedRlEnv,
  soft_ratio: float = 0.95,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize actuator forces approaching MuJoCo torque limits (normalized soft margin)."""
  asset: Entity = env.scene[asset_cfg.name]
  forces = asset.data.actuator_force
  ctrl_ids = asset.indexing.ctrl_ids
  tau_max = env.sim.model.actuator_forcerange[:, ctrl_ids, 1]
  ratio = torch.abs(forces) / tau_max.clamp(min=1.0e-6)
  violation = torch.clamp(ratio - soft_ratio, min=0.0)
  return torch.sum(violation, dim=-1)


def angular_momentum_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  *,
  axes: str = "xy",
) -> torch.Tensor:
  """Penalize whole-body angular momentum (roll/pitch by default)."""
  angmom = env.scene[sensor_name].data
  if axes == "xy":
    return torch.sum(torch.square(angmom[..., :2]), dim=-1)
  if axes == "xyz":
    return torch.sum(torch.square(angmom), dim=-1)
  raise ValueError(f"Unsupported axes {axes!r}; use 'xy' or 'xyz'.")


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


def _joint_torque_and_vel(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
) -> tuple[torch.Tensor, torch.Tensor]:
  asset: Entity = env.scene[asset_cfg.name]
  ids = asset_cfg.joint_ids
  return asset.data.qfrc_actuator[:, ids], asset.data.joint_vel[:, ids]


def mechanical_power_l1(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Sum of positive mechanical power P = τ·ω (motoring only)."""
  tau, qd = _joint_torque_and_vel(env, asset_cfg)
  power = tau * qd
  return torch.sum(torch.clamp(power, min=0.0), dim=-1)


def negative_mechanical_power_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  *,
  power_deadband: float = 0.0,
  penalty_scale: float = 1.0,
  joint_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  """Penalize excessive regenerative braking (OmniXtreme power-safety term)."""
  cfg = (
    SceneEntityCfg(asset_cfg.name, joint_names=joint_names)
    if joint_names is not None
    else asset_cfg
  )
  tau, qd = _joint_torque_and_vel(env, cfg)
  power = tau * qd
  excess = torch.clamp(-power - power_deadband, min=0.0) / max(penalty_scale, 1.0e-6)
  return torch.sum(torch.square(excess), dim=-1)


def torque_envelope_violation_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalty when applied torque exceeds G1 velocity-dependent envelope."""
  from wbc_mjlab.robots.g1.envelope import g1_joint_envelope_tensors

  tau, qd = _joint_torque_and_vel(env, asset_cfg)
  envl = g1_joint_envelope_tensors(env, asset_cfg)
  tau_low, tau_high = torque_speed_limits(qd, envl)
  over_high = torch.clamp(tau - tau_high, min=0.0)
  over_low = torch.clamp(tau_low - tau, min=0.0)
  return torch.sum(torch.square(over_high) + torch.square(over_low), dim=-1)


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
