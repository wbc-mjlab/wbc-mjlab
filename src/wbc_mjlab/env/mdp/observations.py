from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply_inverse,
  subtract_frame_transforms,
)

from .commands import MotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _motion_command(env: ManagerBasedRlEnv, command_name: str) -> MotionCommand:
  return cast(MotionCommand, env.command_manager.get_term(command_name))


# --- Actor reference features (configurable obs terms; were stacked in MotionCommand.command) ---


def ref_base_height(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Reference anchor height relative to env origin (z_I r̂_IB)."""
  return _motion_command(env, command_name).ref_base_height


def ref_base_lin_vel_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Reference anchor linear velocity in anchor frame (B v̂_IB)."""
  return _motion_command(env, command_name).ref_base_lin_vel_b


def ref_base_ang_vel_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Reference anchor angular velocity in anchor frame (B ω̂_IB)."""
  return _motion_command(env, command_name).ref_base_ang_vel_b


def ref_gravity_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Reference gravity in anchor frame (B ĝ_I)."""
  return _motion_command(env, command_name).ref_gravity_b


def ref_joint_pos(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Reference joint positions for tracked DoFs (absolute)."""
  return _motion_command(env, command_name).tracked_joint_pos


def _body_index(asset: Entity, body_name: str) -> int:
  return asset.body_names.index(body_name)


def torso_ang_vel(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  body_name: str = "torso_link",
) -> torch.Tensor:
  """Torso angular velocity in the torso frame (Zest Table S3: T ω_IT)."""
  asset: Entity = env.scene[asset_cfg.name]
  idx = _body_index(asset, body_name)
  quat_w = asset.data.body_link_quat_w[:, idx]
  ang_vel_w = asset.data.body_link_ang_vel_w[:, idx]
  return quat_apply_inverse(quat_w, ang_vel_w)


def torso_projected_gravity(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  body_name: str = "torso_link",
) -> torch.Tensor:
  """Gravity direction in the torso frame (Zest Table S3: T g_I)."""
  asset: Entity = env.scene[asset_cfg.name]
  idx = _body_index(asset, body_name)
  quat_w = asset.data.body_link_quat_w[:, idx]
  return quat_apply_inverse(quat_w, asset.data.gravity_vec_w)


def ref_joint_vel(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reference joint velocities (critic privileged)."""
  command = _motion_command(env, command_name)
  if asset_cfg.joint_ids is not None:
    return command.joint_vel[:, asset_cfg.joint_ids]
  return command.tracked_joint_vel


def ref_base_lin_acc_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Reference anchor linear acceleration in anchor frame (critic privileged)."""
  return _motion_command(env, command_name).ref_base_lin_acc_b


def ref_base_ang_acc_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Reference anchor angular acceleration in anchor frame (critic privileged)."""
  return _motion_command(env, command_name).ref_base_ang_acc_b


# --- Critic privileged keybody / anchor-relative features ---


def _body_lin_vel_in_anchor_frame(
  anchor_quat_w: torch.Tensor,
  body_lin_vel_w: torch.Tensor,
) -> torch.Tensor:
  num_envs, num_bodies, _ = body_lin_vel_w.shape
  quat = anchor_quat_w[:, None, :].expand(num_envs, num_bodies, 4).reshape(-1, 4)
  vel = body_lin_vel_w.reshape(-1, 3)
  vel_b = quat_apply_inverse(quat, vel)
  return vel_b.view(num_envs, num_bodies * 3)


def _body_ang_vel_in_anchor_frame(
  anchor_quat_w: torch.Tensor,
  body_ang_vel_w: torch.Tensor,
) -> torch.Tensor:
  num_envs, num_bodies, _ = body_ang_vel_w.shape
  quat = anchor_quat_w[:, None, :].expand(num_envs, num_bodies, 4).reshape(-1, 4)
  vel = body_ang_vel_w.reshape(-1, 3)
  vel_b = quat_apply_inverse(quat, vel)
  return vel_b.view(num_envs, num_bodies * 3)


def motion_anchor_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = _motion_command(env, command_name)

  pos, _ = subtract_frame_transforms(
    command.robot_anchor_pos_w,
    command.robot_anchor_quat_w,
    command.anchor_pos_w,
    command.anchor_quat_w,
  )

  return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = _motion_command(env, command_name)

  _, ori = subtract_frame_transforms(
    command.robot_anchor_pos_w,
    command.robot_anchor_quat_w,
    command.anchor_pos_w,
    command.anchor_quat_w,
  )
  mat = matrix_from_quat(ori)
  return mat[..., :2].reshape(mat.shape[0], -1)


def robot_body_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = _motion_command(env, command_name)

  num_bodies = len(command.cfg.body_names)
  pos_b, _ = subtract_frame_transforms(
    command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_body_pos_w,
    command.robot_body_quat_w,
  )

  return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = _motion_command(env, command_name)

  num_bodies = len(command.cfg.body_names)
  _, ori_b = subtract_frame_transforms(
    command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_body_pos_w,
    command.robot_body_quat_w,
  )
  mat = matrix_from_quat(ori_b)
  return mat[..., :2].reshape(mat.shape[0], -1)


def ref_body_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Reference keybody positions in the robot anchor frame."""
  command = _motion_command(env, command_name)
  num_bodies = len(command.cfg.body_names)
  pos_b, _ = subtract_frame_transforms(
    command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
    command.body_pos_w,
    command.body_quat_w,
  )
  return pos_b.view(env.num_envs, -1)


def ref_body_ori_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Reference keybody orientations in the robot anchor frame."""
  command = _motion_command(env, command_name)
  num_bodies = len(command.cfg.body_names)
  _, ori_b = subtract_frame_transforms(
    command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
    command.body_pos_w,
    command.body_quat_w,
  )
  mat = matrix_from_quat(ori_b)
  return mat[..., :2].reshape(mat.shape[0], -1)


def motion_body_lin_vel(
  env: ManagerBasedRlEnv, command_name: str
) -> torch.Tensor:
  """Actual keybody linear velocities in the robot anchor frame."""
  command = _motion_command(env, command_name)
  return _body_lin_vel_in_anchor_frame(
    command.robot_anchor_quat_w, command.robot_body_lin_vel_w
  )


def motion_body_ang_vel(
  env: ManagerBasedRlEnv, command_name: str
) -> torch.Tensor:
  """Actual keybody angular velocities in the robot anchor frame."""
  command = _motion_command(env, command_name)
  return _body_ang_vel_in_anchor_frame(
    command.robot_anchor_quat_w, command.robot_body_ang_vel_w
  )


def ref_body_lin_vel(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Reference keybody linear velocities in the robot anchor frame."""
  command = _motion_command(env, command_name)
  return _body_lin_vel_in_anchor_frame(
    command.robot_anchor_quat_w, command.body_lin_vel_w
  )


def ref_body_ang_vel(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Reference keybody angular velocities in the robot anchor frame."""
  command = _motion_command(env, command_name)
  return _body_ang_vel_in_anchor_frame(
    command.robot_anchor_quat_w, command.body_ang_vel_w
  )


def keybody_contact_forces(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Log-scaled contact forces for tracked keybodies (critic privileged)."""
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.force is not None
  forces_flat = sensor_data.force.flatten(start_dim=1)
  return torch.sign(forces_flat) * torch.log1p(torch.abs(forces_flat))
