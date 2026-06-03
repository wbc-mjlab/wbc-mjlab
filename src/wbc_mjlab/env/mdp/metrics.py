from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.utils.lab_api.math import quat_error_magnitude

if TYPE_CHECKING:
  from wbc_mjlab.env.mdp.commands import MotionCommand


def compute_mpkpe(command: MotionCommand) -> torch.Tensor:
  pos_error = command.body_pos_relative_w - command.robot_body_pos_w
  per_body_error = torch.norm(pos_error, dim=-1)
  return per_body_error.mean(dim=-1)


def compute_root_relative_mpkpe(command: MotionCommand) -> torch.Tensor:
  ref_anchor_pos = command.anchor_pos_w.unsqueeze(1)
  ref_rel_pos = command.body_pos_w - ref_anchor_pos

  robot_anchor_pos = command.robot_anchor_pos_w.unsqueeze(1)
  robot_rel_pos = command.robot_body_pos_w - robot_anchor_pos

  pos_error = ref_rel_pos - robot_rel_pos
  per_body_error = torch.norm(pos_error, dim=-1)
  return per_body_error.mean(dim=-1)


def compute_joint_velocity_error(command: MotionCommand) -> torch.Tensor:
  vel_error = command.joint_vel - command.robot_joint_vel
  return torch.norm(vel_error, dim=-1)


def compute_ee_position_error(
  command: MotionCommand,
  ee_body_names: tuple[str, ...],
) -> torch.Tensor:
  ee_indices = _get_body_indices(command, ee_body_names)
  if len(ee_indices) == 0:
    return torch.zeros(command.num_envs, device=command.device)

  ref_ee_pos = command.body_pos_relative_w[:, ee_indices]
  robot_ee_pos = command.robot_body_pos_w[:, ee_indices]

  pos_error = ref_ee_pos - robot_ee_pos
  per_ee_error = torch.norm(pos_error, dim=-1)
  return per_ee_error.mean(dim=-1)


def compute_ee_orientation_error(
  command: MotionCommand,
  ee_body_names: tuple[str, ...],
) -> torch.Tensor:
  ee_indices = _get_body_indices(command, ee_body_names)
  if len(ee_indices) == 0:
    return torch.zeros(command.num_envs, device=command.device)

  ref_ee_quat = command.body_quat_relative_w[:, ee_indices]
  robot_ee_quat = command.robot_body_quat_w[:, ee_indices]

  per_ee_error = quat_error_magnitude(ref_ee_quat, robot_ee_quat)
  return per_ee_error.mean(dim=-1)


def _get_body_indices(
  command: MotionCommand,
  body_names: tuple[str, ...],
) -> list[int]:
  return [i for i, name in enumerate(command.cfg.body_names) if name in body_names]
