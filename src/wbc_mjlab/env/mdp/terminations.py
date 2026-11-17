"""Episode termination terms for WBC motion tracking."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse

from .commands import MotionCommand
from .rewards import _get_body_indexes

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.scene_entity_config import SceneEntityCfg


def bad_anchor_pos(
  env: ManagerBasedRlEnv, command_name: str, threshold: float
) -> torch.Tensor:
  """Terminate when anchor body position drifts beyond *threshold* (m)."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  return (
    torch.norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=1) > threshold
  )


def bad_anchor_pos_z_only(
  env: ManagerBasedRlEnv, command_name: str, threshold: float
) -> torch.Tensor:
  """Terminate when anchor height (z) drifts beyond *threshold* (m)."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  return (
    torch.abs(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1])
    > threshold
  )


def bad_anchor_ori(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float
) -> torch.Tensor:
  """Terminate when projected-gravity z mismatch exceeds *threshold*."""
  asset: Entity = env.scene[asset_cfg.name]
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  motion_projected_gravity_b = quat_apply_inverse(
    command.anchor_quat_w, asset.data.gravity_vec_w
  )
  robot_projected_gravity_b = quat_apply_inverse(
    command.robot_anchor_quat_w, asset.data.gravity_vec_w
  )
  return (
    motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]
  ).abs() > threshold


def bad_motion_body_pos(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  """Terminate when any keybody position error exceeds *threshold* (m).

  Args:
    env: Manager-based RL env.
    command_name: Motion command term name.
    threshold: Max allowed L2 position error per body.
    body_names: Subset of command bodies (default: all).
  """
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.norm(
    command.body_pos_relative_w[:, body_indexes]
    - command.robot_body_pos_w[:, body_indexes],
    dim=-1,
  )
  return torch.any(error > threshold, dim=-1)


def bad_motion_body_pos_z_only(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  """Terminate when any keybody height error exceeds *threshold* (m).

  Typically used for end-effector bodies (EE height termination).
  """
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.abs(
    command.body_pos_relative_w[:, body_indexes, -1]
    - command.robot_body_pos_w[:, body_indexes, -1]
  )
  return torch.any(error > threshold, dim=-1)


def _clearance_by_body(
  env: ManagerBasedRlEnv, sensor_name: str, body_names: tuple[str, ...]
) -> torch.Tensor:
  """Robot body clearance above terrain ``[B, len(body_names)]`` from a per-body
  :class:`~mjlab.sensor.TerrainHeightSensor` (one frame per body, ``reduction="min"``)."""
  sensor = env.scene[sensor_name]
  frames = (
    sensor.cfg.frame if isinstance(sensor.cfg.frame, tuple) else (sensor.cfg.frame,)
  )
  frame_names = [f.name for f in frames]
  missing = [name for name in body_names if name not in frame_names]
  if missing:
    raise ValueError(f"Bodies {missing} are not frames of sensor {sensor_name!r}.")
  idx = [frame_names.index(name) for name in body_names]
  return sensor.data.heights[:, idx]


def bad_anchor_pos_z_terrain(
  env: ManagerBasedRlEnv,
  command_name: str,
  sensor_name: str,
  threshold: float,
) -> torch.Tensor:
  """Terrain-aware :func:`bad_anchor_pos_z_only`: compares clearances.

  Reference clearance is anchor height above the env origin (mocap is
  flat-ground); robot clearance is the anchor's height above the terrain under it.
  """
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  ref_clearance = command.anchor_pos_w[:, 2] - env.scene.env_origins[:, 2]
  robot_clearance = _clearance_by_body(
    env, sensor_name, (command.cfg.anchor_body_name,)
  )[:, 0]
  return torch.abs(ref_clearance - robot_clearance) > threshold


def bad_motion_body_pos_z_terrain(
  env: ManagerBasedRlEnv,
  command_name: str,
  sensor_name: str,
  threshold: float,
  body_names: tuple[str, ...],
) -> torch.Tensor:
  """Terrain-aware :func:`bad_motion_body_pos_z_only`: compares clearances.

  Reference clearance is height above the env origin (mocap is flat-ground);
  robot clearance is read from *sensor_name* frames named after *body_names*.
  """
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  ref_clearance = (
    command.body_pos_relative_w[:, body_indexes, -1] - env.scene.env_origins[:, None, 2]
  )
  robot_clearance = _clearance_by_body(env, sensor_name, body_names)
  return torch.any(torch.abs(ref_clearance - robot_clearance) > threshold, dim=-1)


def excessive_contact_force(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  """Terminate when any contact slot on *sensor_name* exceeds *force_threshold* (N)."""
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    force_mag = torch.norm(data.force_history, dim=-1)
    return (force_mag > force_threshold).any(dim=-1).any(dim=-1)
  if data.force is not None:
    force_mag = torch.norm(data.force, dim=-1)
    return (force_mag > force_threshold).any(dim=-1)
  assert data.found is not None
  return torch.any(data.found, dim=-1)


def excessive_keybody_ground_contact_force(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  body_names: tuple[str, ...],
  body_slot_order: tuple[str, ...],
  force_threshold: float,
) -> torch.Tensor:
  """Terminate when listed keybodies exceed *force_threshold* on the ground sensor."""
  if not body_names:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force is None:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

  name_to_idx = {name: idx for idx, name in enumerate(body_slot_order)}
  missing = [name for name in body_names if name not in name_to_idx]
  if missing:
    raise ValueError(
      f"Unknown keybody names for {sensor_name!r}: {missing}. "
      f"Expected names from body_slot_order."
    )
  indices = [name_to_idx[name] for name in body_names]
  forces = data.force[:, indices, :]
  force_mag = torch.norm(forces, dim=-1)
  return (force_mag > force_threshold).any(dim=-1)
