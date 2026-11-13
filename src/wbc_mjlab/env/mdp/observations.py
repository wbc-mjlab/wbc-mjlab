from __future__ import annotations

import dataclasses
import inspect
from collections.abc import Callable
from typing import TYPE_CHECKING, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply_inverse,
  quat_box_minus,
  subtract_frame_transforms,
)

from wbc_mjlab.env.mdp.torque_envelope import torque_speed_limits

from .commands import MotionCommand, TrackingView

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _motion_command(env: ManagerBasedRlEnv, command_name: str) -> MotionCommand:
  return cast(MotionCommand, env.command_manager.get_term(command_name))


def _ahead(
  env: ManagerBasedRlEnv,
  command_name: str,
  frames: int,
  stride: int,
  value: Callable[[TrackingView], torch.Tensor],
) -> torch.Tensor:
  """``value`` of the reference now and ``stride``, ``2·stride``, … control steps ahead
  (``frames`` in all), concatenated frame by frame."""
  views = _motion_command(env, command_name).ahead(frames, stride)
  if len(views) == 1:
    return value(views[0])
  return torch.cat([value(v).reshape(env.num_envs, -1) for v in views], dim=-1)


def takes_frames(term: ObservationTermCfg) -> bool:
  """Whether an observation term can look ahead (a ``frames`` parameter)."""
  try:
    return "frames" in inspect.signature(term.func).parameters
  except (TypeError, ValueError):
    return False


def look_ahead(
  term: ObservationTermCfg, frames: int, stride: int = 1
) -> ObservationTermCfg:
  """A copy of reference term ``term`` covering ``frames`` reference frames ``stride``
  control steps apart, its per-value noise bounds repeated for each frame."""
  if not takes_frames(term):
    raise ValueError(f"{getattr(term.func, '__name__', term.func)} can't look ahead")
  params = dict(term.params or {})
  before = int(params.get("frames", 1))
  params |= {"frames": int(frames), "stride": int(stride)}
  noise = term.noise

  def per_frame(v):
    if not isinstance(v, (tuple, list)):
      return v
    width = len(v) // before
    return tuple(v[:width]) * int(frames)

  if noise is not None and hasattr(noise, "n_max"):
    noise = dataclasses.replace(
      noise, n_min=per_frame(noise.n_min), n_max=per_frame(noise.n_max)
    )
  return dataclasses.replace(term, params=params, noise=noise)


# --- Actor reference features (configurable obs terms; were stacked in MotionCommand.command) ---
# SE actor layouts: ``apply_se_actor`` in ``presets/se_actor.py``.
# Every reference term takes ``frames`` (reference frames it covers, the current one first)
# and ``stride`` (control steps between them): the reference ahead, as a term's
# ``history_length`` is its past. Frames past a clip's end repeat its last one.


def ref_base_height(
  env: ManagerBasedRlEnv, command_name: str, frames: int = 1, stride: int = 1
) -> torch.Tensor:
  """Reference anchor height relative to env origin (z_I r̂_IB)."""
  return _ahead(env, command_name, frames, stride, lambda c: c.ref_base_height)


def ref_anchor_pos_w(
  env: ManagerBasedRlEnv, command_name: str, frames: int = 1, stride: int = 1
) -> torch.Tensor:
  """Reference anchor xyz position relative to env origin (SE reference command)."""
  return _ahead(
    env,
    command_name,
    frames,
    stride,
    lambda c: c.anchor_pos_w - c._env.scene.env_origins,
  )


def ref_base_lin_vel_b(
  env: ManagerBasedRlEnv, command_name: str, frames: int = 1, stride: int = 1
) -> torch.Tensor:
  """Reference anchor linear velocity in anchor frame (B v̂_IB)."""
  return _ahead(env, command_name, frames, stride, lambda c: c.ref_base_lin_vel_b)


def ref_base_ang_vel_b(
  env: ManagerBasedRlEnv, command_name: str, frames: int = 1, stride: int = 1
) -> torch.Tensor:
  """Reference anchor angular velocity in anchor frame (B ω̂_IB)."""
  return _ahead(env, command_name, frames, stride, lambda c: c.ref_base_ang_vel_b)


def ref_gravity_b(
  env: ManagerBasedRlEnv, command_name: str, frames: int = 1, stride: int = 1
) -> torch.Tensor:
  """Reference gravity in anchor frame (B ĝ_I)."""
  return _ahead(env, command_name, frames, stride, lambda c: c.ref_gravity_b)


def ref_joint_pos(
  env: ManagerBasedRlEnv, command_name: str, frames: int = 1, stride: int = 1
) -> torch.Tensor:
  """Reference joint positions for tracked DoFs (absolute)."""
  return _ahead(env, command_name, frames, stride, lambda c: c.tracked_joint_pos)


def ref_anchor_shift_b(
  env: ManagerBasedRlEnv, command_name: str, frames: int = 1, stride: int = 1
) -> torch.Tensor:
  """Displacement of the reference anchor from where the reference is now, in the robot
  anchor frame: where the motion goes (zero for the current frame). Needs only the robot's
  orientation (IMU), as deployed; SuperMimic (arXiv:2610.28480) looks 10 frames ahead."""
  now = _motion_command(env, command_name).anchor_pos_w
  return _ahead(
    env,
    command_name,
    frames,
    stride,
    lambda c: quat_apply_inverse(c.robot_anchor_quat_w, c.anchor_pos_w - now),
  )


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
  frames: int = 1,
  stride: int = 1,
) -> torch.Tensor:
  """Reference joint velocities (critic privileged)."""

  def value(c: TrackingView) -> torch.Tensor:
    if asset_cfg.joint_ids is not None:
      return c.joint_vel[:, asset_cfg.joint_ids]
    return c.tracked_joint_vel

  return _ahead(env, command_name, frames, stride, value)


def _ori_6d(quat: torch.Tensor) -> torch.Tensor:
  mat = matrix_from_quat(quat)
  return mat[..., :2].reshape(mat.shape[0], -1)


def ref_anchor_ori_6d(
  env: ManagerBasedRlEnv, command_name: str, frames: int = 1, stride: int = 1
) -> torch.Tensor:
  """Reference anchor orientation as 6D rotation matrix columns (world frame)."""
  return _ahead(env, command_name, frames, stride, lambda c: _ori_6d(c.anchor_quat_w))


def ref_base_lin_acc_b(
  env: ManagerBasedRlEnv, command_name: str, frames: int = 1, stride: int = 1
) -> torch.Tensor:
  """Reference anchor linear acceleration in anchor frame (critic privileged)."""
  return _ahead(env, command_name, frames, stride, lambda c: c.ref_base_lin_acc_b)


def ref_base_ang_acc_b(
  env: ManagerBasedRlEnv, command_name: str, frames: int = 1, stride: int = 1
) -> torch.Tensor:
  """Reference anchor angular acceleration in anchor frame (critic privileged)."""
  return _ahead(env, command_name, frames, stride, lambda c: c.ref_base_ang_acc_b)


def height_scan(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  offset: float = 0.0,
  clip: tuple[float, float] = (-1.0, 1.0),
) -> torch.Tensor:
  """Flattened terrain height scan: per-ray frame clearance minus *offset*, clipped.

  Reads a :class:`~mjlab.sensor.TerrainHeightSensor` with ``reduction="none"``.
  Set *offset* to the nominal frame height so flat ground maps to ~0; misses
  (``max_distance``) saturate at ``clip[1]``.
  """
  heights = env.scene.sensors[sensor_name].data.heights.flatten(start_dim=1)
  return torch.clamp(heights - offset, clip[0], clip[1])


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


def motion_anchor_pos_b(
  env: ManagerBasedRlEnv, command_name: str, frames: int = 1, stride: int = 1
) -> torch.Tensor:
  """Full xyz anchor tracking error in robot anchor frame (non-SE layouts)."""

  def value(c: TrackingView) -> torch.Tensor:
    pos, _ = subtract_frame_transforms(
      c.robot_anchor_pos_w, c.robot_anchor_quat_w, c.anchor_pos_w, c.anchor_quat_w
    )
    return pos.view(env.num_envs, -1)

  return _ahead(env, command_name, frames, stride, value)


def motion_anchor_pos_z_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Z-only anchor tracking error in robot anchor frame (SE measurement)."""
  return motion_anchor_pos_b(env, command_name)[:, 2:3]


def motion_anchor_ori_b(
  env: ManagerBasedRlEnv, command_name: str, frames: int = 1, stride: int = 1
) -> torch.Tensor:
  """Reference anchor orientation in the robot anchor frame (6D rotation columns)."""

  def value(c: TrackingView) -> torch.Tensor:
    _, ori = subtract_frame_transforms(
      c.robot_anchor_pos_w, c.robot_anchor_quat_w, c.anchor_pos_w, c.anchor_quat_w
    )
    return _ori_6d(ori)

  return _ahead(env, command_name, frames, stride, value)


def motion_anchor_pos_error_w(
  env: ManagerBasedRlEnv, command_name: str, frames: int = 1, stride: int = 1
) -> torch.Tensor:
  """World-frame anchor position tracking error (ref − robot)."""
  return _ahead(
    env, command_name, frames, stride, lambda c: c.anchor_pos_w - c.robot_anchor_pos_w
  )


def motion_anchor_ori_error(
  env: ManagerBasedRlEnv, command_name: str, frames: int = 1, stride: int = 1
) -> torch.Tensor:
  """Anchor orientation tracking error as axis-angle (3); ‖·‖ = quat_error_magnitude."""
  return _ahead(
    env,
    command_name,
    frames,
    stride,
    lambda c: quat_box_minus(c.anchor_quat_w, c.robot_anchor_quat_w),
  )


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


def _ref_body_indexes(
  command: MotionCommand, body_names: tuple[str, ...] | None
) -> list[int]:
  """Indices into ``command.cfg.body_names`` (command order; all if ``body_names`` is None)."""
  if body_names is None:
    return list(range(len(command.cfg.body_names)))
  name_to_idx = {name: i for i, name in enumerate(command.cfg.body_names)}
  missing = [name for name in body_names if name not in name_to_idx]
  if missing:
    raise ValueError(
      f"body_names {missing} not in motion command bodies {list(command.cfg.body_names)}"
    )
  return [name_to_idx[name] for name in body_names]


def _ref_body_pose_b(
  c: TrackingView, indexes: list[int]
) -> tuple[torch.Tensor, torch.Tensor]:
  n = len(indexes)
  return subtract_frame_transforms(
    c.robot_anchor_pos_w[:, None, :].repeat(1, n, 1),
    c.robot_anchor_quat_w[:, None, :].repeat(1, n, 1),
    c.body_pos_w[:, indexes],
    c.body_quat_w[:, indexes],
  )


def ref_body_pos_b(
  env: ManagerBasedRlEnv,
  command_name: str,
  body_names: tuple[str, ...] | None = None,
  frames: int = 1,
  stride: int = 1,
) -> torch.Tensor:
  """Reference keybody positions in the robot anchor frame.

  Args:
    body_names: Subset of motion command bodies (default: all), in the given order.
  """
  indexes = _ref_body_indexes(_motion_command(env, command_name), body_names)
  return _ahead(
    env,
    command_name,
    frames,
    stride,
    lambda c: _ref_body_pose_b(c, indexes)[0].reshape(env.num_envs, -1),
  )


def ref_body_ori_b(
  env: ManagerBasedRlEnv,
  command_name: str,
  body_names: tuple[str, ...] | None = None,
  frames: int = 1,
  stride: int = 1,
) -> torch.Tensor:
  """Reference keybody orientations in the robot anchor frame (6D rotation columns).

  Args:
    body_names: Subset of motion command bodies (default: all), in the given order.
  """
  indexes = _ref_body_indexes(_motion_command(env, command_name), body_names)
  return _ahead(
    env,
    command_name,
    frames,
    stride,
    lambda c: _ori_6d(_ref_body_pose_b(c, indexes)[1]),
  )


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


def ref_body_lin_vel(
  env: ManagerBasedRlEnv, command_name: str, frames: int = 1, stride: int = 1
) -> torch.Tensor:
  """Reference keybody linear velocities in the robot anchor frame."""
  return _ahead(
    env,
    command_name,
    frames,
    stride,
    lambda c: _body_lin_vel_in_anchor_frame(c.robot_anchor_quat_w, c.body_lin_vel_w),
  )


def ref_body_ang_vel(
  env: ManagerBasedRlEnv, command_name: str, frames: int = 1, stride: int = 1
) -> torch.Tensor:
  """Reference keybody angular velocities in the robot anchor frame."""
  return _ahead(
    env,
    command_name,
    frames,
    stride,
    lambda c: _body_ang_vel_in_anchor_frame(c.robot_anchor_quat_w, c.body_ang_vel_w),
  )


def keybody_contact_forces(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Log-scaled contact forces for tracked keybodies (critic privileged)."""
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.force is not None
  forces_flat = sensor_data.force.flatten(start_dim=1)
  return torch.sign(forces_flat) * torch.log1p(torch.abs(forces_flat))


# --- Actuation (G1 Unitree envelope) ---


def joint_mechanical_power(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Per-joint mechanical power τ·ω (actuator generalized force)."""
  asset: Entity = env.scene[asset_cfg.name]
  ids = asset_cfg.joint_ids
  tau = asset.data.qfrc_actuator[:, ids]
  qd = asset.data.joint_vel[:, ids]
  return tau * qd


def joint_torque_envelope_ratio(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """|τ| / max admissible torque magnitude at current velocity (≈1 at envelope)."""
  from wbc_mjlab.robots.g1.envelope import g1_joint_envelope_tensors

  asset: Entity = env.scene[asset_cfg.name]
  ids = asset_cfg.joint_ids
  tau = asset.data.qfrc_actuator[:, ids]
  qd = asset.data.joint_vel[:, ids]
  envl = g1_joint_envelope_tensors(env, asset_cfg)
  tau_low, tau_high = torque_speed_limits(qd, envl)
  cap = torch.maximum(tau_high, -tau_low).clamp(min=1.0e-6)
  return (tau.abs() / cap).flatten(start_dim=1)


def joint_torque_envelope_margin(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Normalized distance to torque envelope: 0 on boundary, negative if inside."""
  from wbc_mjlab.robots.g1.envelope import g1_joint_envelope_tensors

  asset: Entity = env.scene[asset_cfg.name]
  ids = asset_cfg.joint_ids
  tau = asset.data.qfrc_actuator[:, ids]
  qd = asset.data.joint_vel[:, ids]
  envl = g1_joint_envelope_tensors(env, asset_cfg)
  tau_low, tau_high = torque_speed_limits(qd, envl)
  cap = torch.maximum(tau_high, -tau_low).clamp(min=1.0e-6)
  margin_high = (tau_high - tau) / cap
  margin_low = (tau - tau_low) / cap
  return torch.minimum(margin_high, margin_low).flatten(start_dim=1)


def motion_segment_phase(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Normalized clip progress in ``[0, 1]`` (privileged critic feature)."""
  command = _motion_command(env, command_name)
  seg_start = command.motion.segment_start_idx[command.trajectory_ids]
  seg_len = command.motion.segment_length[command.trajectory_ids].float().clamp(min=1.0)
  local_step = (command.time_steps - seg_start).float()
  phase = local_step / seg_len
  return phase.clamp(0.0, 1.0).unsqueeze(-1)


def motion_tracking_step_rewards(
  env: ManagerBasedRlEnv, command_name: str
) -> torch.Tensor:
  """Unweighted per-step ``motion_*`` reward values (privileged critic feature)."""
  if not hasattr(env, "reward_manager"):
    n_terms = sum(1 for name in env.cfg.rewards if name.startswith("motion_"))
    return torch.zeros(env.num_envs, n_terms, device=env.device)

  command = _motion_command(env, command_name)
  if not hasattr(command, "_critic_motion_reward_indices"):
    reward_manager = env.reward_manager
    command._critic_motion_reward_indices = [
      idx
      for idx, name in enumerate(reward_manager._term_names)
      if name.startswith("motion_")
    ]
  indices = command._critic_motion_reward_indices
  return env.reward_manager._step_reward[:, indices]
