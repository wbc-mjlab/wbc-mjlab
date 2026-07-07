"""Sagittal-plane mirroring for exported motion NPZ clips."""

from __future__ import annotations

from typing import Any

import numpy as np

from wbc_mjlab.robots.symmetry import RobotSymmetryConfig


def build_joint_mirror_map(
  joint_names: list[str],
  config: RobotSymmetryConfig,
) -> tuple[np.ndarray, np.ndarray]:
  """Return ``(source_indices, scales)`` for ``mirrored[:, i] = raw[:, idx[i]] * scale[i]``."""
  n = len(joint_names)
  indices = np.arange(n, dtype=np.int64)
  scales = np.ones(n, dtype=np.float64)
  for entry in config.joints:
    if entry.name_a not in joint_names or entry.name_b not in joint_names:
      continue
    ia = joint_names.index(entry.name_a)
    ib = joint_names.index(entry.name_b)
    indices[[ia, ib]] = indices[[ib, ia]]
    scales[ia] = entry.scale
    scales[ib] = entry.scale
  return indices, scales


def build_body_swap_indices(body_names: list[str]) -> np.ndarray:
  """Permutation swapping ``left_*`` and ``right_*`` body link names."""
  perm = np.arange(len(body_names), dtype=np.int64)
  name_to_idx = {name: idx for idx, name in enumerate(body_names)}
  for idx, name in enumerate(body_names):
    if "left" in name:
      mirror_name = name.replace("left", "right", 1)
      if mirror_name in name_to_idx:
        perm[idx] = name_to_idx[mirror_name]
    elif "right" in name:
      mirror_name = name.replace("right", "left", 1)
      if mirror_name in name_to_idx:
        perm[idx] = name_to_idx[mirror_name]
  return perm


def mirror_joint_array(
  array: np.ndarray,
  *,
  joint_names: list[str],
  config: RobotSymmetryConfig,
) -> np.ndarray:
  """Mirror ``(T, J)`` joint positions or velocities.

  Generalized coordinates and their time derivatives transform identically under
  the robot's left-right symmetry (swap pairs + sign on roll/yaw axes).
  """
  indices, scales = build_joint_mirror_map(joint_names, config)
  return array[:, indices] * scales[np.newaxis, :]


def mirror_body_arrays(
  body_pos_w: np.ndarray,
  body_quat_w: np.ndarray,
  body_lin_vel_w: np.ndarray,
  body_ang_vel_w: np.ndarray,
  *,
  body_names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Mirror body kinematics in world frame (reflection across the XZ plane, ``y -> -y``).

  Transforms (MuJoCo ``xyzw`` quaternions, world-frame velocities):

  - **Position / linear velocity** (polar vectors): ``(x, y, z) -> (x, -y, z)``
  - **Angular velocity** (axial vector): ``(wx, wy, wz) -> (-wx, wy, -wz)``
  - **Orientation**: body quat components ``x`` and ``z`` negated after L/R swap
    (equivalent to ``R' = S R S`` with ``S = diag(1, -1, 1)``)
  """
  perm = build_body_swap_indices(body_names)

  pos = body_pos_w[:, perm].copy()
  pos[:, :, 1] *= -1.0

  quat = body_quat_w[:, perm].copy()
  quat[:, :, [0, 2]] *= -1.0

  lin_vel = body_lin_vel_w[:, perm].copy()
  lin_vel[:, :, 1] *= -1.0

  ang_vel = body_ang_vel_w[:, perm].copy()
  ang_vel[:, :, [0, 2]] *= -1.0

  return pos, quat, lin_vel, ang_vel


def mirror_motion_log(
  log: dict[str, Any],
  *,
  joint_names: list[str],
  body_names: list[str],
  config: RobotSymmetryConfig,
) -> dict[str, Any]:
  """Return a deep-copied motion log with sagittal mirroring applied."""
  joint_pos = mirror_joint_array(
    np.asarray(log["joint_pos"]),
    joint_names=joint_names,
    config=config,
  )
  joint_vel = mirror_joint_array(
    np.asarray(log["joint_vel"]),
    joint_names=joint_names,
    config=config,
  )
  body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w = mirror_body_arrays(
    np.asarray(log["body_pos_w"]),
    np.asarray(log["body_quat_w"]),
    np.asarray(log["body_lin_vel_w"]),
    np.asarray(log["body_ang_vel_w"]),
    body_names=body_names,
  )
  return {
    "fps": list(log["fps"]) if isinstance(log["fps"], list) else log["fps"],
    "joint_pos": joint_pos,
    "joint_vel": joint_vel,
    "body_pos_w": body_pos_w,
    "body_quat_w": body_quat_w,
    "body_lin_vel_w": body_lin_vel_w,
    "body_ang_vel_w": body_ang_vel_w,
  }
