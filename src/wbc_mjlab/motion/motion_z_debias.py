"""Vertical debias for offline motion export (CSV/PKL -> NPZ)."""

from __future__ import annotations

from typing import Any

import numpy as np


def debias_motion_log_vertical(
  log: dict[str, Any],
  *,
  robot_body_names: list[str],
  foot_body_names: tuple[str, ...],
  foot_sole_z: float,
) -> float:
  """Align clip height using logged ``body_pos_w``.

  Subtract ``min(foot_z) - foot_sole_z`` from all body world z coordinates so the
  lowest foot body origin sits ``foot_sole_z`` above the ground (sole at z=0).
  """
  name_to_idx = {name: idx for idx, name in enumerate(robot_body_names)}
  missing = [name for name in foot_body_names if name not in name_to_idx]
  if missing:
    raise ValueError(
      f"Foot bodies not found in robot body list: {missing}. "
      f"Available: {list(robot_body_names)}"
    )
  foot_indices = [name_to_idx[name] for name in foot_body_names]

  body_pos_w = np.asarray(log["body_pos_w"], dtype=np.float32)
  min_foot_z = float(np.min(body_pos_w[:, foot_indices, 2]))
  z_shift = min_foot_z - foot_sole_z
  body_pos_w = body_pos_w.copy()
  body_pos_w[:, :, 2] -= z_shift
  log["body_pos_w"] = body_pos_w
  return z_shift
