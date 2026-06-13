"""Write training motion bundles: stacked NPZ + per-clip ``npz/`` subfolder."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class MotionClipExport:
  log: dict[str, Any]
  source_path: Path
  joint_names: list[str]


def npz_output_dir(output_dir: str | Path) -> Path:
  return Path(output_dir).expanduser().resolve() / "npz"


def export_motion_clip_npz(
  *,
  output_dir: str | Path,
  clip: MotionClipExport,
  robot_id: str,
  robot_body_names: list[str],
) -> Path:
  """Write one clip to ``<output_dir>/npz/<stem>.npz``."""
  per_npz = npz_output_dir(output_dir) / f"{clip.source_path.stem}.npz"
  save_motion_npz(
    output_path=per_npz,
    log=clip.log,
    robot_id=robot_id,
    joint_names=clip.joint_names,
    robot_body_names=robot_body_names,
  )
  print(f"[INFO] Exported clip: {per_npz}")
  return per_npz


def save_motion_npz(
  *,
  output_path: Path,
  log: dict[str, Any],
  robot_id: str,
  joint_names: list[str],
  robot_body_names: list[str],
  segment_start_idx: np.ndarray | None = None,
  segment_length: np.ndarray | None = None,
  segment_source: np.ndarray | None = None,
  segment_names: np.ndarray | None = None,
) -> None:
  _ = robot_body_names
  output_path.parent.mkdir(parents=True, exist_ok=True)
  payload: dict[str, Any] = {
    "fps": np.asarray(log["fps"], dtype=np.float32),
    "joint_names": np.asarray(joint_names, dtype=object),
    "robot": np.asarray(robot_id, dtype=object),
    "joint_pos": log["joint_pos"],
    "joint_vel": log["joint_vel"],
    "body_pos_w": log["body_pos_w"],
    "body_quat_w": log["body_quat_w"],
    "body_lin_vel_w": log["body_lin_vel_w"],
    "body_ang_vel_w": log["body_ang_vel_w"],
  }
  if segment_start_idx is not None:
    payload["segment_start_idx"] = segment_start_idx
    payload["segment_length"] = segment_length
    payload["segment_source"] = segment_source
    if segment_names is not None:
      payload["segment_names"] = segment_names
  np.savez(output_path, **payload)


def export_merged_training_bundle(
  *,
  output_dir: str | Path,
  clips: list[MotionClipExport],
  robot_id: str,
  robot_body_names: list[str],
  output_fps: float,
) -> Path:
  """Stack clips into ``<output_dir>/<dirname>.npz`` (per-clip files must exist already)."""
  from wbc_mjlab.motion.stack_bundle import (
    dataset_bundle_path,
    write_merged_training_bundle,
  )

  root = Path(output_dir).expanduser().resolve()
  root.mkdir(parents=True, exist_ok=True)
  merged_npz = write_merged_training_bundle(
    output_path=dataset_bundle_path(root),
    clips=clips,
    robot_id=robot_id,
    output_fps=output_fps,
  )
  _ = robot_body_names
  print(f"[INFO] Exported training bundle: {merged_npz} ({len(clips)} clips)")
  return merged_npz


def export_motion_training_bundle(
  *,
  output_dir: str | Path,
  clips: list[MotionClipExport],
  robot_id: str,
  robot_body_names: list[str],
  output_fps: float,
) -> Path:
  """Export per-clip ``npz/*.npz`` and stacked ``<name>.npz``."""
  npz_output_dir(output_dir).mkdir(parents=True, exist_ok=True)
  for clip in clips:
    export_motion_clip_npz(
      output_dir=output_dir,
      clip=clip,
      robot_id=robot_id,
      robot_body_names=robot_body_names,
    )
  return export_merged_training_bundle(
    output_dir=output_dir,
    clips=clips,
    robot_id=robot_id,
    robot_body_names=robot_body_names,
    output_fps=output_fps,
  )
