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
  np.savez(output_path, **payload)


def export_motion_training_bundle(
  *,
  output_dir: str | Path,
  clips: list[MotionClipExport],
  robot_id: str,
  robot_body_names: list[str],
  output_fps: float,
) -> Path:
  """Export ``<output_dir>/<name>.npz`` plus per-clip ``npz/*.npz``."""
  if not clips:
    raise ValueError("No motion clips to export")

  root = Path(output_dir).expanduser().resolve()
  root.mkdir(parents=True, exist_ok=True)
  npz_dir = root / "npz"
  npz_dir.mkdir(exist_ok=True)

  merged_stacked: dict[str, list[Any]] = {
    "joint_pos": [],
    "joint_vel": [],
    "body_pos_w": [],
    "body_quat_w": [],
    "body_lin_vel_w": [],
    "body_ang_vel_w": [],
  }
  segment_start_idx: list[int] = []
  segment_length: list[int] = []
  segment_source: list[str] = []
  running_start = 0

  for clip in clips:
    stem = clip.source_path.stem
    per_npz = npz_dir / f"{stem}.npz"
    save_motion_npz(
      output_path=per_npz,
      log=clip.log,
      robot_id=robot_id,
      joint_names=clip.joint_names,
      robot_body_names=robot_body_names,
    )
    print(f"[INFO] Exported clip: {per_npz}")

    length = int(clip.log["joint_pos"].shape[0])
    for key in merged_stacked:
      merged_stacked[key].append(clip.log[key])
    segment_start_idx.append(running_start)
    segment_length.append(length)
    segment_source.append(str(clip.source_path.resolve()))
    running_start += length

  bundle_name = root.name
  merged_npz = root / f"{bundle_name}.npz"
  merged_joint_names = clips[0].joint_names
  save_motion_npz(
    output_path=merged_npz,
    log={
      "fps": [output_fps],
      "joint_pos": np.concatenate(merged_stacked["joint_pos"], axis=0),
      "joint_vel": np.concatenate(merged_stacked["joint_vel"], axis=0),
      "body_pos_w": np.concatenate(merged_stacked["body_pos_w"], axis=0),
      "body_quat_w": np.concatenate(merged_stacked["body_quat_w"], axis=0),
      "body_lin_vel_w": np.concatenate(merged_stacked["body_lin_vel_w"], axis=0),
      "body_ang_vel_w": np.concatenate(merged_stacked["body_ang_vel_w"], axis=0),
    },
    robot_id=robot_id,
    joint_names=merged_joint_names,
    robot_body_names=robot_body_names,
    segment_start_idx=np.asarray(segment_start_idx, dtype=np.int64),
    segment_length=np.asarray(segment_length, dtype=np.int64),
    segment_source=np.asarray(segment_source, dtype=object),
  )
  print(f"[INFO] Exported training bundle: {merged_npz} ({len(clips)} clips)")
  return merged_npz
