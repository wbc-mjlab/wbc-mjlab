"""Stack per-clip motion NPZs into one training bundle."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from wbc_mjlab.motion.motion_export_bundle import (
  MotionClipExport,
  npz_output_dir,
  save_motion_npz,
)


def dataset_bundle_path(dataset_root: Path) -> Path:
  root = dataset_root.resolve()
  return root / f"{root.name}.npz"


def list_clip_npz_files(dataset_root: Path) -> list[Path]:
  return sorted(npz_output_dir(dataset_root).glob("*.npz"))


def _clip_mtime_digest(clips: list[Path]) -> str:
  parts = [f"{p.resolve()}:{p.stat().st_mtime_ns}" for p in clips]
  return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def load_motion_clip_npz(path: Path) -> MotionClipExport:
  data = np.load(path, allow_pickle=True)
  joint_names = [str(x) for x in data["joint_names"].tolist()]
  log: dict[str, Any] = {
    "fps": np.asarray(data["fps"]),
    "joint_pos": np.asarray(data["joint_pos"]),
    "joint_vel": np.asarray(data["joint_vel"]),
    "body_pos_w": np.asarray(data["body_pos_w"]),
    "body_quat_w": np.asarray(data["body_quat_w"]),
    "body_lin_vel_w": np.asarray(data["body_lin_vel_w"]),
    "body_ang_vel_w": np.asarray(data["body_ang_vel_w"]),
  }
  return MotionClipExport(log=log, source_path=path, joint_names=joint_names)


def load_clips_from_npz_dir(dataset_root: Path) -> list[MotionClipExport]:
  clips = list_clip_npz_files(dataset_root)
  if not clips:
    raise FileNotFoundError(f"No clip NPZs in {npz_output_dir(dataset_root)}")
  return [load_motion_clip_npz(path) for path in clips]


def _output_fps_from_clips(clips: list[MotionClipExport]) -> float:
  fps_raw = clips[0].log["fps"]
  fps_arr = np.asarray(fps_raw, dtype=np.float64).reshape(-1)
  return float(fps_arr[0])


def write_merged_training_bundle(
  *,
  output_path: Path,
  clips: list[MotionClipExport],
  robot_id: str,
  output_fps: float | None = None,
) -> Path:
  if not clips:
    raise ValueError("No motion clips to stack")
  fps = output_fps if output_fps is not None else _output_fps_from_clips(clips)

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
    length = int(clip.log["joint_pos"].shape[0])
    for key in merged_stacked:
      merged_stacked[key].append(clip.log[key])
    segment_start_idx.append(running_start)
    segment_length.append(length)
    segment_source.append(str(clip.source_path.resolve()))
    running_start += length

  output_path.parent.mkdir(parents=True, exist_ok=True)
  save_motion_npz(
    output_path=output_path,
    log={
      "fps": [fps],
      "joint_pos": np.concatenate(merged_stacked["joint_pos"], axis=0),
      "joint_vel": np.concatenate(merged_stacked["joint_vel"], axis=0),
      "body_pos_w": np.concatenate(merged_stacked["body_pos_w"], axis=0),
      "body_quat_w": np.concatenate(merged_stacked["body_quat_w"], axis=0),
      "body_lin_vel_w": np.concatenate(merged_stacked["body_lin_vel_w"], axis=0),
      "body_ang_vel_w": np.concatenate(merged_stacked["body_ang_vel_w"], axis=0),
    },
    robot_id=robot_id,
    joint_names=clips[0].joint_names,
    robot_body_names=[],
    segment_start_idx=np.asarray(segment_start_idx, dtype=np.int64),
    segment_length=np.asarray(segment_length, dtype=np.int64),
    segment_source=np.asarray(segment_source, dtype=object),
  )
  return output_path


def _bundle_is_fresh(bundle: Path, clips: list[Path]) -> bool:
  if not bundle.is_file() or not clips:
    return False
  bundle_mtime = bundle.stat().st_mtime
  return all(bundle_mtime >= clip.stat().st_mtime for clip in clips)


def _temp_bundle_path(dataset_root: Path, clips: list[Path]) -> Path:
  digest = _clip_mtime_digest(clips)
  root = dataset_root.resolve()
  cache_dir = Path(tempfile.gettempdir()) / "wbc_mjlab" / "motion_bundles"
  cache_dir.mkdir(parents=True, exist_ok=True)
  return cache_dir / f"{root.name}_{digest}.npz"


def ensure_training_motion_bundle(
  dataset_root: Path,
  *,
  robot_id: str,
  cache_motion_bundle: bool = False,
) -> Path:
  """Resolve stacked motion NPZ from ``npz/*.npz`` (and optional on-disk cache)."""
  root = dataset_root.resolve()
  bundle = dataset_bundle_path(root)
  clips = list_clip_npz_files(root)

  if clips and _bundle_is_fresh(bundle, clips):
    return bundle
  if bundle.is_file() and not clips:
    return bundle

  if not clips:
    raise FileNotFoundError(
      f"No training motion in {root}. Expected {bundle.name} or npz/*.npz "
      f"(convert clips with wbc-mjlab-csv-to-npz)."
    )

  loaded = [load_motion_clip_npz(path) for path in clips]
  if cache_motion_bundle:
    out = write_merged_training_bundle(
      output_path=bundle,
      clips=loaded,
      robot_id=robot_id,
    )
    print(f"[INFO] Cached training bundle: {out} ({len(clips)} clips)")
    return out

  temp_bundle = _temp_bundle_path(root, clips)
  if temp_bundle.is_file() and _bundle_is_fresh(temp_bundle, clips):
    print(f"[INFO] Using temp training bundle: {temp_bundle} ({len(clips)} clips)")
    return temp_bundle

  out = write_merged_training_bundle(
    output_path=temp_bundle,
    clips=loaded,
    robot_id=robot_id,
  )
  print(
    f"[INFO] Stacked training bundle (temp): {out} ({len(clips)} clips). "
    "Pass --cache-motion-bundle to write <dataset>/<dataset>.npz."
  )
  return out
