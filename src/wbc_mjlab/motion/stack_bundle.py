"""Stack per-clip motion NPZs (in memory or optional on-disk cache)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wbc_mjlab.motion.manifest import clip_name_from_path
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


def load_motion_clip_npz(path: Path) -> MotionClipExport:
  with np.load(path, allow_pickle=True) as data:
    joint_names = [str(x) for x in data["joint_names"].tolist()]
    log = {
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


def _output_fps_from_clip_paths(clip_paths: list[Path]) -> float:
  with np.load(clip_paths[0], mmap_mode="r") as data:
    fps_arr = np.asarray(data["fps"], dtype=np.float64).reshape(-1)
  return float(fps_arr[0])


def _load_joint_names(clip_paths: list[Path]) -> list[str]:
  with np.load(clip_paths[0], allow_pickle=True) as data:
    return [str(x) for x in data["joint_names"].tolist()]


def _robot_id_from_clip_paths(clip_paths: list[Path]) -> str:
  with np.load(clip_paths[0], allow_pickle=True) as data:
    if "robot" in data:
      return str(np.asarray(data["robot"]).flat[0])
  return "g1"


@dataclass(frozen=True)
class StackedMotionArrays:
  joint_pos: np.ndarray
  joint_vel: np.ndarray
  body_pos_w: np.ndarray
  body_quat_w: np.ndarray
  body_lin_vel_w: np.ndarray
  body_ang_vel_w: np.ndarray
  segment_start_idx: np.ndarray
  segment_length: np.ndarray
  clip_paths: tuple[Path, ...]
  clip_names: tuple[str, ...]
  fps: float


def stack_clip_arrays_from_paths(clip_paths: list[Path]) -> StackedMotionArrays:
  """Concatenate clip NPZs in memory (no disk write)."""
  if not clip_paths:
    raise ValueError("No motion clips to stack")

  clip_paths = [p.resolve() for p in clip_paths]
  clip_names = tuple(clip_name_from_path(path) for path in clip_paths)
  fps = _output_fps_from_clip_paths(clip_paths)

  starts: list[int] = []
  lengths: list[int] = []
  offset = 0
  for path in clip_paths:
    with np.load(path, mmap_mode="r") as data:
      length = int(data["joint_pos"].shape[0])
    starts.append(offset)
    lengths.append(length)
    offset += length
  total_frames = offset

  with np.load(clip_paths[0], mmap_mode="r") as probe:
    n_joints = int(probe["joint_pos"].shape[1])
    n_bodies = int(probe["body_pos_w"].shape[1])

  arr_joint_pos = np.empty((total_frames, n_joints), dtype=np.float32)
  arr_joint_vel = np.empty((total_frames, n_joints), dtype=np.float32)
  arr_body_pos_w = np.empty((total_frames, n_bodies, 3), dtype=np.float32)
  arr_body_quat_w = np.empty((total_frames, n_bodies, 4), dtype=np.float32)
  arr_body_lin_vel_w = np.empty((total_frames, n_bodies, 3), dtype=np.float32)
  arr_body_ang_vel_w = np.empty((total_frames, n_bodies, 3), dtype=np.float32)

  for i, path in enumerate(clip_paths):
    start = starts[i]
    length = lengths[i]
    sl = slice(start, start + length)
    with np.load(path, mmap_mode="r") as data:
      arr_joint_pos[sl] = data["joint_pos"]
      arr_joint_vel[sl] = data["joint_vel"]
      arr_body_pos_w[sl] = data["body_pos_w"]
      arr_body_quat_w[sl] = data["body_quat_w"]
      arr_body_lin_vel_w[sl] = data["body_lin_vel_w"]
      arr_body_ang_vel_w[sl] = data["body_ang_vel_w"]

  return StackedMotionArrays(
    joint_pos=arr_joint_pos,
    joint_vel=arr_joint_vel,
    body_pos_w=arr_body_pos_w,
    body_quat_w=arr_body_quat_w,
    body_lin_vel_w=arr_body_lin_vel_w,
    body_ang_vel_w=arr_body_ang_vel_w,
    segment_start_idx=np.asarray(starts, dtype=np.int64),
    segment_length=np.asarray(lengths, dtype=np.int64),
    clip_paths=tuple(clip_paths),
    clip_names=clip_names,
    fps=fps,
  )


def write_merged_training_bundle_from_paths(
  *,
  output_path: Path,
  clip_paths: list[Path],
  robot_id: str,
  dataset: str | None = None,
  output_fps: float | None = None,
) -> Path:
  """Write stacked clip NPZs to a single bundle file on disk."""
  stacked = stack_clip_arrays_from_paths(clip_paths)
  fps = output_fps if output_fps is not None else stacked.fps
  joint_names = _load_joint_names(list(stacked.clip_paths))
  output_path.parent.mkdir(parents=True, exist_ok=True)
  save_motion_npz(
    output_path=output_path,
    log={
      "fps": [fps],
      "joint_pos": stacked.joint_pos,
      "joint_vel": stacked.joint_vel,
      "body_pos_w": stacked.body_pos_w,
      "body_quat_w": stacked.body_quat_w,
      "body_lin_vel_w": stacked.body_lin_vel_w,
      "body_ang_vel_w": stacked.body_ang_vel_w,
    },
    robot_id=robot_id,
    joint_names=joint_names,
    robot_body_names=[],
    segment_start_idx=stacked.segment_start_idx,
    segment_length=stacked.segment_length,
    segment_source=np.asarray([str(p) for p in stacked.clip_paths], dtype=object),
    segment_names=np.asarray(stacked.clip_names, dtype=object),
  )
  return output_path


def write_merged_training_bundle(
  *,
  output_path: Path,
  clips: list[MotionClipExport],
  robot_id: str,
  dataset: str | None = None,
  output_fps: float | None = None,
) -> Path:
  clip_paths = [clip.source_path for clip in clips]
  fps = output_fps
  if fps is None and clips:
    fps_arr = np.asarray(clips[0].log["fps"], dtype=np.float64).reshape(-1)
    fps = float(fps_arr[0])
  return write_merged_training_bundle_from_paths(
    output_path=output_path,
    clip_paths=clip_paths,
    robot_id=robot_id,
    dataset=dataset,
    output_fps=fps,
  )


def _bundle_is_fresh(bundle: Path, clips: list[Path]) -> bool:
  if not bundle.is_file() or not clips:
    return False
  bundle_mtime = bundle.stat().st_mtime
  return all(bundle_mtime >= clip.stat().st_mtime for clip in clips)


def resolve_training_motion_source(
  dataset_root: Path,
  *,
  robot_id: str,
  cache_motion_bundle: bool = False,
) -> Path:
  """Return a dataset directory (load ``npz/*.npz`` in memory) or a bundled ``.npz`` path."""
  root = dataset_root.resolve()
  bundle = dataset_bundle_path(root)
  clips = list_clip_npz_files(root)

  if cache_motion_bundle:
    if clips:
      if not _bundle_is_fresh(bundle, clips):
        write_merged_training_bundle_from_paths(
          output_path=bundle,
          clip_paths=clips,
          robot_id=robot_id,
          dataset=root.name,
        )
        print(f"[INFO] Cached training bundle: {bundle} ({len(clips)} clips)")
      else:
        print(f"[INFO] Using cached training bundle: {bundle} ({len(clips)} clips)")
      return bundle
    if bundle.is_file():
      return bundle
    raise FileNotFoundError(
      f"No training motion in {root}. Expected npz/*.npz or {bundle.name} "
      f"(convert clips with wbc-mjlab-data-to-npz)."
    )

  if clips:
    print(
      f"[INFO] Motion library: {len(clips)} clips in {npz_output_dir(root)} "
      "(in-memory stack; pass --cache-motion-bundle to write "
      f"{bundle.name})"
    )
    return root

  if bundle.is_file():
    return bundle

  raise FileNotFoundError(
    f"No training motion in {root}. Expected npz/*.npz or {bundle.name} "
    f"(convert clips with wbc-mjlab-data-to-npz)."
  )


# Back-compat alias.
ensure_training_motion_bundle = resolve_training_motion_source
