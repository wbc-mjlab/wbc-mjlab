"""Motion library manifest built from a loaded bundle (written on play/export)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from wbc_mjlab.deploy_paths import PLAY_MOTION_LIBRARY_MANIFEST

MANIFEST_SCHEMA = "wbc_motion_library_v1"


def clip_name_from_path(path: str | Path) -> str:
  return Path(path).stem


def infer_dataset_name(bundle_path: Path, segment_sources: tuple[str, ...] = ()) -> str:
  """Guess dataset folder name from bundle path or clip source paths."""
  bundle_path = bundle_path.resolve()
  if bundle_path.name == f"{bundle_path.parent.name}.npz":
    return bundle_path.parent.name

  for source in segment_sources:
    clip_path = Path(source).resolve()
    if clip_path.parent.name == "npz" and clip_path.parent.parent.name:
      return clip_path.parent.parent.name

  stem = bundle_path.stem
  suffix = stem.rsplit("_", 1)[-1]
  if "_" in stem and len(suffix) == 16 and suffix.isalnum():
    return stem.rsplit("_", 1)[0]
  return stem


def build_motion_library_manifest(motion_file: str | Path) -> dict[str, Any]:
  """Build manifest metadata from a dataset directory or bundled NPZ."""
  source = Path(motion_file).expanduser().resolve()
  if source.is_dir():
    return build_motion_library_manifest_from_dataset(source)
  with np.load(source, allow_pickle=True) as data:
    return build_motion_library_manifest_from_npz(data, source)


def build_motion_library_manifest_from_dataset(dataset_root: Path) -> dict[str, Any]:
  """Build manifest metadata from ``npz/*.npz`` under a dataset directory."""
  from wbc_mjlab.motion.stack_bundle import (
    _output_fps_from_clip_paths,
    _robot_id_from_clip_paths,
    list_clip_npz_files,
  )

  dataset_root = dataset_root.resolve()
  clip_paths = list_clip_npz_files(dataset_root)
  if not clip_paths:
    raise FileNotFoundError(f"No clip NPZs in {dataset_root / 'npz'}")

  clip_names = [clip_name_from_path(path) for path in clip_paths]
  total_frames = 0
  for path in clip_paths:
    with np.load(path, mmap_mode="r") as data:
      total_frames += int(data["joint_pos"].shape[0])

  return {
    "schema": MANIFEST_SCHEMA,
    "dataset": dataset_root.name,
    "robot": _robot_id_from_clip_paths(clip_paths),
    "bundle": None,
    "bundle_path": str(dataset_root),
    "num_clips": len(clip_names),
    "total_frames": total_frames,
    "fps": _output_fps_from_clip_paths(clip_paths),
    "clips": clip_names,
  }


def build_motion_library_manifest_from_npz(
  data: np.lib.npyio.NpzFile,
  bundle_path: Path,
) -> dict[str, Any]:
  """Build manifest metadata from an already-open NPZ handle."""
  bundle_path = bundle_path.expanduser().resolve()
  fps = float(np.asarray(data["fps"], dtype=np.float64).reshape(-1)[0])
  total_frames = int(data["joint_pos"].shape[0])
  robot_id = str(np.asarray(data["robot"]).flat[0]) if "robot" in data else "g1"
  segment_sources = (
    tuple(str(x) for x in data["segment_source"].tolist())
    if "segment_source" in data
    else ()
  )

  if "segment_names" in data:
    clip_names = [str(x) for x in data["segment_names"].tolist()]
  elif segment_sources:
    clip_names = [clip_name_from_path(source) for source in segment_sources]
  else:
    clip_names = [bundle_path.stem]

  dataset = infer_dataset_name(bundle_path, segment_sources)

  return {
    "schema": MANIFEST_SCHEMA,
    "dataset": dataset,
    "robot": robot_id,
    "bundle": bundle_path.name,
    "bundle_path": str(bundle_path),
    "num_clips": len(clip_names),
    "total_frames": total_frames,
    "fps": fps,
    "clips": clip_names,
  }


def write_motion_library_manifest(path: Path, doc: dict[str, Any]) -> Path:
  path = path.resolve()
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(
    yaml.safe_dump(doc, sort_keys=False, default_flow_style=None),
    encoding="utf-8",
  )
  return path


def export_motion_library_manifest(
  params_dir: Path,
  doc: dict[str, Any],
  *,
  filename: str = PLAY_MOTION_LIBRARY_MANIFEST,
) -> Path:
  """Write ``params/motion_library.yaml`` from an in-memory manifest."""
  return write_motion_library_manifest(params_dir / filename, doc)
