"""Discover source clips and pack or restore the dataset cache."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CACHE_DIR = ".cache"
CACHE_SCHEMA = 1
"""Bump when the packed layout changes (old caches are then rebuilt)."""


# ---------------------------------------------------------------------------------- sources
def dataset_sources_dir(root: Path) -> Path:
  """Folder holding a dataset's source clips: ``raw/`` when it has any, else the root."""
  from wbc_mjlab.motion.motion_formats import _extensions_for_format

  raw = root / "raw"
  exts = _extensions_for_format(None)
  if raw.is_dir() and any(p.suffix.lower() in exts for p in raw.iterdir()):
    return raw
  return root


def list_source_files(root: Path, *, format_name: str | None = None) -> list[Path]:
  """Source clips (CSV / PKL …) of a dataset folder, sorted by name; empty when none."""
  from wbc_mjlab.motion.motion_formats import _extensions_for_format

  root = Path(root)
  if root.is_file():
    return [root]
  src = dataset_sources_dir(root)
  if not src.is_dir():
    return []
  exts = _extensions_for_format(format_name)
  return sorted(
    (p for p in src.iterdir() if p.is_file() and p.suffix.lower() in exts),
    key=lambda p: p.stem,
  )


def has_sources(root: Path | str) -> bool:
  root = Path(root)
  return root.is_dir() and bool(list_source_files(root))


# ---------------------------------------------------------------------------------- packing
@dataclass
class PackedClips:
  """Source clips back to back: ``frames[start[i] : start[i] + length[i]]`` is clip ``i``.

  ``frames`` columns: base position (3), base quaternion (4, wxyz), joint angles in
  ``joint_names`` order. ``fps`` is each clip's source rate.
  """

  names: list[str]
  sources: list[str]
  joint_names: list[str]
  fps: np.ndarray
  start: np.ndarray
  length: np.ndarray
  frames: np.ndarray

  @property
  def num_clips(self) -> int:
    return len(self.names)

  @property
  def num_frames(self) -> int:
    return int(self.frames.shape[0])


def _cache_key(
  files: Sequence[Path],
  joint_names: Sequence[str],
  format_name: str | None,
  input_fps: float | None,
) -> str:
  h = hashlib.sha1()
  h.update(
    json.dumps([CACHE_SCHEMA, list(joint_names), format_name, input_fps]).encode()
  )
  for p in files:
    st = p.stat()
    h.update(f"{p.name}\0{st.st_size}\0{st.st_mtime_ns}\n".encode())
  return h.hexdigest()[:16]


def _load_one(
  path: Path,
  joint_names: Sequence[str],
  format_name: str | None,
  input_fps: float | None,
  line_range: tuple[int, int] | None = None,
) -> tuple[np.ndarray, float]:
  from wbc_mjlab.motion.motion_formats import load_motion_clip
  from wbc_mjlab.motion.robot_assets import remap_dof_columns

  raw = load_motion_clip(
    path, format_name=format_name, input_fps=input_fps, line_range=line_range
  )
  dof = raw.dof_pos
  if raw.source_joint_names is not None:
    missing = [n for n in joint_names if n not in raw.source_joint_names]
    if missing:
      raise ValueError(f"{path.name}: no column for joints {missing}")
    dof = remap_dof_columns(dof, list(raw.source_joint_names), list(joint_names))
  if dof.shape[1] != len(joint_names):
    raise ValueError(
      f"{path.name}: {dof.shape[1]} joint columns, the robot has {len(joint_names)} "
      "(wrong robot for this dataset?)"
    )
  quat = np.asarray(raw.base_rot_xyzw, np.float64)[:, [3, 0, 1, 2]]
  quat /= np.linalg.norm(quat, axis=1, keepdims=True).clip(min=1e-12)
  rows = np.concatenate([raw.base_pos, quat, dof], axis=1).astype(np.float32)
  return rows, float(raw.fps)


def pack_sources(
  files: Sequence[Path],
  joint_names: Sequence[str],
  *,
  format_name: str | None = None,
  input_fps: float | None = None,
  line_range: tuple[int, int] | None = None,
  workers: int | None = None,
) -> PackedClips:
  """Read the source clips into one :class:`PackedClips` (no cache)."""
  files = [Path(p) for p in files]
  if not files:
    raise FileNotFoundError("no source clips to pack")
  workers = workers or min(8, os.cpu_count() or 1)
  with ThreadPoolExecutor(max_workers=workers) as pool:
    loaded = list(
      pool.map(
        lambda p: _load_one(p, joint_names, format_name, input_fps, line_range), files
      )
    )
  length = np.array([rows.shape[0] for rows, _ in loaded], np.int64)
  empty = [f.name for f, n in zip(files, length, strict=True) if n == 0]
  if empty:
    raise ValueError(f"empty clips: {', '.join(empty[:5])}")
  start = np.concatenate([[0], np.cumsum(length)[:-1]]).astype(np.int64)
  return PackedClips(
    names=[f.stem for f in files],
    sources=[str(f) for f in files],
    joint_names=list(joint_names),
    fps=np.array([fps for _, fps in loaded], np.float64),
    start=start,
    length=length,
    frames=np.concatenate([rows for rows, _ in loaded], axis=0),
  )


def _save_packed(path: Path, packed: PackedClips) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
  with open(tmp, "wb") as f:
    np.savez(
      f,
      names=np.array(packed.names),
      sources=np.array(packed.sources),
      joint_names=np.array(packed.joint_names),
      fps=packed.fps,
      start=packed.start,
      length=packed.length,
      frames=packed.frames,
    )
  os.replace(tmp, path)


def _load_packed(path: Path) -> PackedClips:
  with np.load(path) as d:
    return PackedClips(
      names=[str(x) for x in d["names"]],
      sources=[str(x) for x in d["sources"]],
      joint_names=[str(x) for x in d["joint_names"]],
      fps=np.asarray(d["fps"], np.float64),
      start=np.asarray(d["start"], np.int64),
      length=np.asarray(d["length"], np.int64),
      frames=np.asarray(d["frames"], np.float32),
    )


def load_dataset(
  root: Path | str,
  joint_names: Sequence[str],
  *,
  format_name: str | None = None,
  input_fps: float | None = None,
  cache: bool = True,
  verbose: bool = True,
) -> PackedClips:
  """Packed source clips of a dataset folder (or one clip file), through the cache.

  The cache file is ``<dataset>/.cache/motion_<key>.npz``; the key covers every source
  file's name, size and mtime, the joint list and the format options, so editing, adding or
  removing a clip rebuilds it. Older cache files of the dataset are removed.
  """
  import time

  started = time.perf_counter()
  root = Path(root).expanduser().resolve()
  files = list_source_files(root, format_name=format_name)
  if not files:
    raise FileNotFoundError(f"No source clips (CSV / PKL) in {root} or {root / 'raw'}")
  if not cache or root.is_file():
    return pack_sources(
      files, joint_names, format_name=format_name, input_fps=input_fps
    )

  key = _cache_key(files, joint_names, format_name, input_fps)
  cache_dir = root / CACHE_DIR
  path = cache_dir / f"motion_{key}.npz"
  if path.is_file():
    try:
      packed = _load_packed(path)
      if verbose:
        print(
          f"[INFO] Motion library: {packed.num_clips} clips, {packed.num_frames} frames "
          f"(cache {path.name}, {time.perf_counter() - started:.2f}s)"
        )
      return packed
    except (OSError, ValueError, KeyError) as e:
      print(f"[WARN] Motion cache {path} unreadable ({e}); rebuilding")
  if verbose:
    print(
      f"[INFO] Motion library: packing {len(files)} source clips from {files[0].parent}"
    )
  packed = pack_sources(
    files, joint_names, format_name=format_name, input_fps=input_fps
  )
  try:
    _save_packed(path, packed)
    for old in cache_dir.glob("motion_*.npz"):
      if old != path:
        old.unlink(missing_ok=True)
  except OSError as e:  # read-only dataset: run without a cache
    print(f"[WARN] Could not write motion cache {path}: {e}")
  if verbose:
    print(
      f"[INFO] Motion library: source packing ready in {time.perf_counter() - started:.2f}s",
      flush=True,
    )
  return packed
