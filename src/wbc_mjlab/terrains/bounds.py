"""Vertical bounds without model compilation or simulation."""

from pathlib import Path

import mujoco
import numpy as np


def rotation_matrix(quat) -> np.ndarray:
  quat = np.asarray(quat, dtype=float)
  quat = quat / np.linalg.norm(quat)
  matrix = np.empty(9)
  mujoco.mju_quat2Mat(matrix, quat)
  return matrix.reshape(3, 3)


def geometry_min_z(geom, spec: mujoco.MjSpec) -> float:
  """Bound a geom whose pose has been flattened into world coordinates."""
  kind = geom.type
  row = rotation_matrix(geom.quat)[2]
  size = np.asarray(geom.size)
  z = float(geom.pos[2])
  if kind == mujoco.mjtGeom.mjGEOM_MESH:
    mesh = spec.mesh(geom.meshname)
    vertices = np.asarray(mesh.uservert, dtype=float).reshape(-1, 3)
    if not len(vertices):
      import trimesh

      vertices = np.asarray(
        trimesh.load_mesh(mesh.file, process=False, skip_materials=True).vertices,
        dtype=float,
      )
    if not len(vertices) or not np.isfinite(vertices).all():
      raise ValueError(f"Invalid mesh vertices: {mesh.name}")
    # Match MuJoCo's refpos subtraction, inverse refquat rotation, then scale.
    vertices = ((vertices - mesh.refpos) @ rotation_matrix(mesh.refquat)) * mesh.scale
    return z + float(np.min(vertices @ row))
  if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
    extent = size[0]
  elif kind == mujoco.mjtGeom.mjGEOM_BOX:
    extent = np.abs(row) @ size
  elif kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
    extent = np.linalg.norm(row * size)
  elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
    extent = size[0] + abs(row[2]) * size[1]
  elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
    extent = size[0] * np.linalg.norm(row[:2]) + abs(row[2]) * size[1]
  elif kind == mujoco.mjtGeom.mjGEOM_HFIELD:
    field = spec.hfield(geom.hfieldname)
    sx, sy, height, base = field.size
    z += row[2] * (height - base) / 2
    extent = np.abs(row) @ np.array([sx, sy, (height + base) / 2])
  else:
    raise ValueError(f"Cannot compute terrain bounds for geom type {kind}")
  return z - float(extent)


def trajectory_min_z(motion_path: str) -> float:
  """Minimum recorded root height, without converting or changing the clip."""
  if not motion_path:
    return float("inf")
  path = Path(motion_path)
  if path.suffix.lower() == ".csv":
    heights = np.loadtxt(path, delimiter=",", usecols=(2,), ndmin=1)
  else:
    from wbc_mjlab.motion.motion_formats import load_motion_clip

    heights = load_motion_clip(path).base_pos[:, 2]
  if not len(heights) or not np.isfinite(heights).all():
    raise ValueError(f"Invalid trajectory heights: {motion_path}")
  return float(np.min(heights))
