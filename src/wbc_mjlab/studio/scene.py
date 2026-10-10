"""What the Studio shows, as plain data: N robot instances, N reference ghosts and overlays.

Both frontends (web and viser) render a :class:`SceneState`; services only write it.
Robot instances are body poses ``(N, nbody, 3|4)`` of the Studio robot model (instanced on
the client: one draw call per mesh, whatever ``N``). *Ghosts* are a second set of instances
of the same model drawn translucent: the motion reference a policy tracks. Updates replace
whole arrays, so a reader in another thread always sees a consistent frame.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import mujoco
import numpy as np


def _empty_poses() -> tuple[np.ndarray, np.ndarray]:
  return np.zeros((0, 0, 3), np.float32), np.zeros((0, 0, 4), np.float32)


@dataclass
class Overlay:
  """``points`` (M, 3) · ``segments`` (K, 2, 3) · ``mesh`` (V, 3) + ``faces`` (F, 3)."""

  kind: str
  data: np.ndarray
  color: np.ndarray  # (3,) or per point / segment (M, 3), uint8
  size: float = 0.01  # point size or line thickness [m]
  opacity: float = 1.0
  faces: np.ndarray | None = None
  on_top: bool = False  # drawn through the robot (x-ray markers)


@dataclass
class SceneState:
  body_pos: np.ndarray = field(default_factory=lambda: _empty_poses()[0])
  body_quat: np.ndarray = field(default_factory=lambda: _empty_poses()[1])
  ghost_pos: np.ndarray = field(default_factory=lambda: _empty_poses()[0])
  ghost_quat: np.ndarray = field(default_factory=lambda: _empty_poses()[1])
  labels: list[str] = field(default_factory=list)  # per instance (clip names in grids)
  overlays: dict[str, Overlay] = field(default_factory=dict)
  focus: int = 0  # instance the camera follows / stats refer to
  hud: dict = field(
    default_factory=dict
  )  # playback readout of the owner (frame, time, stats)
  model_version: int = 0
  pose_version: int = 0
  overlay_version: int = 0
  terrain: dict | None = None  # {verts, faces, colors}: terrain under the robots (Play)
  terrain_version: int = 0
  lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

  @property
  def num_instances(self) -> int:
    return int(self.body_pos.shape[0])

  def set_poses(
    self,
    pos: np.ndarray,
    quat: np.ndarray,
    ghost: tuple[np.ndarray, np.ndarray] | None = None,
    labels: list[str] | None = None,
  ) -> None:
    with self.lock:
      self.body_pos, self.body_quat = pos, quat
      self.ghost_pos, self.ghost_quat = ghost if ghost is not None else _empty_poses()
      if labels is not None:
        self.labels = labels
      elif len(self.labels) != pos.shape[0]:
        self.labels = []
      self.focus = min(self.focus, max(0, pos.shape[0] - 1))
      self.pose_version += 1

  def set_terrain(self, mesh: dict | None) -> None:
    with self.lock:
      if mesh is None and self.terrain is None:
        return
      self.terrain = mesh
      self.terrain_version += 1

  def set_hud(self, hud: dict) -> None:
    with self.lock:
      self.hud = hud
      self.pose_version += 1

  def set_overlay(self, key: str, overlay: Overlay) -> None:
    with self.lock:
      self.overlays[key] = overlay
      self.overlay_version += 1

  def remove_overlays(self, *keys: str, prefix: str | None = None) -> None:
    with self.lock:
      gone = [
        k
        for k in self.overlays
        if k in keys or (prefix is not None and k.startswith(prefix))
      ]
      for k in gone:
        del self.overlays[k]
      if gone:
        self.overlay_version += 1

  def snapshot(self) -> dict:
    with self.lock:
      return {
        "pos": self.body_pos,
        "quat": self.body_quat,
        "ghost_pos": self.ghost_pos,
        "ghost_quat": self.ghost_quat,
        "labels": list(self.labels),
        "overlays": dict(self.overlays),
        "focus": self.focus,
        "hud": dict(self.hud),
      }


class Kinematics:
  """Batched forward kinematics of the Studio robot model on the CPU (qpos → body poses)."""

  def __init__(self, model: mujoco.MjModel):
    self.model = model
    self.data = mujoco.MjData(model)

  def __call__(self, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    qpos = np.atleast_2d(np.asarray(qpos, np.float64))
    n, nb = qpos.shape[0], self.model.nbody
    pos = np.empty((n, nb, 3), np.float32)
    quat = np.empty((n, nb, 4), np.float32)
    for i in range(n):
      self.data.qpos[:] = qpos[i]
      mujoco.mj_kinematics(self.model, self.data)
      pos[i] = self.data.xpos
      quat[i] = self.data.xquat
    return pos, quat


def error_colors(values: np.ndarray, vmax: float) -> np.ndarray:
  """Tracking error → uint8 RGB: brand green (0) → amber → danger red (``vmax``)."""
  t = np.clip(np.asarray(values, np.float32) / max(vmax, 1e-9), 0.0, 1.0)[:, None]
  green = np.array([79, 199, 138], np.float32)
  amber = np.array([230, 162, 60], np.float32)
  red = np.array([239, 106, 106], np.float32)
  lo = green + (amber - green) * np.clip(2 * t, 0, 1)
  return np.where(t < 0.5, lo, amber + (red - amber) * np.clip(2 * t - 1, 0, 1)).astype(
    np.uint8
  )


def grid_offsets(n: int, spacing: float = 1.6) -> np.ndarray:
  """XY offsets ``(n, 3)`` on rows of ⌈√n⌉, instance 0 front-left, centred on the origin."""
  cols = max(1, int(np.ceil(np.sqrt(n))))
  i = np.arange(n)
  xy = np.stack([(i % cols) * spacing, -(i // cols) * spacing], -1).astype(np.float32)
  if n:
    xy -= xy.mean(0)
  return np.concatenate([xy, np.zeros((n, 1), np.float32)], -1)


__all__ = ["Kinematics", "Overlay", "SceneState", "error_colors", "grid_offsets"]
