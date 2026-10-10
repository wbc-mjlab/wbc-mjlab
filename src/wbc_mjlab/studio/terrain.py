"""Terrain for the Studio's 3D view: a height map sampled once from a play env's (static)
terrain geoms, and patches of it around the robots as one mesh.

Procedural terrain (``-Terrain-Scan`` tasks) is boxes and height fields welded to the world.
They're rasterised into a grid (box tops, bilinear height-field samples; elsewhere 0), so a
patch around any robot is a slice — no ray casting per frame.
"""

from __future__ import annotations

import mujoco
import numpy as np

LOW = np.array((118, 132, 144), np.float32)  # flat: grey-blue (sRGB)
HIGH = np.array((120, 210, 224), np.float32)  # 0.12 m up: towards the accent teal


class TerrainMap:
  def __init__(self, heights: np.ndarray, origin: tuple[float, float], res: float):
    self.h = heights  # (ny, nx), row = y
    self.x0, self.y0 = origin
    self.res = res

  @classmethod
  def from_model(cls, m: mujoco.MjModel, res: float = 0.1) -> TerrainMap | None:
    """Rasterise the world-welded boxes and height fields of ``m`` (None if it has none)."""
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    world = [
      g
      for g in range(m.ngeom)
      if m.body_weldid[m.geom_bodyid[g]] == 0
      and m.geom_type[g] in (mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_HFIELD)
    ]
    if not world:
      return None
    boxes, fields = [], []
    lo, hi = np.full(2, np.inf), np.full(2, -np.inf)
    for g in world:
      pos, rot = d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3)
      if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
        half = np.abs(rot) @ m.geom_size[g]
        top = pos[2] + half[2]
        boxes.append(
          (pos[0] - half[0], pos[0] + half[0], pos[1] - half[1], pos[1] + half[1], top)
        )
        ext = half[:2]
      else:
        hid = int(m.geom_dataid[g])
        rx, ry, ez, _ = m.hfield_size[hid]
        nr, nc = int(m.hfield_nrow[hid]), int(m.hfield_ncol[hid])
        adr = int(m.hfield_adr[hid])
        data = m.hfield_data[adr : adr + nr * nc].reshape(nr, nc) * ez + pos[2]
        fields.append((pos[0], pos[1], rx, ry, data))
        ext = np.array((rx, ry))
      lo = np.minimum(lo, pos[:2] - ext)
      hi = np.maximum(hi, pos[:2] + ext)
    nx = int(np.ceil((hi[0] - lo[0]) / res)) + 1
    ny = int(np.ceil((hi[1] - lo[1]) / res)) + 1
    if nx * ny > 40_000_000:  # absurd extents: sample coarser
      return cls.from_model(m, res * 2)
    h = np.full((ny, nx), -np.inf, np.float32)
    xs = lo[0] + np.arange(nx) * res
    ys = lo[1] + np.arange(ny) * res
    # large boxes first (borders, platforms), small ones on top
    for x0, x1, y0, y1, top in sorted(
      boxes, key=lambda b: -(b[1] - b[0]) * (b[3] - b[2])
    ):
      i0, i1 = np.searchsorted(xs, [x0, x1])
      j0, j1 = np.searchsorted(ys, [y0, y1])
      cell = h[j0:j1, i0:i1]
      np.maximum(cell, top, out=cell)
    for cx, cy, rx, ry, data in fields:
      nr, nc = data.shape
      i0, i1 = np.searchsorted(xs, [cx - rx, cx + rx])
      j0, j1 = np.searchsorted(ys, [cy - ry, cy + ry])
      if i1 <= i0 or j1 <= j0:
        continue
      u = (xs[i0:i1] - (cx - rx)) / (2 * rx) * (nc - 1)
      v = (ys[j0:j1] - (cy - ry)) / (2 * ry) * (nr - 1)
      c0 = np.clip(np.floor(u).astype(int), 0, nc - 2)
      r0 = np.clip(np.floor(v).astype(int), 0, nr - 2)
      fu, fv = (u - c0)[None, :], (v - r0)[:, None]
      a = data[r0][:, c0] * (1 - fu) + data[r0][:, c0 + 1] * fu
      b = data[r0 + 1][:, c0] * (1 - fu) + data[r0 + 1][:, c0 + 1] * fu
      h[j0:j1, i0:i1] = np.maximum(h[j0:j1, i0:i1], a * (1 - fv) + b * fv)
    h[~np.isfinite(h)] = 0.0
    return cls(h, (float(lo[0]), float(lo[1])), res)

  def patch(
    self, center: np.ndarray, half: float, res: float
  ) -> tuple[np.ndarray, np.ndarray]:
    """Grid ``(k, k, 3)`` world points around ``center`` (xy) with terrain heights."""
    k = int(round(2 * half / res)) + 1
    g = center[None, :2] + (np.arange(k) * res - half)[:, None]
    gx, gy = np.meshgrid(g[:, 0], g[:, 1])
    i = np.clip(np.rint((gx - self.x0) / self.res).astype(int), 0, self.h.shape[1] - 1)
    j = np.clip(np.rint((gy - self.y0) / self.res).astype(int), 0, self.h.shape[0] - 1)
    return np.stack([gx, gy, self.h[j, i]], axis=-1), np.array((k, k))

  def mesh(
    self, centers: np.ndarray, shifts: np.ndarray, half: float = 2.5, res: float = 0.1
  ) -> dict:
    """One mesh of patches around ``centers`` (world xy), each moved by its ``shifts``
    (display offset): ``{verts (V,3) f32, faces (F,3) u32, colors (V,3) u8}``."""
    verts, faces, colors = [], [], []
    base = 0
    for c, s in zip(centers, shifts, strict=True):
      pts, (k, _) = self.patch(np.asarray(c, np.float64), half, res)
      t = np.clip(pts[..., 2:3] / 0.12, 0.0, 1.0)  # colour by height: 0 → 0.12 m
      col = (
        (LOW + (HIGH - LOW) * t) / 255.0
      ) ** 2.2 * 255.0  # sRGB → linear (three.js)
      verts.append((pts + np.asarray(s)).reshape(-1, 3))
      colors.append(col.reshape(-1, 3))
      r, q = np.meshgrid(np.arange(k - 1), np.arange(k - 1), indexing="ij")
      a = (r * k + q).ravel() + base
      faces.append(
        np.stack([a, a + 1, a + k, a + 1, a + k + 1, a + k], axis=1).reshape(-1, 3)
      )
      base += k * k
    return {
      "verts": np.concatenate(verts).astype(np.float32),
      "faces": np.concatenate(faces).astype(np.uint32),
      "colors": np.concatenate(colors).astype(np.uint8),
    }


__all__ = ["TerrainMap"]
