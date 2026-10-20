"""Motion library: raw clips (CSV / PKL) → reference kinematics on the GPU, online.

Nothing is converted ahead of time. A dataset's source clips are packed once into a single
array of generalized coordinates (base position, base quaternion, joint angles at the source
rate) and cached next to the dataset (``.cache/motion_<key>.npz``, rebuilt when a source
file, the format options or the robot's joint list change). Everything a converted NPZ used
to hold is computed when it is asked for:

* **Resampling** — any time ``t`` is interpolated from the two neighbouring source frames
  (lerp for positions and joints, slerp for the base rotation), so the reference follows the
  control rate of the env instead of a rate fixed at conversion.
* **Velocities** — central differences of the interpolated coordinates at ``t ± dt``
  (one-sided at the clip ends), as the converter's ``torch.gradient`` did.
* **Mirroring** — left/right mirrored clips are virtual segments: the coordinates are
  mirrored (joint swap + sign, base reflected across the XZ plane) before FK, so nothing is
  stored twice. Clips whose name already ends with the mirror suffix are not mirrored again.
* **Vertical debias** — a per-clip height shift (lowest foot origin at the sole height),
  found with one batched FK pass over the clip when the library is built.
* **Body kinematics** — GPU forward kinematics (``mujoco_warp``: kinematics, com_pos,
  com_vel; one CUDA graph) on a robot model with one world per env, read for the bodies the
  caller asks for.
* **Anchor accelerations** — differences of the anchor body velocity over the source frames,
  computed with the same FK pass at build time and interpolated like the coordinates.

:class:`OnlineMotion` is what the training ``MotionCommand`` uses: global frame indices on a
``step_dt`` grid (segments back to back, as a stacked NPZ bundle had them), resolved to clip
and time. :func:`bake_clip` produces the arrays of a converted clip NPZ for the tools that
still want one (exports, ``wbc-mjlab-data-to-npz``).
"""

from __future__ import annotations

import dataclasses
import gc
import hashlib
import json
import os
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import mujoco
import numpy as np
import torch

CACHE_DIR = ".cache"
CACHE_SCHEMA = 1
"""Bump when the packed layout changes (old caches are then rebuilt)."""

Storage = Literal["device", "host"]


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
          f"(cache {path.name})"
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
  return packed


# ---------------------------------------------------------------------------------- math
def _quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
  from mjlab.utils.lab_api.math import quat_mul

  return quat_mul(a, b)


def _slerp(q0: torch.Tensor, q1: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
  """Shortest-arc slerp of ``(..., 4)`` quaternions. The angle comes from
  ``2·atan2(|q0 - q1|, |q0 + q1|)``, accurate in float32 for the small rotations between
  neighbouring frames (``acos`` of the dot product is not, and velocities difference them)."""
  q1 = torch.where((q0 * q1).sum(-1, keepdim=True) < 0.0, -q1, q1)
  angle = 2.0 * torch.atan2(
    (q0 - q1).norm(dim=-1, keepdim=True), (q0 + q1).norm(dim=-1, keepdim=True)
  )
  sin = torch.sin(angle)
  small = sin < 1e-6
  safe = torch.where(small, 1.0, sin)
  w0 = torch.where(small, 1.0 - tau, torch.sin((1.0 - tau) * angle) / safe)
  w1 = torch.where(small, tau, torch.sin(tau * angle) / safe)
  out = w0 * q0 + w1 * q1
  return out / out.norm(dim=-1, keepdim=True)


_POS_MIRROR = (1.0, -1.0, 1.0)  # polar vectors: y → -y
_ROT_MIRROR = (-1.0, 1.0, -1.0)  # axial vectors (angular velocity / acceleration)
_QUAT_MIRROR = (1.0, -1.0, 1.0, -1.0)  # wxyz: R' = S R S with S = diag(1, -1, 1)


# ---------------------------------------------------------------------------------- FK
class RobotKinematics:
  """Batched forward kinematics of one robot in a MuJoCo model, on the GPU (or CPU).

  ``fk(qpos, qvel)`` takes robot coordinates (free joint + ``joint_names`` in that order)
  for any number of rows and returns world pose and velocity of ``body_names``; rows are
  run ``nworld`` at a time through ``kinematics → com_pos → com_vel`` (a CUDA graph on GPU).
  """

  def __init__(
    self,
    mj_model: mujoco.MjModel,
    *,
    joint_names: Sequence[str],
    body_names: Sequence[str],
    root_body: str,
    nworld: int,
    device: str,
    entity_name: str = "robot",
  ):
    import mujoco_warp as mjw
    import warp as wp

    self.device = torch.device(device)
    self.nworld = max(1, int(nworld))
    self._wp_device = wp.get_device(str(device))

    def body_id(name: str) -> int:
      i = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, f"{entity_name}/{name}")
      if i < 0:
        i = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
      if i < 0:
        raise KeyError(f"body {name!r} not in the model")
      return i

    def joint_qadr(name: str) -> int:
      i = mujoco.mj_name2id(
        mj_model, mujoco.mjtObj.mjOBJ_JOINT, f"{entity_name}/{name}"
      )
      if i < 0:
        i = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
      if i < 0:
        raise KeyError(f"joint {name!r} not in the model")
      return int(mj_model.jnt_qposadr[i]), int(mj_model.jnt_dofadr[i])

    root = body_id(root_body)
    jnt = int(mj_model.body_jntadr[root])
    if jnt < 0 or mj_model.jnt_type[jnt] != mujoco.mjtJoint.mjJNT_FREE:
      raise ValueError(f"root body {root_body!r} has no free joint")
    self._root_body = root
    self._free_q = int(mj_model.jnt_qposadr[jnt])
    self._free_v = int(mj_model.jnt_dofadr[jnt])
    adr = [joint_qadr(n) for n in joint_names]
    self._joint_q = torch.tensor([a for a, _ in adr], device=self.device)
    self._joint_v = torch.tensor([v for _, v in adr], device=self.device)
    self._body_ids = torch.tensor([body_id(n) for n in body_names], device=self.device)

    mjd = mujoco.MjData(mj_model)
    mujoco.mj_kinematics(mj_model, mjd)  # no collision: the FK data holds no contacts
    with wp.ScopedDevice(self._wp_device):
      self._m = mjw.put_model(mj_model)
    # A full-width batch (one world per env) and a narrow one for the few rows of a reset.
    sizes = sorted({self.nworld, min(self.nworld, SMALL_BATCH)})
    self._batches = [
      _FkBatch(self._m, mj_model, mjd, n, self._wp_device) for n in sizes
    ]

  def fk(
    self, qpos: torch.Tensor, qvel: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``qpos`` (M, 7+J), ``qvel`` (M, 6+J) → body pos, quat (wxyz), lin vel, ang vel (world)."""
    m = int(qpos.shape[0])
    nb = int(self._body_ids.numel())
    out = [
      torch.empty(m, nb, k, device=self.device, dtype=torch.float32)
      for k in (3, 4, 3, 3)
    ]
    batch = self.batch_for(m)
    for a in range(0, m, batch.nworld):
      b = min(m, a + batch.nworld)
      self.write(batch, qpos[a:b], qvel[a:b])
      batch.launch(self.device)
      for o, x in zip(out, self.read(batch, b - a), strict=True):
        o[a:b] = x
    return out[0], out[1], out[2], out[3]

  def batch_for(self, m: int) -> _FkBatch:
    """The narrowest FK batch holding ``m`` rows (the widest one when none does)."""
    return next((x for x in self._batches if x.nworld >= m), self._batches[-1])

  def write(self, batch: _FkBatch, qpos: torch.Tensor, qvel: torch.Tensor) -> None:
    n, fq, fv = int(qpos.shape[0]), self._free_q, self._free_v
    batch.qpos[:n, fq : fq + 7] = qpos[:, :7]
    batch.qpos[:n, self._joint_q] = qpos[:, 7:]
    batch.qvel[:n, fv : fv + 6] = qvel[:, :6]
    batch.qvel[:n, self._joint_v] = qvel[:, 6:]

  def read(
    self, batch: _FkBatch, n: int
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from mjlab.entity.data import compute_velocity_from_cvel

    pos = batch.xpos[:n, self._body_ids]
    com = batch.subtree_com[:n, self._root_body].unsqueeze(1)
    vel = compute_velocity_from_cvel(pos, com, batch.cvel[:n, self._body_ids])
    return pos, batch.xquat[:n, self._body_ids], vel[..., :3], vel[..., 3:]


SMALL_BATCH = 256
"""Worlds of the narrow FK batch (reset rows); wider requests use the full one."""


class _FkBatch:
  """One ``mujoco_warp`` data of ``nworld`` worlds and its captured FK graph."""

  def __init__(
    self, m, mj_model: mujoco.MjModel, mjd: mujoco.MjData, nworld: int, device
  ):
    import mujoco_warp as mjw
    import warp as wp

    self.nworld = nworld
    self._wp_device = device
    with wp.ScopedDevice(device):
      d = mjw.put_data(mj_model, mjd, nworld=nworld, nconmax=1, njmax=1)
    self.qpos, self.qvel = wp.to_torch(d.qpos), wp.to_torch(d.qvel)
    self.xpos, self.xquat = wp.to_torch(d.xpos), wp.to_torch(d.xquat)
    self.cvel, self.subtree_com = wp.to_torch(d.cvel), wp.to_torch(d.subtree_com)

    def kernels() -> None:
      mjw.kinematics(m, d)
      mjw.com_pos(m, d)
      mjw.com_vel(m, d)

    self._data = d
    self._kernels = kernels
    self._graph = None
    with wp.ScopedDevice(device):
      kernels()  # compile / warm up before capture
      if device.is_cuda and wp.is_mempool_enabled(device):
        enabled = gc.isenabled()
        gc.disable()
        try:
          with wp.ScopedCapture() as cap:
            kernels()
          self._graph = cap.graph
        finally:
          if enabled:
            gc.enable()

  def launch(self, torch_device: torch.device) -> None:
    import warp as wp

    with wp.ScopedDevice(self._wp_device):
      if not self._wp_device.is_cuda:
        self._kernels()
        return
      # On the torch stream: ordered after the qpos writes, before the reads.
      stream = wp.stream_from_torch(torch.cuda.current_stream(torch_device))
      if self._graph is not None:
        wp.capture_launch(self._graph, stream=stream)
      else:
        with wp.ScopedStream(stream):
          self._kernels()


# ---------------------------------------------------------------------------------- frames
@dataclass
class MotionFrame:
  """Reference state at a batch of frames (tracked bodies only)."""

  joint_pos: torch.Tensor  # (M, J)
  joint_vel: torch.Tensor  # (M, J)
  body_pos_w: torch.Tensor  # (M, B, 3)
  body_quat_w: torch.Tensor  # (M, B, 4) wxyz
  body_lin_vel_w: torch.Tensor  # (M, B, 3)
  body_ang_vel_w: torch.Tensor  # (M, B, 3)
  anchor_lin_acc_w: torch.Tensor  # (M, 3)
  anchor_ang_acc_w: torch.Tensor  # (M, 3)


@dataclass(frozen=True)
class MirrorSpec:
  joint_index: tuple[int, ...]
  joint_scale: tuple[float, ...]
  suffix: str


def mirror_spec_for(robot_id: str, joint_names: Sequence[str]) -> MirrorSpec:
  from wbc_mjlab.motion.motion_mirror import build_joint_mirror_map
  from wbc_mjlab.robots.symmetry import get_robot_symmetry_config

  sym = get_robot_symmetry_config(robot_id)
  if sym is None:
    raise ValueError(
      f"mirror needs a left/right symmetry config for robot {robot_id!r} "
      "(robots/<id>/symmetry.py, or symmetry_config on an extension's WbcRobotSpec)"
    )
  idx, scale = build_joint_mirror_map(list(joint_names), sym)
  return MirrorSpec(
    tuple(int(i) for i in idx), tuple(float(s) for s in scale), sym.mirror_suffix
  )


class MotionLibrary:
  """Packed clips on a device (or pinned host memory), sampled at any time.

  Segments are the clips plus, with ``mirror``, a mirrored copy of each (sorted by name, as a
  folder of converted clips listed). ``qpos_at(seg, t)`` interpolates, mirrors and applies
  the clip's height shift.
  """

  def __init__(
    self,
    packed: PackedClips,
    *,
    device: str,
    storage: Storage = "device",
    mirror: MirrorSpec | None = None,
  ):
    self.packed = packed
    self.device = torch.device(device)
    self.storage = storage
    frames = torch.from_numpy(packed.frames)
    if storage == "host":
      self._frames = frames.pin_memory() if self.device.type == "cuda" else frames
    else:
      self._frames = frames.to(self.device)
    self.clip_start = torch.as_tensor(packed.start, device=self.device)
    self.clip_length = torch.as_tensor(packed.length, device=self.device)
    self.clip_fps = torch.as_tensor(packed.fps, dtype=torch.float64, device=self.device)
    self.z_shift = torch.zeros(packed.num_clips, device=self.device)
    self.acc_table: torch.Tensor | None = (
      None  # (F, 6) anchor lin / ang acc, source rate
    )

    segs = [(name, i, False) for i, name in enumerate(packed.names)]
    self.mirror = mirror
    if mirror is not None:
      segs += [
        (f"{name}{mirror.suffix}", i, True)
        for i, name in enumerate(packed.names)
        if not name.endswith(mirror.suffix)
      ]
      self._mirror_index = torch.tensor(mirror.joint_index, device=self.device)
      self._mirror_scale = torch.tensor(mirror.joint_scale, device=self.device)
    segs.sort(key=lambda s: s[0])
    self.segment_names = tuple(s[0] for s in segs)
    self.segment_clip = torch.tensor([s[1] for s in segs], device=self.device)
    self.segment_mirrored = torch.tensor([s[2] for s in segs], device=self.device)
    self._pos_m = torch.tensor(_POS_MIRROR, device=self.device)
    self._rot_m = torch.tensor(_ROT_MIRROR, device=self.device)
    self._quat_m = torch.tensor(_QUAT_MIRROR, device=self.device)

  @property
  def num_segments(self) -> int:
    return len(self.segment_names)

  def segment_duration(self) -> torch.Tensor:
    """Time of each segment's last source frame [s]."""
    clip = self.segment_clip
    return ((self.clip_length[clip] - 1).double() / self.clip_fps[clip]).float()

  def _rows(self, index: torch.Tensor) -> torch.Tensor:
    if self.storage == "host":
      return self._frames[index.cpu()].to(self.device, non_blocking=True)
    return self._frames[index]

  def _table(self, table: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    if table.device != self.device:
      return table[index.cpu()].to(self.device, non_blocking=True)
    return table[index]

  def _interp_index(
    self, clip: torch.Tensor, t: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = (t.double() * self.clip_fps[clip]).clamp(min=0.0)
    last = self.clip_length[clip] - 1
    x = torch.minimum(x, last.double())
    i0 = torch.minimum(x.floor().long(), last)
    i1 = torch.minimum(i0 + 1, last)
    base = self.clip_start[clip]
    return base + i0, base + i1, (x - i0.double()).float().unsqueeze(-1)

  def _coords(
    self,
    clip: torch.Tensor,
    t: torch.Tensor,
    mirrored: torch.Tensor | None,
    shift: bool,
  ) -> torch.Tensor:
    g0, g1, b = self._interp_index(clip, t)
    r0, r1 = self._rows(g0), self._rows(g1)
    pos = torch.lerp(r0[:, :3], r1[:, :3], b)
    quat = _slerp(r0[:, 3:7], r1[:, 3:7], b)
    dof = torch.lerp(r0[:, 7:], r1[:, 7:], b)
    if shift:
      pos[:, 2] -= self.z_shift[clip]
    if mirrored is not None:
      m = mirrored.unsqueeze(-1)
      pos = torch.where(m, pos * self._pos_m, pos)
      quat = torch.where(m, quat * self._quat_m, quat)
      dof = torch.where(m, dof[:, self._mirror_index] * self._mirror_scale, dof)
    return torch.cat([pos, quat, dof], dim=-1)

  def _mirrored(self, segment: torch.Tensor) -> torch.Tensor | None:
    return self.segment_mirrored[segment] if self.mirror is not None else None

  def qpos_at(self, segment: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Robot coordinates ``(M, 7+J)`` of ``segment`` at time ``t`` [s]."""
    return self._coords(self.segment_clip[segment], t, self._mirrored(segment), True)

  def _state(
    self,
    clip: torch.Tensor,
    t: torch.Tensor,
    h: torch.Tensor | float,
    mirrored: torch.Tensor | None,
    shift: bool,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    from mjlab.utils.lab_api.math import (
      axis_angle_from_quat,
      quat_apply_inverse,
      quat_conjugate,
    )

    end = (self.clip_length[clip] - 1).double() / self.clip_fps[clip]
    t = torch.minimum(t.double().clamp(min=0.0), end)
    t0 = (t - h).clamp(min=0.0)
    t1 = torch.minimum(t + h, end)
    m = int(t.shape[0])
    rep = None if mirrored is None else mirrored.repeat(3)
    q = self._coords(clip.repeat(3), torch.cat([t0, t, t1]), rep, shift)
    q0, q, q1 = q[:m], q[m : 2 * m], q[2 * m :]
    span = (t1 - t0).float().unsqueeze(-1)
    moving = span > 1e-9
    span = span.clamp(min=1e-9)
    lin = torch.where(moving, (q1[:, :3] - q0[:, :3]) / span, 0.0)
    dof = torch.where(moving, (q1[:, 7:] - q0[:, 7:]) / span, 0.0)
    ang_w = axis_angle_from_quat(_quat_mul(q1[:, 3:7], quat_conjugate(q0[:, 3:7])))
    ang_w = torch.where(moving, ang_w / span, 0.0)
    qvel = torch.cat([lin, quat_apply_inverse(q[:, 3:7], ang_w), dof], dim=-1)
    return q, qvel

  def state_at(
    self, segment: torch.Tensor, t: torch.Tensor, h: float
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """``qpos`` at ``t`` and ``qvel`` by central differences over ``t ± h`` (clamped to the
    segment: one-sided at its ends). Free-joint angular velocity is in the base frame."""
    return self._state(self.segment_clip[segment], t, h, self._mirrored(segment), True)

  def anchor_acc_at(self, segment: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Anchor ``(lin, ang)`` acceleration ``(M, 6)`` from the build-time table."""
    if self.acc_table is None:
      return torch.zeros(int(t.shape[0]), 6, device=self.device)
    clip = self.segment_clip[segment]
    g0, g1, b = self._interp_index(clip, t)
    acc = torch.lerp(
      self._table(self.acc_table, g0), self._table(self.acc_table, g1), b
    )
    if self.mirror is not None:
      m = self.segment_mirrored[segment].unsqueeze(-1)
      acc = torch.where(m, acc * torch.cat([self._pos_m, self._rot_m]), acc)
    return acc

  def source_pass(
    self,
    kin: RobotKinematics,
    *,
    anchor: int | None,
    feet: Sequence[int] = (),
    foot_sole_z: float | None = None,
  ) -> None:
    """One FK pass over every source frame (velocities from the neighbouring source frames):
    the anchor acceleration table (``anchor``: index into ``kin``'s bodies) and, with
    ``feet`` and ``foot_sole_z``, each clip's height shift (lowest foot origin → sole height).
    """
    p = self.packed
    n = p.num_frames
    idx = torch.arange(n, device=self.device)
    clip_of = torch.repeat_interleave(
      torch.arange(p.num_clips, device=self.device), self.clip_length
    )
    local = idx - self.clip_start[clip_of]
    fps = self.clip_fps[clip_of]
    vel = torch.zeros(n, 6, device=self.device)
    foot_min = torch.full((p.num_clips,), float("inf"), device=self.device)
    for a in range(0, n, kin.nworld):
      b = min(n, a + kin.nworld)
      clip = clip_of[a:b]
      q, qv = self._state(
        clip, local[a:b].double() / fps[a:b], 1.0 / fps[a:b], None, False
      )
      pos, _, lin, ang = kin.fk(q, qv)
      if anchor is not None:
        vel[a:b] = torch.cat([lin[:, anchor], ang[:, anchor]], dim=-1)
      if feet:
        z = pos[:, list(feet), 2].amin(dim=1)
        foot_min.scatter_reduce_(0, clip, z, reduce="amin")
    if feet and foot_sole_z is not None:
      self.z_shift = foot_min - float(foot_sole_z)
    if anchor is not None:
      last = self.clip_length[clip_of] - 1
      prev = idx - (local > 0).long()
      nxt = idx + (local < last).long()
      steps = (nxt - prev).clamp(min=1).float().unsqueeze(-1)
      acc = (vel[nxt] - vel[prev]) * fps.float().unsqueeze(-1) / steps
      if self.storage == "host":
        acc = acc.cpu()
        self.acc_table = acc.pin_memory() if self.device.type == "cuda" else acc
      else:
        self.acc_table = acc


# ---------------------------------------------------------------------------------- training
def _grid_lengths(lib: MotionLibrary, step_dt: float) -> torch.Tensor:
  """Frames of each segment on a ``step_dt`` grid (at least one): the converter's
  ``len(arange(0, (n - 1) / fps, step_dt))``, so a clip keeps its frame count."""
  p = lib.packed
  per_clip = [
    max(1, torch.arange(0, (int(n) - 1) / float(f), step_dt).numel())
    for n, f in zip(p.length, p.fps, strict=True)
  ]
  counts = torch.tensor(per_clip, dtype=torch.long, device=lib.device)
  return counts[lib.segment_clip]


class OnlineMotion:
  """Motion source of the training command, computed from the raw clips each step.

  Exposes the layout of a stacked NPZ bundle — segments back to back on a ``step_dt`` grid
  (``segment_start_idx`` / ``segment_length`` / ``segment_end_idx`` / ``segment_names``) —
  and :meth:`frame` returns the reference at any batch of global frame indices.
  """

  def __init__(
    self,
    source: Path | str,
    *,
    mj_model: mujoco.MjModel,
    joint_names: Sequence[str],
    body_names: Sequence[str],
    anchor_body: str,
    root_body: str,
    step_dt: float,
    nworld: int,
    device: str,
    robot_id: str,
    entity_name: str = "robot",
    mirror: bool = False,
    debias_z: bool = False,
    debias_foot_sole_z: float | None = None,
    storage: Storage = "device",
    format_name: str | None = None,
    input_fps: float | None = None,
    cache: bool = True,
  ):
    from wbc_mjlab.motion.robot_assets import (
      get_robot_motion_spec,
      qpos_dof_joint_names,
    )

    self.motion_source = str(Path(source).expanduser().resolve())
    self.step_dt = float(step_dt)
    self.device = torch.device(device)
    qpos_names = qpos_dof_joint_names(mj_model, entity_name=entity_name)
    missing = [n for n in joint_names if n not in qpos_names]
    if missing:
      raise ValueError(f"joints {missing} are not in the model")
    packed = load_dataset(
      source, qpos_names, format_name=format_name, input_fps=input_fps, cache=cache
    )
    spec = mirror_spec_for(robot_id, qpos_names) if mirror else None
    self.library = MotionLibrary(packed, device=device, storage=storage, mirror=spec)

    feet: list[str] = []
    sole_z = None
    if debias_z:
      _, mspec = get_robot_motion_spec(robot_id)
      if not mspec.foot_body_names or mspec.foot_sole_z is None:
        raise ValueError(f"vertical debias is not configured for robot {robot_id!r}")
      feet = list(mspec.foot_body_names)
      sole_z = (
        debias_foot_sole_z if debias_foot_sole_z is not None else mspec.foot_sole_z
      )
    bodies = list(body_names) + [f for f in feet if f not in body_names]
    self._num_bodies = len(body_names)
    self.kin = RobotKinematics(
      mj_model,
      joint_names=qpos_names,
      body_names=bodies,
      root_body=root_body,
      nworld=nworld,
      device=device,
      entity_name=entity_name,
    )
    self.library.source_pass(
      self.kin,
      anchor=bodies.index(anchor_body),
      feet=[bodies.index(f) for f in feet],
      foot_sole_z=sole_z,
    )
    self._use_graphs = self.device.type == "cuda" and storage == "device"
    self._graphs: dict[int, tuple] = {}
    self._joint_perm = (
      None
      if list(joint_names) == qpos_names
      else torch.tensor([qpos_names.index(n) for n in joint_names], device=self.device)
    )

    self.segment_names = self.library.segment_names
    self.segment_length = _grid_lengths(self.library, self.step_dt)
    self.segment_start_idx = torch.cumsum(self.segment_length, 0) - self.segment_length
    self.segment_end_idx = self.segment_start_idx + self.segment_length
    self.num_trajectories = len(self.segment_names)
    self.time_step_total = int(self.segment_length.sum().item())
    mirrored = int(self.library.segment_mirrored.sum().item())
    print(
      f"[INFO] Online motion: {self.num_trajectories} segments ({mirrored} mirrored), "
      f"{self.time_step_total} frames at {1.0 / self.step_dt:g} Hz, "
      f"{packed.num_frames} source frames on {storage}"
      + (", height debiased" if debias_z else "")
    )

  def locate(self, steps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Global frame indices → (segment, time [s])."""
    seg = torch.searchsorted(self.segment_end_idx, steps, right=True)
    seg = seg.clamp(max=self.num_trajectories - 1)
    return seg, (steps - self.segment_start_idx[seg]).double() * self.step_dt

  def frame(self, steps: torch.Tensor) -> MotionFrame:
    """Reference at global frames ``steps``.

    On CUDA (device storage) the work is a few hundred small kernels, so it runs as three
    graph launches per batch width: a torch graph up to the FK input (locate, interpolate,
    differentiate, mirror), the warp FK graph, and a torch graph from the FK output on.
    """
    m = int(steps.shape[0])
    if not self._use_graphs or m == 0 or m > self.kin.nworld:
      return self._frame(steps)
    batch = self.kin.batch_for(m)
    pre, post, buf, out = self._graphs_for(batch)
    buf[:m] = steps
    if m < batch.nworld:
      buf[m:] = 0
    pre.replay()
    batch.launch(self.device)
    post.replay()
    return MotionFrame(*(x[:m].clone() for x in out))

  def _graphs_for(self, batch) -> tuple:
    entry = self._graphs.get(batch.nworld)
    if entry is not None:
      return entry
    w = batch.nworld
    buf = torch.zeros(w, dtype=torch.long, device=self.device)

    def before():
      seg, t = self.locate(buf)
      q, qv = self.library.state_at(seg, t, self.step_dt)
      self.kin.write(batch, q, qv)
      return q, qv, self.library.anchor_acc_at(seg, t)

    def after(q, qv, acc):
      return self._assemble(q, qv, *self.kin.read(batch, w), acc)

    cur = torch.cuda.current_stream(self.device)
    side = torch.cuda.Stream(self.device)
    side.wait_stream(cur)
    with torch.cuda.stream(side):
      for _ in range(2):  # warm-up outside capture (allocator, lazy init)
        mid = before()
        batch.launch(self.device)
        after(*mid)
    cur.wait_stream(side)
    pre, post = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    with torch.cuda.graph(pre):
      mid = before()
    batch.launch(self.device)
    with torch.cuda.graph(post):
      f = after(*mid)
    out = tuple(getattr(f, x.name) for x in dataclasses.fields(MotionFrame))
    entry = self._graphs[w] = (pre, post, buf, out)
    return entry

  def _frame(self, steps: torch.Tensor) -> MotionFrame:
    seg, t = self.locate(steps)
    q, qv = self.library.state_at(seg, t, self.step_dt)
    pos, quat, lin, ang = self.kin.fk(q, qv)
    return self._assemble(
      q, qv, pos, quat, lin, ang, self.library.anchor_acc_at(seg, t)
    )

  def _assemble(self, q, qv, pos, quat, lin, ang, acc) -> MotionFrame:
    joint_pos, joint_vel = q[:, 7:], qv[:, 6:]
    if self._joint_perm is not None:
      joint_pos = joint_pos[:, self._joint_perm]
      joint_vel = joint_vel[:, self._joint_perm]
    b = self._num_bodies
    return MotionFrame(
      joint_pos=joint_pos,
      joint_vel=joint_vel,
      body_pos_w=pos[:, :b],
      body_quat_w=quat[:, :b],
      body_lin_vel_w=lin[:, :b],
      body_ang_vel_w=ang[:, :b],
      anchor_lin_acc_w=acc[:, :3],
      anchor_ang_acc_w=acc[:, 3:],
    )


# ---------------------------------------------------------------------------------- baking
class Baker:
  """Converted-clip arrays (all robot bodies, at ``fps``) from a dataset's raw clips.

  For the tools that still want an NPZ: ``wbc-mjlab-data-to-npz``, the web reference export
  and the Studio. Uses the robot's conversion scene; ``clips`` limits which clips are packed.
  """

  def __init__(
    self,
    source: Path | str,
    *,
    robot_id: str,
    fps: float = 50.0,
    mirror: bool = False,
    debias_z: bool = False,
    debias_foot_sole_z: float | None = None,
    device: str = "cuda:0",
    batch: int = 4096,
    format_name: str | None = None,
    input_fps: float | None = None,
    clips: Sequence[str] | None = None,
    line_range: tuple[int, int] | None = None,
    cache: bool = True,
  ):
    from mjlab.scene import Scene

    from wbc_mjlab.motion.robot_assets import (
      conversion_scene_cfg,
      get_robot_motion_spec,
      qpos_dof_joint_names,
    )

    if device.startswith("cuda") and not torch.cuda.is_available():
      device = "cpu"
    self.robot_id, mspec = get_robot_motion_spec(robot_id)
    scene = Scene(conversion_scene_cfg(mspec, num_envs=1), device=device)
    model = scene.compile()
    self.body_names = list(scene["robot"].body_names)
    self.joint_names = qpos_dof_joint_names(model)
    self.fps = float(fps)
    root = Path(source).expanduser().resolve()
    if clips is None and line_range is None:
      packed = load_dataset(
        root,
        self.joint_names,
        format_name=format_name,
        input_fps=input_fps,
        cache=cache,
      )
    else:
      files = list_source_files(root, format_name=format_name)
      if clips is not None:
        wanted = set(clips)
        files = [p for p in files if p.stem in wanted]
      if not files:
        raise FileNotFoundError(f"no source clips to bake in {root}")
      packed = pack_sources(
        files,
        self.joint_names,
        format_name=format_name,
        input_fps=input_fps,
        line_range=line_range,
      )
    spec = mirror_spec_for(self.robot_id, self.joint_names) if mirror else None
    self.library = MotionLibrary(packed, device=device, mirror=spec)
    self.kin = RobotKinematics(
      model,
      joint_names=self.joint_names,
      body_names=self.body_names,
      root_body=self.body_names[0],
      nworld=batch,
      device=device,
    )
    feet: list[int] = []
    sole_z = None
    if debias_z:
      if not mspec.foot_body_names or mspec.foot_sole_z is None:
        raise ValueError(
          f"vertical debias is not configured for robot {self.robot_id!r}"
        )
      feet = [self.body_names.index(f) for f in mspec.foot_body_names]
      sole_z = (
        debias_foot_sole_z if debias_foot_sole_z is not None else mspec.foot_sole_z
      )
      self.library.source_pass(self.kin, anchor=None, feet=feet, foot_sole_z=sole_z)
    self.segment_names = self.library.segment_names
    self.lengths = _grid_lengths(self.library, 1.0 / self.fps)

  def source_of(self, segment: int) -> Path:
    clip = int(self.library.segment_clip[segment])
    path = Path(self.library.packed.sources[clip])
    if bool(self.library.segment_mirrored[segment]):
      return path.with_stem(self.segment_names[segment])
    return path

  def bake(self, segment: int) -> dict:
    """The converter's log of one segment: ``fps``, ``joint_pos`` / ``joint_vel`` (qpos
    order), ``body_*_w`` for every robot body."""
    n = int(self.lengths[segment])
    dev = self.library.device
    seg = torch.full((n,), segment, dtype=torch.long, device=dev)
    t = torch.arange(n, device=dev, dtype=torch.float64) / self.fps
    q, qv = self.library.state_at(seg, t, 1.0 / self.fps)
    pos, quat, lin, ang = self.kin.fk(q, qv)
    npy = lambda x: x.detach().cpu().numpy().astype(np.float32)  # noqa: E731
    return {
      "fps": [self.fps],
      "joint_pos": npy(q[:, 7:]),
      "joint_vel": npy(qv[:, 6:]),
      "body_pos_w": npy(pos),
      "body_quat_w": npy(quat),
      "body_lin_vel_w": npy(lin),
      "body_ang_vel_w": npy(ang),
    }

  def segments(self, names: Sequence[str] | None = None) -> list[int]:
    if names is None:
      return list(range(len(self.segment_names)))
    index = {n: i for i, n in enumerate(self.segment_names)}
    return [index[n] for n in names if n in index]


# ---------------------------------------------------------------------------------- runs
RUN_MOTION_OPTIONS = ("mirror", "debias_z", "source_format", "source_fps")
"""Motion command options a policy was trained with, re-applied when it is played."""


def run_motion_options(run_dir: Path | str) -> dict:
  """The :data:`RUN_MOTION_OPTIONS` of a training run (``params/env.yaml``); empty when the
  run predates them or has no params."""
  import yaml

  path = Path(run_dir) / "params" / "env.yaml"
  if not path.is_file():
    return {}

  class _Loader(yaml.SafeLoader):
    pass

  # mjlab writes tuples (and other python objects) with python/* tags: read them as lists
  _Loader.add_multi_constructor(
    "tag:yaml.org,2002:python/",
    lambda loader, _suffix, node: (
      loader.construct_sequence(node)
      if isinstance(node, yaml.SequenceNode)
      else loader.construct_mapping(node)
      if isinstance(node, yaml.MappingNode)
      else loader.construct_scalar(node)
    ),
  )
  try:
    doc = yaml.load(path.read_text(errors="replace"), Loader=_Loader)  # noqa: S506
    motion = doc["commands"]["motion"]
  except (yaml.YAMLError, KeyError, TypeError):
    return {}
  return {k: motion[k] for k in RUN_MOTION_OPTIONS if k in motion}


def apply_run_motion_options(motion_cfg, run_dir: Path | str | None) -> dict:
  """Set a run's motion options on a ``MotionCommandCfg`` (what it was trained with)."""
  options = run_motion_options(run_dir) if run_dir is not None else {}
  for key, value in options.items():
    setattr(motion_cfg, key, value)
  if options:
    print(f"[INFO] Motion options of the run: {options}")
  return options
