"""Source motion formats and resampling.

Canonical clip layout (every format must produce this)::

  base_pos      (T, 3)   world root position
  base_rot_xyzw (T, 4)   root quaternion, xyzw
  dof_pos       (T, n)   joint positions (robot DOF order in file)
  fps           float    source frame rate

``MotionLoader`` resamples a ``MotionClipRaw`` clip to the training output rate.

Adding a new format
-------------------
1. If the layout matches an existing factory (CSV rows or PKL dict), append one
   line to **MOTION FORMAT REGISTRY** below.
2. Otherwise implement ``load`` + ``peek_dof_width`` in the loader helpers section
   and register a ``MotionFormatSpec`` in the registry.
"""

from __future__ import annotations

import pickle
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mjlab.utils.lab_api.math import (
  axis_angle_from_quat,
  quat_conjugate,
  quat_mul,
)

from wbc_mjlab.motion.robot_assets import normalize_joint_name_list, remap_dof_columns

__all__ = [
  "DEFAULT_MOTION_FORMAT",
  "MotionClipRaw",
  "MotionFormatSpec",
  "MotionLoader",
  "MotionPlayback",
  "MOTION_FORMATS",
  "ResampledClip",
  "get_motion_format",
  "infer_format_name",
  "list_motion_format_names",
  "load_motion_clip",
  "peek_dof_width",
  "resample_motion_clips_batch",
  "resolve_input_motion_paths",
]

DEFAULT_MOTION_FORMAT = "default"


# =============================================================================
# Canonical clip + format descriptor
# =============================================================================


@dataclass(frozen=True)
class MotionClipRaw:
  """Raw clip before MuJoCo FK / NPZ export."""

  base_pos: np.ndarray
  base_rot_xyzw: np.ndarray
  dof_pos: np.ndarray
  fps: float
  source_joint_names: list[str] | None = None

  def __post_init__(self) -> None:
    n = self.base_pos.shape[0]
    if self.base_rot_xyzw.shape[0] != n or self.dof_pos.shape[0] != n:
      raise ValueError(
        f"Frame count mismatch: base_pos={n}, "
        f"base_rot={self.base_rot_xyzw.shape[0]}, dof_pos={self.dof_pos.shape[0]}"
      )


@dataclass(frozen=True)
class MotionFormatSpec:
  """One registered source file format."""

  name: str
  extensions: frozenset[str]
  default_fps: float | None
  load: Callable[..., MotionClipRaw]
  peek_dof_width: Callable[..., int]


# =============================================================================
# MOTION FORMAT REGISTRY — add new formats here
# =============================================================================
#
# | name      | file   | layout                                           | default fps |
# |-----------|--------|--------------------------------------------------|-------------|
# | default   | .csv   | root_pos(m)+root_rot_xyzw(4)+dof_pos(rad), no hdr| 30          |
# | gmr_pkl   | .pkl   | dict: fps, root_pos, root_rot, dof_pos           | from file   |
#
# Example — new CSV dataset, same columns as LAFAN, different fps::
#
#   "my_csv": csv_row_format("my_csv", default_fps=60.0),
#
# Example — new PKL dict with GMR keys and optional joint metadata::
#
#   "my_pkl": gmr_pkl_format("my_pkl"),
# =============================================================================

MOTION_FORMATS: dict[str, MotionFormatSpec] = {}


# =============================================================================
# Loader helpers (shared building blocks for format factories)
# =============================================================================


def _as_float32(arr: Any) -> np.ndarray:
  return np.asarray(arr, dtype=np.float32)


def _make_clip(
  base_pos: Any,
  base_rot_xyzw: Any,
  dof_pos: Any,
  fps: float,
  *,
  source_joint_names: list[str] | None = None,
) -> MotionClipRaw:
  return MotionClipRaw(
    base_pos=_as_float32(base_pos),
    base_rot_xyzw=_as_float32(base_rot_xyzw),
    dof_pos=_as_float32(dof_pos),
    fps=float(fps),
    source_joint_names=source_joint_names,
  )


def _read_csv_matrix(
  path: Path,
  *,
  line_range: tuple[int, int] | None = None,
) -> np.ndarray:
  if line_range is None:
    matrix = np.loadtxt(path, delimiter=",")
  else:
    matrix = np.loadtxt(
      path,
      delimiter=",",
      skiprows=line_range[0] - 1,
      max_rows=line_range[1] - line_range[0] + 1,
    )
  matrix = _as_float32(matrix)
  if matrix.ndim != 2 or matrix.shape[1] < 8:
    raise ValueError(
      f"CSV {path} must have at least 8 columns "
      f"(root_pos + root_rot_xyzw + joints), got shape {matrix.shape}"
    )
  return matrix


def _peek_csv_row_dof_width(
  path: Path,
  *,
  line_range: tuple[int, int] | None = None,
) -> int:
  if line_range is None:
    row = np.loadtxt(path, delimiter=",", max_rows=1)
  else:
    row = np.loadtxt(
      path,
      delimiter=",",
      skiprows=line_range[0] - 1,
      max_rows=1,
    )
  row = np.atleast_1d(_as_float32(row))
  if row.size < 8:
    raise ValueError(
      f"CSV {path} must have at least 8 columns (root pos/rot + joints), "
      f"got {row.size}"
    )
  return int(row.size - 7)


def _load_csv_row_clip(
  path: Path,
  *,
  line_range: tuple[int, int] | None = None,
  default_fps: float = 30.0,
) -> MotionClipRaw:
  matrix = _read_csv_matrix(path, line_range=line_range)
  return _make_clip(
    matrix[:, :3],
    matrix[:, 3:7],
    matrix[:, 7:],
    default_fps,
  )


def _pick_joint_names_from_pkl(data: dict[str, Any]) -> list[str] | None:
  for key in ("dof_joint_names", "joint_names", "joint_order"):
    raw = data.get(key)
    if raw is None:
      continue
    names = normalize_joint_name_list(list(raw))
    if names:
      return names
  return None


def _load_gmr_pkl_dict(path: Path) -> dict[str, Any]:
  with path.open("rb") as handle:
    data = pickle.load(handle)
  if not isinstance(data, dict):
    raise TypeError(f"Expected dict in {path}, got {type(data)}")
  required = ("fps", "root_pos", "root_rot", "dof_pos")
  missing = [k for k in required if k not in data]
  if missing:
    raise KeyError(f"Missing keys {missing} in {path}")
  return data


def _load_gmr_pkl_clip(
  path: Path,
  *,
  line_range: tuple[int, int] | None = None,
) -> MotionClipRaw:
  if line_range is not None:
    raise ValueError(f"line_range is not supported for PKL inputs ({path})")
  data = _load_gmr_pkl_dict(path)
  return _make_clip(
    data["root_pos"],
    data["root_rot"],
    data["dof_pos"],
    data["fps"],
    source_joint_names=_pick_joint_names_from_pkl(data),
  )


def _peek_gmr_pkl_dof_width(path: Path, **_: Any) -> int:
  data = _load_gmr_pkl_dict(path)
  return int(_as_float32(data["dof_pos"]).shape[1])


# =============================================================================
# Format factories — wire helpers into MotionFormatSpec
# =============================================================================


def csv_row_format(
  name: str,
  *,
  extensions: frozenset[str] = frozenset({".csv"}),
  default_fps: float = 30.0,
  description: str = "",
) -> MotionFormatSpec:
  """LAFAN-style CSV: ``root_pos(3) + root_rot_xyzw(4) + dof_pos(n)`` per row, no header."""
  _ = description
  fmt_default_fps = default_fps

  def load(
    path: Path,
    *,
    line_range: tuple[int, int] | None = None,
    default_fps: float = fmt_default_fps,
  ) -> MotionClipRaw:
    return _load_csv_row_clip(path, line_range=line_range, default_fps=default_fps)

  def peek_dof_width(
    path: Path,
    *,
    line_range: tuple[int, int] | None = None,
  ) -> int:
    return _peek_csv_row_dof_width(path, line_range=line_range)

  return MotionFormatSpec(
    name=name,
    extensions=extensions,
    default_fps=default_fps,
    load=load,
    peek_dof_width=peek_dof_width,
  )


def gmr_pkl_format(
  name: str,
  *,
  extensions: frozenset[str] = frozenset({".pkl"}),
) -> MotionFormatSpec:
  """PKL dict: ``fps``, ``root_pos``, ``root_rot`` (xyzw), ``dof_pos`` (+ optional joint metadata)."""
  return MotionFormatSpec(
    name=name,
    extensions=extensions,
    default_fps=None,
    load=_load_gmr_pkl_clip,
    peek_dof_width=_peek_gmr_pkl_dof_width,
  )


def _register_formats() -> dict[str, MotionFormatSpec]:
  return {
    "default": csv_row_format(
      "default",
      description="LAFAN / retarget CSV (meters, quat xyzw, joint radians)",
    ),
    "gmr_pkl": gmr_pkl_format("gmr_pkl"),
  }


MOTION_FORMATS.update(_register_formats())

_EXTENSION_TO_FORMATS: dict[str, list[str]] = {}
for _name, _spec in MOTION_FORMATS.items():
  for _ext in _spec.extensions:
    _EXTENSION_TO_FORMATS.setdefault(_ext, []).append(_name)


# =============================================================================
# Public API — load / resolve paths
# =============================================================================


def list_motion_format_names() -> list[str]:
  return sorted(MOTION_FORMATS.keys())


def get_motion_format(name: str) -> MotionFormatSpec:
  key = name.strip().lower()
  if key not in MOTION_FORMATS:
    known = ", ".join(list_motion_format_names())
    raise ValueError(f"Unknown motion format {name!r}. Known formats: {known}")
  return MOTION_FORMATS[key]


def infer_format_name(path: Path, *, format_name: str | None = None) -> str:
  if format_name is not None:
    return get_motion_format(format_name).name
  ext = path.suffix.lower()
  candidates = _EXTENSION_TO_FORMATS.get(ext)
  if not candidates:
    raise ValueError(
      f"Cannot infer motion format for {path} (extension {ext!r}). "
      f"Pass --format explicitly."
    )
  if len(candidates) == 1:
    return candidates[0]
  if ext == ".csv":
    return DEFAULT_MOTION_FORMAT
  raise ValueError(
    f"Ambiguous motion format for {path}. Pass --format explicitly "
    f"(candidates: {', '.join(candidates)})."
  )


def load_motion_clip(
  path: Path,
  *,
  format_name: str | None = None,
  input_fps: float | None = None,
  line_range: tuple[int, int] | None = None,
) -> MotionClipRaw:
  """Load one clip from disk into canonical raw layout."""
  path = path.expanduser().resolve()
  name = infer_format_name(path, format_name=format_name)
  spec = get_motion_format(name)
  if spec.extensions and path.suffix.lower() not in spec.extensions:
    raise ValueError(
      f"Format {name!r} expects {sorted(spec.extensions)}, got {path.suffix}"
    )
  load_kwargs: dict[str, Any] = {}
  if line_range is not None:
    load_kwargs["line_range"] = line_range
  if spec.default_fps is not None:
    load_kwargs["default_fps"] = spec.default_fps
  raw = spec.load(path, **load_kwargs)
  fps = float(input_fps if input_fps is not None else raw.fps)
  return MotionClipRaw(
    base_pos=raw.base_pos,
    base_rot_xyzw=raw.base_rot_xyzw,
    dof_pos=raw.dof_pos,
    fps=fps,
    source_joint_names=raw.source_joint_names,
  )


def peek_dof_width(
  path: Path,
  *,
  format_name: str | None = None,
  line_range: tuple[int, int] | None = None,
) -> int:
  path = path.expanduser().resolve()
  name = infer_format_name(path, format_name=format_name)
  spec = get_motion_format(name)
  return spec.peek_dof_width(path, line_range=line_range)


def _extensions_for_format(format_name: str | None) -> frozenset[str]:
  if format_name is None:
    exts: set[str] = set()
    for spec in MOTION_FORMATS.values():
      exts.update(spec.extensions)
    return frozenset(exts)
  return get_motion_format(format_name).extensions


def resolve_input_motion_paths(
  input_path: str,
  *,
  format_name: str | None = None,
) -> list[Path]:
  """Resolve a file or directory to motion source paths."""
  path = Path(input_path).expanduser().resolve()
  extensions = _extensions_for_format(format_name)

  if path.is_file():
    if path.suffix.lower() not in extensions:
      raise ValueError(
        f"{path}: expected extension {sorted(extensions)}, got {path.suffix}"
      )
    return [path]

  if path.is_dir():
    files: list[Path] = []
    for ext in sorted(extensions):
      files.extend(path.glob(f"*{ext}"))
    files = sorted({p.resolve() for p in files})
    if not files:
      raise ValueError(
        f"No motion files ({', '.join(sorted(extensions))}) found in: {path}"
      )
    return files

  raise ValueError(f"Input path does not exist: {path}")


# =============================================================================
# Resampling — raw clip(s) → output-rate tensors (batched on GPU)
# =============================================================================


@dataclass
class ResampledClip:
  """Interpolated clip ready for FK playback."""

  base_pos: torch.Tensor
  base_rot: torch.Tensor
  dof_pos: torch.Tensor
  base_lin_vel: torch.Tensor
  base_ang_vel: torch.Tensor
  dof_vel: torch.Tensor
  output_frames: int


def _raw_clip_to_input_tensors(
  raw: MotionClipRaw,
  *,
  model_joint_names: list[str] | None,
  device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, float]:
  dof_pos = raw.dof_pos
  if raw.source_joint_names is not None and model_joint_names is not None:
    dof_pos = remap_dof_columns(dof_pos, raw.source_joint_names, model_joint_names)

  motion = np.concatenate(
    [raw.base_pos, raw.base_rot_xyzw, dof_pos],
    axis=1,
    dtype=np.float32,
  )
  motion = torch.from_numpy(motion).to(device=device)
  base_pos = motion[:, :3]
  base_rot_wxyz = motion[:, 3:7][:, [3, 0, 1, 2]]
  dof_pos_t = motion[:, 7:]
  input_frames = int(motion.shape[0])
  input_fps = float(raw.fps)
  return base_pos, base_rot_wxyz, dof_pos_t, input_frames, input_fps


def _batched_quat_slerp(
  q0: torch.Tensor,
  q1: torch.Tensor,
  tau: torch.Tensor,
) -> torch.Tensor:
  """SLERP for ``(..., 4)`` quaternions (wxyz) and broadcastable ``tau``."""
  eps = torch.finfo(q0.dtype).eps * 4.0
  d = (q0 * q1).sum(dim=-1, keepdim=True)
  q1 = torch.where(d < 0.0, -q1, q1)
  d = d.abs().clamp(-1.0, 1.0)
  angle = torch.acos(d)
  near_parallel = angle < eps
  sin_angle = torch.sin(angle).clamp_min(eps)
  w0 = torch.sin((1.0 - tau) * angle) / sin_angle
  w1 = torch.sin(tau * angle) / sin_angle
  out = w0 * q0 + w1 * q1
  return torch.where(near_parallel, q0, out)


def _batched_lerp_gather(
  values: torch.Tensor,
  index_0: torch.Tensor,
  index_1: torch.Tensor,
  blend: torch.Tensor,
) -> torch.Tensor:
  """Linear interpolation with per-step gather indices.

  ``values``: ``(B, T_in, F)``; ``index_*``: ``(B, T_out)``; ``blend``: ``(B, T_out)``.
  """
  feature_dim = values.shape[-1]
  idx0 = index_0.unsqueeze(-1).expand(-1, -1, feature_dim)
  idx1 = index_1.unsqueeze(-1).expand(-1, -1, feature_dim)
  blend_f = blend.unsqueeze(-1)
  a = torch.gather(values, 1, idx0)
  b = torch.gather(values, 1, idx1)
  return a * (1.0 - blend_f) + b * blend_f


def _compute_output_frame_count(input_frames: int, input_fps: float, output_fps: int) -> int:
  if input_frames <= 1:
    return input_frames
  duration = (input_frames - 1) / input_fps
  output_dt = 1.0 / output_fps
  return int(torch.arange(0, duration, output_dt).numel())


def _compute_velocities_for_clip(
  base_pos: torch.Tensor,
  base_rot: torch.Tensor,
  dof_pos: torch.Tensor,
  output_dt: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  base_lin_vel = torch.gradient(base_pos, spacing=output_dt, dim=0)[0]
  dof_vel = torch.gradient(dof_pos, spacing=output_dt, dim=0)[0]
  q_prev, q_next = base_rot[:-2], base_rot[2:]
  q_rel = quat_mul(q_next, quat_conjugate(q_prev))
  omega = axis_angle_from_quat(q_rel) / (2.0 * output_dt)
  base_ang_vel = torch.cat([omega[:1], omega, omega[-1:]], dim=0)
  return base_lin_vel, base_ang_vel, dof_vel


def resample_motion_clips_batch(
  raw_clips: list[MotionClipRaw],
  output_fps: int,
  device: torch.device | str,
  *,
  model_joint_names: list[str] | None = None,
) -> list[ResampledClip]:
  """Resample multiple clips on GPU in one batched interpolation pass."""
  if not raw_clips:
    return []

  device = torch.device(device)
  output_dt = 1.0 / float(output_fps)

  parsed = [
    _raw_clip_to_input_tensors(
      raw, model_joint_names=model_joint_names, device=device
    )
    for raw in raw_clips
  ]
  batch_size = len(parsed)
  input_frames = torch.tensor(
    [item[3] for item in parsed], device=device, dtype=torch.long
  )
  input_fps = torch.tensor(
    [item[4] for item in parsed], device=device, dtype=torch.float32
  )
  t_in_max = int(input_frames.max().item())
  dof_dim = parsed[0][2].shape[1]

  base_pos_in = torch.zeros(batch_size, t_in_max, 3, device=device)
  base_rot_in = torch.zeros(batch_size, t_in_max, 4, device=device)
  dof_pos_in = torch.zeros(batch_size, t_in_max, dof_dim, device=device)
  for batch_idx, (base_pos, base_rot, dof_pos, n_in, _) in enumerate(parsed):
    base_pos_in[batch_idx, :n_in] = base_pos
    base_rot_in[batch_idx, :n_in] = base_rot
    dof_pos_in[batch_idx, :n_in] = dof_pos
    if n_in < t_in_max:
      base_pos_in[batch_idx, n_in:] = base_pos[-1]
      base_rot_in[batch_idx, n_in:] = base_rot[-1]
      dof_pos_in[batch_idx, n_in:] = dof_pos[-1]

  output_counts = [
    _compute_output_frame_count(int(n), float(fps), output_fps)
    for n, fps in zip(input_frames.tolist(), input_fps.tolist(), strict=True)
  ]
  t_out_max = max(output_counts)

  durations = (input_frames.float() - 1.0) / input_fps
  times = (
    torch.arange(t_out_max, device=device, dtype=torch.float32).unsqueeze(0)
    * output_dt
  )
  safe_durations = durations.unsqueeze(1).clamp_min(1.0e-8)
  phase = times / safe_durations
  index_0 = (phase * (input_frames.unsqueeze(1) - 1).float()).floor().long()
  index_1 = torch.minimum(
    index_0 + 1, (input_frames - 1).unsqueeze(1).expand(-1, t_out_max)
  )
  blend = phase * (input_frames.unsqueeze(1) - 1).float() - index_0.float()

  base_pos_out = _batched_lerp_gather(base_pos_in, index_0, index_1, blend)
  dof_pos_out = _batched_lerp_gather(dof_pos_in, index_0, index_1, blend)

  rot0 = torch.gather(base_rot_in, 1, index_0.unsqueeze(-1).expand(-1, -1, 4))
  rot1 = torch.gather(base_rot_in, 1, index_1.unsqueeze(-1).expand(-1, -1, 4))
  base_rot_out = _batched_quat_slerp(rot0, rot1, blend.unsqueeze(-1))

  clips: list[ResampledClip] = []
  for batch_idx, n_out in enumerate(output_counts):
    if n_out <= 0:
      raise ValueError(
        f"Clip {batch_idx} produced no output frames "
        f"(input_frames={int(input_frames[batch_idx])})."
      )
    pos = base_pos_out[batch_idx, :n_out]
    rot = base_rot_out[batch_idx, :n_out]
    dof = dof_pos_out[batch_idx, :n_out]
    lin_vel, ang_vel, dof_vel = _compute_velocities_for_clip(
      pos, rot, dof, output_dt
    )
    clips.append(
      ResampledClip(
        base_pos=pos,
        base_rot=rot,
        dof_pos=dof,
        base_lin_vel=lin_vel,
        base_ang_vel=ang_vel,
        dof_vel=dof_vel,
        output_frames=n_out,
      )
    )
  return clips


class MotionPlayback:
  """Step through a pre-resampled clip."""

  def __init__(self, clip: ResampledClip):
    self._clip = clip
    self.current_idx = 0
    self.output_frames = clip.output_frames

  def advance_state(
    self,
  ) -> tuple[
    tuple[
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
    ],
    bool,
  ]:
    if self.current_idx >= self.output_frames:
      raise RuntimeError("Motion clip already finished")
    idx = self.current_idx
    state = (
      self._clip.base_pos[idx : idx + 1],
      self._clip.base_rot[idx : idx + 1],
      self._clip.base_lin_vel[idx : idx + 1],
      self._clip.base_ang_vel[idx : idx + 1],
      self._clip.dof_pos[idx : idx + 1],
      self._clip.dof_vel[idx : idx + 1],
    )
    self.current_idx += 1
    return state, self.current_idx >= self.output_frames


class MotionLoader:
  """Interpolate a raw clip and derive root / joint velocities."""

  def __init__(
    self,
    raw: MotionClipRaw,
    output_fps: int,
    device: torch.device | str,
    *,
    model_joint_names: list[str] | None = None,
  ):
    self.output_fps = output_fps
    self.input_fps = int(raw.fps)
    self.current_idx = 0
    clip = resample_motion_clips_batch(
      [raw],
      output_fps,
      device,
      model_joint_names=model_joint_names,
    )[0]
    self.output_frames = clip.output_frames
    self.motion_base_poss = clip.base_pos
    self.motion_base_rots = clip.base_rot
    self.motion_dof_poss = clip.dof_pos
    self.motion_base_lin_vels = clip.base_lin_vel
    self.motion_base_ang_vels = clip.base_ang_vel
    self.motion_dof_vels = clip.dof_vel
    self._playback = MotionPlayback(clip)

  def advance_state(
    self,
  ) -> tuple[
    tuple[
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
    ],
    bool,
  ]:
    return self._playback.advance_state()

  def get_next_state(
    self,
  ) -> tuple[
    tuple[
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
    ],
    bool,
  ]:
    state, done = self._playback.advance_state()
    if done:
      self._playback.current_idx = 0
    return state, done
