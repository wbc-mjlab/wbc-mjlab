"""Per-clip reference-command streams for the wbc-demo live policy engine.

Implements the ``wbc_reference_stream_v1`` wire format (issue wbc-mjlab-ow5, the
live-engine pivot). The browser runs the policy ONNX live and builds the full
actor observation by concatenating this **reference command** with live
proprioception from its own sim. So we ship only the reference stream, never
full-body render poses.

The reference command is 39 floats per frame, in the deploy ``config.yaml``
``tracking.reference_observation_names`` order::

  ref_base_height   (1)   anchor z relative to env origin
  ref_base_lin_vel_b(3)   anchor linear velocity, anchor frame
  ref_base_ang_vel_b(3)   anchor angular velocity, anchor frame
  ref_gravity_b     (3)   gravity unit vector (0,0,-1), anchor frame
  ref_joint_pos     (29)  reference joint positions (config joint order)

This matches mjlab's ``MotionCommand`` (``env/mdp/commands.py``) and the deploy
C++ ``WbcMotionLoader`` (``wbc-g1-deploy/src/WbcMotionLoader.cpp``) term-for-term:

- The anchor body is ``torso_link``; its world pose/velocity is read straight
  from the clip's stored ``body_*_w`` arrays (NO forward kinematics).
- ``quat_apply_inverse(q, v)`` rotates a world vector into the anchor frame,
  identical to the deploy ``q.conjugate() * v`` (quat stored ``w, x, y, z``).
- Gravity is the unit vector ``(0, 0, -1)`` in the MuJoCo Z-up world frame.
- ``env_origin_z`` is 0 for an exported clip, so ``ref_base_height`` is the raw
  anchor world z.

The wire format per clip is a raw little-endian Float32 ``.bin`` with NO header,
frame-major ``frames x 39``. ``reference/index.json`` lists every clip and the
term layout.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from wbc_mjlab.motion.robot_assets import normalize_joint_name_list, remap_dof_columns

REFERENCE_SCHEMA = "wbc_reference_stream_v1"
REFERENCE_SUBDIR = "reference"
REFERENCE_INDEX_NAME = "index.json"

# Gravity unit vector in the MuJoCo Z-up world frame (matches deploy kGravityW
# and mjlab's robot.data.gravity_vec_w for a Z-up world).
_GRAVITY_W = np.array([0.0, 0.0, -1.0], dtype=np.float64)

# Float32 little-endian, no header.
_DTYPE = np.dtype("<f4")

# Reference command term layout (name, dim) in reference_observation_names order.
# Sums to wbc_command_dim = 39 for G1 (29-dof).
REF_TERMS: tuple[tuple[str, int], ...] = (
  ("ref_base_height", 1),
  ("ref_base_lin_vel_b", 3),
  ("ref_base_ang_vel_b", 3),
  ("ref_gravity_b", 3),
  ("ref_joint_pos", 29),
)


@dataclass(frozen=True)
class ReferenceClip:
  """One exported reference-command clip and its ``index.json`` metadata."""

  id: str
  name: str
  file: str
  frames: int
  duration_sec: float
  tags: tuple[str, ...]

  def index_entry(self) -> dict[str, Any]:
    return {
      "id": self.id,
      "name": self.name,
      "file": self.file,
      "frames": self.frames,
      "durationSec": round(self.duration_sec, 4),
      "tags": list(self.tags),
    }


def command_dim(joint_names: list[str]) -> int:
  """Total reference command dim for a given joint set (39 for G1 29-dof)."""
  fixed = sum(dim for name, dim in REF_TERMS if name != "ref_joint_pos")
  return fixed + len(joint_names)


def quat_apply_inverse(quat_wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:
  """Rotate world vectors into the frame of ``quat`` (its conjugate applied).

  Batched: ``quat_wxyz`` is ``(N, 4)`` (``w, x, y, z``), ``vec`` is ``(N, 3)``.
  Mirrors mjlab ``quat_apply_inverse`` and deploy ``q.conjugate() * v``.
  """
  quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64)
  vec = np.asarray(vec, dtype=np.float64)
  w = quat_wxyz[:, 0]
  xyz = quat_wxyz[:, 1:4]
  # v' = v - 2 w (xyz x v) + 2 xyz x (xyz x v) ... use the conjugate (negate xyz).
  # Equivalent to the active rotation by the conjugate quaternion.
  t = 2.0 * np.cross(-xyz, vec)
  return vec + w[:, None] * t + np.cross(-xyz, t)


def reference_terms_from_clip(
  *,
  anchor_pos_w: np.ndarray,
  anchor_quat_w: np.ndarray,
  anchor_lin_vel_w: np.ndarray,
  anchor_ang_vel_w: np.ndarray,
  joint_pos: np.ndarray,
  env_origin_z: float = 0.0,
) -> np.ndarray:
  """Compute the per-frame reference command ``(frames, 39)`` for one clip.

  All inputs are already at the anchor body (``torso_link``) and in the clip's
  joint order. ``joint_pos`` columns must already be in the target (config)
  joint order. Returns Float32, term-major per the layout in :data:`REF_TERMS`.
  """
  anchor_pos_w = np.asarray(anchor_pos_w, dtype=np.float64)
  anchor_quat_w = np.asarray(anchor_quat_w, dtype=np.float64)
  joint_pos = np.asarray(joint_pos, dtype=np.float64)
  frames = anchor_pos_w.shape[0]

  ref_base_height = anchor_pos_w[:, 2:3] - env_origin_z
  ref_base_lin_vel_b = quat_apply_inverse(anchor_quat_w, anchor_lin_vel_w)
  ref_base_ang_vel_b = quat_apply_inverse(anchor_quat_w, anchor_ang_vel_w)
  gravity_w = np.broadcast_to(_GRAVITY_W, (frames, 3))
  ref_gravity_b = quat_apply_inverse(anchor_quat_w, gravity_w)

  stacked = np.concatenate(
    [
      ref_base_height,
      ref_base_lin_vel_b,
      ref_base_ang_vel_b,
      ref_gravity_b,
      joint_pos,
    ],
    axis=1,
  )
  return np.ascontiguousarray(stacked, dtype=_DTYPE)


def _anchor_body_index(robot_body_names: list[str], anchor_body_name: str) -> int:
  """Index of ``anchor_body_name`` in the clip's full robot body order.

  The NPZ ``body_*_w`` arrays are stored in ``robot.body_names`` order (the
  MJCF body order excluding the MuJoCo ``world`` body), which equals the model
  body ids 1..nbody-1.
  """
  try:
    return robot_body_names.index(anchor_body_name)
  except ValueError as exc:
    raise ValueError(
      f"Anchor body {anchor_body_name!r} not in robot body names {robot_body_names}"
    ) from exc


def reference_command_from_npz(
  data: np.lib.npyio.NpzFile,
  *,
  robot_body_names: list[str],
  anchor_body_name: str,
  config_joint_names: list[str],
  env_origin_z: float = 0.0,
) -> np.ndarray:
  """Per-frame reference command ``(frames, dim)`` for one motion-clip NPZ.

  - The anchor world state is read directly from ``body_*_w`` at the anchor body
    index (resolved against ``robot_body_names``), matching the deploy loader.
  - ``ref_joint_pos`` is remapped from the NPZ ``joint_names`` order to
    ``config_joint_names`` order, so the stream matches the config obs layout.
  """
  anchor = _anchor_body_index(robot_body_names, anchor_body_name)
  body_pos_w = np.asarray(data["body_pos_w"], dtype=np.float64)
  body_quat_w = np.asarray(data["body_quat_w"], dtype=np.float64)
  body_lin_vel_w = np.asarray(data["body_lin_vel_w"], dtype=np.float64)
  body_ang_vel_w = np.asarray(data["body_ang_vel_w"], dtype=np.float64)

  npz_joint_names = normalize_joint_name_list(
    [str(x) for x in data["joint_names"].tolist()]
  )
  joint_pos = np.asarray(data["joint_pos"], dtype=np.float64)
  if joint_pos.shape[1] != len(config_joint_names):
    raise ValueError(
      f"NPZ joint width {joint_pos.shape[1]} != config joint count "
      f"{len(config_joint_names)}"
    )
  joint_pos = remap_dof_columns(joint_pos, npz_joint_names, config_joint_names)

  return reference_terms_from_clip(
    anchor_pos_w=body_pos_w[:, anchor],
    anchor_quat_w=body_quat_w[:, anchor],
    anchor_lin_vel_w=body_lin_vel_w[:, anchor],
    anchor_ang_vel_w=body_ang_vel_w[:, anchor],
    joint_pos=joint_pos,
    env_origin_z=env_origin_z,
  )


def write_reference_bin(path: Path, command: np.ndarray) -> int:
  """Write ``command`` (frames, dim) as raw little-endian Float32. Returns bytes."""
  if command.ndim != 2:
    raise ValueError(f"command must be (frames, dim), got {command.shape}")
  path.parent.mkdir(parents=True, exist_ok=True)
  flat = np.ascontiguousarray(command, dtype=_DTYPE).reshape(-1)
  flat.tofile(path)
  return int(flat.nbytes)


def expected_bin_bytes(frames: int, dim: int) -> int:
  """Byte length of a ``.bin`` for ``frames`` frames of width ``dim``."""
  return frames * dim * _DTYPE.itemsize


def ref_terms_index(joint_names: list[str]) -> list[dict[str, int]]:
  """``refTerms`` index entries (name + dim), with ref_joint_pos sized to dofs."""
  out: list[dict[str, int]] = []
  for name, dim in REF_TERMS:
    out.append(
      {"name": name, "dim": len(joint_names) if name == "ref_joint_pos" else dim}
    )
  return out


def write_reference_index(
  reference_dir: Path,
  *,
  robot_id: str,
  fps: float,
  joint_names: list[str],
  clips: list[ReferenceClip],
) -> Path:
  """Write ``reference/index.json`` (``wbc_reference_stream_v1``)."""
  reference_dir.mkdir(parents=True, exist_ok=True)
  doc = {
    "schema": REFERENCE_SCHEMA,
    "robot": robot_id,
    "commandDim": command_dim(joint_names),
    "fps": fps,
    "refTerms": ref_terms_index(joint_names),
    "clips": [clip.index_entry() for clip in clips],
  }
  out = reference_dir / REFERENCE_INDEX_NAME
  out.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
  return out
