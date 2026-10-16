"""Motion clips for the Studio: load (converted NPZ or raw CSV / PKL), FK once, and QA.

A :class:`Clip` holds a clip as Studio-model ``qpos`` rows plus the body poses of every frame
(FK runs once at load, playback only indexes), and :func:`analyze` turns it into the numbers
and warnings the data pane shows: duration, root height / speed, joint-limit violations, joint
speed spikes, foot penetration and airborne time.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

RAW_EXTENSIONS = (".csv", ".pkl")


@dataclass
class Clip:
  name: str
  path: Path
  kind: str  # "npz" | "raw"
  fps: float
  qpos: np.ndarray  # (T, nq)
  body_pos: np.ndarray  # (T, nbody, 3)
  body_quat: np.ndarray  # (T, nbody, 4)
  joint_names: list[str] = field(default_factory=list)
  qa: dict = field(default_factory=dict)

  @property
  def frames(self) -> int:
    return int(self.qpos.shape[0])

  @property
  def duration(self) -> float:
    return self.frames / self.fps

  def poses(
    self, frame: int, offset: np.ndarray | None = None
  ) -> tuple[np.ndarray, np.ndarray]:
    f = int(np.clip(frame, 0, self.frames - 1))
    pos = self.body_pos[f]
    return (pos + offset if offset is not None else pos), self.body_quat[f]

  def frame_at(self, t: float) -> int:
    """Frame shown at time ``t`` [s] (clamped)."""
    return int(np.clip(round(t * self.fps), 0, self.frames - 1))


# ---------------------------------------------------------------------- headers (cheap)
def npz_header(path: Path) -> dict:
  """Frames / fps / joint count of a clip NPZ without loading its arrays."""
  out: dict = {}
  with zipfile.ZipFile(path) as z:
    with z.open("joint_pos.npy") as f:
      version = np.lib.format.read_magic(f)
      read = (
        np.lib.format.read_array_header_1_0
        if version == (1, 0)
        else np.lib.format.read_array_header_2_0
      )
      shape, _, _ = read(f)
      out["frames"], out["dof"] = int(shape[0]), int(shape[1])
    if "fps.npy" in z.namelist():
      with z.open("fps.npy") as f:
        out["fps"] = float(np.lib.format.read_array(f).reshape(-1)[0])
  return out


def raw_header(path: Path) -> dict:
  if path.suffix.lower() == ".csv":
    data = path.read_bytes()
    frames = data.count(b"\n") + (0 if data.endswith(b"\n") or not data else 1)
    first = data.split(b"\n", 1)[0]
    return {"frames": frames, "fps": 30.0, "dof": max(0, first.count(b",") + 1 - 7)}
  from wbc_mjlab.motion.motion_formats import load_motion_clip

  raw = load_motion_clip(path)
  return {
    "frames": int(raw.dof_pos.shape[0]),
    "fps": float(raw.fps),
    "dof": int(raw.dof_pos.shape[1]),
  }


# ---------------------------------------------------------------------- loading
def _model_dof_names(model: mujoco.MjModel) -> list[str]:
  from wbc_mjlab.motion.robot_assets import qpos_dof_joint_names

  return qpos_dof_joint_names(model)


def load_clip(path: Path, core, *, fps_override: float | None = None) -> Clip:
  """Load a converted NPZ or a raw CSV / PKL clip as Studio-model qpos + FK body poses."""
  path = Path(path)
  if path.suffix.lower() == ".npz":
    with np.load(path, allow_pickle=True) as d:
      joint_names = [str(x) for x in d["joint_names"].tolist()]
      root_pos = np.asarray(d["body_pos_w"][:, 0], np.float64)
      root_quat = np.asarray(d["body_quat_w"][:, 0], np.float64)
      joint_pos = np.asarray(d["joint_pos"], np.float64)
      fps = float(np.asarray(d["fps"]).reshape(-1)[0]) if "fps" in d else 50.0
    kind = "npz"
  else:
    from wbc_mjlab.motion.motion_formats import load_motion_clip

    raw = load_motion_clip(path)
    root_pos = np.asarray(raw.base_pos, np.float64)
    root_quat = np.asarray(raw.base_rot_xyzw, np.float64)[
      :, [3, 0, 1, 2]
    ]  # xyzw → wxyz
    joint_pos = np.asarray(raw.dof_pos, np.float64)
    joint_names = raw.source_joint_names or _model_dof_names(core.model)
    if len(joint_names) != joint_pos.shape[1]:
      raise ValueError(
        f"{path.name}: {joint_pos.shape[1]} joint columns, robot {core.robot_id!r} has "
        f"{len(joint_names)} — wrong robot for this dataset?"
      )
    fps = float(raw.fps)
    kind = "raw"
  if fps_override:
    fps = float(fps_override)
  root_quat /= np.linalg.norm(root_quat, axis=1, keepdims=True).clip(1e-9)
  qpos = core.qpos_from_named(root_pos, root_quat, joint_names, joint_pos)
  body_pos, body_quat = core.kin(qpos)
  return Clip(
    name=path.stem,
    path=path,
    kind=kind,
    fps=fps,
    qpos=qpos,
    body_pos=body_pos,
    body_quat=body_quat,
    joint_names=joint_names,
  )


# ---------------------------------------------------------------------- QA
def _hinges(model: mujoco.MjModel) -> list[tuple[str, int, int, bool, float, float]]:
  out = []
  for j in range(model.njnt):
    if model.jnt_type[j] not in (
      mujoco.mjtJoint.mjJNT_HINGE,
      mujoco.mjtJoint.mjJNT_SLIDE,
    ):
      continue
    lo, hi = model.jnt_range[j]
    out.append(
      (
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j),
        int(model.jnt_qposadr[j]),
        int(model.jnt_dofadr[j]),
        bool(model.jnt_limited[j]),
        float(lo),
        float(hi),
      )
    )
  return out


def foot_heights(
  clip: Clip, model: mujoco.MjModel, feet: list[str], sole_z: float
) -> dict:
  """Sole height of each foot body per frame (body z − sole offset)."""
  out = {}
  for name in feet:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid >= 0:
      out[name] = clip.body_pos[:, bid, 2] - sole_z
  return out


def analyze(
  clip: Clip, model: mujoco.MjModel, *, feet: list[str], sole_z: float
) -> dict:
  """Summary numbers + warnings of one clip. ``issues``: ``{level, text}``, plus ``plot``
  (``{channels, frame, quantity}``: the Plots window channels that show it and the first frame
  where it happens) when there is a curve to look at."""
  q, dt = clip.qpos, 1.0 / clip.fps
  t = clip.frames
  root_z = q[:, 2]
  vel_xy = (
    np.linalg.norm(np.diff(q[:, 0:2], axis=0), axis=1) / dt if t > 1 else np.zeros(1)
  )
  path_len = (
    float(np.sum(np.linalg.norm(np.diff(q[:, 0:3], axis=0), axis=1))) if t > 1 else 0.0
  )
  issues: list[dict] = []

  limits = []
  speeds = []
  for name, qadr, _, limited, lo, hi in _hinges(model):
    col = q[:, qadr]
    if limited:
      over = np.maximum(col - hi, 0.0) + np.maximum(lo - col, 0.0)
      frac = float(np.mean(over > 1e-3))
      if frac > 0:
        limits.append(
          {
            "joint": name,
            "fraction": frac,
            "max_deg": float(np.degrees(over.max())),
            "frame": int(np.argmax(over > 1e-3)),
          }
        )
    if t > 2:
      jumps = np.abs(np.diff(col))
      speeds.append(
        {"joint": name, "max": float(jumps.max()) / dt, "frame": int(jumps.argmax())}
      )
  limits.sort(key=lambda x: -x["max_deg"])
  speeds.sort(key=lambda x: -x["max"])

  for lim in limits[:3]:
    level = "error" if lim["max_deg"] > 10 else "warn"
    issues.append(
      {
        "level": level,
        "text": f"{lim['joint']} outside its range on {100 * lim['fraction']:.0f}% of frames "
        f"(max {lim['max_deg']:.1f}°)",
        "plot": {"channels": [lim["joint"]], "frame": lim["frame"]},
      }
    )
  if speeds and speeds[0]["max"] > 25.0:
    issues.append(
      {
        "level": "warn",
        "text": f"joint speed spike: {speeds[0]['joint']} reaches {speeds[0]['max']:.0f} rad/s "
        "(a glitch in the source?)",
        "plot": {
          "channels": [speeds[0]["joint"]],
          "frame": speeds[0]["frame"],
          "quantity": "vel",
        },
      }
    )

  feet_h = foot_heights(clip, model, feet, sole_z)
  feet_out = {}
  if feet_h:
    stacked = np.stack(list(feet_h.values()))
    lowest = stacked.min(axis=0)
    pen = float(lowest.min())
    foot_ids = [f"foot_{n}" for n in feet_h]
    airborne = float(np.mean(lowest > 0.05))
    feet_out = {"min": pen, "airborne": airborne}
    if pen < -0.02:
      issues.append(
        {
          "level": "warn",
          "text": f"feet go {-100 * pen:.1f} cm below the ground (try --debias-z)",
          "plot": {"channels": foot_ids, "frame": int(lowest.argmin())},
        }
      )
    elif float(np.median(lowest)) > 0.05 and airborne > 0.8:
      issues.append(
        {
          "level": "warn",
          "text": f"feet float {100 * float(np.median(lowest)):.0f} cm above the ground most of "
          "the clip (try --debias-z)",
          "plot": {"channels": foot_ids},
        }
      )
  if clip.kind == "raw" and abs(clip.fps - 50.0) > 1e-3:
    issues.append(
      {
        "level": "info",
        "text": f"source at {clip.fps:g} Hz: conversion resamples to 50 Hz",
      }
    )
  if root_z.min() < 0.25:
    issues.append(
      {
        "level": "info",
        "text": f"root goes down to {root_z.min():.2f} m (lying / crawling)",
        "plot": {"channels": ["root_z"], "frame": int(root_z.argmin())},
      }
    )
  if not issues:
    issues.append({"level": "ok", "text": "no issues found"})

  return {
    "frames": t,
    "fps": clip.fps,
    "duration": clip.duration,
    "root_z": [float(root_z.min()), float(root_z.max())],
    "speed_mean": float(vel_xy.mean()),
    "speed_max": float(vel_xy.max()),
    "path_length": path_len,
    "limits": limits[:8],
    "speeds": speeds[:5],
    "feet": feet_out,
    "issues": issues,
  }


__all__ = [
  "RAW_EXTENSIONS",
  "Clip",
  "analyze",
  "load_clip",
  "npz_header",
  "raw_header",
]
