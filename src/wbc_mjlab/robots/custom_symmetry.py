"""Suggest and validate sagittal reflection maps for Studio robot imports."""

from __future__ import annotations

import re
from dataclasses import dataclass

import mujoco

from wbc_mjlab.robots.symmetry import JointSymmetryEntry, RobotSymmetryConfig


@dataclass
class CustomSymmetry:
  joints: list[JointSymmetryEntry]
  body_pairs: list[list[str]]

  def config(self) -> RobotSymmetryConfig:
    return RobotSymmetryConfig(
      joints=tuple(self.joints), body_pairs=tuple((a, b) for a, b in self.body_pairs)
    )

  @classmethod
  def from_dict(cls, data: dict) -> CustomSymmetry:
    try:
      joints = [JointSymmetryEntry(**row) for row in data["joints"]]
      pairs = data["body_pairs"]
      if not isinstance(pairs, list) or not all(
        isinstance(p, list) and len(p) == 2 for p in pairs
      ):
        raise ValueError("body_pairs must be a list of body name pairs")
      return cls(joints=joints, body_pairs=pairs)
    except (TypeError, KeyError, ValueError) as exc:
      raise ValueError(f"invalid robot symmetry: {exc}") from exc


def _named_pairs(names: list[str]) -> dict[str, str]:
  # Long side names (LeftHip, hip_right), delimited L/R, and LHip/RHip.
  by_name = {n.lower(): n for n in names}
  partners = {}
  for name in names:
    match = re.search(r"left|right|(?<![a-z0-9])[lr](?![a-z0-9])", name, re.I)
    if match is None:
      match = re.match(r"[LR](?=[A-Z])", name)
    if match:
      other = {"left": "right", "right": "left", "l": "r", "r": "l"}[match[0].lower()]
      partner = by_name.get(
        (name[: match.start()] + other + name[match.end() :]).lower()
      )
      if partner:
        partners[name] = partner
  return {a: b for a, b in partners.items() if partners.get(b) == a}


def suggest_symmetry(model: mujoco.MjModel) -> dict:
  """Reflect joint axes in a common frame at the model's reference configuration."""
  data = mujoco.MjData(model)
  data.qpos[:3] = 0
  data.qpos[3:7] = (1, 0, 0, 0)
  mujoco.mj_kinematics(model, data)
  hinge = mujoco.mjtJoint.mjJNT_HINGE
  joint_ids = {
    model.joint(i).name: i
    for i in range(model.njnt)
    if model.jnt_type[i] in (hinge, mujoco.mjtJoint.mjJNT_SLIDE)
  }
  partners = _named_pairs(list(joint_ids))
  joints = []
  for name, i in joint_ids.items():
    partner = partners.get(name, name)
    j = joint_ids[partner]
    if j < i or model.jnt_type[j] != model.jnt_type[i]:
      continue
    if j == i and abs(data.xanchor[i, 1]) > 1e-4:
      continue  # An unmatched off-center joint needs a user-selected partner.
    reflected = data.xaxis[i] * [1, -1, 1]
    if model.jnt_type[i] == hinge:
      reflected = -reflected  # Rotation axes are axial vectors, slide axes are polar.
    dot = float(reflected @ data.xaxis[j])
    if abs(dot) > 0.99:
      joints.append(dict(name_a=name, name_b=partner, scale=1 if dot > 0 else -1))
  bodies = _named_pairs([model.body(i).name for i in range(1, model.nbody)])
  return {"joints": joints, "body_pairs": [[a, b] for a, b in bodies.items() if a < b]}


def validate_symmetry(config: CustomSymmetry, model: mujoco.MjModel) -> None:
  joints = {
    model.joint(i).name
    for i in range(model.njnt)
    if model.jnt_type[i] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
  }
  bodies = {model.body(i).name for i in range(1, model.nbody)}

  def check_pairs(pairs, names, kind):
    seen = set()
    for a, b in pairs:
      if not isinstance(a, str) or not isinstance(b, str):
        raise ValueError(f"symmetry: {kind} names must be strings")
      for name in {a, b}:
        if name not in names:
          raise ValueError(f"symmetry: unknown {kind} {name!r}")
        if name in seen:
          raise ValueError(f"symmetry: repeated {kind} {name!r}")
        seen.add(name)
    return seen

  for entry in config.joints:
    if type(entry.scale) not in (float, int) or entry.scale not in (-1, 1):
      raise ValueError("symmetry: each joint scale must be +1 or -1")
  covered = check_pairs(((e.name_a, e.name_b) for e in config.joints), joints, "joint")
  if missing := joints - covered:
    raise ValueError(
      f"symmetry: choose mirror partners for {', '.join(sorted(missing))}"
    )
  check_pairs(config.body_pairs, bodies, "body")
