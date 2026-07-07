"""Robot left-right symmetry configs for motion mirroring."""

from __future__ import annotations

from dataclasses import dataclass

from wbc_mjlab.robots.ids import resolve_robot_id


@dataclass(frozen=True)
class JointSymmetryEntry:
  """One mirrored joint pair (or a central joint reflected onto itself)."""

  name_a: str
  name_b: str
  scale: float
  """Sign applied after the swap (+1 pitch-like, -1 roll/yaw-like)."""


@dataclass(frozen=True)
class RobotSymmetryConfig:
  """Sagittal-plane mirror spec for a robot asset."""

  joints: tuple[JointSymmetryEntry, ...]
  mirror_suffix: str = "_mirror"
  """Appended to the source clip stem for mirrored NPZ exports."""


_REGISTRY: dict[str, RobotSymmetryConfig] = {}


def register_robot_symmetry_config(robot_id: str, config: RobotSymmetryConfig) -> None:
  rid = robot_id.strip().lower()
  _REGISTRY[rid] = config


def get_robot_symmetry_config(robot_id: str) -> RobotSymmetryConfig | None:
  rid = resolve_robot_id(robot_id)
  return _REGISTRY.get(rid)
