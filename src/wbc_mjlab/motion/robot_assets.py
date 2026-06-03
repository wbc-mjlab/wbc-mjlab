"""Robot assets for offline motion conversion (scene + debias metadata)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import mujoco
import numpy as np
from mjlab.scene import SceneCfg

from wbc_mjlab.robots.ids import RobotId, resolve_robot_id


@dataclass(frozen=True)
class RobotMotionSpec:
  """Scene + optional foot metadata for motion NPZ conversion."""

  scene_cfg_fn: Callable[[], SceneCfg]
  actuated_joint_names: tuple[str, ...] | None = None
  foot_body_names: tuple[str, ...] | None = None
  foot_sole_z: float | None = None


def _g1_scene() -> SceneCfg:
  from wbc_mjlab.robots.env import make_wbc_env_cfg

  return make_wbc_env_cfg("g1").scene


def _g1_motion_spec() -> RobotMotionSpec:
  from wbc_mjlab.robots.g1.constants import (
    MOTION_Z_DEBIAS_FOOT_BODY_NAMES,
    MOTION_Z_DEBIAS_FOOT_SOLE_Z,
  )

  return RobotMotionSpec(
    scene_cfg_fn=_g1_scene,
    actuated_joint_names=None,
    foot_body_names=MOTION_Z_DEBIAS_FOOT_BODY_NAMES,
    foot_sole_z=MOTION_Z_DEBIAS_FOOT_SOLE_Z,
  )


_ROBOT_SPECS: dict[RobotId, RobotMotionSpec] = {
  "g1": _g1_motion_spec(),
}


def get_robot_motion_spec(name: str) -> tuple[RobotId, RobotMotionSpec]:
  robot_id = resolve_robot_id(name)
  return robot_id, _ROBOT_SPECS[robot_id]


def conversion_scene_cfg(motion_spec: RobotMotionSpec) -> SceneCfg:
  scene_cfg = motion_spec.scene_cfg_fn()
  scene_cfg.num_envs = 1
  return scene_cfg


def strip_entity_prefix(joint_name: str, entity_name: str = "robot") -> str:
  prefix = f"{entity_name}/"
  return joint_name[len(prefix) :] if joint_name.startswith(prefix) else joint_name


def normalize_joint_name_list(
  names: list[str] | tuple[str, ...],
  *,
  entity_name: str = "robot",
) -> list[str]:
  out: list[str] = []
  for raw in names:
    name = strip_entity_prefix(str(raw), entity_name=entity_name)
    if name in ("floating_base_joint", "freejoint", "root_joint"):
      continue
    out.append(name)
  return out


def qpos_dof_joint_names(
  model: mujoco.MjModel, *, entity_name: str = "robot"
) -> list[str]:
  names: list[str] = []
  for j in range(model.njnt):
    jtype = int(model.jnt_type[j])
    if jtype == mujoco.mjtJoint.mjJNT_FREE:
      continue
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
    if name is None:
      raise RuntimeError(f"Unnamed joint at index {j}")
    names.append(strip_entity_prefix(name, entity_name=entity_name))
  expected = model.nq - 7
  if len(names) != expected:
    expanded: list[str] = []
    for j in range(model.njnt):
      jtype = int(model.jnt_type[j])
      if jtype == mujoco.mjtJoint.mjJNT_FREE:
        continue
      name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
      adr = int(model.jnt_qposadr[j])
      if j + 1 < model.njnt:
        next_adr = int(model.jnt_qposadr[j + 1])
      else:
        next_adr = model.nq
      width = next_adr - adr
      plain = strip_entity_prefix(name, entity_name=entity_name)
      if width == 1:
        expanded.append(plain)
      else:
        for k in range(width):
          expanded.append(f"{plain}__qpos{k}")
    if len(expanded) != expected:
      raise RuntimeError(
        f"Could not derive {expected} qpos tail names from model "
        f"(got {len(expanded)} expanded, {len(names)} joints)."
      )
    return expanded
  return names


def remap_dof_columns(
  dof_pos: np.ndarray,
  source_names: list[str],
  target_names: list[str],
) -> np.ndarray:
  if list(source_names) == list(target_names):
    return dof_pos
  index = [source_names.index(n) for n in target_names]
  return dof_pos[:, index]


def resolve_dof_joint_names(
  model: mujoco.MjModel,
  dof_dim: int,
  *,
  source_joint_names: list[str] | None = None,
  entity_name: str = "robot",
) -> tuple[list[str], list[str] | None]:
  model_names = qpos_dof_joint_names(model, entity_name=entity_name)
  if dof_dim != len(model_names):
    raise ValueError(
      f"Motion DOF width {dof_dim} != model qpos tail {len(model_names)} "
      f"(model.nq={model.nq}). Check --robot matches the retargeted asset."
    )
  if source_joint_names is None:
    return model_names, None
  names = normalize_joint_name_list(source_joint_names, entity_name=entity_name)
  if len(names) != dof_dim:
    raise ValueError(
      f"Source joint name list length {len(names)} != motion DOF width {dof_dim}"
    )
  return model_names, names


def peek_csv_dof_width(
  csv_path: Path,
  line_range: tuple[int, int] | None = None,
) -> int:
  if line_range is None:
    row = np.loadtxt(csv_path, delimiter=",", max_rows=1)
  else:
    row = np.loadtxt(
      csv_path,
      delimiter=",",
      skiprows=line_range[0] - 1,
      maxrows=1,
    )
  row = np.atleast_1d(np.asarray(row, dtype=np.float64))
  if row.size < 8:
    raise ValueError(
      f"CSV {csv_path} must have at least 8 columns (root pos/rot + joints), "
      f"got {row.size}"
    )
  return int(row.size - 7)
