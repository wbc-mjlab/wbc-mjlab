"""G1 Unitree actuators: motor model, PD gains, and articulation groups."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

import torch

from mjlab.actuator.actuator import ActuatorCmd
from mjlab.actuator.pd_actuator import IdealPdActuator, IdealPdActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg
from mjlab.utils.actuator import reflected_inertia_from_two_stage_planetary

from wbc_mjlab.actuation.envelope import (
  TorqueSpeedEnvelope,
  TorqueSpeedEnvelopeTensors,
  clip_unitree_effort,
)

if TYPE_CHECKING:
  from mjlab.entity import Entity

UnitreeCfgT = TypeVar("UnitreeCfgT", bound="UnitreeActuatorCfg")

_NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10 Hz
_DAMPING_RATIO = 2.0

# Policy→motor command delay (physics steps). Disabled by default (OmniXtreme [0, 0]).
_DELAY_ENABLED = False
_DELAY_MIN_LAG = 0
_DELAY_MAX_LAG = 3  # one policy period at decimation=4
_DELAY_UPDATE_PERIOD = 4

_DELAY_KWARGS: dict[str, int] = (
  {
    "delay_min_lag": _DELAY_MIN_LAG,
    "delay_max_lag": _DELAY_MAX_LAG,
    "delay_update_period": _DELAY_UPDATE_PERIOD,
  }
  if _DELAY_ENABLED
  else {}
)


@dataclass(frozen=True)
class UnitreeMotorPreset:
  """Motor SKU parameters from unitree_rl_lab ``unitree_actuators.py``."""

  x1: float
  x2: float
  y1: float
  y2: float
  fs: float = 0.0
  fd: float = 0.0
  va: float = 0.01


N7520_22P5 = UnitreeMotorPreset(14.5, 22.7, 111.0, 131.0, fs=2.4, fd=0.24)
N7520_14P3 = UnitreeMotorPreset(22.63, 35.52, 71.0, 83.3, fs=1.6, fd=0.16)
N5020_16 = UnitreeMotorPreset(30.86, 40.13, 24.8, 31.9, fs=0.6, fd=0.06)
W4010_25 = UnitreeMotorPreset(15.3, 24.76, 4.8, 8.6, fs=0.6, fd=0.06)


def _armature(rotor_inertias: tuple[float, ...], gears: tuple[float, ...]) -> float:
  return reflected_inertia_from_two_stage_planetary(rotor_inertias, gears)


def _pd_gains(armature: float) -> tuple[float, float]:
  stiffness = armature * _NATURAL_FREQ**2
  damping = 2.0 * _DAMPING_RATIO * armature * _NATURAL_FREQ
  return stiffness, damping


_ARMATURE_5020 = _armature((0.139e-4, 0.017e-4, 0.169e-4), (1, 1 + 46 / 18, 1 + 56 / 16))
_ARMATURE_7520_14 = _armature(
  (0.489e-4, 0.098e-4, 0.533e-4), (1, 4.5, 1 + 48 / 22)
)
_ARMATURE_7520_22 = _armature((0.489e-4, 0.109e-4, 0.738e-4), (1, 4.5, 5))
_ARMATURE_4010 = _armature((0.068e-4, 0.0, 0.0), (1, 5, 5))


@dataclass(kw_only=True)
class UnitreeActuatorCfg(IdealPdActuatorCfg):
  """Ideal PD actuator with Unitree torque–speed limits and friction."""

  x1: float
  x2: float
  y1: float
  y2: float | None = None
  fs: float = 0.0
  fd: float = 0.0
  va: float = 0.01

  def __post_init__(self) -> None:
    if self.y2 is None:
      self.y2 = self.y1
    super().__post_init__()

  @property
  def envelope(self) -> TorqueSpeedEnvelope:
    assert self.y2 is not None
    return TorqueSpeedEnvelope(self.x1, self.x2, self.y1, self.y2)

  def build(
    self, entity: Entity, target_ids: list[int], target_names: list[str]
  ) -> UnitreeActuator:
    return UnitreeActuator(self, entity, target_ids, target_names)


def _actuator_cfg(
  preset: UnitreeMotorPreset,
  *,
  target_names_expr: tuple[str, ...],
  armature: float,
  torque_scale: float = 1.0,
) -> UnitreeActuatorCfg:
  stiffness, damping = _pd_gains(armature)
  scale = torque_scale
  return UnitreeActuatorCfg(
    target_names_expr=target_names_expr,
    stiffness=stiffness,
    damping=damping,
    effort_limit=float("inf"),
    armature=armature,
    x1=preset.x1,
    x2=preset.x2,
    y1=preset.y1 * scale,
    y2=preset.y2 * scale,
    fs=preset.fs * scale,
    fd=preset.fd * scale,
    va=preset.va,
    **_DELAY_KWARGS,
  )


class UnitreeActuator(IdealPdActuator[UnitreeCfgT], Generic[UnitreeCfgT]):
  """PD actuator with Unitree torque–speed envelope clipping and friction."""

  def __init__(
    self,
    cfg: UnitreeCfgT,
    entity: Entity,
    target_ids: list[int],
    target_names: list[str],
  ) -> None:
    super().__init__(cfg, entity, target_ids, target_names)
    self._joint_vel: torch.Tensor | None = None
    self._envelope: TorqueSpeedEnvelopeTensors | None = None
    self._friction_static: torch.Tensor | None = None
    self._friction_dynamic: torch.Tensor | None = None
    self._activation_vel: torch.Tensor | None = None

  def initialize(
    self,
    mj_model,
    model,
    data,
    device: str,
  ) -> None:
    super().initialize(mj_model, model, data, device)
    num_envs = data.nworld
    num_joints = len(self._target_names)
    cfg = self.cfg

    self._joint_vel = torch.zeros(num_envs, num_joints, device=device)
    self._envelope = TorqueSpeedEnvelopeTensors(
      x1=torch.full((num_envs, num_joints), cfg.x1, device=device),
      x2=torch.full((num_envs, num_joints), cfg.x2, device=device),
      y1=torch.full((num_envs, num_joints), cfg.y1, device=device),
      y2=torch.full((num_envs, num_joints), cfg.y2, device=device),
    )
    self._friction_static = torch.full((num_envs, num_joints), cfg.fs, device=device)
    self._friction_dynamic = torch.full((num_envs, num_joints), cfg.fd, device=device)
    self._activation_vel = torch.full((num_envs, num_joints), cfg.va, device=device)

  def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
    assert self._joint_vel is not None
    self._joint_vel[:] = cmd.vel
    effort = super().compute(cmd)
    assert self._friction_static is not None
    assert self._friction_dynamic is not None
    assert self._activation_vel is not None
    return effort - (
      self._friction_static * torch.tanh(cmd.vel / self._activation_vel)
      + self._friction_dynamic * cmd.vel
    )

  def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
    assert self._joint_vel is not None
    assert self._envelope is not None
    return clip_unitree_effort(effort, self._joint_vel, self._envelope)


##
# G1 joint groups (order matters for ``G1_ACTUATOR_GROUPS`` envelope lookup).
##

G1_ACTUATOR_5020 = _actuator_cfg(
  N5020_16,
  target_names_expr=(
    ".*_elbow_joint",
    ".*_shoulder_pitch_joint",
    ".*_shoulder_roll_joint",
    ".*_shoulder_yaw_joint",
    ".*_wrist_roll_joint",
  ),
  armature=_ARMATURE_5020,
)
G1_ACTUATOR_7520_14 = _actuator_cfg(
  N7520_14P3,
  target_names_expr=(".*_hip_pitch_joint", ".*_hip_yaw_joint", "waist_yaw_joint"),
  armature=_ARMATURE_7520_14,
)
G1_ACTUATOR_7520_22 = _actuator_cfg(
  N7520_22P5,
  target_names_expr=(".*_hip_roll_joint", ".*_knee_joint"),
  armature=_ARMATURE_7520_22,
)
G1_ACTUATOR_4010 = _actuator_cfg(
  W4010_25,
  target_names_expr=(".*_wrist_pitch_joint", ".*_wrist_yaw_joint"),
  armature=_ARMATURE_4010,
)

# Waist pitch/roll and ankles: parallel 4-bar with two 5020 motors (nominal 2× torque/armature).
_DUAL_5020_ARMATURE = _ARMATURE_5020 * 2
G1_ACTUATOR_WAIST = _actuator_cfg(
  N5020_16,
  target_names_expr=("waist_pitch_joint", "waist_roll_joint"),
  armature=_DUAL_5020_ARMATURE,
  torque_scale=2.0,
)
G1_ACTUATOR_ANKLE = _actuator_cfg(
  N5020_16,
  target_names_expr=(".*_ankle_pitch_joint", ".*_ankle_roll_joint"),
  armature=_DUAL_5020_ARMATURE,
  torque_scale=2.0,
)

G1_ACTUATOR_GROUPS: tuple[UnitreeActuatorCfg, ...] = (
  G1_ACTUATOR_7520_22,
  G1_ACTUATOR_7520_14,
  G1_ACTUATOR_5020,
  G1_ACTUATOR_4010,
  G1_ACTUATOR_WAIST,
  G1_ACTUATOR_ANKLE,
)

G1_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    G1_ACTUATOR_5020,
    G1_ACTUATOR_7520_14,
    G1_ACTUATOR_7520_22,
    G1_ACTUATOR_4010,
    G1_ACTUATOR_WAIST,
    G1_ACTUATOR_ANKLE,
  ),
  soft_joint_pos_limit_factor=0.9,
)

G1_ACTION_SCALE: dict[str, float] = {}
for _a in G1_ARTICULATION.actuators:
  assert isinstance(_a, UnitreeActuatorCfg)
  _effort = _a.y2
  _stiffness = _a.stiffness
  for _name in _a.target_names_expr:
    G1_ACTION_SCALE[_name] = 0.25 * _effort / _stiffness
