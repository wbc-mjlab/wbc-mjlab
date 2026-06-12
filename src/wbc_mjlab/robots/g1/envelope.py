"""G1 torque–speed envelope lookup for MDP terms."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import torch

from wbc_mjlab.actuation.envelope import (
  TorqueSpeedEnvelope,
  TorqueSpeedEnvelopeTensors,
  build_envelope_tensors,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.scene_entity_config import SceneEntityCfg

_ENVELOPE_CACHE: dict[tuple[str, str], TorqueSpeedEnvelopeTensors] = {}


def g1_torque_speed_envelope_for_joint(joint_name: str) -> TorqueSpeedEnvelope:
  from wbc_mjlab.robots.g1.actuators import G1_ACTUATOR_GROUPS

  for actuator_cfg in G1_ACTUATOR_GROUPS:
    for pattern in actuator_cfg.target_names_expr:
      if re.fullmatch(pattern, joint_name):
        return actuator_cfg.envelope
  raise KeyError(f"No G1 torque-speed envelope for joint {joint_name!r}")


def build_g1_joint_envelope_tensors(
  joint_names: tuple[str, ...] | list[str],
  *,
  device: torch.device | str,
  dtype: torch.dtype = torch.float32,
) -> TorqueSpeedEnvelopeTensors:
  """Per-joint envelope tensors (matches ``UnitreeActuator`` sim limits)."""
  return build_envelope_tensors(
    [g1_torque_speed_envelope_for_joint(n) for n in joint_names],
    device=device,
    dtype=dtype,
  )


def g1_joint_envelope_tensors(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> TorqueSpeedEnvelopeTensors:
  """Cached per-joint envelope tensors for the robot in *env*."""
  key = (asset_cfg.name, str(env.device))
  if key not in _ENVELOPE_CACHE:
    from mjlab.entity import Entity

    asset: Entity = env.scene[asset_cfg.name]
    _ENVELOPE_CACHE[key] = build_g1_joint_envelope_tensors(
      asset.joint_names, device=env.device
    )
  return _ENVELOPE_CACHE[key]
