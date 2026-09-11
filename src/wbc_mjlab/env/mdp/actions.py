"""Tracking-specific action terms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.actuator.actuator import TransmissionType
from mjlab.envs.mdp.actions.actions import (
  BaseAction,
  BaseActionCfg,
  JointPositionAction,
  JointPositionActionCfg,
)

from wbc_mjlab.env.mdp.commands import MotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class ReferenceJointPositionActionCfg(BaseActionCfg):
  """Joint position targets as motion reference plus scaled residual actions.

  Implements Zest-style control: q_cmd = q_ref + scale * a_t (minus encoder bias).
  """

  command_name: str = "motion"

  def __post_init__(self):
    self.transmission_type = TransmissionType.JOINT
    if self.offset != 0.0:
      raise ValueError(
        "ReferenceJointPositionActionCfg does not support 'offset'. "
        "Reference joint positions come from the motion command."
      )

  def build(self, env: ManagerBasedRlEnv) -> ReferenceJointPositionAction:
    return ReferenceJointPositionAction(self, env)


class ReferenceJointPositionAction(BaseAction):
  """Control joints via position targets relative to the motion reference."""

  cfg: ReferenceJointPositionActionCfg
  _command: MotionCommand

  def __init__(self, cfg: ReferenceJointPositionActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)
    command = env.command_manager.get_term(cfg.command_name)
    if not isinstance(command, MotionCommand):
      raise TypeError(
        f"ReferenceJointPositionAction requires MotionCommand, "
        f"got {type(command).__name__} for '{cfg.command_name}'."
      )
    self._command = command

  def apply_actions(self) -> None:
    ref_joint_pos = self._command.tracked_joint_pos
    encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
    target = ref_joint_pos + self._processed_actions - encoder_bias
    self._entity.set_joint_position_target(target, joint_ids=self._target_ids)


@dataclass(kw_only=True)
class DefaultOffsetJointPositionActionCfg(JointPositionActionCfg):
  """Default-relative joint targets; seed ``a`` from ``q_ref`` on RSI resample.

  Live command stays deployable: ``q_cmd = q_default + scale · a``.
  After motion resample, ``a`` is set to ``(q_ref - q_default) / scale`` so the
  first PD target matches the clip. Deploy uses the same formula with
  measured joints instead of ``q_ref``.
  """

  command_name: str = "motion"

  def build(self, env: ManagerBasedRlEnv) -> DefaultOffsetJointPositionAction:
    return DefaultOffsetJointPositionAction(self, env)


class DefaultOffsetJointPositionAction(JointPositionAction):
  """``q_cmd = q_default + scale · a`` with reset warm-start from ``q_ref``."""

  cfg: DefaultOffsetJointPositionActionCfg
  _command: MotionCommand

  def __init__(
    self, cfg: DefaultOffsetJointPositionActionCfg, env: ManagerBasedRlEnv
  ):
    super().__init__(cfg=cfg, env=env)
    command = env.command_manager.get_term(cfg.command_name)
    if not isinstance(command, MotionCommand):
      raise TypeError(
        f"DefaultOffsetJointPositionAction requires MotionCommand, "
        f"got {type(command).__name__} for '{cfg.command_name}'."
      )
    self._command = command

  def seed_from_motion_reference(self, env_ids: torch.Tensor) -> None:
    """Set raw actions so ``q_default + scale · a = q_ref`` (resample only)."""
    q_ref = self._command.tracked_joint_pos[env_ids]
    q_default = self._entity.data.default_joint_pos[env_ids][:, self._target_ids]
    scale = self._scale[env_ids] if isinstance(self._scale, torch.Tensor) else self._scale
    raw = (q_ref - q_default) / torch.as_tensor(
      scale, device=q_ref.device, dtype=q_ref.dtype
    ).clamp(min=1.0e-6)
    self._raw_actions[env_ids] = raw
    self._processed_actions[env_ids] = raw * scale + q_default
    self._write_manager_action(env_ids, raw)

  def _write_manager_action(self, env_ids: torch.Tensor, raw: torch.Tensor) -> None:
    manager = self._env.action_manager
    idx = 0
    for term in manager._terms.values():
      width = term.action_dim
      if term is self:
        manager._action[env_ids, idx : idx + width] = raw
        manager._prev_action[env_ids, idx : idx + width] = raw
        return
      idx += width
