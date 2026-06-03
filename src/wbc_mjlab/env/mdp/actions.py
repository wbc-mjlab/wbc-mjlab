"""Tracking-specific action terms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from mjlab.actuator.actuator import TransmissionType
from mjlab.envs.mdp.actions.actions import BaseAction, BaseActionCfg
from mjlab.managers.action_manager import ActionTermCfg

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
