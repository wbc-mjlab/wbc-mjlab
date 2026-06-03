"""Shared WBC task preset type (one mjlab task id per preset)."""

from __future__ import annotations

from dataclasses import dataclass

from wbc_mjlab.robots.ids import RobotId


@dataclass(frozen=True)
class WbcTaskPreset:
  """One registered mjlab task: env config name + separate log experiment name."""

  task_id: str
  robot_id: RobotId
  description: str
  experiment_name: str
  env_cfg: str = "default"

  def env_kwargs(self) -> dict:
    return {"env_cfg": self.env_cfg}
