"""WBC task config: one mjlab task id bound to an env builder."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from mjlab.envs import ManagerBasedRlEnvCfg

from wbc_mjlab.robots.ids import RobotId

EnvCfgBuilder = Callable[[], ManagerBasedRlEnvCfg]


@dataclass(frozen=True)
class WbcTaskConfig:
  """One registered mjlab task: metadata + env builder."""

  task_id: str
  robot_id: RobotId
  description: str
  experiment_name: str
  build_env_cfg: EnvCfgBuilder
