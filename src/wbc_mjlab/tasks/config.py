"""WBC task config: one mjlab task id bound to an env builder."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from wbc_mjlab.robots.ids import RobotId

# Avoid importing mjlab here — tasks.config is loaded while ``wbc_mjlab.tasks``
# initializes, and mjlab loads the ``wbc_mjlab.mjlab_entry`` hook (circular import).
EnvCfgBuilder = Callable[[], Any]


@dataclass(frozen=True)
class WbcTaskConfig:
  """One registered mjlab task: metadata + env builder."""

  task_id: str
  robot_id: RobotId
  description: str
  experiment_name: str
  build_env_cfg: EnvCfgBuilder
