"""Public API for external WBC robot + task extensions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from wbc_mjlab.motion.robot_assets import RobotMotionSpec
from wbc_mjlab.tasks.config import WbcTaskConfig

EnvCfgBuilder = Callable[..., Any]
RlCfgBuilder = Callable[[], Any]


@dataclass(frozen=True)
class WbcRobotSpec:
  """Robot wiring registered into wbc-mjlab from an external package."""

  robot_id: str
  make_env_cfg: EnvCfgBuilder
  make_rl_cfg: RlCfgBuilder
  motion_spec: RobotMotionSpec | None = None
  aliases: tuple[str, ...] = ()


def register_robot(spec: WbcRobotSpec) -> None:
  """Register robot id, env/RL builders, and optional motion-conversion metadata."""
  from wbc_mjlab.motion.robot_assets import register_robot_motion_spec
  from wbc_mjlab.robots.env import register_robot_builders
  from wbc_mjlab.robots.ids import register_robot_id

  rid = spec.robot_id.strip().lower()
  register_robot_id(rid, aliases=spec.aliases)
  register_robot_builders(rid, spec.make_env_cfg, spec.make_rl_cfg)
  if spec.motion_spec is not None:
    register_robot_motion_spec(rid, spec.motion_spec)


def register_wbc_extension(
  robot: WbcRobotSpec,
  tasks: WbcTaskConfig | tuple[WbcTaskConfig, ...],
) -> None:
  """One-shot helper: register a robot and its WBC task table."""
  register_robot(robot)
  from wbc_mjlab.tasks import register_wbc_tasks

  register_wbc_tasks(tasks)


__all__ = [
  "EnvCfgBuilder",
  "RlCfgBuilder",
  "WbcRobotSpec",
  "register_robot",
  "register_wbc_extension",
]
