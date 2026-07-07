"""Public API for external WBC robot + task extensions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wbc_mjlab.motion.robot_assets import RobotMotionSpec
from wbc_mjlab.robots.symmetry import RobotSymmetryConfig
from wbc_mjlab.tasks.config import WbcTaskConfig

EnvCfgBuilder = Callable[..., Any]
RlCfgBuilder = Callable[[], Any]


@dataclass(frozen=True)
class WbcRobotSpec:
  """Robot wiring registered into wbc-mjlab from an external package."""

  robot_id: str
  """Canonical robot id (e.g. ``\"h2\"``)."""
  make_env_cfg: EnvCfgBuilder
  """Builder returning a ``ManagerBasedRlEnvCfg``."""
  make_rl_cfg: RlCfgBuilder
  """Builder returning the RL runner config."""
  motion_spec: RobotMotionSpec | None = None
  """Optional FK / debias metadata for ``wbc-mjlab-data-to-npz``."""
  symmetry_config: RobotSymmetryConfig | None = None
  """Optional :class:`~wbc_mjlab.robots.symmetry.RobotSymmetryConfig` for ``--mirror``."""
  aliases: tuple[str, ...] = ()
  """Alternate ids accepted by CLIs."""
  project_root: Path | str | None = None
  """Extension package root (for ``data/<robot_id>/`` resolution)."""
  mjcf_path: Path | str | None = None
  """Optional MJCF path when not set on ``motion_spec``."""


def register_robot(spec: WbcRobotSpec) -> None:
  """Register robot id, env/RL builders, and optional motion-conversion metadata.

  Prefer :func:`register_wbc_extension` when also registering tasks.
  """
  from wbc_mjlab.motion.robot_assets import register_robot_motion_spec
  from wbc_mjlab.robots.env import register_robot_builders
  from wbc_mjlab.robots.ids import register_robot_id

  rid = spec.robot_id.strip().lower()
  register_robot_id(rid, aliases=spec.aliases)
  register_robot_builders(rid, spec.make_env_cfg, spec.make_rl_cfg)
  if spec.motion_spec is not None:
    motion_spec = spec.motion_spec
    if spec.mjcf_path is not None and motion_spec.mjcf_path is None:
      from dataclasses import replace

      motion_spec = replace(motion_spec, mjcf_path=Path(spec.mjcf_path))
    register_robot_motion_spec(rid, motion_spec)
  elif spec.mjcf_path is not None:
    from wbc_mjlab.motion.robot_assets import register_robot_mjcf_path

    register_robot_mjcf_path(rid, spec.mjcf_path)
  if spec.symmetry_config is not None:
    from wbc_mjlab.robots.symmetry import register_robot_symmetry_config

    register_robot_symmetry_config(rid, spec.symmetry_config)
  if spec.project_root is not None:
    from wbc_mjlab.data_paths import register_project_root

    register_project_root(spec.project_root)


def register_wbc_extension(
  robot: WbcRobotSpec,
  tasks: WbcTaskConfig | tuple[WbcTaskConfig, ...],
) -> None:
  """Register a robot and its WBC task table in one call.

  Call at extension import time (e.g. from the package ``__init__`` or the
  ``mjlab.tasks`` entry point) so ``wbc-mjlab-list-envs`` / train / play discover
  the robot.
  """
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
