"""Build WBC env / RL configs for a physical robot (used by task registration)."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from wbc_mjlab.robots.ids import RobotId, resolve_robot_id

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnvCfg
  from mjlab.rl import RslRlOnPolicyRunnerCfg

EnvCfgBuilder = Callable[..., Any]
RlCfgBuilder = Callable[[], Any]

_ENV_BUILDERS: dict[str, EnvCfgBuilder] = {}
_RL_BUILDERS: dict[str, RlCfgBuilder] = {}
_G1_REGISTERED = False


def _register_g1() -> None:
  global _G1_REGISTERED
  if _G1_REGISTERED:
    return

  from wbc_mjlab.robots.g1.rl_cfg import g1_wbc_rl_cfg
  from wbc_mjlab.robots.g1.tasks import make_g1_wbc_env_cfg

  _ENV_BUILDERS["g1"] = make_g1_wbc_env_cfg
  _RL_BUILDERS["g1"] = g1_wbc_rl_cfg
  _G1_REGISTERED = True


def register_robot_builders(
  robot_id: str,
  make_env_cfg: EnvCfgBuilder,
  make_rl_cfg: RlCfgBuilder,
) -> None:
  """Register env/RL config builders for an external or in-tree robot.

  Prefer :func:`wbc_mjlab.extension.register_wbc_extension` for the full
  robot + task registration path.
  """
  rid = robot_id.strip().lower()
  _ENV_BUILDERS[rid] = make_env_cfg
  _RL_BUILDERS[rid] = make_rl_cfg


def _ensure_builders() -> None:
  _register_g1()


def make_wbc_env_cfg(
  robot_id: str | RobotId = "g1",
  *,
  play: bool = False,
  task_id: str = "Wbc-G1",
  **kwargs,
) -> ManagerBasedRlEnvCfg:
  """Build a WBC env config for a registered robot.

  Args:
    robot_id: Registered robot id (default ``\"g1\"``).
    play: If True, disable train-only DR / assistive wrench (play mode).
    task_id: Task id used by the robot builder to select the preset stack.
    **kwargs: Forwarded to the robot's env builder.
  """
  _ensure_builders()
  rid = resolve_robot_id(robot_id) if isinstance(robot_id, str) else robot_id
  try:
    return _ENV_BUILDERS[rid](play=play, task_id=task_id, **kwargs)
  except KeyError as exc:
    raise KeyError(f"No WBC env builder for robot {rid!r}") from exc


def make_wbc_rl_cfg(robot_id: str | RobotId = "g1") -> RslRlOnPolicyRunnerCfg:
  """Build the RSL-RL runner config for a registered robot."""
  _ensure_builders()
  rid = resolve_robot_id(robot_id) if isinstance(robot_id, str) else robot_id
  try:
    return _RL_BUILDERS[rid]()
  except KeyError as exc:
    raise KeyError(f"No WBC RL config for robot {rid!r}") from exc
