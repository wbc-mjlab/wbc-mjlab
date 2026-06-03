"""Build WBC env / RL configs for a physical robot (used by task preset registration)."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from wbc_mjlab.robots.ids import RobotId, resolve_robot_id

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnvCfg
  from mjlab.rl import RslRlOnPolicyRunnerCfg

EnvCfgBuilder = Callable[..., Any]
RlCfgBuilder = Callable[[], Any]

_ENV_BUILDERS: dict[RobotId, EnvCfgBuilder] = {}
_RL_BUILDERS: dict[RobotId, RlCfgBuilder] = {}


def _register_g1() -> None:
  from wbc_mjlab.robots.g1.configs import make_g1_wbc_env_cfg
  from wbc_mjlab.robots.g1.rl_cfg import g1_wbc_rl_cfg

  _ENV_BUILDERS["g1"] = make_g1_wbc_env_cfg
  _RL_BUILDERS["g1"] = g1_wbc_rl_cfg


def _ensure_builders() -> None:
  if _ENV_BUILDERS:
    return
  _register_g1()


def make_wbc_env_cfg(
  robot_id: str | RobotId = "g1",
  *,
  play: bool = False,
  **kwargs,
) -> ManagerBasedRlEnvCfg:
  _ensure_builders()
  rid = resolve_robot_id(robot_id) if isinstance(robot_id, str) else robot_id
  try:
    return _ENV_BUILDERS[rid](play=play, **kwargs)
  except KeyError as exc:
    raise KeyError(f"No WBC env builder for robot {rid!r}") from exc


def make_wbc_rl_cfg(robot_id: str | RobotId = "g1") -> RslRlOnPolicyRunnerCfg:
  _ensure_builders()
  rid = resolve_robot_id(robot_id) if isinstance(robot_id, str) else robot_id
  try:
    return _RL_BUILDERS[rid]()
  except KeyError as exc:
    raise KeyError(f"No WBC RL config for robot {rid!r}") from exc
