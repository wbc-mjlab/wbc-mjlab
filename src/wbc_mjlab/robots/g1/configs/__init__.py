"""G1 WBC env config builders (one module per method)."""

from __future__ import annotations

from collections.abc import Callable

from mjlab.envs import ManagerBasedRlEnvCfg

from wbc_mjlab.env.mdp.commands import MotionCommandCfg
from wbc_mjlab.robots.g1.configs.binary_failure import g1_wbc_binary_failure_env_cfg
from wbc_mjlab.robots.g1.configs.nose import g1_wbc_nose_env_cfg
from wbc_mjlab.robots.g1.configs.wbc import g1_wbc_env_cfg
from wbc_mjlab.robots.g1.configs.zest import g1_wbc_zest_env_cfg

G1EnvCfgBuilder = Callable[[], ManagerBasedRlEnvCfg]

G1_ENV_CFG_BUILDERS: dict[str, G1EnvCfgBuilder] = {
  "default": g1_wbc_env_cfg,
  "wbc": g1_wbc_env_cfg,
  "nose": g1_wbc_nose_env_cfg,
  "zest": g1_wbc_zest_env_cfg,
  "binary_failure": g1_wbc_binary_failure_env_cfg,
}


def make_g1_wbc_env_cfg(
  *,
  play: bool = False,
  env_cfg: str = "default",
  **kwargs,
) -> ManagerBasedRlEnvCfg:
  if kwargs:
    unknown = ", ".join(sorted(kwargs))
    raise TypeError(
      f"Unknown env cfg kwargs for G1: {unknown}. "
      f"Add a builder under robots/g1/configs/ or pass env_cfg=<name>."
    )
  try:
    cfg = G1_ENV_CFG_BUILDERS[env_cfg]()
  except KeyError as exc:
    known = ", ".join(sorted(G1_ENV_CFG_BUILDERS))
    raise KeyError(f"Unknown G1 env cfg {env_cfg!r}. Known: {known}") from exc

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
    motion_cmd.assistive_wrench_enabled = False
    if "assistive_wrench" in cfg.events:
      cfg.events["assistive_wrench"].params["enabled"] = False

  return cfg


__all__ = [
  "G1EnvCfgBuilder",
  "G1_ENV_CFG_BUILDERS",
  "g1_wbc_binary_failure_env_cfg",
  "g1_wbc_env_cfg",
  "g1_wbc_nose_env_cfg",
  "g1_wbc_zest_env_cfg",
  "make_g1_wbc_env_cfg",
]
