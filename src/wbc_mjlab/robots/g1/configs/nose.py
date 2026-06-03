"""Deploy-style env config (no state estimation, no obs history)."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg

from wbc_mjlab.robots.g1.configs.zest import g1_wbc_zest_env_cfg


def g1_wbc_nose_env_cfg() -> ManagerBasedRlEnvCfg:
  """Same obs and RSI as Zest (no history) for deploy export."""
  return g1_wbc_zest_env_cfg()
