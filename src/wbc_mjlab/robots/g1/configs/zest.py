"""Zest-style env config."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg

from wbc_mjlab.env.mdp.commands import (
  MotionCommandCfg,
  wbc_joint_only_similarity_terms,
)
from wbc_mjlab.robots.g1.configs.base import g1_base_cfg, without_state_estimation


def g1_wbc_zest_env_cfg() -> ManagerBasedRlEnvCfg:
  """No SE, joint-only similarity RSI, adaptive bins, assistive wrench."""
  cfg = g1_base_cfg()
  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.adaptive_sampling_strategy = "similarity_ema"
  motion_cmd.adaptive_similarity_terms = wbc_joint_only_similarity_terms()
  motion_cmd.adaptive_bin_width_s = 4.0
  motion_cmd.assistive_wrench_enabled = True
  without_state_estimation(cfg)
  return cfg
