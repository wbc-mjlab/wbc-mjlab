"""BeyondMimic-style env config."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg

from wbc_mjlab.env.mdp.commands import (
  MotionCommandCfg,
  whole_body_adaptive_similarity_terms,
)
from wbc_mjlab.robots.g1.configs.base import g1_base_cfg


def g1_wbc_binary_failure_env_cfg() -> ManagerBasedRlEnvCfg:
  """Full obs, binary failure RSI, assistive wrench."""
  cfg = g1_base_cfg()
  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.adaptive_sampling_strategy = "binary_failure"
  motion_cmd.adaptive_similarity_terms = whole_body_adaptive_similarity_terms()
  motion_cmd.adaptive_bin_width_s = 4.0
  motion_cmd.assistive_wrench_enabled = True
  return cfg
