"""Default WBC env config."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg

from wbc_mjlab.env.mdp.commands import (
  MotionCommandCfg,
  wbc_joint_only_similarity_terms,
)
from wbc_mjlab.robots.g1.configs.base import g1_base_cfg


def g1_wbc_env_cfg() -> ManagerBasedRlEnvCfg:
  """Zest-style RSI + assistive wrench, deploy-style obs, actor history length 10."""
  cfg = g1_base_cfg()
  rw = cfg.rewards
  rw["motion_global_root_pos"].weight = 0.5
  rw["motion_global_root_ori"].weight = 1.0
  rw["motion_root_lin_vel_b"].weight = 1.0
  rw["motion_root_ang_vel_b"].weight = 1.0
  rw["motion_body_pos"].weight = 1.5
  rw["motion_body_ori"].weight = 1.0
  rw["motion_body_lin_vel"].weight = 1.0
  rw["motion_body_ang_vel"].weight = 1.0
  rw["motion_joint_pos"].weight = 1.0
  rw["motion_joint_vel"].weight = 0.5
  rw["action_rate_l1"].weight = -0.1
  rw["joint_acc"].weight = -5.0e-6
  rw["survival"].weight = 1.0
  rw["foot_slip"].weight = -0.0

  cfg.observations["actor"].history_length = 1
  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.adaptive_sampling_strategy = "similarity_ema"
  motion_cmd.adaptive_similarity_terms = wbc_joint_only_similarity_terms()
  motion_cmd.adaptive_bin_width_s = 4.0
  motion_cmd.assistive_wrench_enabled = True

  actor = cfg.observations["actor"]
  for key in ("motion_anchor_pos_b", "base_lin_vel", "ref_joint_vel"):
    actor.terms.pop(key, None)

  return cfg
