"""BeyondMimic-style env config."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.termination_manager import TerminationTermCfg

import wbc_mjlab.env.mdp as mdp
from wbc_mjlab.env.mdp.commands import (
  MotionCommandCfg,
  whole_body_adaptive_similarity_terms,
)
from wbc_mjlab.robots.g1.configs.base import (
  G1_EE_TERMINATION_BODY_NAMES,
  g1_base_cfg,
)


def g1_wbc_binary_failure_env_cfg() -> ManagerBasedRlEnvCfg:
  """Full obs, binary failure RSI, assistive wrench."""
  cfg = g1_base_cfg()
  rw = cfg.rewards
  rw["motion_global_root_pos"].weight = 0.5
  rw["motion_global_root_ori"].weight = 0.5
  rw["motion_body_pos"].weight = 1.0
  rw["motion_body_ori"].weight = 1.0
  rw["motion_body_lin_vel"].weight = 1.0
  rw["motion_body_ang_vel"].weight = 1.0
  rw["motion_joint_pos"].weight = 0.0
  rw["motion_joint_vel"].weight = 0.0
  rw["action_rate_l1"].weight = -0.1
  rw["joint_acc"].weight = -2.0e-6
  rw["survival"].weight = 0.0
  rw["foot_slip"].weight = 0.0

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.adaptive_sampling_strategy = "binary_failure"
  motion_cmd.adaptive_similarity_terms = whole_body_adaptive_similarity_terms()
  motion_cmd.adaptive_bin_width_s = 4.0
  motion_cmd.assistive_wrench_enabled = True

  cfg.terminations["ee_body_pos"] = TerminationTermCfg(
    func=mdp.bad_motion_body_pos_z_only,
    params={
      "command_name": "motion",
      "threshold": 0.25,
      "body_names": G1_EE_TERMINATION_BODY_NAMES,
    },
  )
  return cfg
