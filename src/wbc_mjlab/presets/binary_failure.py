"""BeyondMimic-style preset (binary-failure adaptive RSI)."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg

from wbc_mjlab.env.mdp.commands import MotionCommandCfg
from wbc_mjlab.env.mdp.sampling import keybody_similarity_preset


def apply_binary_failure(cfg: ManagerBasedRlEnvCfg) -> None:
  """Full obs, binary failure RSI, assistive wrench."""
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
  motion_cmd.assistive_wrench_enabled = True
  motion_cmd.rsi.strategy = "binary_failure"
  motion_cmd.rsi.similarity_terms = keybody_similarity_preset()
  motion_cmd.rsi.bin_width_s = 4.0
