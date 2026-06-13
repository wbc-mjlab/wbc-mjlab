"""Default WBC env config (deploy-style tracking, standalone from Zest)."""

from __future__ import annotations

from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg

from wbc_mjlab.env.mdp.commands import MotionCommandCfg
from wbc_mjlab.env.mdp.sampling import joint_pos_similarity_preset
from wbc_mjlab.robots.g1.configs.base import g1_base_cfg

# Zest Table S4 joint kernel: exp(-κ‖e‖²/σ²) with κ=1/4, σ=0.3 per DoF.
_JOINT_TRACKING_KAPPA = 0.25
_JOINT_TRACKING_SIGMA = 0.4


def g1_wbc_env_cfg() -> ManagerBasedRlEnvCfg:
  """Joint-only RSI, assistive wrench, deploy-style actor obs, mjlab tracking rewards."""
  cfg = g1_base_cfg()
  rw = cfg.rewards

  rw["motion_global_root_pos"].weight = 0.5
  rw["motion_global_root_pos"].params["std"] = 0.5
  rw["motion_global_root_ori"].weight = 1.0
  rw["motion_global_root_ori"].params["std"] = 0.4
  rw["motion_root_lin_vel_b"].weight = 1.0
  rw["motion_root_lin_vel_b"].params["std"] = 1.0
  rw["motion_root_ang_vel_b"].weight = 1.0
  rw["motion_root_ang_vel_b"].params["std"] = 1.5
  rw["motion_body_pos"].weight = 1.5
  rw["motion_body_pos"].params["std"] = 0.2
  rw["motion_body_ori"].weight = 1.0
  rw["motion_body_ori"].params["std"] = 0.4
  rw["motion_body_lin_vel"].weight = 1.0
  rw["motion_body_lin_vel"].params["std"] = 1.0
  rw["motion_body_ang_vel"].weight = 1.0
  rw["motion_body_ang_vel"].params["std"] = 3.14
  rw["motion_joint_pos"].weight = 1.0
  rw["motion_joint_pos"].params["std"] = 0.5
  rw["motion_joint_pos"].params.pop("per_joint", None)
  rw["motion_joint_pos"].params["sigma_per_joint"] = _JOINT_TRACKING_SIGMA
  rw["motion_joint_pos"].params["kappa"] = _JOINT_TRACKING_KAPPA
  rw["motion_joint_vel"].weight = 0.5
  rw["motion_joint_vel"].params["std"] = 2.0

  rw["action_rate_l1"].weight = -0.12
  rw["joint_acc"].weight = -5.0e-6
  rw["survival"].weight = 1.0
  rw["foot_slip"].weight = 0.0
  rw["joint_limit"].weight = 0.0
  rw["actuator_torque_soft_limit"].weight = 0.0
  rw["neg_regen_power"].weight = 0.0
  rw["anti_shake"].weight = 0.0
  rw["angular_momentum"].weight = 0.0

  cfg.observations["actor"].history_length = 1

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.assistive_wrench_enabled = True
  motion_cmd.rsi = replace(
    motion_cmd.rsi,
    strategy="similarity_ema",
    similarity_terms=joint_pos_similarity_preset(),
    similarity_from_rewards=False,
    bin_width_s=4.0,
    min_bin_span_ratio=0.0,
    similarity_norm_by_remaining_clip=False,
    persist_failure_levels=False,
  )

  actor = cfg.observations["actor"]
  for key in ("motion_anchor_pos_b", "base_lin_vel", "ref_joint_vel"):
    actor.terms.pop(key, None)

  return cfg
