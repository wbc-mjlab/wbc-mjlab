"""Default full-stack WBC env (Zest core + extended tracking and safety rewards)."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.termination_manager import TerminationTermCfg

import wbc_mjlab.env.mdp as mdp
from wbc_mjlab.robots.g1.configs.base import (
  G1_EE_TERMINATION_BODY_NAMES,
  G1_MOTION_BODY_NAMES,
  G1_WRIST_BODY_NAMES,
)
from wbc_mjlab.robots.g1.configs.zest import (
  _apply_tracking_kappa,
  g1_wbc_zest_env_cfg,
)

# Whole-body tracking extras beyond Zest Table S4 (no joint vel term).
_WBC_EXTRA_TRACKING = (
  "motion_body_lin_vel",
  "motion_body_ang_vel",
)


def g1_wbc_env_cfg() -> ManagerBasedRlEnvCfg:
  """Full WBC: Zest RSI + SONIC whole-body tracking + Omni power safety."""
  cfg = g1_wbc_zest_env_cfg()
  rw = cfg.rewards

  # --- Whole-body tracking (14 keybodies) ---
  # Root terms stay at Zest Table S4 values from ``g1_wbc_zest_env_cfg()``.
  rw["motion_body_pos"].weight = 1.0
  rw["motion_body_pos"].params.pop("sigma_per_keybody", None)
  rw["motion_body_pos"].params["std"] = 0.3
  rw["motion_body_pos"].params["body_error_aggregate"] = "mean"
  rw["motion_body_pos"].params["body_names"] = G1_MOTION_BODY_NAMES

  rw["motion_body_ori"].weight = 1.0
  rw["motion_body_ori"].params.pop("sigma_per_keybody", None)
  rw["motion_body_ori"].params["std"] = 0.4
  rw["motion_body_ori"].params["body_error_aggregate"] = "mean"
  rw["motion_body_ori"].params["body_names"] = G1_MOTION_BODY_NAMES

  rw["motion_body_lin_vel"].weight = 1.0
  rw["motion_body_lin_vel"].params["std"] = 1.0
  rw["motion_body_lin_vel"].params["body_names"] = G1_MOTION_BODY_NAMES
  rw["motion_body_ang_vel"].weight = 1.0
  rw["motion_body_ang_vel"].params["std"] = 3.14
  rw["motion_body_ang_vel"].params["body_names"] = G1_MOTION_BODY_NAMES

  rw["motion_joint_pos"].params.pop("sigma_per_joint", None)
  rw["motion_joint_pos"].params.pop("per_joint", None)
  rw["motion_joint_pos"].params["std"] = 1.5

  rw["motion_joint_vel"].weight = 0.0

  _apply_tracking_kappa(rw, *_WBC_EXTRA_TRACKING)

  # --- Regularization: Zest L1 + Omni knee regen power (not soft torque limit) ---
  rw["action_rate_l1"].weight = -0.1
  rw["joint_acc"].weight = -1.0e-5
  rw["actuator_torque_soft_limit"].weight = 0.0
  rw["neg_regen_power"].weight = -1.0
  rw["neg_regen_power"].params["power_deadband"] = 150.0
  rw["neg_regen_power"].params["penalty_scale"] = 500.0
  rw["foot_slip"].weight = 0.0

  # SONIC-style ``anti_shake`` on wrists only (G1).
  rw["anti_shake"].weight = -5.0e-3
  rw["anti_shake"].params["body_names"] = G1_WRIST_BODY_NAMES
  rw["anti_shake"].params["threshold"] = 1.5

  # Deploy stack: actor history + stricter EE height termination.
  cfg.observations["actor"].history_length = 1
  cfg.terminations["anchor_pos"].params["threshold"] = 0.25
  cfg.terminations["ee_body_pos"] = TerminationTermCfg(
    func=mdp.bad_motion_body_pos_z_only,
    params={
      "command_name": "motion",
      "threshold": 0.25,
      "body_names": G1_EE_TERMINATION_BODY_NAMES,
    },
  )

  return cfg
