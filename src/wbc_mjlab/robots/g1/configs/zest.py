"""Zest env config (arXiv:2511.02367)."""

from __future__ import annotations

from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.termination_manager import TerminationTermCfg

import wbc_mjlab.env.mdp as mdp
from wbc_mjlab.env.mdp.commands import MotionCommandCfg
from wbc_mjlab.robots.g1.configs.base import (
  G1_ENDEFFECTOR_BODY_NAMES,
  G1_MOTION_BODY_NAMES,
  g1_base_cfg,
)

# Catastrophic ground-contact spike only (Zest §Early Terminations); allows multi-contact skills.
KEYBODY_GROUND_CONTACT_FORCE_THRESHOLD = 2000.0

# Zest Table S4: exp(-κ‖e‖²/σ²) with κ = 1/4.
_TRACKING_KAPPA = 0.25

# Seven positive tracking terms (Table S4); no separate keybody vel / joint vel terms.
_ZEST_TRACKING_REWARDS = (
  "motion_global_root_pos",
  "motion_global_root_ori",
  "motion_root_lin_vel_b",
  "motion_root_ang_vel_b",
  "motion_body_pos",
  "motion_body_ori",
  "motion_joint_pos",
)

# SE actor terms today (interim); see env TODO(SE obs) for planned swap in task configs.
# TODO(SE): replace with z-only tracking-error terms + full xyz/ori reference command obs;
# drop ref_gravity_b / projected_gravity from SE layouts. Non-SE tasks unchanged.
_ZEST_SE_OBS = ("motion_anchor_pos_b", "base_lin_vel")


def _apply_tracking_kappa(rw, *names: str) -> None:
  for name in names:
    rw[name].params["kappa"] = _TRACKING_KAPPA


def _configure_zest_actor_obs(cfg: ManagerBasedRlEnvCfg, *, state_estimation: bool) -> None:
  actor = cfg.observations["actor"]
  actor.terms.pop("ref_joint_vel", None)
  if not state_estimation:
    for key in _ZEST_SE_OBS:
      actor.terms.pop(key, None)


def _g1_zest_env_cfg(*, state_estimation: bool) -> ManagerBasedRlEnvCfg:
  """Shared Zest rewards, RSI, and terminations; optional actor SE terms."""
  cfg = g1_base_cfg()
  rw = cfg.rewards

  # --- Table S4 tracking (all weight 1.0) ---
  rw["motion_global_root_pos"].weight = 1.0
  rw["motion_global_root_pos"].params["std"] = 0.4
  rw["motion_global_root_ori"].weight = 1.0
  rw["motion_global_root_ori"].params["std"] = 0.5
  rw["motion_root_lin_vel_b"].weight = 1.0
  rw["motion_root_lin_vel_b"].params["std"] = 0.6
  rw["motion_root_ang_vel_b"].weight = 1.0
  rw["motion_root_ang_vel_b"].params["std"] = 1.5
  rw["motion_body_pos"].weight = 1.0
  rw["motion_body_pos"].params.pop("std", None)
  rw["motion_body_pos"].params["sigma_per_keybody"] = 0.2
  rw["motion_body_pos"].params["body_error_aggregate"] = "sum"
  rw["motion_body_pos"].params["body_names"] = G1_ENDEFFECTOR_BODY_NAMES
  rw["motion_body_ori"].weight = 1.0
  rw["motion_body_ori"].params.pop("std", None)
  rw["motion_body_ori"].params["sigma_per_keybody"] = 0.4
  rw["motion_body_ori"].params["body_error_aggregate"] = "sum"
  rw["motion_body_ori"].params["body_names"] = G1_ENDEFFECTOR_BODY_NAMES
  rw["motion_joint_pos"].weight = 1.0
  rw["motion_joint_pos"].params.pop("std", None)
  rw["motion_joint_pos"].params.pop("per_joint", None)
  rw["motion_joint_pos"].params["sigma_per_joint"] = 0.3

  _apply_tracking_kappa(rw, *_ZEST_TRACKING_REWARDS)

  # Not in Table S4 (disable extras inherited from the WBC template).
  rw["motion_body_lin_vel"].weight = 0.0
  rw["motion_body_ang_vel"].weight = 0.0
  rw["motion_joint_vel"].weight = 0.0
  rw["foot_slip"].weight = 0.0
  rw["neg_regen_power"].weight = 0.0
  rw["angular_momentum"].weight = 0.0

  rw["survival"].weight = 1.0
  rw["action_rate_l1"].weight = -0.1
  rw["joint_acc"].weight = -5.0e-6
  rw["joint_limit"].weight = -10.0
  rw["actuator_torque_soft_limit"].weight = -1.0
  rw["actuator_torque_soft_limit"].params["soft_ratio"] = 0.9

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.assistive_wrench_enabled = True
  motion_cmd.rsi = replace(
    motion_cmd.rsi,
    similarity_from_rewards=True,
    bin_width_s=4.0,
    similarity_norm_by_remaining_clip=True,
    min_bin_span_ratio=0.5,
    persist_failure_levels=True,
  )

  _configure_zest_actor_obs(cfg, state_estimation=state_estimation)

  # Zest paper: anchor early termination + catastrophic contact only (no EE tracking cutoff).
  cfg.terminations.pop("ee_body_pos", None)
  cfg.terminations["anchor_pos"].params["threshold"] = 0.35

  cfg.terminations["keybody_ground_contact_force"] = TerminationTermCfg(
    func=mdp.excessive_keybody_ground_contact_force,
    params={
      "sensor_name": "keybodies_ground_contact",
      "body_names": G1_MOTION_BODY_NAMES,
      "body_slot_order": G1_MOTION_BODY_NAMES,
      "force_threshold": KEYBODY_GROUND_CONTACT_FORCE_THRESHOLD,
    },
  )
  return cfg


def g1_wbc_zest_env_cfg() -> ManagerBasedRlEnvCfg:
  """Zest paper repro: no actor SE terms, reward-aligned RSI, assistive wrench."""
  return _g1_zest_env_cfg(state_estimation=False)


def g1_wbc_zest_se_env_cfg() -> ManagerBasedRlEnvCfg:
  """Zest rewards/RSI with ``motion_anchor_pos_b`` and ``base_lin_vel`` in the actor."""
  return _g1_zest_env_cfg(state_estimation=True)
