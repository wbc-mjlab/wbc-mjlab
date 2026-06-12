"""Zest-style env config."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.termination_manager import TerminationTermCfg

import wbc_mjlab.env.mdp as mdp
from wbc_mjlab.env.mdp.commands import (
  MotionCommandCfg,
  wbc_joint_only_similarity_terms,
)
from wbc_mjlab.robots.g1.configs.base import G1_MOTION_BODY_NAMES, g1_base_cfg

# Catastrophic ground-contact spike only (Zest §Early Terminations); allows multi-contact skills.
KEYBODY_GROUND_CONTACT_FORCE_THRESHOLD = 2000.0


def g1_wbc_zest_env_cfg() -> ManagerBasedRlEnvCfg:
  """No SE, joint-only similarity RSI, adaptive bins, assistive wrench."""
  cfg = g1_base_cfg()
  rw = cfg.rewards
  rw["motion_global_root_pos"].weight = 0.8
  rw["motion_global_root_ori"].weight = 1.0
  rw["motion_root_lin_vel_b"].weight = 1.0
  rw["motion_root_ang_vel_b"].weight = 1.0
  rw["motion_body_pos"].weight = 1.5
  rw["motion_body_ori"].weight = 1.0
  rw["motion_body_lin_vel"].weight = 1.0
  rw["motion_body_ang_vel"].weight = 1.0
  rw["motion_joint_pos"].weight = 1.0
  rw["motion_joint_vel"].weight = 0.5
  rw["action_rate_l1"].weight = -0.12
  rw["joint_acc"].weight = -6.0e-6
  rw["survival"].weight = 1.0
  rw["foot_slip"].weight = -0.0

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.adaptive_sampling_strategy = "similarity_ema"
  motion_cmd.adaptive_similarity_terms = wbc_joint_only_similarity_terms()
  motion_cmd.adaptive_bin_width_s = 4.0
  motion_cmd.assistive_wrench_enabled = True

  actor = cfg.observations["actor"]
  for key in ("motion_anchor_pos_b", "base_lin_vel", "ref_joint_vel"):
    actor.terms.pop(key, None)

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
