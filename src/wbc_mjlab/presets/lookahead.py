"""Lookahead preset: the task of the SuperMimic paper (Liao et al., arXiv:2610.28480,
Appendix A and D.1), trained with either method — with the world model (``--algo wm``) it is
SuperMimic, with PPO its baseline.

A policy that looks ahead: BeyondMimic's tracking reward without the self-collision term, anchor-only
terminations, physical domain randomization without pushes, a policy that sees 10 steps of
proprioception and 10 reference frames ahead (separate reference terms with ``frames``),
actions about the default pose, and episode starts sampled where the policy earns low
return.
"""

from __future__ import annotations

from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

import wbc_mjlab.env.mdp as mdp
from wbc_mjlab.env.mdp.actions import DefaultOffsetJointPositionActionCfg
from wbc_mjlab.env.mdp.commands import MotionCommandCfg

HISTORY = 10
"""Proprioceptive steps the policy sees (Table A.1)."""
FUTURE_FRAMES = 10
"""Reference frames the policy sees, the current one first (K, Appendix A)."""


def apply_lookahead(cfg: ManagerBasedRlEnvCfg) -> None:
  """Turn the shared template (after a robot's base config) into the Lookahead task."""
  # --- Observation (Table A.1): proprioception with history + the reference ahead ---
  actor = cfg.observations["actor"]
  base = actor.terms
  noise = {
    "joint_pos": Unoise(n_min=-0.01, n_max=0.01),
    "joint_vel": Unoise(n_min=-0.5, n_max=0.5),
    "base_ang_vel": Unoise(n_min=-0.2, n_max=0.2),
    "projected_gravity": Unoise(n_min=-0.05, n_max=0.05),
    "actions": None,
  }
  terms = {
    name: replace(base[name], noise=n, history_length=HISTORY)
    for name, n in noise.items()
  }
  # the reference ahead: joints, the anchor's orientation (robot frame) and where it goes
  ahead = {"command_name": "motion", "frames": FUTURE_FRAMES}
  goal = {
    name: ObservationTermCfg(func=func, params=dict(ahead))
    for name, func in (
      ("ref_joint_pos", mdp.ref_joint_pos),
      ("ref_joint_vel", mdp.ref_joint_vel),
      ("ref_anchor_ori_b", mdp.motion_anchor_ori_b),
      ("ref_anchor_shift_b", mdp.ref_anchor_shift_b),
    )
  }
  actor.terms = {**goal, **terms}
  actor.history_length = None  # per term: proprioception only
  critic = cfg.observations["critic"].terms
  critic |= {name: replace(term, params=dict(ahead)) for name, term in goal.items()}

  # --- Action: joint setpoints about the default pose ---
  joint_pos = cfg.actions["joint_pos"]
  cfg.actions["joint_pos"] = DefaultOffsetJointPositionActionCfg(
    entity_name=joint_pos.entity_name,
    actuator_names=joint_pos.actuator_names,
    scale=joint_pos.scale,
    use_default_offset=True,
    command_name="motion",
  )

  # --- Reward (Table A.2): BeyondMimic without self-collision, joint limit -2 ---
  rw = cfg.rewards
  for term in rw.values():
    term.weight = 0.0
  tracking = {  # name: (weight, std); exp(-mean_b ‖e_b‖² / std²)
    "motion_body_pos": (1.0, 0.3),
    "motion_body_ori": (1.0, 0.4),
    "motion_body_lin_vel": (1.0, 1.0),
    "motion_body_ang_vel": (1.0, 3.14),
    "motion_global_root_pos": (0.5, 0.3),
    "motion_global_root_ori": (0.5, 0.4),
  }
  for name, (weight, std) in tracking.items():
    rw[name].weight = weight
    rw[name].params = {"command_name": "motion", "std": std}
  rw["action_rate_l2"] = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1)
  rw["joint_limit"].weight = -2.0

  # --- Termination (Appendix A): anchor height and tilt only ---
  cfg.terminations = {
    name: cfg.terminations[name] for name in ("time_out", "anchor_pos", "anchor_ori")
  }
  cfg.terminations["anchor_pos"].params["threshold"] = 0.25
  cfg.terminations["anchor_ori"].params["threshold"] = 0.8

  # --- Domain randomization (Table A.3): physical parameters only, no pushes ---
  for name in ("push_robot", "pull_robot", "assistive_wrench"):
    cfg.events.pop(name, None)
  cfg.events["foot_friction"].params["ranges"] = (0.3, 1.2)

  # --- Episode starts (Appendix D.1): 1 s bins, 50 % uniform, 40 % low return, 10 % memory ---
  motion = cfg.commands["motion"]
  assert isinstance(motion, MotionCommandCfg)
  motion.assistive_wrench_enabled = False
  motion.rsi = replace(
    motion.rsi,
    sampling_mode="adaptive",
    strategy="discounted_return",
    bin_width_s=1.0,
    uniform_ratio=0.5,
    memory_ratio=0.1,
    temperature_base=1.0 / 25.0,
    temperature_scale_by_bins=False,
    alpha=0.1,
    min_bin_span_ratio=0.0,
    persist_failure_levels=True,
  )
