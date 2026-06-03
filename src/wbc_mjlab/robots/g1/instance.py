"""G1 robot instance: scene, tracking bodies, sensors, and action scales."""

from __future__ import annotations

from typing import Literal

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from wbc_mjlab.env.mdp.commands import (
  AdaptiveSimilarityTermCfg,
  MotionCommandCfg,
  wbc_joint_only_similarity_terms,
)
from wbc_mjlab.robots.g1.constants import G1_ACTION_SCALE, get_g1_robot_cfg


def configure_g1_wbc_env(
  cfg: ManagerBasedRlEnvCfg,
  *,
  play: bool = False,
  has_state_estimation: bool = True,
  sampling_strategy: Literal["binary_failure", "similarity_ema"] | None = None,
  adaptive_similarity_terms: tuple[AdaptiveSimilarityTermCfg, ...] | None = None,
) -> ManagerBasedRlEnvCfg:
  """Apply G1-specific scene, command, and sensor wiring to the shared WBC env."""
  cfg.scene.entities = {"robot": get_g1_robot_cfg()}

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg)

  cfg.actions["joint_pos"].scale = G1_ACTION_SCALE

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.sampling_mode = "adaptive"
  motion_cmd.adaptive_sampling_strategy = sampling_strategy or "similarity_ema"
  motion_cmd.adaptive_bin_width_s = 4.0
  motion_cmd.adaptive_similarity_terms = (
    adaptive_similarity_terms or wbc_joint_only_similarity_terms()
  )
  motion_cmd.assistive_wrench_enabled = True
  motion_cmd.anchor_body_name = "torso_link"
  motion_cmd.body_names = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
  )

  keybody_names = motion_cmd.body_names
  keybodies_ground_cfg = ContactSensorCfg(
    name="keybodies_ground_contact",
    primary=ContactMatch(
      mode="body",
      pattern=r"^(" + "|".join(keybody_names) + r")$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=len(keybody_names),
    track_air_time=False,
  )
  cfg.scene.sensors = (*cfg.scene.sensors, keybodies_ground_cfg)

  cfg.events["foot_friction"].params[
    "asset_cfg"
  ].geom_names = r"^(left|right)_foot[1-7]_collision$"
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = ("left_foot", "right_foot")
  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
  )

  cfg.viewer.body_name = "torso_link"

  if not has_state_estimation:
    new_actor_terms = {
      k: v
      for k, v in cfg.observations["actor"].terms.items()
      if k not in ("motion_anchor_pos_b", "base_lin_vel")
    }
    cfg.observations["actor"] = ObservationGroupCfg(
      terms=new_actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    )

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
    motion_cmd.assistive_wrench_enabled = False
    if "assistive_wrench" in cfg.events:
      cfg.events["assistive_wrench"].params["enabled"] = False

  return cfg
