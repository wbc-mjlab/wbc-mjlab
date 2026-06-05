"""G1 robot instance + shared env cfg scaffolding."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from wbc_mjlab.env.mdp.commands import MotionCommandCfg
from wbc_mjlab.env.wbc_env_cfg import make_base_wbc_env_cfg
from wbc_mjlab.robots.g1.constants import G1_ACTION_SCALE, get_g1_robot_cfg

G1_ANCHOR_BODY_NAME = "torso_link"

G1_MOTION_BODY_NAMES: tuple[str, ...] = (
  "pelvis",
  "left_hip_roll_link",
  "left_knee_link",
  "left_ankle_roll_link",
  "right_hip_roll_link",
  "right_knee_link",
  "right_ankle_roll_link",
  G1_ANCHOR_BODY_NAME,
  "left_shoulder_roll_link",
  "left_elbow_link",
  "left_wrist_yaw_link",
  "right_shoulder_roll_link",
  "right_elbow_link",
  "right_wrist_yaw_link",
)

G1_EE_TERMINATION_BODY_NAMES: tuple[str, ...] = (
  "left_ankle_roll_link",
  "right_ankle_roll_link",
  "left_wrist_yaw_link",
  "right_wrist_yaw_link",
)


def g1_base_cfg() -> ManagerBasedRlEnvCfg:
  """Shared WBC template + G1 scene, tracking bodies, and sensors."""
  cfg = make_base_wbc_env_cfg(use_reference_residual_action=True)

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
  motion_cmd.anchor_body_name = G1_ANCHOR_BODY_NAME
  motion_cmd.body_names = G1_MOTION_BODY_NAMES

  keybodies_ground_cfg = ContactSensorCfg(
    name="keybodies_ground_contact",
    primary=ContactMatch(
      mode="body",
      pattern=r"^(" + "|".join(G1_MOTION_BODY_NAMES) + r")$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=len(G1_MOTION_BODY_NAMES),
    track_air_time=False,
  )
  cfg.scene.sensors = (*cfg.scene.sensors, keybodies_ground_cfg)

  cfg.events["assistive_wrench"].params["asset_cfg"].body_names = (
    G1_ANCHOR_BODY_NAME,
  )
  cfg.events["assistive_wrench"].params["body_name"] = G1_ANCHOR_BODY_NAME
  cfg.events["pull_robot"].params["asset_cfg"].body_names = (G1_ANCHOR_BODY_NAME,)
  cfg.events["foot_friction"].params[
    "asset_cfg"
  ].geom_names = r"^(left|right)_foot[1-7]_collision$"
  cfg.events["base_com"].params["asset_cfg"].body_names = (G1_ANCHOR_BODY_NAME,)
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = ("left_foot", "right_foot")
  cfg.terminations["ee_body_pos"].params["body_names"] = G1_EE_TERMINATION_BODY_NAMES

  cfg.viewer.body_name = G1_ANCHOR_BODY_NAME

  return cfg


def without_state_estimation(cfg: ManagerBasedRlEnvCfg) -> None:
  actor = cfg.observations["actor"]
  cfg.observations["actor"] = ObservationGroupCfg(
    terms={
      k: v
      for k, v in actor.terms.items()
      if k not in ("motion_anchor_pos_b", "base_lin_vel")
    },
    concatenate_terms=actor.concatenate_terms,
    enable_corruption=actor.enable_corruption,
    history_length=actor.history_length,
    flatten_history_dim=actor.flatten_history_dim,
  )
