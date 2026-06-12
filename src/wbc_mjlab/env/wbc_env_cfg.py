"""Robot-agnostic WBC environment template.

Per-robot wiring lives in ``wbc_mjlab.robots.<id>.configs`` (registered via task configs).
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg

from wbc_mjlab.env.mdp.actions import ReferenceJointPositionActionCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

import wbc_mjlab.env.mdp as mdp
from wbc_mjlab.env.mdp import AssistiveWrenchEvent, MotionCommandCfg

VELOCITY_RANGE = {
  "x": (-0.5, 0.5),
  "y": (-0.5, 0.5),
  "z": (-0.2, 0.2),
  "roll": (-0.52, 0.52),
  "pitch": (-0.52, 0.52),
  "yaw": (-0.78, 0.78),
}

_MOTION_COMMAND = "motion"


def make_base_wbc_env_cfg(
  use_reference_residual_action: bool = True,
) -> ManagerBasedRlEnvCfg:
  motion = {"command_name": _MOTION_COMMAND}
  # Actor: WBC reference terms first, then proprio (+ SE terms when enabled).
  # Reference obs noise: SONIC Table 2 target motion perturbations (actor only;
  # critic has enable_corruption=False and play mode disables actor corruption).
  actor_terms = {
    "ref_base_height": ObservationTermCfg(
      func=mdp.ref_base_height,
      params=motion,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "ref_base_lin_vel_b": ObservationTermCfg(
      func=mdp.ref_base_lin_vel_b,
      params=motion,
      noise=Unoise(n_min=(-0.1, -0.1, -0.05), n_max=(0.1, 0.1, 0.05)),
    ),
    "ref_base_ang_vel_b": ObservationTermCfg(
      func=mdp.ref_base_ang_vel_b,
      params=motion,
      noise=Unoise(n_min=(-0.2, -0.2, -0.3), n_max=(0.2, 0.2, 0.3)),
    ),
    "ref_gravity_b": ObservationTermCfg(
      func=mdp.ref_gravity_b,
      params=motion,
      noise=Unoise(n_min=-0.02, n_max=0.02),
    ),
    "ref_joint_pos": ObservationTermCfg(
      func=mdp.ref_joint_pos,
      params=motion,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "ref_joint_vel": ObservationTermCfg(
      func=mdp.ref_joint_vel,
      params=motion,
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": ""},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-1.5, n_max=1.5),
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
    "motion_anchor_pos_b": ObservationTermCfg(
      func=mdp.motion_anchor_pos_b,
      params={"command_name": _MOTION_COMMAND},
      noise=Unoise(n_min=-0.25, n_max=0.25),
    ),
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": ""},
      noise=Unoise(n_min=-0.5, n_max=0.5),
    ),
  }

  # Critic: actor (no noise) + privileged keybody / contact features.
  critic_terms = {
    **{
      k: ObservationTermCfg(func=v.func, params=dict(v.params) if v.params else {})
      for k, v in actor_terms.items()
    },
    "motion_anchor_ori_b": ObservationTermCfg(
      func=mdp.motion_anchor_ori_b, params={"command_name": "motion"}
    ),
    "ref_body_pos": ObservationTermCfg(
      func=mdp.ref_body_pos_b, params={"command_name": "motion"}
    ),
    "ref_body_ori": ObservationTermCfg(
      func=mdp.ref_body_ori_b, params={"command_name": "motion"}
    ),
    "body_pos": ObservationTermCfg(
      func=mdp.robot_body_pos_b, params={"command_name": "motion"}
    ),
    "body_ori": ObservationTermCfg(
      func=mdp.robot_body_ori_b, params={"command_name": "motion"}
    ),
    "ref_body_lin_vel": ObservationTermCfg(
      func=mdp.ref_body_lin_vel, params={"command_name": "motion"}
    ),
    "ref_body_ang_vel": ObservationTermCfg(
      func=mdp.ref_body_ang_vel, params={"command_name": "motion"}
    ),
    "motion_body_lin_vel": ObservationTermCfg(
      func=mdp.motion_body_lin_vel, params={"command_name": "motion"}
    ),
    "motion_body_ang_vel": ObservationTermCfg(
      func=mdp.motion_body_ang_vel, params={"command_name": "motion"}
    ),
    "ref_joint_vel": ObservationTermCfg(
      func=mdp.ref_joint_vel, params={"command_name": "motion"}
    ),
    "ref_base_lin_acc": ObservationTermCfg(
      func=mdp.ref_base_lin_acc_b, params={"command_name": "motion"}
    ),
    "ref_base_ang_acc": ObservationTermCfg(
      func=mdp.ref_base_ang_acc_b, params={"command_name": "motion"}
    ),
    "base_lin_vel": ObservationTermCfg(
      func=mdp.base_lin_vel,
      params={"asset_cfg": SceneEntityCfg("robot")},
    ),
    "assistive_force": ObservationTermCfg(
      func=mdp.assistive_wrench_force,
      params={"command_name": "motion"},
    ),
    "assistive_torque": ObservationTermCfg(
      func=mdp.assistive_wrench_torque,
      params={"command_name": "motion"},
    ),
    "assistive_gain": ObservationTermCfg(
      func=mdp.assistive_wrench_gain,
      params={"command_name": "motion"},
    ),
  }

  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  if use_reference_residual_action:
    joint_pos_action = ReferenceJointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=0.25,
      command_name="motion",
    )
  else:
    joint_pos_action = JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=0.25,
      use_default_offset=True,
    )

  actions: dict[str, ActionTermCfg] = {"joint_pos": joint_pos_action}

  commands: dict[str, CommandTermCfg] = {
    "motion": MotionCommandCfg(
      entity_name="robot",
      resampling_time_range=(1.0e9, 1.0e9),
      debug_vis=True,
      pose_range={
        "x": (-0.05, 0.05),
        "y": (-0.05, 0.05),
        "z": (-0.01, 0.01),
        "roll": (-0.1, 0.1),
        "pitch": (-0.1, 0.1),
        "yaw": (-0.2, 0.2),
      },
      velocity_range=VELOCITY_RANGE,
      joint_position_range=(-0.1, 0.1),
      adaptive_bin_width_s=4.0,
      adaptive_sampling_strategy="similarity_ema",
      assistive_wrench_enabled=True,
      motion_file="",
      anchor_body_name="",
      body_names=(),
    )
  }

  events: dict[str, EventTermCfg] = {
    "assistive_wrench": EventTermCfg(
      func=AssistiveWrenchEvent,
      mode="step",
      params={
        "command_name": "motion",
        "asset_cfg": SceneEntityCfg("robot", body_names=()),
        "body_name": "",
        "kvp": 1.0,
        "kvd": 10.0,
        "kwp": 200.0,
        "kwd": 10.0,
        "enabled": True,
      },
    ),
    "push_robot": EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(1.0, 3.0),
      params={"velocity_range": VELOCITY_RANGE},
    ),
    "base_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=()),
        "operation": "add",
        "ranges": {
          0: (-0.025, 0.025),
          1: (-0.05, 0.05),
          2: (-0.05, 0.05),
        },
      },
    ),
    "pull_robot": EventTermCfg(
      func=mdp.apply_external_force_torque,
      mode="interval",
      interval_range_s=(1.0, 4.0),
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=()),
        "force_range": (-0.0, 0.0),
        "torque_range": (-0.0, 0.0),
      },
    ),
    "encoder_bias": EventTermCfg(
      mode="startup",
      func=dr.encoder_bias,
      params={"asset_cfg": SceneEntityCfg("robot"), "bias_range": (-0.01, 0.01)},
    ),
    "foot_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=()),
        "operation": "abs",
        "ranges": (0.3, 1.5),
        "shared_random": True,
      },
    ),
  }

  # Tracking rewards (std/weights aligned with mjlab ``tracking_env_cfg``).
  rewards: dict[str, RewardTermCfg] = {
    "motion_global_root_pos": RewardTermCfg(
      func=mdp.motion_global_anchor_position_error_exp,
      weight=0.5,
      params={"command_name": "motion", "std": 0.5},
    ),
    "motion_global_root_ori": RewardTermCfg(
      func=mdp.motion_global_anchor_orientation_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 0.4},
    ),
    "motion_root_lin_vel_b": RewardTermCfg(
      func=mdp.motion_anchor_linear_velocity_body_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 1.0},
    ),
    "motion_root_ang_vel_b": RewardTermCfg(
      func=mdp.motion_anchor_angular_velocity_body_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 1.5},
    ),
    "motion_body_pos": RewardTermCfg(
      func=mdp.motion_relative_body_position_error_exp,
      weight=1.5,
      params={"command_name": "motion", "std": 0.15},
    ),
    "motion_body_ori": RewardTermCfg(
      func=mdp.motion_relative_body_orientation_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 0.4},
    ),
    "motion_body_lin_vel": RewardTermCfg(
      func=mdp.motion_global_body_linear_velocity_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 1.0},
    ),
    "motion_body_ang_vel": RewardTermCfg(
      func=mdp.motion_global_body_angular_velocity_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 3.14},
    ),
    "motion_joint_pos": RewardTermCfg(
      func=mdp.motion_joint_position_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 0.5},
    ),
    "motion_joint_vel": RewardTermCfg(
      func=mdp.motion_joint_velocity_error_exp,
      weight=0.5,
      params={"command_name": "motion", "std": 2.0},
    ),
    "action_rate_l1": RewardTermCfg(func=mdp.action_rate_l1, weight=-0.08),
    "joint_acc": RewardTermCfg(
      func=mdp.joint_acc_l1,
      weight=-2.0e-6,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "neg_regen_power": RewardTermCfg(
      func=mdp.negative_mechanical_power_l2,
      weight=0.0,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "power_deadband": 0.0,
        "penalty_scale": 1.0,
      },
    ),
    "joint_limit": RewardTermCfg(
      func=mdp.joint_pos_limits,
      weight=-10.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "survival": RewardTermCfg(func=mdp.is_alive, weight=1.0),
    "foot_slip": RewardTermCfg(
      func=mdp.feet_slip,
      weight=-0.0,
      params={
        "sensor_name": "feet_ground_contact",
        "asset_cfg": SceneEntityCfg("robot", site_names=()),
      },
    ),
    # "self_collisions": RewardTermCfg(
    #   func=mdp.self_collision_cost,
    #   weight=-10.0,
    #   params={"sensor_name": "self_collision", "force_threshold": 10.0},
    # ),
  }

  # Common tracking terminations; ``ee_body_pos`` is full 3D on end effectors (wired in base).
  terminations: dict[str, TerminationTermCfg] = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "anchor_pos": TerminationTermCfg(
      func=mdp.bad_anchor_pos_z_only,
      params={"command_name": "motion", "threshold": 0.25},
    ),
    "anchor_ori": TerminationTermCfg(
      func=mdp.bad_anchor_ori,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "command_name": "motion",
        "threshold": 0.8,
      },
    ),
    "ee_body_pos": TerminationTermCfg(
      func=mdp.bad_motion_body_pos,
      params={
        "command_name": "motion",
        "threshold": 0.25,
        "body_names": (),
      },
    ),
  }

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(terrain=TerrainEntityCfg(terrain_type="plane"), num_envs=4096),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="",
      distance=2.8,
      fovy=55.0,
      elevation=-5.0,
      azimuth=120.0,
    ),
    sim=SimulationCfg(
      nconmax=45,
      njmax=250,
      mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20),
    ),
    decimation=4,
    episode_length_s=10.0,
  )
