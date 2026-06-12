"""G1 robot MJCF, scene entity, and motion metadata."""

from __future__ import annotations

from pathlib import Path

import mujoco

from mjlab.entity import EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

_HERE = Path(__file__).parent
G1_XML: Path = _HERE / "xmls" / "g1.xml"
assert G1_XML.exists()


def get_spec() -> mujoco.MjSpec:
  # Empty spec.assets: MuJoCo resolves mesh files from disk (mjlab #873).
  return mujoco.MjSpec.from_file(str(G1_XML))


##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.8),
  joint_pos={
    ".*_hip_pitch_joint": -0.1,
    ".*_knee_joint": 0.3,
    ".*_ankle_pitch_joint": -0.2,
    ".*_shoulder_pitch_joint": 0.35,
    ".*_elbow_joint": 0.87,
    "left_shoulder_roll_joint": 0.18,
    "right_shoulder_roll_joint": -0.18,
  },
  joint_vel={".*": 0.0},
)

KNEES_BENT_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.78),
  joint_pos={
    ".*_hip_pitch_joint": -0.312,
    ".*_knee_joint": 0.669,
    ".*_ankle_pitch_joint": -0.363,
    ".*_elbow_joint": 0.6,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_pitch_joint": 0.2,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot[1-7]_collision$": 1},
  friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
)

FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=0,
  conaffinity=1,
  condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot[1-7]_collision$": 1},
  friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
)

FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(r"^(left|right)_foot[1-7]_collision$",),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

##
# Entity config.
##


def get_g1_robot_cfg() -> EntityCfg:
  """Get a fresh G1 robot configuration instance."""
  from wbc_mjlab.robots.g1.actuators import G1_ARTICULATION

  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=G1_ARTICULATION,
  )


# Motion conversion: foot bodies and sole height for vertical debias.
MOTION_Z_DEBIAS_FOOT_BODY_NAMES: tuple[str, ...] = (
  "left_ankle_roll_link",
  "right_ankle_roll_link",
)
MOTION_Z_DEBIAS_FOOT_SOLE_Z: float = 0.035


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_g1_robot_cfg())
  viewer.launch(robot.spec.compile())
