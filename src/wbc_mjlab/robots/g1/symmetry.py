"""G1 left-right symmetry for motion mirroring."""

from __future__ import annotations

from wbc_mjlab.robots.g1.constants import G1_MOTION_BODY_NAMES
from wbc_mjlab.robots.symmetry import (
  JointSymmetryEntry,
  RobotSymmetryConfig,
  register_robot_symmetry_config,
)

# Joint pairs: (left, right, scale). scale=-1 for roll/yaw axes, +1 for pitch.
_G1_JOINT_SYMMETRY: tuple[JointSymmetryEntry, ...] = (
  JointSymmetryEntry("left_hip_pitch_joint", "right_hip_pitch_joint", 1.0),
  JointSymmetryEntry("left_hip_roll_joint", "right_hip_roll_joint", -1.0),
  JointSymmetryEntry("left_hip_yaw_joint", "right_hip_yaw_joint", -1.0),
  JointSymmetryEntry("left_knee_joint", "right_knee_joint", 1.0),
  JointSymmetryEntry("left_ankle_pitch_joint", "right_ankle_pitch_joint", 1.0),
  JointSymmetryEntry("left_ankle_roll_joint", "right_ankle_roll_joint", -1.0),
  JointSymmetryEntry("left_shoulder_pitch_joint", "right_shoulder_pitch_joint", 1.0),
  JointSymmetryEntry("left_shoulder_roll_joint", "right_shoulder_roll_joint", -1.0),
  JointSymmetryEntry("left_shoulder_yaw_joint", "right_shoulder_yaw_joint", -1.0),
  JointSymmetryEntry("left_elbow_joint", "right_elbow_joint", 1.0),
  JointSymmetryEntry("left_wrist_roll_joint", "right_wrist_roll_joint", -1.0),
  JointSymmetryEntry("left_wrist_pitch_joint", "right_wrist_pitch_joint", 1.0),
  JointSymmetryEntry("left_wrist_yaw_joint", "right_wrist_yaw_joint", -1.0),
  JointSymmetryEntry("waist_yaw_joint", "waist_yaw_joint", -1.0),
  JointSymmetryEntry("waist_roll_joint", "waist_roll_joint", -1.0),
  JointSymmetryEntry("waist_pitch_joint", "waist_pitch_joint", 1.0),
)

G1_SYMMETRY_CONFIG = RobotSymmetryConfig(joints=_G1_JOINT_SYMMETRY)

register_robot_symmetry_config("g1", G1_SYMMETRY_CONFIG)

# Tracking keybodies that participate in left-right swap (documentation / phase 2).
G1_SYMMETRY_BODY_NAMES: tuple[str, ...] = G1_MOTION_BODY_NAMES
