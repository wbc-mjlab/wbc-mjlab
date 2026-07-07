"""Tests for sagittal motion mirroring."""

from __future__ import annotations

import numpy as np
import pytest

from wbc_mjlab.motion.motion_mirror import mirror_joint_array, mirror_motion_log
from wbc_mjlab.robots.g1.symmetry import G1_SYMMETRY_CONFIG

_G1_JOINT_NAMES = [
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
]

_G1_BODY_NAMES = [
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
]


def _quat_xyzw_to_rot(q: np.ndarray) -> np.ndarray:
  x, y, z, w = q
  return np.array(
    [
      [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
  )


def _synthetic_log(t: int = 4) -> dict:
  j = len(_G1_JOINT_NAMES)
  b = len(_G1_BODY_NAMES)
  rng = np.random.default_rng(0)
  joint_pos = rng.normal(size=(t, j))
  joint_vel = rng.normal(size=(t, j))
  body_pos_w = rng.normal(size=(t, b, 3))
  body_quat_w = rng.normal(size=(t, b, 4))
  body_quat_w /= np.linalg.norm(body_quat_w, axis=-1, keepdims=True)
  body_lin_vel_w = rng.normal(size=(t, b, 3))
  body_ang_vel_w = rng.normal(size=(t, b, 3))
  return {
    "fps": [50.0],
    "joint_pos": joint_pos,
    "joint_vel": joint_vel,
    "body_pos_w": body_pos_w,
    "body_quat_w": body_quat_w,
    "body_lin_vel_w": body_lin_vel_w,
    "body_ang_vel_w": body_ang_vel_w,
  }


def test_mirror_twice_restores_joint_channels() -> None:
  log = _synthetic_log()
  once = mirror_motion_log(
    log,
    joint_names=_G1_JOINT_NAMES,
    body_names=_G1_BODY_NAMES,
    config=G1_SYMMETRY_CONFIG,
  )
  twice = mirror_motion_log(
    once,
    joint_names=_G1_JOINT_NAMES,
    body_names=_G1_BODY_NAMES,
    config=G1_SYMMETRY_CONFIG,
  )
  np.testing.assert_allclose(twice["joint_pos"], log["joint_pos"], atol=1e-12)
  np.testing.assert_allclose(twice["joint_vel"], log["joint_vel"], atol=1e-12)


def test_mirror_swaps_left_right_hip_roll() -> None:
  joint_pos = np.zeros((1, len(_G1_JOINT_NAMES)))
  joint_pos[0, _G1_JOINT_NAMES.index("left_hip_roll_joint")] = 0.25
  joint_pos[0, _G1_JOINT_NAMES.index("right_hip_roll_joint")] = -0.40
  mirrored = mirror_joint_array(
    joint_pos,
    joint_names=_G1_JOINT_NAMES,
    config=G1_SYMMETRY_CONFIG,
  )
  assert mirrored[0, _G1_JOINT_NAMES.index("left_hip_roll_joint")] == pytest.approx(0.40)
  assert mirrored[0, _G1_JOINT_NAMES.index("right_hip_roll_joint")] == pytest.approx(-0.25)


def test_mirror_twice_restores_body_kinematics() -> None:
  log = _synthetic_log()
  twice = mirror_motion_log(
    mirror_motion_log(
      log,
      joint_names=_G1_JOINT_NAMES,
      body_names=_G1_BODY_NAMES,
      config=G1_SYMMETRY_CONFIG,
    ),
    joint_names=_G1_JOINT_NAMES,
    body_names=_G1_BODY_NAMES,
    config=G1_SYMMETRY_CONFIG,
  )
  np.testing.assert_allclose(twice["body_pos_w"], log["body_pos_w"], atol=1e-12)
  np.testing.assert_allclose(twice["body_lin_vel_w"], log["body_lin_vel_w"], atol=1e-12)
  np.testing.assert_allclose(twice["body_ang_vel_w"], log["body_ang_vel_w"], atol=1e-12)
  quats_a = twice["body_quat_w"].reshape(-1, 4)
  quats_b = log["body_quat_w"].reshape(-1, 4)
  rot_err = np.array(
    [np.linalg.norm(_quat_xyzw_to_rot(a) - _quat_xyzw_to_rot(b)) for a, b in zip(quats_a, quats_b)]
  )
  assert float(np.max(rot_err)) < 1e-12


@pytest.mark.slow
def test_mirror_matches_fk_body_kinematics() -> None:
  """Mirrored joint state + FK should match mirrored FK body arrays."""
  import torch
  from mjlab.entity import Entity
  from mjlab.scene import Scene
  from mjlab.sim.sim import Simulation, SimulationCfg

  from wbc_mjlab.motion.motion_mirror import mirror_body_arrays, mirror_joint_array
  from wbc_mjlab.motion.robot_assets import (
    conversion_scene_cfg,
    get_robot_motion_spec,
    qpos_dof_joint_names,
  )

  _, motion_spec = get_robot_motion_spec("g1")
  scene = Scene(conversion_scene_cfg(motion_spec, num_envs=1), device="cpu")
  model = scene.compile()
  joint_names = qpos_dof_joint_names(model)
  body_names = list(scene["robot"].body_names)

  sim = Simulation(num_envs=1, cfg=SimulationCfg(), model=model, device="cpu")
  scene.initialize(sim.mj_model, sim.model, sim.data)
  robot: Entity = scene["robot"]
  scene.reset()

  rng = np.random.default_rng(7)
  root = robot.data.default_root_state.clone()
  root[0, 0:3] = torch.tensor([0.4, -0.15, 0.88])
  root[0, 3:7] = torch.tensor([0.05, 0.12, -0.08, 0.99])
  root[0, 3:7] /= torch.linalg.norm(root[0, 3:7])
  root[0, 7:10] = torch.tensor([0.2, 0.05, -0.1])
  root[0, 10:13] = torch.tensor([-0.03, 0.08, 0.02])

  joint_pos = robot.data.default_joint_pos.clone()
  joint_vel = robot.data.default_joint_vel.clone()
  joint_idx = robot.find_joints(joint_names, preserve_order=True)[0]
  joint_pos[0, joint_idx] = torch.tensor(rng.normal(len(joint_names)), dtype=torch.float32)
  joint_vel[0, joint_idx] = torch.tensor(rng.normal(len(joint_names)), dtype=torch.float32)

  robot.write_root_state_to_sim(root)
  robot.write_joint_state_to_sim(joint_pos, joint_vel)
  sim.forward()

  jp = robot.data.joint_pos[0, joint_idx].cpu().numpy()[None]
  jv = robot.data.joint_vel[0, joint_idx].cpu().numpy()[None]
  bp = robot.data.body_link_pos_w[0].cpu().numpy()[None]
  bq = robot.data.body_link_quat_w[0].cpu().numpy()[None]
  bl = robot.data.body_link_lin_vel_w[0].cpu().numpy()[None]
  ba = robot.data.body_link_ang_vel_w[0].cpu().numpy()[None]

  mjp = mirror_joint_array(jp, joint_names=joint_names, config=G1_SYMMETRY_CONFIG)
  mjv = mirror_joint_array(jv, joint_names=joint_names, config=G1_SYMMETRY_CONFIG)
  mbp, mbq, mbl, mba = mirror_body_arrays(bp, bq, bl, ba, body_names=body_names)

  pelvis = body_names.index("pelvis")
  root_m = robot.data.default_root_state.clone()
  root_m[0, 0:3] = torch.tensor(mbp[0, pelvis], dtype=torch.float32)
  root_m[0, 3:7] = torch.tensor(mbq[0, pelvis], dtype=torch.float32)
  root_m[0, 7:10] = torch.tensor(mbl[0, pelvis], dtype=torch.float32)
  root_m[0, 10:13] = torch.tensor(mba[0, pelvis], dtype=torch.float32)

  joint_pos_m = robot.data.default_joint_pos.clone()
  joint_vel_m = robot.data.default_joint_vel.clone()
  joint_pos_m[0, joint_idx] = torch.tensor(mjp[0], dtype=torch.float32)
  joint_vel_m[0, joint_idx] = torch.tensor(mjv[0], dtype=torch.float32)

  robot.write_root_state_to_sim(root_m)
  robot.write_joint_state_to_sim(joint_pos_m, joint_vel_m)
  sim.forward()

  np.testing.assert_allclose(
    robot.data.body_link_pos_w[0].cpu().numpy(),
    mbp[0],
    atol=1e-4,
  )
  np.testing.assert_allclose(
    robot.data.body_link_lin_vel_w[0].cpu().numpy(),
    mbl[0],
    atol=5e-4,
  )
  np.testing.assert_allclose(
    robot.data.body_link_ang_vel_w[0].cpu().numpy(),
    mba[0],
    atol=1e-4,
  )
  fk_quat = robot.data.body_link_quat_w[0].cpu().numpy()
  rot_err = np.array(
    [
      np.linalg.norm(_quat_xyzw_to_rot(a) - _quat_xyzw_to_rot(b))
      for a, b in zip(fk_quat, mbq[0])
    ]
  )
  assert float(np.max(rot_err)) < 1e-4
  np.testing.assert_allclose(robot.data.joint_vel[0, joint_idx].cpu().numpy(), mjv[0], atol=1e-6)
