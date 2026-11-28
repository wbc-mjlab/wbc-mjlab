"""Synchronized terrain/reference yaw without simulation or real datasets."""

import json
import math
from types import MethodType, SimpleNamespace

import mujoco
import numpy as np
import torch

from wbc_mjlab.env.mdp.commands import MotionCommand
from wbc_mjlab.motion.library import MotionFrame
from wbc_mjlab.presets.terrain_dataset import terrain_from_dataset_cfg
from wbc_mjlab.terrains.dataset import DATASET_FLOOR_NAME, XmlTerrainCfg


def frame(count):
  vector = torch.tensor([[1.0, 0.0, 0.0]]).repeat(count, 1)
  return MotionFrame(
    joint_pos=torch.full((count, 1), 0.25),
    joint_vel=torch.full((count, 1), 0.5),
    body_pos_w=vector[:, None, :].clone(),
    body_quat_w=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]).repeat(count, 1, 1),
    body_lin_vel_w=vector[:, None, :].clone(),
    body_ang_vel_w=vector[:, None, :].clone(),
    anchor_lin_acc_w=vector.clone(),
    anchor_ang_acc_w=vector.clone(),
  )


def test_reference_cache_patch_and_lookahead_apply_yaw_once():
  c = math.sqrt(0.5)
  command = SimpleNamespace(
    _trajectory_terrain_quats=torch.tensor([[c, 0.0, 0.0, c], [c, 0.0, 0.0, -c]]),
    trajectory_ids=torch.tensor([0, 1]),
    time_steps=torch.tensor([0, 0]),
    _ref_frame=None,
    _ref_key=None,
    num_envs=2,
    device="cpu",
    motion=SimpleNamespace(
      frame=lambda steps: frame(len(steps)), segment_end_idx=torch.tensor([10, 10])
    ),
  )
  command._rotate_terrain_reference = MethodType(
    MotionCommand._rotate_terrain_reference, command
  )
  ref = MotionCommand.reference(command)
  expected = torch.tensor([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])
  for values in (ref.body_pos_w, ref.body_lin_vel_w, ref.body_ang_vel_w):
    torch.testing.assert_close(values[:, 0], expected, atol=1e-6, rtol=0.0)
  for values in (ref.anchor_lin_acc_w, ref.anchor_ang_acc_w):
    torch.testing.assert_close(values, expected, atol=1e-6, rtol=0.0)
  torch.testing.assert_close(ref.body_quat_w[:, 0], command._trajectory_terrain_quats)
  torch.testing.assert_close(ref.joint_pos, torch.full((2, 1), 0.25))
  assert MotionCommand.reference(command) is ref
  window = MotionCommand.reference_window(command, 3)
  torch.testing.assert_close(
    window.body_pos_w[:, :, 0], expected[:, None].expand(-1, 3, -1), atol=1e-6, rtol=0.0
  )
  command.trajectory_ids[0] = 1
  command.time_steps[0] += 1
  MotionCommand._patch_reference(command, torch.tensor([0]))
  updated = MotionCommand.reference(command)
  torch.testing.assert_close(
    updated.body_pos_w[:, 0], expected[1:].expand(2, -1), atol=1e-6, rtol=0.0
  )


def test_geometry_rotates_about_tile_origin_while_floor_stays_fixed(tmp_path):
  path = tmp_path / "scene.xml"
  path.write_text(
    '<mujoco><worldbody><geom name="step" type="box" size="1 .5 .2" pos="2 0 .2"/></worldbody></mujoco>'
  )
  cfg = XmlTerrainCfg(
    xml_path=str(path), trajectory_name="clip", yaw=math.pi / 2, floor_clearance=None
  )
  spec = mujoco.MjSpec()
  spec.worldbody.add_body(name="terrain")
  output = cfg.function(0.0, spec, np.random.default_rng(0))
  geom = output.geometries[0].geom
  np.testing.assert_allclose(geom.pos, output.origin + [0.0, 2.0, 0.2], atol=1e-12)
  np.testing.assert_allclose(geom.quat, [math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)])
  np.testing.assert_allclose(spec.geom(DATASET_FLOOR_NAME).quat, [1.0, 0.0, 0.0, 0.0])


def test_random_yaw_is_stable_across_subset_selection(tmp_path):
  records = [dict(id=f"clip{i}", motion=f"{i}.csv", fps=30) for i in range(3)]
  (tmp_path / "manifest.json").write_text(json.dumps(dict(clips=records)))
  full = terrain_from_dataset_cfg(str(tmp_path))
  subset = terrain_from_dataset_cfg(str(tmp_path), clip_ids=("clip1",))
  assert full.sub_terrains["clip1"].yaw == subset.sub_terrains["clip1"].yaw
  angles = [cfg.yaw for cfg in full.sub_terrains.values()]
  assert len(set(angles)) == 3
  assert all(-math.pi <= yaw < math.pi for yaw in angles)
