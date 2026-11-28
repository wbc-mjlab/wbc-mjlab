"""Vertical placement from small synthetic assets; no model compilation."""

import mujoco
import numpy as np
import pytest

from wbc_mjlab.terrains.dataset import DATASET_FLOOR_NAME, XmlTerrainCfg


@pytest.mark.parametrize("root_z, expected_shift", [(1.0, 2.51), (-3.0, 3.01)])
def test_terrain_and_reference_receive_identical_shift(
  tmp_path, root_z, expected_shift
):
  scene = tmp_path / "scene.xml"
  scene.write_text("""<mujoco><compiler angle="degree"/><worldbody>
    <body pos="0 0 -2" euler="0 90 0">
      <geom type="box" size=".5 .2 .25"/>
    </body></worldbody></mujoco>""")
  motion = tmp_path / "motion.csv"
  motion.write_text(f"0,0,{root_z},0,0,0,1,0\n" * 2)
  cfg = XmlTerrainCfg(
    xml_path=str(scene), motion_path=str(motion), trajectory_name="box"
  )
  spec = mujoco.MjSpec()
  spec.worldbody.add_body(name="terrain")
  output = cfg.function(0.0, spec, np.random.default_rng(0))
  shift = output.origin[2]
  assert shift == pytest.approx(expected_shift)
  geom = output.geometries[0].geom
  assert geom.pos[2] == pytest.approx(-2.0 + shift)
  assert geom.pos[2] - 0.5 >= 0.01 - 1e-12
  # This is MotionCommand.body_pos_w's reference + env_origin transformation.
  reference_z = root_z + shift
  assert reference_z >= 0.01 - 1e-12
  assert geom.pos[2] - reference_z == pytest.approx(-2.0 - root_z)


def test_rotated_scaled_mesh_vertices_stay_above_floor(tmp_path):
  mesh = tmp_path / "mesh.obj"
  mesh.write_text(
    "v 0 -2 0\nv 1 -2 0\nv 0 1 0\nv 0 0 1\nf 1 2 3\nf 1 4 2\nf 1 3 4\nf 2 4 3\n"
  )
  scene = tmp_path / "scene.xml"
  scene.write_text("""<mujoco><compiler angle="degree"/>
    <asset><mesh name="obstacle" file="mesh.obj" scale="2 3 4" refpos="0 0 1"/></asset>
    <worldbody><geom type="mesh" mesh="obstacle" euler="90 0 0" pos="0 0 -1"/></worldbody>
  </mujoco>""")
  cfg = XmlTerrainCfg(xml_path=str(scene), trajectory_name="mesh")
  spec = mujoco.MjSpec()
  spec.worldbody.add_body(name="terrain")
  output = cfg.function(0.0, spec, np.random.default_rng(0))
  assert output.origin[2] == pytest.approx(7.01)
  assert output.geometries[0].geom.pos[2] == pytest.approx(6.01)


def test_flat_fallback_is_also_lifted_with_trajectory(tmp_path):
  motion = tmp_path / "motion.csv"
  motion.write_text("0,0,-2,0,0,0,1,0\n")
  cfg = XmlTerrainCfg(trajectory_name="flat", motion_path=str(motion))
  spec = mujoco.MjSpec()
  spec.worldbody.add_body(name="terrain")
  output = cfg.function(0.0, spec, np.random.default_rng(0))
  assert output.origin[2] == pytest.approx(2.01)
  geom = output.geometries[0].geom
  assert geom.pos[2] - geom.size[2] >= 0.01
  assert -2.0 + output.origin[2] == pytest.approx(0.01)


def test_shared_world_floor_stays_at_zero_when_tiles_move():
  spec = mujoco.MjSpec()
  spec.worldbody.add_body(name="terrain")
  for name, offset in (("first", [10.0, 20.0, 0.0]), ("second", [-10.0, -20.0, 0.0])):
    output = XmlTerrainCfg(trajectory_name=name).function(
      0.0, spec, np.random.default_rng(0)
    )
    assert output.origin[2] > 0.0
    for item in output.geometries:
      assert item.geom.type != mujoco.mjtGeom.mjGEOM_PLANE
      item.geom.pos += np.array(offset)
  planes = [geom for geom in spec.geoms if geom.type == mujoco.mjtGeom.mjGEOM_PLANE]
  assert len(planes) == 1
  floor = planes[0]
  assert floor.name == DATASET_FLOOR_NAME
  np.testing.assert_allclose(floor.pos, [0.0, 0.0, 0.0])
  np.testing.assert_allclose(floor.quat, [1.0, 0.0, 0.0, 0.0])
  assert floor.group == 0
  assert floor.contype == floor.conaffinity == 1
