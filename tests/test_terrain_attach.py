"""Ensure terrain import never enumerates the growing target geometry list."""

import mujoco
import numpy as np
import pytest

from wbc_mjlab.terrains.dataset import XmlTerrainCfg


class NoGlobalGeomScan:
  """Forward spec operations but reject target-wide geometry enumeration."""

  def __init__(self, spec):
    self.spec = spec

  @property
  def geoms(self):
    raise AssertionError("Terrain import must not enumerate target spec.geoms")

  def __getattr__(self, name):
    return getattr(self.spec, name)


@pytest.mark.parametrize("copy_attach", [False, True])
def test_attachment_returns_live_isolated_geometries_without_global_scan(
  tmp_path, copy_attach
):
  path = tmp_path / "scene.xml"
  path.write_text("""<mujoco><worldbody>
    <geom name="step" type="box" size="1 .5 .2" pos="2 0 .2"/>
  </worldbody></mujoco>""")
  target = mujoco.MjSpec()
  target.copy_during_attach = copy_attach
  target.worldbody.add_body(name="terrain")
  spec = NoGlobalGeomScan(target)
  cfg = XmlTerrainCfg(xml_path=str(path), trajectory_name="clip", floor_clearance=None)
  first = cfg.function(0.0, spec, np.random.default_rng(0))
  second = cfg.function(0.0, spec, np.random.default_rng(0))
  first_geom, second_geom = first.geometries[0].geom, second.geometries[0].geom
  assert first_geom.name != second_geom.name
  before = first_geom.pos.copy()
  second_geom.pos += np.array([100.0, 200.0, 0.0])
  np.testing.assert_allclose(target.geom(second_geom.name).pos, second_geom.pos)
  np.testing.assert_allclose(target.geom(first_geom.name).pos, before)
  assert len(target.geoms) == 3  # Shared floor plus two imported steps.
