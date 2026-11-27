"""Check grid placement and clip origins without model compilation."""

import json

import mujoco
import numpy as np

from wbc_mjlab.presets.terrain_dataset import terrain_from_dataset_cfg
from wbc_mjlab.terrains.dataset import DATASET_FLOOR_NAME


def test_five_clips_form_centered_grid_with_matching_geometry_and_origins(tmp_path):
  records = [dict(id=f"clip{i}", motion=f"{i}.csv", fps=30) for i in range(5)]
  (tmp_path / "manifest.json").write_text(json.dumps(dict(clips=records)))
  generator = terrain_from_dataset_cfg(str(tmp_path), size=(12.0, 8.0))
  assert generator.num_rows == 1  # Clip IDs still index logical columns.
  spec = mujoco.MjSpec()
  spec.worldbody.add_body(name="terrain")
  expected = [(-6.0, -8.0), (-6.0, 0.0), (-6.0, 8.0), (6.0, -8.0), (6.0, 0.0)]
  for index, cfg in enumerate(generator.sub_terrains.values()):
    assert cfg.trajectory_name == f"clip{index}"
    cfg.motion_path = ""  # No trajectory data needs to be loaded for placement.
    cfg.progress_total = 0
    cfg.size = generator.size
    output = cfg.function(0.0, spec, np.random.default_rng(0))
    # Same world-position translation as TerrainGenerator._create_terrain_geom.
    logical_corner = np.array([-6.0, (index - 2.5) * 8.0, 0.0])
    origin = output.origin + logical_corner
    np.testing.assert_allclose(origin[:2], expected[index])
    for item in output.geometries:
      item.geom.pos += logical_corner
      np.testing.assert_allclose(item.geom.pos[:2], origin[:2])
      assert item.geom.pos[2] - item.geom.size[2] >= 0.01 - 1e-12
  np.testing.assert_allclose(spec.geom(DATASET_FLOOR_NAME).pos, [0.0, 0.0, 0.0])


def test_subset_recomputes_grid_indices(tmp_path):
  records = [dict(id=f"clip{i}", motion=f"{i}.csv", fps=30) for i in range(5)]
  (tmp_path / "manifest.json").write_text(json.dumps(dict(clips=records)))
  generator = terrain_from_dataset_cfg(str(tmp_path), clip_ids=("clip1", "clip4"))
  assert [cfg.grid_index for cfg in generator.sub_terrains.values()] == [0, 1]
  assert all(cfg.grid_count == 2 for cfg in generator.sub_terrains.values())
