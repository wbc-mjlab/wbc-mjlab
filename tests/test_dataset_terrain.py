"""Small, CPU-only checks; no model compilation or simulation."""

from __future__ import annotations

import json
from types import SimpleNamespace

import mujoco
import numpy as np
import torch

from wbc_mjlab.env.mdp.commands import MotionCommand
from wbc_mjlab.motion.dataset import load_dataset
from wbc_mjlab.presets.terrain_dataset import (
  apply_terrain_dataset,
  terrain_from_dataset_cfg,
)
from wbc_mjlab.robots.g1.tasks import g1_wbc_env_cfg
from wbc_mjlab.terrains.dataset import XmlTerrainCfg


def test_xml_import_preserves_geometry_and_excludes_robot(tmp_path):
  path = tmp_path / "scene.xml"
  path.write_text("""<mujoco><worldbody>
    <geom name="ground" type="plane" size="10 10 .1"/>
    <geom name="step" type="box" size="1 .5 .2" pos="2 -1 .2"
          quat=".9238795325 0 0 .3826834324"/>
    <body name="robot"><freejoint/><geom type="sphere" size=".1"/></body>
  </worldbody></mujoco>""")
  cfg = XmlTerrainCfg(
    xml_path=str(path), trajectory_name="clip", size=(12.0, 8.0), floor_clearance=None
  )
  spec = mujoco.MjSpec()
  spec.worldbody.add_body(name="terrain")
  original = path.read_bytes()
  first = cfg.function(0, spec, np.random.default_rng(1))
  second = cfg.function(1, spec, np.random.default_rng(2))
  assert len(first.geometries) == len(second.geometries) == 2
  assert len(spec.joints) == 0
  assert len({geom.name for geom in spec.geoms}) == 5
  ground, step = [item.geom for item in first.geometries]
  assert ground.type == mujoco.mjtGeom.mjGEOM_BOX
  np.testing.assert_allclose(ground.size, [6, 4, 0.05])
  np.testing.assert_allclose(ground.pos, [6, 4, -0.05])
  np.testing.assert_allclose(step.pos, [8, 3, 0.2])
  np.testing.assert_allclose(step.quat, [0.9238795325, 0, 0, 0.3826834324])
  assert path.read_bytes() == original


def test_nested_static_transforms_and_fromto_survive_tile_translation(tmp_path):
  path = tmp_path / "nested.xml"
  path.write_text("""<mujoco><compiler angle="degree"/>
    <default><default class="obstacle"><geom friction=".7 .01 .002"/></default></default>
    <worldbody>
      <body name="obstacle" pos="1 2 0" euler="0 0 90" childclass="obstacle">
        <body pos="2 0 0"><frame pos="0 1 0">
          <geom name="box" type="box" size=".5 .4 .3" pos="1 0 0"/>
          <geom name="bar" type="capsule" size=".1" fromto="0 0 0 2 0 0"/>
        </frame></body>
      </body>
      <body name="robot"><freejoint/><geom type="sphere" size=".1"/></body>
    </worldbody></mujoco>""")
  spec = mujoco.MjSpec()
  spec.worldbody.add_body(name="terrain")
  cfg = XmlTerrainCfg(
    xml_path=str(path), trajectory_name="nested", size=(12.0, 8.0), floor_clearance=None
  )
  output = cfg.function(0.0, spec, np.random.default_rng(0))
  assert len(spec.bodies) == 2
  assert len(spec.joints) == 0
  assert len(output.geometries) == 2
  for item in output.geometries:
    geom = item.geom
    np.testing.assert_allclose(geom.pos, [6.0, 9.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(geom.friction, [0.7, 0.01, 0.002])
    assert geom.parent.name == "terrain"
    axis = (
      np.array([0.0, 0.0, 1.0])
      if geom.type == mujoco.mjtGeom.mjGEOM_CAPSULE
      else np.array([1.0, 0.0, 0.0])
    )
    rotated = np.empty(3)
    mujoco.mju_rotVecQuat(rotated, axis, geom.quat)
    np.testing.assert_allclose(rotated, [0.0, 1.0, 0.0], atol=1e-12)
    # TerrainGenerator adds the tile offset to geom.pos in world axes.
    geom.pos += np.array([10.0, -20.0, 0.0])
    np.testing.assert_allclose(geom.pos, [16.0, -11.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(geom.frame.pos, [0.0, 0.0, 0.0])
    if geom.type == mujoco.mjtGeom.mjGEOM_CAPSULE:
      np.testing.assert_allclose(geom.size[:2], [0.1, 1.0])
      assert np.isnan(geom.fromto[0])


def test_manifest_motion_names_rates_and_selected_scene(tmp_path):
  records = []
  for name, fps in (("a", 30), ("b", 60)):
    directory = tmp_path / "clips" / name
    directory.mkdir(parents=True)
    (directory / "motion.csv").write_text("0,0,1,0,0,0,1,.2,.4\n" * 2)
    (directory / "scene.xml").write_text(
      '<mujoco><worldbody><geom type="plane" size="1 1 .1"/></worldbody></mujoco>'
    )
    records.append(
      dict(
        id=name,
        fps=fps,
        motion=f"clips/{name}/motion.csv",
        scene=f"clips/{name}/scene.xml",
      )
    )
  (tmp_path / "manifest.json").write_text(
    json.dumps(dict(clips=records, joint_names=["j2", "j1"]))
  )
  packed = load_dataset(tmp_path, ["j1", "j2"], cache=False, verbose=False)
  assert packed.names == ["a", "b"]
  np.testing.assert_allclose(packed.fps, [30, 60])
  np.testing.assert_allclose(packed.frames[:, 7:], [[0.4, 0.2]] * 4)
  selected = load_dataset(
    tmp_path, ["j1", "j2"], cache=False, verbose=False, clip_ids=("b",)
  )
  assert selected.names == ["b"]
  cfg = g1_wbc_env_cfg()
  original_anchor = cfg.terminations["anchor_pos"]
  apply_terrain_dataset(
    cfg,
    scan_body_name="pelvis",
    scan_offset=0.8,
    ee_termination_bodies=("pelvis",),
    dataset_root=str(tmp_path),
    clip_ids=("b",),
  )
  assert list(cfg.scene.terrain.terrain_generator.sub_terrains) == ["b"]
  assert cfg.commands["motion"].dataset_terrain
  assert not cfg.commands["motion"].ground_reference
  assert cfg.terminations["anchor_pos"] is original_anchor
  assert "terrain_levels" not in cfg.curriculum


def test_reset_origin_follows_trajectory_for_reset_environments_only():
  terrain = SimpleNamespace(
    terrain_types=torch.zeros(3, dtype=torch.long),
    terrain_levels=torch.zeros(3, dtype=torch.long),
    env_origins=torch.zeros(3, 3),
    terrain_origins=torch.tensor([[[10.0, 0.0, 0.0], [20.0, 0.0, 0.0]]]),
  )
  command = SimpleNamespace(
    _trajectory_terrain_columns=torch.tensor([1, 0]),
    trajectory_ids=torch.tensor([0, 1, 0]),
    _env=SimpleNamespace(scene=SimpleNamespace(terrain=terrain)),
  )
  MotionCommand._update_terrain_origin(command, torch.tensor([0, 1]))
  torch.testing.assert_close(terrain.env_origins[:, 0], torch.tensor([20.0, 10.0, 0.0]))
  torch.testing.assert_close(terrain.terrain_types, torch.tensor([1, 0, 0]))


def test_missing_scenes_use_flat_tiles_and_preserve_pairing(tmp_path):
  (tmp_path / "scene.xml").write_text(
    '<mujoco><worldbody><geom name="step" type="box" '
    'size="1 1 .2" pos="0 0 .2"/></worldbody></mujoco>'
  )
  records = [
    dict(id="absent", motion="absent.csv", fps=30),
    dict(id="empty", motion="empty.csv", fps=30, scene=""),
    dict(id="missing", motion="missing.csv", fps=30, scene="missing.xml"),
    dict(id="null", motion="null.csv", fps=30, scene=None),
    dict(id="xml", motion="xml.csv", fps=30, scene="scene.xml"),
  ]
  (tmp_path / "manifest.json").write_text(json.dumps(dict(clips=records)))
  generator = terrain_from_dataset_cfg(str(tmp_path), size=(12.0, 8.0))
  assert list(generator.sub_terrains) == [r["id"] for r in records]
  for name, cfg in generator.sub_terrains.items():
    assert cfg.trajectory_name == name
    assert cfg.motion_path == str(tmp_path / f"{name}.csv")
    cfg.floor_clearance = None
    cfg.grid_index = None  # Check local fallback geometry independently of placement.
    # TerrainGenerator normally propagates its tile size before generation.
    cfg.size = generator.size
    spec = mujoco.MjSpec()
    spec.worldbody.add_body(name="terrain")
    output = cfg.function(0.0, spec, np.random.default_rng(0))
    np.testing.assert_allclose(output.origin, [6.0, 4.0, 0.0])
    assert len(output.geometries) == 1
    geom = output.geometries[0].geom
    assert geom.type == mujoco.mjtGeom.mjGEOM_BOX
    if name == "xml":
      np.testing.assert_allclose(geom.size, [1.0, 1.0, 0.2])
    else:
      np.testing.assert_allclose(geom.size[:2], [6.0, 4.0])
      np.testing.assert_allclose(geom.pos[:2], output.origin[:2])
      assert np.isclose(geom.pos[2] + geom.size[2], 0.0)
