"""Height-scan tracking on XML scenes paired with custom dataset trajectories."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

from wbc_mjlab.env.mdp.commands import MotionCommandCfg
from wbc_mjlab.motion.custom_dataset import dataset_records
from wbc_mjlab.presets.terrain import apply_terrain_scan
from wbc_mjlab.terrains.dataset import XmlTerrainCfg

DEFAULT_DATASET_ROOT = "data/g1/custom"


def terrain_yaw(trajectory_name: str, seed: int = 0) -> float:
  """Stable uniform yaw in [-pi, pi), independent of filtering and clip order."""
  digest = hashlib.sha256(f"{seed}:{trajectory_name}".encode()).digest()
  return (int.from_bytes(digest[:8], "little") / 2**64) * math.tau - math.pi


def terrain_from_dataset_cfg(
  dataset_root: str = DEFAULT_DATASET_ROOT,
  *,
  size: tuple[float, float] = (10.0, 10.0),
  clip_ids: tuple[str, ...] | None = None,
) -> TerrainGeneratorCfg:
  """One logical column per clip, physically arranged on a near-square grid.

  Tile size must cover the full scene and motion extents, including margins.
  No scene or motion data is opened while constructing this configuration.
  """
  root = Path(dataset_root).expanduser().resolve()
  # Task registration must also work on machines without the private dataset.
  records = (
    dataset_records(root, clip_ids) if (root / "manifest.json").is_file() else []
  )
  return TerrainGeneratorCfg(
    seed=0,
    size=size,
    num_rows=1,
    curriculum=True,
    color_scheme="none",
    sub_terrains={
      record["id"]: XmlTerrainCfg(
        xml_path=str(root / record["scene"]) if record.get("scene") else None,
        trajectory_name=record["id"],
        motion_path=str(root / record["motion"]),
        progress_total=len(records),
        grid_index=index,
        grid_count=len(records),
        yaw=terrain_yaw(record["id"]),
      )
      for index, record in enumerate(records)
    },
  )


def apply_terrain_dataset(
  cfg: ManagerBasedRlEnvCfg,
  *,
  scan_body_name: str,
  scan_offset: float,
  ee_termination_bodies: tuple[str, ...],
  dataset_root: str = DEFAULT_DATASET_ROOT,
  size: tuple[float, float] = (30.0, 30.0),
  clip_ids: tuple[str, ...] | None = None,
) -> None:
  """Reuse scan sensors, but keep recorded coordinates and pair reset origins."""
  tracking_terminations = {
    name: cfg.terminations[name] for name in ("anchor_pos", "ee_body_pos")
  }
  apply_terrain_scan(
    cfg,
    scan_body_name=scan_body_name,
    scan_offset=scan_offset,
    ee_termination_bodies=ee_termination_bodies,
  )
  cfg.scene.terrain = TerrainEntityCfg(
    terrain_type="generator",
    terrain_generator=terrain_from_dataset_cfg(
      dataset_root, size=size, clip_ids=clip_ids
    ),
    max_init_terrain_level=0,
  )
  cfg.curriculum.pop("terrain_levels", None)
  # Dataset reference heights already include obstacles; use world-space errors.
  cfg.terminations.update(tracking_terminations)
  motion = cfg.commands["motion"]
  assert isinstance(motion, MotionCommandCfg)
  motion.motion_file = str(Path(dataset_root).expanduser().resolve())
  motion.motion_source = "online"
  motion.ground_reference = False
  motion.dataset_terrain = True
  motion.mirror = False
  motion.debias_z = False
