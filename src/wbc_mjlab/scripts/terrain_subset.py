"""Select a deterministic motion/terrain subset before environment creation."""

from __future__ import annotations

from pathlib import Path

DATASET_SOURCES = frozenset({"ego", "grail", "lafan"})


def pop_exclude_sources(argv: list[str]) -> tuple[list[str], tuple[str, ...]]:
  """Accept space/comma-separated sources and repeated exclusion flags."""
  rest = []
  excluded = set()
  i = 0
  while i < len(argv):
    arg = argv[i]
    if arg == "--exclude-sources" or arg.startswith("--exclude-sources="):
      values = []
      if "=" in arg:
        values.append(arg.split("=", 1)[1])
      else:
        while i + 1 < len(argv) and not argv[i + 1].startswith("--"):
          i += 1
          values.append(argv[i])
      names = [name.strip().lower() for value in values for name in value.split(",")]
      if not names or any(name not in DATASET_SOURCES for name in names):
        raise ValueError("--exclude-sources expects ego, grail, or lafan")
      excluded.update(names)
    else:
      rest.append(arg)
    i += 1
  return rest, tuple(sorted(excluded))


def pop_num_trajectories(argv: list[str]) -> tuple[list[str], int | None]:
  rest = []
  limit = None
  i = 0
  while i < len(argv):
    arg = argv[i]
    if arg == "--num-trajectories" or arg.startswith("--num-trajectories="):
      if limit is not None:
        raise ValueError("--num-trajectories may only be specified once")
      if "=" in arg:
        value = arg.split("=", 1)[1]
      else:
        i += 1
        if i == len(argv):
          raise ValueError("--num-trajectories requires a positive integer")
        value = argv[i]
      try:
        limit = int(value)
      except ValueError as error:
        raise ValueError("--num-trajectories requires a positive integer") from error
      if limit <= 0:
        raise ValueError("--num-trajectories requires a positive integer")
    else:
      rest.append(arg)
    i += 1
  return rest, limit


def apply_trajectory_limit(
  env_cfg,
  limit: int | None,
  *,
  motion_file: str | None = None,
  exclude_sources: tuple[str, ...] = (),
):
  """Exclude sources, then select the first N sorted motion/terrain pairs."""
  from wbc_mjlab.motion.custom_dataset import dataset_records
  from wbc_mjlab.presets.terrain_dataset import terrain_from_dataset_cfg

  motion = env_cfg.commands.get("motion")
  if motion is None or not getattr(motion, "dataset_terrain", False):
    raise ValueError("Subset selection currently requires a dataset-terrain task")
  if limit is not None and limit <= 0:
    raise ValueError("--num-trajectories requires a positive integer")
  terrain = env_cfg.scene.terrain
  if terrain is None or terrain.terrain_generator is None:
    raise ValueError("Dataset terrain generator is not configured")
  root = Path(motion_file or motion.motion_file).expanduser().resolve()
  records = dataset_records(root)
  excluded = set(exclude_sources)
  if excluded - DATASET_SOURCES:
    raise ValueError("--exclude-sources expects ego, grail, or lafan")
  eligible = [
    record
    for record in records
    if str(record.get("source") or record["id"].split("/", 1)[0]).lower()
    not in excluded
  ]
  clip_ids = tuple(record["id"] for record in eligible[:limit])
  if not clip_ids:
    raise ValueError("No trajectories remain after source exclusions")
  terrain.terrain_generator = terrain_from_dataset_cfg(
    str(root), size=terrain.terrain_generator.size, clip_ids=clip_ids
  )
  motion.motion_file = str(root)
  print(
    f"[INFO] Selected {len(clip_ids)}/{len(records)} trajectories and paired terrain tiles "
    f"(sorted IDs; excluded sources: {', '.join(sorted(excluded)) or 'none'})"
  )


def configure_terrain_selection(
  env_cfg,
  limit: int | None = None,
  *,
  motion_file: str | None = None,
  exclude_sources: tuple[str, ...] = (),
):
  """Synchronize terrain source when the motion dataset changes, even unfiltered."""
  motion = env_cfg.commands.get("motion")
  requested = limit is not None or bool(exclude_sources)
  if motion is None or not getattr(motion, "dataset_terrain", False):
    if requested:
      raise ValueError("Subset selection currently requires a dataset-terrain task")
    return
  if requested:
    apply_trajectory_limit(
      env_cfg, limit, motion_file=motion_file, exclude_sources=exclude_sources
    )
    return
  root = Path(motion_file or motion.motion_file).expanduser().resolve()
  terrain = env_cfg.scene.terrain
  generator = terrain.terrain_generator if terrain is not None else None
  patches = generator.sub_terrains.values() if generator is not None else ()
  paths = [getattr(patch, "motion_path", "") for patch in patches]
  source_changed = not paths or any(
    not path or not Path(path).is_relative_to(root) for path in paths
  )
  if source_changed:
    apply_trajectory_limit(
      env_cfg, limit, motion_file=str(root), exclude_sources=exclude_sources
    )


def apply_training_trajectory_limit(
  task_id: str,
  argv: list[str],
  limit: int | None,
  *,
  exclude_sources: tuple[str, ...] = (),
):
  """Update the training template before mjlab parses environment overrides."""
  from mjlab.tasks import registry

  motion_file = None
  for index, arg in enumerate(argv):
    for flag in ("--motion-file", "--env.commands.motion.motion-file"):
      if arg == flag and index + 1 < len(argv):
        motion_file = argv[index + 1]
      elif arg.startswith(flag + "="):
        motion_file = arg.split("=", 1)[1]
  configure_terrain_selection(
    registry._REGISTRY[task_id].env_cfg,
    limit,
    motion_file=motion_file,
    exclude_sources=exclude_sources,
  )
