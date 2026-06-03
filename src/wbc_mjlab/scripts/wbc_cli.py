"""Shared CLI flags for WBC mjlab tasks (``Wbc-G1``, ``Wbc-G1-NoSE``, …)."""

from __future__ import annotations

from wbc_mjlab.tasks import (
  LEGACY_TASK_TO_ID,
  list_wbc_task_ids,
  resolve_task_id,
  robot_id_for_run,
)

DEFAULT_ROBOT = "g1"
_KNOWN_TASK_IDS = frozenset(list_wbc_task_ids()) | frozenset(LEGACY_TASK_TO_ID)


def parse_wbc_argv(
  argv: list[str],
) -> tuple[list[str], str, str, bool, bool, str | None, str | None, bool]:
  """Strip WBC flags; return (rest, robot, task_id, no_se, legacy, dataset, dataset_path, cache_motion_bundle).

  ``robot`` is taken from the task preset unless ``--robot`` was passed explicitly.
  """
  rest: list[str] = []
  robot = DEFAULT_ROBOT
  robot_explicit = False
  task: str | None = None
  no_se = False
  used_legacy = False
  dataset: str | None = None
  dataset_path: str | None = None
  cache_motion_bundle = False
  i = 0
  while i < len(argv):
    arg = argv[i]
    if arg.startswith("--robot="):
      robot = arg.split("=", 1)[1]
      robot_explicit = True
      i += 1
      continue
    if arg == "--robot":
      if i + 1 >= len(argv):
        raise ValueError("--robot requires a value (e.g. g1)")
      robot = argv[i + 1]
      robot_explicit = True
      i += 2
      continue
    if arg.startswith("--task="):
      task = arg.split("=", 1)[1]
      i += 1
      continue
    if arg == "--task":
      if i + 1 >= len(argv):
        raise ValueError("--task requires a value (e.g. Wbc-G1-Zest)")
      task = argv[i + 1]
      i += 2
      continue
    if arg.startswith("--dataset-path="):
      dataset_path = arg.split("=", 1)[1]
      i += 1
      continue
    if arg == "--dataset-path":
      if i + 1 >= len(argv):
        raise ValueError("--dataset-path requires a directory or .npz path")
      dataset_path = argv[i + 1]
      i += 2
      continue
    if arg.startswith("--dataset="):
      dataset = arg.split("=", 1)[1]
      i += 1
      continue
    if arg == "--dataset":
      if i + 1 >= len(argv):
        raise ValueError("--dataset requires a name (e.g. lafan)")
      dataset = argv[i + 1]
      i += 2
      continue
    if arg == "--no-state-estimation":
      no_se = True
      i += 1
      continue
    if arg == "--cache-motion-bundle":
      cache_motion_bundle = True
      i += 1
      continue
    if arg in _KNOWN_TASK_IDS:
      used_legacy = True
      if task is None:
        task = arg
      if arg.endswith("No-State-Estimation"):
        no_se = True
      i += 1
      continue
    rest.append(arg)
    i += 1

  task_id = resolve_task_id(
    robot if robot_explicit else None,
    task=task,
    no_state_estimation=no_se,
  )
  robot = robot_id_for_run(
    task_id=task_id,
    robot_id=robot,
    robot_explicit=robot_explicit,
  )
  return rest, robot, task_id, no_se, used_legacy, dataset, dataset_path, cache_motion_bundle


def _has_explicit_motion_file(argv: list[str]) -> bool:
  return any(
    a == "--motion-file"
    or a.startswith("--motion-file=")
    or a == "--env.commands.motion.motion-file"
    or a.startswith("--env.commands.motion.motion-file=")
    for a in argv
  )


def apply_dataset_motion_file(
  argv: list[str],
  *,
  robot: str,
  dataset: str | None = None,
  dataset_path: str | None = None,
  cache_motion_bundle: bool = False,
) -> list[str]:
  """Map ``--dataset`` / ``--dataset-path`` to ``--motion-file`` when not set explicitly."""
  if dataset is None and dataset_path is None:
    return argv
  if _has_explicit_motion_file(argv):
    return argv

  from wbc_mjlab.data_paths import resolve_training_motion_file

  path = resolve_training_motion_file(
    robot,
    dataset=dataset,
    dataset_path=dataset_path,
    cache_motion_bundle=cache_motion_bundle,
  )
  return [*argv, "--motion-file", str(path)]


def ensure_task_id(argv: list[str], task_id: str) -> list[str]:
  """Prepend the mjlab task id when not already present."""
  if argv and argv[0] in _KNOWN_TASK_IDS:
    return argv
  return [task_id, *argv]
