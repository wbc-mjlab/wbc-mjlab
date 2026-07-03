"""WBC mjlab task configs and registration."""

from __future__ import annotations

from collections.abc import Iterable

from wbc_mjlab.robots.ids import resolve_robot_id
from wbc_mjlab.tasks.config import WbcTaskConfig

DEFAULT_TASK_ID = "Wbc-G1"
TASK_ID = DEFAULT_TASK_ID  # back-compat

LEGACY_TASK_TO_ID: dict[str, str] = {
  "Wbc": DEFAULT_TASK_ID,
  "Wbc-Tracking-G1": "Wbc-G1",
  "Wbc-Tracking-G1-No-State-Estimation": "Wbc-G1-Zest",
  "Wbc-G1-NoSE": "Wbc-G1-Zest",
}

_TASK_BY_ID: dict[str, WbcTaskConfig] = {}
_BUILTIN_TASKS_LOADED = False
_TASKS_REGISTERED = False
_REGISTERING = False
_MJLAB_REGISTERED: set[str] = set()

_LAST_ROBOT_ID = "g1"
_LAST_TASK_ID = DEFAULT_TASK_ID


def last_registered_robot_id() -> str:
  return _LAST_ROBOT_ID


def last_registered_task_id() -> str:
  return _LAST_TASK_ID


def _set_active_run(robot_id: str, task_id: str) -> None:
  global _LAST_ROBOT_ID, _LAST_TASK_ID
  _LAST_ROBOT_ID = robot_id
  _LAST_TASK_ID = task_id


def _load_builtin_tasks() -> None:
  global _BUILTIN_TASKS_LOADED
  if _BUILTIN_TASKS_LOADED:
    return
  from wbc_mjlab.robots.g1.tasks import G1_WBC_TASKS

  for task in G1_WBC_TASKS:
    _TASK_BY_ID.setdefault(task.task_id, task)
  _BUILTIN_TASKS_LOADED = True


def _ensure_tasks() -> None:
  _load_builtin_tasks()


def all_wbc_tasks() -> tuple[WbcTaskConfig, ...]:
  _ensure_tasks()
  return tuple(_TASK_BY_ID[tid] for tid in sorted(_TASK_BY_ID))


def get_task_config(task_id: str) -> WbcTaskConfig:
  _ensure_tasks()
  key = LEGACY_TASK_TO_ID.get(task_id, task_id)
  try:
    return _TASK_BY_ID[key]
  except KeyError as exc:
    known = ", ".join(sorted(_TASK_BY_ID))
    raise KeyError(f"Unknown WBC task {task_id!r}. Registered: {known}") from exc


def list_wbc_task_ids() -> list[str]:
  _ensure_tasks()
  return sorted(_TASK_BY_ID)


def list_robot_ids_from_tasks() -> list[str]:
  _ensure_tasks()
  return sorted({t.robot_id for t in _TASK_BY_ID.values()})


def _rl_cfg_for_task(task: WbcTaskConfig):
  from mjlab.rl import RslRlOnPolicyRunnerCfg

  from wbc_mjlab.robots.env import make_wbc_rl_cfg

  cfg: RslRlOnPolicyRunnerCfg = make_wbc_rl_cfg(task.robot_id)
  cfg.experiment_name = task.experiment_name
  return cfg


def register_wbc_task(task: WbcTaskConfig) -> None:
  from mjlab.tasks.registry import register_mjlab_task

  from wbc_mjlab.robots.env import make_wbc_env_cfg
  from wbc_mjlab.rl.runner import PolicyOnlyMotionTrackingRunner

  register_mjlab_task(
    task_id=task.task_id,
    env_cfg=make_wbc_env_cfg(task.robot_id, task_id=task.task_id),
    play_env_cfg=make_wbc_env_cfg(task.robot_id, play=True, task_id=task.task_id),
    rl_cfg=_rl_cfg_for_task(task),
    runner_cls=PolicyOnlyMotionTrackingRunner,
  )
  _MJLAB_REGISTERED.add(task.task_id)


def register_wbc_tasks(
  tasks: WbcTaskConfig | Iterable[WbcTaskConfig],
) -> None:
  """Add tasks to the WBC task table; register with mjlab when bootstrap is done."""
  if not _REGISTERING:
    _load_builtin_tasks()

  batch = (tasks,) if isinstance(tasks, WbcTaskConfig) else tuple(tasks)
  for task in batch:
    _TASK_BY_ID[task.task_id] = task

  if _TASKS_REGISTERED and not _REGISTERING:
    for task in batch:
      if task.task_id not in _MJLAB_REGISTERED:
        register_wbc_task(task)


def register_all_wbc_tasks() -> None:
  """Register every known WBC task into mjlab (idempotent)."""
  global _TASKS_REGISTERED, _REGISTERING
  if _TASKS_REGISTERED or _REGISTERING:
    return
  _REGISTERING = True
  try:
    _load_builtin_tasks()
    from wbc_mjlab.robots.env import _register_g1

    _register_g1()
    for task_id in sorted(_TASK_BY_ID):
      if task_id not in _MJLAB_REGISTERED:
        register_wbc_task(_TASK_BY_ID[task_id])
    _TASKS_REGISTERED = True
  finally:
    _REGISTERING = False


def resolve_task_id(
  robot_id: str | None = None,
  *,
  task: str | None = None,
  no_state_estimation: bool = False,
) -> str:
  """Pick mjlab task id from ``--task``, legacy alias, or robot + NoSE flag."""
  if task is not None:
    return get_task_config(task).task_id
  rid = resolve_robot_id(robot_id or "g1")
  if no_state_estimation:
    return f"Wbc-{rid.upper()}-Zest"
  return f"Wbc-{rid.upper()}"


def robot_id_for_run(
  *,
  task_id: str,
  robot_id: str | None = None,
  robot_explicit: bool = False,
) -> str:
  """Return robot id for data paths; default comes from the task config."""
  task = get_task_config(task_id)
  if robot_explicit:
    rid = resolve_robot_id(robot_id or "g1")
    if task.robot_id != rid:
      raise ValueError(
        f"Task {task.task_id!r} is for robot {task.robot_id!r}, "
        f"but --robot is {rid!r}."
      )
    return rid
  return task.robot_id


def prepare_wbc_run(*, task_id: str) -> str:
  """Register tasks and record the active robot/task for this CLI invocation."""
  import mjlab  # noqa: F401 — load mjlab.tasks entry points (external robots)

  register_all_wbc_tasks()
  task = get_task_config(task_id)
  _set_active_run(task.robot_id, task.task_id)
  return task.task_id
