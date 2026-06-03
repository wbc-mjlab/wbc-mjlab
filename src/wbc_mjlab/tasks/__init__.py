"""WBC mjlab task presets and registration."""

from __future__ import annotations

from wbc_mjlab.robots.env import make_wbc_env_cfg, make_wbc_rl_cfg
from wbc_mjlab.robots.ids import resolve_robot_id
from wbc_mjlab.tasks.preset import WbcTaskPreset

DEFAULT_TASK_ID = "Wbc-G1"
TASK_ID = DEFAULT_TASK_ID  # back-compat

LEGACY_TASK_TO_ID: dict[str, str] = {
  "Wbc": DEFAULT_TASK_ID,
  "Wbc-Tracking-G1": "Wbc-G1",
  "Wbc-Tracking-G1-No-State-Estimation": "Wbc-G1-NoSE",
}

_ALL_PRESETS: tuple[WbcTaskPreset, ...] | None = None
_PRESET_BY_ID: dict[str, WbcTaskPreset] = {}
_TASKS_REGISTERED = False

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


def _load_all_presets() -> tuple[WbcTaskPreset, ...]:
  from wbc_mjlab.robots.g1.presets import G1_WBC_TASK_PRESETS

  return G1_WBC_TASK_PRESETS


def _ensure_presets() -> None:
  global _ALL_PRESETS, _PRESET_BY_ID
  if _ALL_PRESETS is not None:
    return
  _ALL_PRESETS = _load_all_presets()
  _PRESET_BY_ID.update({p.task_id: p for p in _ALL_PRESETS})


def all_wbc_task_presets() -> tuple[WbcTaskPreset, ...]:
  _ensure_presets()
  return _ALL_PRESETS  # type: ignore[return-value]


def get_task_preset(task_id: str) -> WbcTaskPreset:
  _ensure_presets()
  key = LEGACY_TASK_TO_ID.get(task_id, task_id)
  try:
    return _PRESET_BY_ID[key]
  except KeyError as exc:
    known = ", ".join(sorted(_PRESET_BY_ID))
    raise KeyError(f"Unknown WBC task {task_id!r}. Registered: {known}") from exc


def list_wbc_task_ids() -> list[str]:
  _ensure_presets()
  return sorted(_PRESET_BY_ID)


def list_robot_ids_from_presets() -> list[str]:
  _ensure_presets()
  return sorted({p.robot_id for p in _ALL_PRESETS})  # type: ignore[union-attr]


def _rl_cfg_for_preset(preset: WbcTaskPreset):
  from mjlab.rl import RslRlOnPolicyRunnerCfg

  cfg: RslRlOnPolicyRunnerCfg = make_wbc_rl_cfg(preset.robot_id)
  cfg.experiment_name = preset.experiment_name
  return cfg


def register_wbc_task_preset(preset: WbcTaskPreset) -> None:
  from mjlab.tasks.registry import register_mjlab_task

  from wbc_mjlab.rl.runner import PolicyOnlyMotionTrackingRunner

  env_kw = preset.env_kwargs()
  register_mjlab_task(
    task_id=preset.task_id,
    env_cfg=make_wbc_env_cfg(preset.robot_id, **env_kw),
    play_env_cfg=make_wbc_env_cfg(preset.robot_id, play=True, **env_kw),
    rl_cfg=_rl_cfg_for_preset(preset),
    runner_cls=PolicyOnlyMotionTrackingRunner,
  )


def register_all_wbc_tasks() -> None:
  """Register every WBC preset into mjlab (idempotent)."""
  global _TASKS_REGISTERED
  if _TASKS_REGISTERED:
    return
  _ensure_presets()
  for preset in _ALL_PRESETS:  # type: ignore[union-attr]
    register_wbc_task_preset(preset)
  _TASKS_REGISTERED = True


def prepare_wbc_run(robot_id: str, *, task_id: str) -> str:
  """Register tasks and record the active robot/task for this CLI invocation."""
  register_all_wbc_tasks()
  rid = resolve_robot_id(robot_id)
  preset = get_task_preset(task_id)
  if preset.robot_id != rid:
    raise ValueError(
      f"Task {preset.task_id!r} is for robot {preset.robot_id!r}, "
      f"but --robot is {rid!r}."
    )
  _set_active_run(rid, preset.task_id)
  return preset.task_id


def resolve_task_id(
  robot_id: str,
  *,
  task: str | None = None,
  no_state_estimation: bool = False,
) -> str:
  """Pick mjlab task id from ``--task``, legacy alias, or robot + NoSE flag."""
  rid = resolve_robot_id(robot_id)
  if task is not None:
    preset = get_task_preset(task)
    if preset.robot_id != rid:
      raise ValueError(
        f"Task {preset.task_id!r} is for robot {preset.robot_id!r}, "
        f"but --robot is {rid!r}."
      )
    return preset.task_id
  if no_state_estimation:
    return f"Wbc-{rid.upper()}-NoSE"
  return f"Wbc-{rid.upper()}"
