"""Mjlab ``mjlab.tasks`` entry point (standalone module — avoids ``tasks`` import cycles)."""

from __future__ import annotations

_BOOTSTRAPPED = False


def register_default() -> None:
  global _BOOTSTRAPPED
  if _BOOTSTRAPPED:
    return
  from wbc_mjlab.tasks import DEFAULT_TASK_ID, register_all_wbc_tasks, _set_active_run

  register_all_wbc_tasks()
  _set_active_run("g1", DEFAULT_TASK_ID)
  _BOOTSTRAPPED = True


register_default()
