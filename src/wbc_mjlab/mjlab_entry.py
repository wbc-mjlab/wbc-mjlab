"""Mjlab ``mjlab.tasks`` entry point (standalone module — avoids ``tasks`` import cycles)."""

from __future__ import annotations

_BOOTSTRAPPED = False
_DEFAULT_TASK_ID = "Wbc-G1"


def register_default() -> None:
  """Register WBC tasks with mjlab (idempotent)."""
  global _BOOTSTRAPPED
  if _BOOTSTRAPPED:
    return

  # Import the package module only after ``tasks.config`` finished loading
  # (``tasks.config`` must not import mjlab — see tasks/config.py).
  from wbc_mjlab import tasks as tasks_mod

  tasks_mod.register_all_wbc_tasks()
  tasks_mod._set_active_run("g1", _DEFAULT_TASK_ID)
  _BOOTSTRAPPED = True


register_default()
