"""Smoke test for wbc-mjlab package artifacts."""

from __future__ import annotations

import importlib.metadata as md
import sys
import traceback

try:
  import pytest
except ModuleNotFoundError:
  pytest = None  # type: ignore[assignment]


def test_package_metadata() -> None:
  """Wheel/sdist install: package imports and entry points are present."""
  import wbc_mjlab

  assert wbc_mjlab.__name__ == "wbc_mjlab"
  assert wbc_mjlab.__version__

  dist = md.distribution("wbc-mjlab")
  scripts = {ep.name for ep in dist.entry_points if ep.group == "console_scripts"}
  assert "wbc-mjlab-list-envs" in scripts
  task_plugins = {ep.name for ep in dist.entry_points if ep.group == "mjlab.tasks"}
  assert "wbc_mjlab" in task_plugins


@pytest.mark.slow if pytest else lambda f: f
def test_import_and_list_tasks() -> None:
  """Full stack: mjlab import + WBC task registry."""
  from wbc_mjlab.tasks import list_wbc_task_ids, register_all_wbc_tasks

  register_all_wbc_tasks()
  task_ids = list_wbc_task_ids()
  assert "Wbc-G1" in task_ids
  assert "Wbc-G1-Zest" in task_ids


if __name__ == "__main__":
  try:
    test_package_metadata()
    test_import_and_list_tasks()
    print("Smoke test passed!")
    sys.exit(0)
  except Exception:
    print("Smoke test failed:")
    traceback.print_exc()
    sys.exit(1)
