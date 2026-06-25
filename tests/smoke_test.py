"""Smoke test for wbc-mjlab package artifacts."""

from __future__ import annotations

import sys

try:
  import pytest
except ModuleNotFoundError:
  pytest = None  # type: ignore[assignment]


@pytest.mark.slow if pytest else lambda f: f
def test_import_and_list_tasks() -> None:
  import wbc_mjlab
  from wbc_mjlab.tasks import list_wbc_task_ids, register_all_wbc_tasks

  assert wbc_mjlab.__name__ == "wbc_mjlab"
  register_all_wbc_tasks()
  task_ids = list_wbc_task_ids()
  assert "Wbc-G1" in task_ids
  assert "Wbc-G1-Zest" in task_ids


if __name__ == "__main__":
  try:
    test_import_and_list_tasks()
    print("Smoke test passed!")
    sys.exit(0)
  except Exception as exc:
    print(f"Smoke test failed: {exc}")
    sys.exit(1)
