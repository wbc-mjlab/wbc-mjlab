"""List WBC task presets."""

from __future__ import annotations

import sys

from wbc_mjlab.tasks import (
  all_wbc_task_presets,
  list_robot_ids_from_presets,
  register_all_wbc_tasks,
)


def main() -> None:
  register_all_wbc_tasks()
  from mjlab.scripts.list_envs import main as _mjlab_list

  print("Robots (from presets):", ", ".join(list_robot_ids_from_presets()))
  print("\nWBC task presets:")
  for p in all_wbc_task_presets():
    print(f"  {p.task_id:24}  logs/rsl_rl/{p.experiment_name}/  — {p.description}")
  print()
  _mjlab_list()


if __name__ == "__main__":
  main()
