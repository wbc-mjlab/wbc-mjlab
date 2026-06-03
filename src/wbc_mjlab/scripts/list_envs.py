"""List WBC tasks."""

from __future__ import annotations

from wbc_mjlab.tasks import (
  all_wbc_tasks,
  list_robot_ids_from_tasks,
  register_all_wbc_tasks,
)


def main() -> None:
  register_all_wbc_tasks()
  from mjlab.scripts.list_envs import main as _mjlab_list

  print("Robots:", ", ".join(list_robot_ids_from_tasks()))
  print("\nWBC tasks:")
  for task in all_wbc_tasks():
    print(f"  {task.task_id:24}  logs/rsl_rl/{task.experiment_name}/  — {task.description}")
  print()
  _mjlab_list()


if __name__ == "__main__":
  main()
