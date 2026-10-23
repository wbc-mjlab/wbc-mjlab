"""Can an algorithm train a task or preset? Builds a 2-env copy of its env and checks.

Usage (what the Studio runs, in its own process)::

  python -m wbc_mjlab.studio.algo_check --task <task id> --algo wm --dataset samples

Prints one line ``ALGO_CHECK <json>`` (:func:`wbc_mjlab.rl.algos.check_env`).
"""

from __future__ import annotations

import argparse
import json

MARK = "ALGO_CHECK "


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--task", required=True)
  p.add_argument("--algo", required=True)
  p.add_argument("--dataset", required=True)
  a = p.parse_args()

  import mjlab  # noqa: F401 — entry points before wbc_mjlab task modules

  from wbc_mjlab.data_paths import resolve_dataset_root
  from wbc_mjlab.rl.algos import check_env
  from wbc_mjlab.studio import presets
  from wbc_mjlab.tasks import get_task_config, prepare_wbc_run

  preset = presets.split_task_id(a.task)
  prepare_wbc_run(task_id=preset[0] if preset else a.task)
  if preset:
    presets.register(preset[1])
  robot = get_task_config(a.task).robot_id
  result = check_env(a.task, a.algo, str(resolve_dataset_root(robot, a.dataset)))
  print(MARK + json.dumps(result), flush=True)


if __name__ == "__main__":
  main()
