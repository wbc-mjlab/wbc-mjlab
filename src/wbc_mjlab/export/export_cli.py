#!/usr/bin/env python3
"""CLI: write ``wbc_tracking_params.yaml`` from a WBC task preset."""

from __future__ import annotations

import argparse
from pathlib import Path

from wbc_mjlab.export.tracking_params_yaml import write_wbc_tracking_params_yaml
from wbc_mjlab.robots.env import make_wbc_env_cfg
from wbc_mjlab.robots.ids import resolve_robot_id
from wbc_mjlab.tasks import get_task_preset, register_all_wbc_tasks, resolve_task_id


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--robot", default="g1")
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--no-state-estimation", action="store_true")
  parser.add_argument(
    "--task",
    default=None,
    help="WBC task preset (e.g. Wbc-G1-NoSE). Overrides --no-state-estimation.",
  )
  parser.add_argument(
    "--play",
    action="store_true",
    help="use play env overrides (long episode, no DR)",
  )
  args = parser.parse_args()
  register_all_wbc_tasks()

  rid = resolve_robot_id(args.robot)
  task_id = resolve_task_id(
    rid,
    task=args.task,
    no_state_estimation=args.no_state_estimation,
  )
  preset = get_task_preset(task_id)
  env_kw = preset.env_kwargs()
  cfg = make_wbc_env_cfg(rid, play=args.play, **env_kw)

  doc = write_wbc_tracking_params_yaml(
    args.out,
    cfg,
    robot_id=rid,
    has_state_estimation=preset.has_state_estimation,
  )
  print(
    f"Wrote {args.out} (task={task_id}, robot={rid}, "
    f"wbc_command_dim={doc['tracking']['wbc_command_dim']})"
  )


if __name__ == "__main__":
  main()
