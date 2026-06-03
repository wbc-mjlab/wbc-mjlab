#!/usr/bin/env python3
"""CLI: write ``wbc_tracking_params.yaml`` from a WBC task config."""

from __future__ import annotations

import argparse
from pathlib import Path

from wbc_mjlab.export.tracking_params_yaml import write_wbc_tracking_params_yaml
from wbc_mjlab.robots.env import make_wbc_env_cfg
from wbc_mjlab.tasks import (
  get_task_config,
  register_all_wbc_tasks,
  resolve_task_id,
  robot_id_for_run,
)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--robot",
    default=None,
    help="optional; inferred from --task when omitted",
  )
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--no-state-estimation", action="store_true")
  parser.add_argument(
    "--task",
    default=None,
    help="WBC task id (e.g. Wbc-G1-NoSE). Overrides --no-state-estimation.",
  )
  parser.add_argument(
    "--play",
    action="store_true",
    help="use play env overrides (long episode, no DR)",
  )
  args = parser.parse_args()
  register_all_wbc_tasks()

  task_id = resolve_task_id(
    args.robot,
    task=args.task,
    no_state_estimation=args.no_state_estimation,
  )
  rid = robot_id_for_run(
    task_id=task_id,
    robot_id=args.robot,
    robot_explicit=args.robot is not None,
  )
  task = get_task_config(task_id)
  cfg = make_wbc_env_cfg(rid, play=args.play, task_id=task.task_id)

  has_se = "motion_anchor_pos_b" in cfg.observations["actor"].terms
  doc = write_wbc_tracking_params_yaml(
    args.out,
    cfg,
    robot_id=rid,
    has_state_estimation=has_se,
  )
  print(
    f"Wrote {args.out} (task={task_id}, robot={rid}, "
    f"wbc_command_dim={doc['tracking']['wbc_command_dim']})"
  )


if __name__ == "__main__":
  main()
