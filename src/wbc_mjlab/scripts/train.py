"""Train WBC motion tracking for a chosen task preset and robot.

Usage::

  wbc-mjlab-train --robot g1 --dataset lafan
  wbc-mjlab-train --task Wbc-G1-Zest --robot g1 --dataset lafan
  wbc-mjlab-train --robot g1 --no-state-estimation --dataset lafan
"""

from __future__ import annotations

import logging
import sys

import mjlab.utils.os as _mjlab_os

from wbc_mjlab.tasks import prepare_wbc_run
from wbc_mjlab.scripts.wbc_cli import (
  apply_dataset_motion_file,
  ensure_task_id,
  parse_wbc_argv,
)

log = logging.getLogger(__name__)


def _has_explicit_agent_logger(argv: list[str]) -> bool:
  return any(a == "--agent.logger" or a.startswith("--agent.logger=") for a in argv)


def _filter_argv_for_wandb_flag(argv: list[str]) -> tuple[list[str], bool]:
  use = False
  out: list[str] = []
  i = 0
  while i < len(argv):
    if argv[i] == "--use_wandb":
      use = True
      i += 1
      continue
    out.append(argv[i])
    i += 1
  return out, use


def _apply_motion_file_shortcut(argv: list[str]) -> list[str]:
  out: list[str] = []
  i = 0
  while i < len(argv):
    arg = argv[i]
    if arg.startswith("--motion-file="):
      out.extend(["--env.commands.motion.motion-file", arg.split("=", 1)[1]])
      i += 1
      continue
    if arg == "--motion-file":
      out.extend(["--env.commands.motion.motion-file", argv[i + 1]])
      i += 2
      continue
    out.append(arg)
    i += 1
  return out


def _apply_default_logger(argv: list[str], *, use_wandb: bool) -> list[str]:
  if use_wandb or _has_explicit_agent_logger(argv):
    return argv
  return [*argv, "--agent.logger", "tensorboard"]


def _safe_dump_yaml(filename, data, sort_keys: bool = False) -> None:
  import yaml
  from pathlib import Path

  def _sanitize(obj):
    if isinstance(obj, dict):
      return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
      return type(obj)(_sanitize(v) for v in obj)
    if isinstance(obj, (str, int, float, bool, type(None))):
      return obj
    if callable(obj):
      return f"{obj.__module__}.{obj.__qualname__}"
    try:
      yaml.dump(obj)
      return obj
    except Exception:
      return f"<non-serializable: {type(obj).__name__}>"

  filename = Path(filename)
  if not filename.suffix:
    filename = filename.with_suffix(".yaml")
  filename.parent.mkdir(parents=True, exist_ok=True)
  with open(filename, "w") as f:
    yaml.dump(_sanitize(data), f, sort_keys=sort_keys)


_mjlab_os.dump_yaml = _safe_dump_yaml


def main() -> None:
  prog = sys.argv[0]
  rest, robot, task_id, no_se, legacy, dataset, dataset_path = parse_wbc_argv(
    sys.argv[1:]
  )
  if legacy:
    log.warning("Legacy task id mapped to --task / --no-state-estimation")
  prepare_wbc_run(robot, task_id=task_id)

  rest, use_wandb = _filter_argv_for_wandb_flag(rest)
  rest = apply_dataset_motion_file(
    rest, robot=robot, dataset=dataset, dataset_path=dataset_path
  )
  rest = _apply_motion_file_shortcut(rest)
  rest = _apply_default_logger(rest, use_wandb=use_wandb)
  rest = ensure_task_id(rest, task_id)
  sys.argv = [prog, *rest]

  from mjlab.scripts.train import main as _mjlab_train_main

  _mjlab_train_main()


if __name__ == "__main__":
  main()
