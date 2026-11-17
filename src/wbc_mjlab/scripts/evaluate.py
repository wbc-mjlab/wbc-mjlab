"""Evaluate a saved tracking run; the shared backend also serves Studio.

Protocol and attribution: wbc_mjlab.rl.evaluation.
"""

import argparse
from pathlib import Path


def _split(text: str) -> tuple[str, str]:
  name, sep, path = text.partition("=")
  if not sep or not name or not path:
    raise argparse.ArgumentTypeError("expected NAME=PATH")
  return name, path


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--task", help="override task identity for legacy runs only")
  parser.add_argument("--checkpoint-file", required=True, type=Path)
  data = parser.add_mutually_exclusive_group()
  data.add_argument("--dataset")
  data.add_argument("--dataset-path")
  data.add_argument(
    "--split",
    type=_split,
    action="append",
    metavar="NAME=PATH",
    help="held-out set (repeatable); reports each split and their mean",
  )
  parser.add_argument("--num-envs", type=int, default=128)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--clips", nargs="+")
  parser.add_argument("--max-step-reward", type=float)
  parser.add_argument("--out", type=Path)
  args = parser.parse_args()
  import json

  from wbc_mjlab.rl.evaluation import (
    EvaluationConfig,
    evaluate_checkpoint,
    evaluate_splits,
  )

  splits = dict(args.split or ())
  if len(splits) != len(args.split or ()):
    parser.error("split names must be unique")
  options = {k: v for k, v in vars(args).items() if k != "split"}
  if splits:
    if args.clips:
      parser.error("--clips selects clips of one dataset; not with --split")
    result, out = evaluate_splits(EvaluationConfig(**options), splits)
    print(json.dumps(result["mean"], indent=2))
  else:
    result, out = evaluate_checkpoint(EvaluationConfig(**options))
    print(json.dumps(result["summary"], indent=2))
  print(f"Wrote {out}")


if __name__ == "__main__":
  main()
