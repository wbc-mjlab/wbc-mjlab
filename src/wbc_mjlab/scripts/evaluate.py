"""Evaluate a saved tracking run; the shared backend also serves Studio.

Protocol and attribution: wbc_mjlab.rl.evaluation.
"""

import argparse
from pathlib import Path


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--task", help="override task identity for legacy runs only")
  parser.add_argument("--checkpoint-file", required=True, type=Path)
  data = parser.add_mutually_exclusive_group()
  data.add_argument("--dataset")
  data.add_argument("--dataset-path")
  parser.add_argument("--num-envs", type=int, default=128)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--clips", nargs="+")
  parser.add_argument("--max-step-reward", type=float)
  parser.add_argument("--out", type=Path)
  args = parser.parse_args()
  from wbc_mjlab.rl.evaluation import EvaluationConfig, evaluate_checkpoint

  result, out = evaluate_checkpoint(EvaluationConfig(**vars(args)))
  import json

  print(json.dumps(result["summary"], indent=2))
  print(f"Wrote {out}")


if __name__ == "__main__":
  main()
