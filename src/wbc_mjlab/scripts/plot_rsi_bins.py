"""Plot adaptive RSI bin failure levels saved during training."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "stats_path",
    type=Path,
    help="Path to params/rsi_bin_stats.npz (e.g. logs/.../params/rsi_bin_stats.npz)",
  )
  parser.add_argument(
    "--trajectory",
    type=int,
    default=0,
    help="Trajectory index to visualize (default: 0)",
  )
  parser.add_argument(
    "--output",
    type=Path,
    default=None,
    help="Optional path to save the figure instead of showing it",
  )
  args = parser.parse_args()

  data = np.load(args.stats_path)
  if "failure_levels" in data:
    failure_levels = data["failure_levels"]
  else:
    failure_levels = data["failure"]
  valid = data["valid_mask"].astype(bool)
  traj_idx = int(np.clip(args.trajectory, 0, failure_levels.shape[0] - 1))

  row = np.where(valid[traj_idx], failure_levels[traj_idx], np.nan)
  fig, ax = plt.subplots(figsize=(10, 2.5))
  ax.plot(row, marker="o", linewidth=1.0)
  ax.set_title(f"RSI bin failure — trajectory {traj_idx}")
  ax.set_xlabel("bin index")
  ax.set_ylabel("failure level")
  ax.set_ylim(0.0, 1.0)
  ax.grid(True, alpha=0.3)
  fig.tight_layout()

  if args.output is not None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    print(f"Wrote {args.output}")
  else:
    plt.show()


if __name__ == "__main__":
  main()
