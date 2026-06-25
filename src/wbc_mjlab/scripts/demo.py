"""Interactive play demo: bundled checkpoint + samples clip library.

Requires a one-time conversion of bundled CSVs to NPZ:

  uv run wbc-mjlab-data-to-npz --robot g1 --dataset samples --batch-size 16
"""

from __future__ import annotations

import sys

import mjlab
import tyro
from mjlab.scripts.play import PlayConfig

from wbc_mjlab.demo_assets import ensure_demo_checkpoint, ensure_samples_motion
from wbc_mjlab.scripts.play import run_play
from wbc_mjlab.tasks import prepare_wbc_run

DEFAULT_TASK_ID = "Wbc-G1"


def main() -> None:
  print("Setting up wbc-mjlab demo (bundled checkpoint + samples clip library)...")

  try:
    checkpoint = ensure_demo_checkpoint()
    motion_file = ensure_samples_motion()
  except FileNotFoundError as exc:
    print(f"Demo setup failed:\n{exc}")
    sys.exit(1)

  print(f"[INFO] Checkpoint: {checkpoint}")
  print(f"[INFO] Motion library: {motion_file}")

  prepare_wbc_run(task_id=DEFAULT_TASK_ID)
  args = tyro.cli(
    PlayConfig,
    default=PlayConfig(
      checkpoint_file=str(checkpoint),
      motion_file=str(motion_file),
      num_envs=8,
      viewer="viser",
      _demo_mode=True,
    ),
    config=mjlab.TYRO_FLAGS,
  )
  run_play(DEFAULT_TASK_ID, args)


if __name__ == "__main__":
  main()
