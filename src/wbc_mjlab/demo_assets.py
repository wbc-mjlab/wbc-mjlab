"""Bundled demo checkpoint and samples motion prerequisites."""

from __future__ import annotations

from pathlib import Path

from wbc_mjlab.data_paths import repo_root

_DEMO_CHECKPOINT_REL = Path("demos/wbc_g1/model.pt")
_SAMPLES_DATASET = "samples"


def ensure_demo_checkpoint() -> Path:
  """Return the bundled G1 demo checkpoint (``demos/wbc_g1/model.pt``)."""
  for candidate in (Path.cwd() / _DEMO_CHECKPOINT_REL, repo_root() / _DEMO_CHECKPOINT_REL):
    if candidate.is_file():
      return candidate.resolve()
  raise FileNotFoundError(
    f"Demo checkpoint not found at {_DEMO_CHECKPOINT_REL}. "
    "Use a full clone of wbc-mjlab (includes demos/wbc_g1/model.pt)."
  )


def ensure_samples_motion() -> Path:
  """Return samples motion source; requires ``wbc-mjlab-data-to-npz`` first."""
  from wbc_mjlab.data_paths import resolve_dataset_motion_file
  from wbc_mjlab.motion.stack_bundle import list_clip_npz_files

  samples_npz = repo_root() / "data" / "g1" / _SAMPLES_DATASET / "npz"
  try:
    motion_path = resolve_dataset_motion_file("g1", _SAMPLES_DATASET)
  except FileNotFoundError:
    motion_path = None

  if motion_path is None:
    raise _samples_not_converted_error(samples_npz)

  if motion_path.is_dir() and not list_clip_npz_files(motion_path):
    raise _samples_not_converted_error(samples_npz)

  if motion_path.is_file() and not motion_path.exists():
    raise _samples_not_converted_error(samples_npz)

  return motion_path.resolve()


def _samples_not_converted_error(samples_npz: Path) -> FileNotFoundError:
  return FileNotFoundError(
    "Samples motion NPZ not found. Convert bundled CSVs once, then re-run demo:\n"
    f"  uv run wbc-mjlab-data-to-npz --robot g1 --dataset {_SAMPLES_DATASET} --batch-size 16\n"
    f"Expected clips under: {samples_npz}/"
  )
