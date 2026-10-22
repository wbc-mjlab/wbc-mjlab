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
  """Return the bundled samples dataset (source CSVs, played online: nothing to convert)."""
  from wbc_mjlab.data_paths import resolve_dataset_motion_file

  return resolve_dataset_motion_file("g1", _SAMPLES_DATASET).resolve()
