"""Metadata for custom datasets with explicitly paired motions and XML scenes."""

from __future__ import annotations

import json
from pathlib import Path


def dataset_records(
  root: Path | str, clip_ids: tuple[str, ...] | None = None
) -> list[dict]:
  """Read metadata only; defer asset and motion loading to their consumers."""
  root = Path(root).expanduser().resolve()
  manifest = json.loads((root / "manifest.json").read_text())
  records = sorted(manifest["clips"], key=lambda record: record["id"])
  ids = [record["id"] for record in records]
  if not ids or len(ids) != len(set(ids)):
    raise ValueError("Dataset clip IDs must be nonempty and unique")
  if clip_ids is not None:
    missing = set(clip_ids) - set(ids)
    if not clip_ids or missing:
      raise ValueError(f"Empty selection or unknown clip IDs: {sorted(missing)[:5]}")
    selected = set(clip_ids)
    records = [record for record in records if record["id"] in selected]
  for record in records:
    if record.get("object_mode", "static") != "static":
      raise ValueError(f"Dynamic scene is not supported: {record['id']}")
    for key in ("motion", "scene"):
      if key == "scene" and not record.get(key):
        continue
      path = Path(record[key])
      if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Invalid dataset {key} path: {record[key]}")
  return records
