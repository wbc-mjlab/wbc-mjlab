"""Repository ``data/<robot>/<dataset>/`` layout for WBC tracking motions.

Per-robot dataset folders under repo ``data/<robot>/<dataset_name>/``
(e.g. ``data/g1/<dataset_name>/`` at the project root).

Typical layout::

  data/g1/lafan/
    *.csv              # retargeted clips (or under ``raw/``)
    lafan.npz          # stacked training bundle (``wbc-mjlab-csv-to-npz``)
    npz/<clip>.npz     # per-clip exports
"""

from __future__ import annotations

from pathlib import Path

from wbc_mjlab.robots.ids import RobotId, resolve_robot_id


def repo_root() -> Path:
  """Root of the wbc_mjlab project (directory containing ``pyproject.toml``)."""
  here = Path(__file__).resolve()
  for parent in here.parents:
    if (parent / "pyproject.toml").is_file():
      return parent
  raise RuntimeError("Could not locate wbc_mjlab repo root (pyproject.toml)")


def data_root() -> Path:
  return repo_root() / "data"


def robot_data_dir(robot_id: str | RobotId) -> Path:
  rid = resolve_robot_id(robot_id) if isinstance(robot_id, str) else robot_id
  return data_root() / rid


def dataset_dir(robot_id: str | RobotId, dataset: str) -> Path:
  name = dataset.strip()
  if not name or "/" in name or name.startswith("."):
    raise ValueError(f"Invalid dataset name {dataset!r}")
  return robot_data_dir(robot_id) / name


def dataset_bundle_npz(robot_id: str | RobotId, dataset: str) -> Path:
  """Stacked NPZ used for training (``<dataset>/<dataset>.npz``)."""
  d = dataset_dir(robot_id, dataset)
  return d / f"{d.name}.npz"


def dataset_npz_subdir(robot_id: str | RobotId, dataset: str) -> Path:
  return dataset_dir(robot_id, dataset) / "npz"


def dataset_raw_dir(robot_id: str | RobotId, dataset: str) -> Path:
  return dataset_dir(robot_id, dataset) / "raw"


def resolve_dataset_input_dir(robot_id: str | RobotId, dataset: str) -> Path:
  """Directory to read CSV/PKL clips from (``raw/`` if present, else dataset root)."""
  raw = dataset_raw_dir(robot_id, dataset)
  if raw.is_dir() and (any(raw.glob("*.csv")) or any(raw.glob("*.pkl"))):
    return raw
  return dataset_dir(robot_id, dataset)


def list_datasets(robot_id: str | RobotId = "g1") -> list[str]:
  root = robot_data_dir(robot_id)
  if not root.is_dir():
    return []
  return sorted(
    p.name
    for p in root.iterdir()
    if p.is_dir() and not p.name.startswith(".")
  )


def resolve_dataset_motion_file(robot_id: str | RobotId, dataset: str) -> Path:
  path = dataset_bundle_npz(robot_id, dataset)
  if not path.is_file():
    raise FileNotFoundError(
      f"No training bundle at {path}. "
      f"Convert clips first, e.g. wbc-mjlab-csv-to-npz --robot {resolve_robot_id(robot_id)} "
      f"--dataset {dataset.strip()}"
    )
  return path


def resolve_motion_path(path: str | Path) -> Path:
  """Resolve a training motion NPZ from an explicit file or dataset directory."""
  p = Path(path).expanduser()
  if not p.is_absolute():
    p = (Path.cwd() / p).resolve()
  else:
    p = p.resolve()

  if p.is_file():
    if p.suffix.lower() != ".npz":
      raise ValueError(f"Expected a .npz motion file, got: {p}")
    return p

  if p.is_dir():
    named = p / f"{p.name}.npz"
    if named.is_file():
      return named
    npzs = sorted(x for x in p.glob("*.npz") if x.is_file())
    if len(npzs) == 1:
      return npzs[0]
    if len(npzs) > 1:
      names = ", ".join(x.name for x in npzs[:8])
      raise FileNotFoundError(
        f"Multiple .npz files in {p}; pass --motion-file explicitly. Found: {names}"
      )
    raise FileNotFoundError(
      f"No training .npz in {p}. Expected {p.name}.npz or a single *.npz in the folder."
    )

  raise FileNotFoundError(f"Motion path does not exist: {p}")


def resolve_training_motion_file(
  robot_id: str | RobotId,
  *,
  dataset: str | None = None,
  dataset_path: str | None = None,
) -> Path:
  """Resolve bundled motion NPZ for train/play from dataset name or explicit path."""
  if dataset_path is not None and dataset is not None:
    raise ValueError("Use only one of --dataset or --dataset-path")
  if dataset_path is not None:
    return resolve_motion_path(dataset_path)
  if dataset is not None:
    return resolve_dataset_motion_file(robot_id, dataset)
  raise ValueError("No motion source provided")


def resolve_conversion_paths(
  *,
  robot_id: str | RobotId,
  dataset: str | None = None,
  dataset_path: str | None = None,
  input_path: str | None = None,
  output_dir: str | None = None,
) -> tuple[str, str]:
  """Resolve motion conversion I/O from ``--dataset``, ``--dataset-path``, or explicit paths."""
  if dataset_path is not None and dataset is not None:
    raise ValueError("Use only one of --dataset or --dataset-path")

  rid = resolve_robot_id(robot_id) if isinstance(robot_id, str) else robot_id

  if dataset_path is not None:
    root = Path(dataset_path).expanduser()
    if not root.is_absolute():
      root = (Path.cwd() / root).resolve()
    else:
      root = root.resolve()
    if root.is_file():
      raise ValueError("--dataset-path must be a directory for conversion, not a .npz file")
    if not root.is_dir():
      raise FileNotFoundError(f"Dataset path does not exist: {root}")
    out = str(output_dir or root)
    if input_path is not None:
      inp = input_path
    else:
      raw = root / "raw"
      inp = str(raw if raw.is_dir() else root)
    return inp, out

  if dataset is not None:
    out = str(output_dir or dataset_dir(rid, dataset))
    inp = str(input_path or resolve_dataset_input_dir(rid, dataset))
    return inp, out

  if input_path is None or output_dir is None:
    raise ValueError(
      "Provide --dataset <name>, --dataset-path <dir>, or both --input-path and --output-dir"
    )
  return input_path, output_dir
