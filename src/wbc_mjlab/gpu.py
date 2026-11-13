"""NVIDIA telemetry and GPU selection, usable on the host with only Python's stdlib.

Queries follow https://docs.nvidia.com/deploy/nvidia-smi/index.html.
UUIDs identify hardware; CUDA ordinals are resolved separately in the Studio process.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def number(value: str) -> float | None:
  try:
    result = float(value.strip())
    return result if math.isfinite(result) else None
  except ValueError:
    return None


def parse_gpus(output: str) -> list[dict]:
  rows = []
  for row in csv.reader(io.StringIO(output), skipinitialspace=True):
    if not row:
      continue
    if len(row) != 7:
      raise ValueError("Unexpected nvidia-smi response")
    index, uuid, name, total, free, used, utilization = [x.strip() for x in row]
    total, free, used, utilization = map(number, (total, free, used, utilization))
    fraction = free / total if free is not None and total else None
    status = (
      "unknown"
      if fraction is None or utilization is None
      else "busy"
      if utilization >= 80 or fraction < 0.2
      else "available"
      if utilization < 30 and fraction >= 0.5
      else "moderate"
    )
    rows.append(
      dict(
        index=int(index),
        uuid=uuid,
        name=name,
        total_mib=total,
        free_mib=free,
        used_mib=used,
        utilization=utilization,
        status=status,
      )
    )
  return rows


def query_gpus() -> list[dict]:
  try:
    result = subprocess.run(
      [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total,memory.free,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
      ],
      capture_output=True,
      text=True,
      timeout=5,
    )
  except (OSError, subprocess.TimeoutExpired) as error:
    raise ValueError(f"GPU telemetry unavailable: {error}") from error
  if result.returncode:
    raise ValueError(f"GPU telemetry unavailable: {result.stderr.strip()}")
  return parse_gpus(result.stdout)


def visible_gpus() -> list[dict]:
  """Attach CUDA ordinals, respecting container and CUDA_VISIBLE_DEVICES restrictions."""
  import torch

  ordinal = {
    str(torch.cuda.get_device_properties(i).uuid).removeprefix("GPU-"): i
    for i in range(torch.cuda.device_count())
  }
  return [
    row | {"cuda_index": ordinal[row["uuid"].removeprefix("GPU-")]}
    for row in query_gpus()
    if row["uuid"].removeprefix("GPU-") in ordinal
  ]


def training_gpu_indices(uuids: list[str], rows: list[dict]) -> list[int]:
  if not uuids or len(set(uuids)) != len(uuids):
    raise ValueError("Select at least one GPU, without duplicates")
  by_uuid = {row["uuid"]: row["cuda_index"] for row in rows}
  if any(uuid not in by_uuid for uuid in uuids):
    raise ValueError("Selected GPU is no longer visible; refresh the GPU list")
  return [by_uuid[uuid] for uuid in uuids]


def resolve_device(gpu_uuid: str | None, default: str) -> str:
  """One device for playback/evaluation; None preserves the caller's default."""
  if gpu_uuid is None:
    return default
  return f"cuda:{training_gpu_indices([gpu_uuid], visible_gpus())[0]}"


def label(row: dict) -> str:
  free = "?" if row["free_mib"] is None else f"{row['free_mib'] / 1024:.1f}"
  total = "?" if row["total_mib"] is None else f"{row['total_mib'] / 1024:.1f}"
  util = "?" if row["utilization"] is None else f"{row['utilization']:.0f}"
  return f"GPU {row['index']} · {row['name']} · {util}% load · {free}/{total} GiB free · {row['status']}"


def atomic_write(path: Path, content: str) -> None:
  # .env may contain credentials: preserve its permissions when replacing it.
  with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
    temporary = Path(stream.name)
    try:
      if path.exists():
        os.fchmod(stream.fileno(), path.stat().st_mode & 0o777)
      stream.write(content)
      stream.flush()
      temporary.replace(path)
    finally:
      temporary.unlink(missing_ok=True)


def save_selection(root: Path, uuids: list[str], compose: str) -> None:
  """Persist only our GPU config; refuse to change a running workspace's hardware."""
  target = root / ".devcontainer/gpus.local.yaml"
  config = {"services": {"gpu": {"gpus": [{"driver": "nvidia", "device_ids": uuids}]}}}
  if (
    target.is_file()
    and json.loads(target.read_text()) == config
    and (root / ".env").is_file()
    and "WBC_GPU_CONFIG=.devcontainer/gpus.local.yaml"
    in (root / ".env").read_text().splitlines()
  ):
    return
  result = subprocess.run(
    [*shlex.split(compose), "ps", "-q", "--status", "running", "dev"],
    cwd=root,
    text=True,
    capture_output=True,
    timeout=15,
  )
  if result.returncode:
    raise ValueError(f"Cannot check the Docker workspace: {result.stderr.strip()}")
  if result.stdout.strip():
    raise ValueError(
      "Stop the workspace with make docker-stop before changing its GPUs"
    )
  # JSON is valid YAML and needs no host-side YAML dependency.
  atomic_write(target, json.dumps(config, indent=2) + "\n")
  env_path = root / ".env"
  lines = env_path.read_text().splitlines() if env_path.exists() else []
  lines = [line for line in lines if not line.strip().startswith("WBC_GPU_CONFIG=")]
  lines.append("WBC_GPU_CONFIG=.devcontainer/gpus.local.yaml")
  atomic_write(env_path, "\n".join(lines) + "\n")


def choose(rows: list[dict], selected: set[str]) -> list[str] | None:
  import curses

  def menu(screen):
    nonlocal rows, selected
    curses.curs_set(0)
    if curses.has_colors():
      curses.start_color()
      curses.use_default_colors()
      for i, color in enumerate(
        (curses.COLOR_GREEN, curses.COLOR_YELLOW, curses.COLOR_RED, curses.COLOR_WHITE),
        1,
      ):
        curses.init_pair(i, color, -1)
    cursor, refreshed, error = 0, 0.0, ""
    screen.timeout(500)
    while True:
      if time.monotonic() - refreshed >= 2:
        try:
          rows = query_gpus()
          error = ""
        except ValueError as exc:
          error = str(exc)
        refreshed = time.monotonic()
      screen.erase()
      height, width = screen.getmaxyx()

      def line(y, text, style=0, height=height, width=width):
        if y < height - 1:
          screen.addnstr(y, 0, text, max(0, width - 1), style)

      line(
        0,
        "GPU picker · Up/Down move · Space toggle · A all · Enter save · Q cancel",
      )
      line(
        1,
        "Live every 2s. Available: <30% load & >=50% free; busy: >=80% load or <20% free.",
      )
      cursor = min(cursor, max(0, len(rows) - 1))
      page = max(1, height - 5)
      start = max(0, cursor - page + 1)
      for offset, row in enumerate(rows[start : start + page]):
        color = ("available", "moderate", "busy", "unknown").index(row["status"]) + 1
        style = curses.color_pair(color) if curses.has_colors() else 0
        if start + offset == cursor:
          style |= curses.A_REVERSE
        line(
          offset + 3, f"[{'x' if row['uuid'] in selected else ' '}] {label(row)}", style
        )
      line(
        height - 2,
        error
        or f"{len(selected)} selected · utilization is advisory, not a GPU reservation",
      )
      screen.refresh()
      key = screen.getch()
      if key in (ord("q"), ord("Q"), 27):
        return None
      if key == curses.KEY_DOWN:
        cursor = min(len(rows) - 1, cursor + 1)
      elif key == curses.KEY_UP:
        cursor = max(0, cursor - 1)
      elif key == ord(" ") and rows:
        uuid = rows[cursor]["uuid"]
        selected.symmetric_difference_update({uuid})
      elif key in (ord("a"), ord("A")):
        ids = {row["uuid"] for row in rows}
        selected = ids if selected != ids else set()
      elif key in (10, 13, curses.KEY_ENTER):
        chosen = [row["uuid"] for row in rows if row["uuid"] in selected]
        if chosen and not error:
          return chosen
        error = "Select at least one currently visible GPU"

  # selected is mutated in the nested menu.
  try:
    return curses.wrapper(menu)
  except curses.error as exc:
    raise ValueError("GPU picker needs a larger, color-capable terminal") from exc


def main():
  parser = argparse.ArgumentParser(description="GPU load and Docker GPU selection")
  parser.add_argument(
    "--select", action="store_true", help="interactive checkbox picker"
  )
  parser.add_argument("--set", help="comma-separated host GPU indices/UUIDs, or all")
  parser.add_argument("--compose", default="docker compose")
  args = parser.parse_args()
  try:
    rows = query_gpus()
    if not rows:
      raise ValueError("No NVIDIA GPUs found")
    root = Path(__file__).resolve().parents[2]
    if args.select:
      if not sys.stdin.isatty() or not sys.stdout.isatty():
        raise ValueError(
          "Interactive selection needs a terminal; use --set 0,2 for scripts"
        )
      target = root / ".devcontainer/gpus.local.yaml"
      selected = {row["uuid"] for row in rows}
      if target.is_file():
        selected = set(
          json.loads(target.read_text())["services"]["gpu"]["gpus"][0]["device_ids"]
        )
      chosen = choose(rows, selected)
      if chosen is None:
        return
    elif args.set:
      requested = [part.strip() for part in args.set.split(",")]
      chosen = [
        row["uuid"]
        for row in rows
        if args.set == "all"
        or str(row["index"]) in requested
        or row["uuid"] in requested
      ]
      known = {str(row["index"]) for row in rows} | {row["uuid"] for row in rows}
      if args.set != "all" and (not chosen or any(x not in known for x in requested)):
        raise ValueError(
          "Unknown GPU; use --set with the indices/UUIDs shown in the GPU list"
        )
    else:
      for row in rows:
        color = {"available": 32, "moderate": 33, "busy": 31, "unknown": 37}[
          row["status"]
        ]
        print(
          f"\033[{color}m{label(row)}\033[0m"
          if sys.stdout.isatty() and "NO_COLOR" not in os.environ
          else label(row)
        )
      return
    save_selection(root, chosen, args.compose)
    print(
      "Saved Docker GPUs: "
      + ", ".join(chosen)
      + ". Start with make docker-shell or make docker-studio."
    )
  except KeyboardInterrupt:
    parser.exit(130, "GPU selection cancelled\n")
  except (ValueError, OSError, subprocess.TimeoutExpired) as exc:
    parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
  main()
