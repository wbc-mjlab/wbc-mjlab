"""Synchronous progress reporting for dataset terrain import."""

from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from weakref import WeakKeyDictionary

import mujoco
from tqdm import tqdm

_BARS: WeakKeyDictionary[mujoco.MjSpec, tqdm] = WeakKeyDictionary()


@contextmanager
def terrain_load_progress(spec, total: int, trajectory: str, xml_path: str | None):
  """Share a progress bar across patches of one spec, never across model builds."""
  if total <= 0:
    yield lambda stage: None
    return
  bar = _BARS.get(spec)
  if bar is None:
    bar = tqdm(
      total=total,
      desc="Terrain import",
      unit="scene",
      mininterval=1.0,
      bar_format="{desc}: {n_fmt}/{total_fmt} ({percentage:3.0f}%) [{elapsed}<{remaining}] {postfix}",
    )
    _BARS[spec] = bar

  def stage(name: str):
    bar.set_postfix_str(f"{trajectory} | {name}", refresh=False)
    # Redraw the terminal in place; throttle redirected logs to one update/sec.
    if sys.stderr.isatty() or bar.n == 0 or time.time() - bar.last_print_t >= 1.0:
      bar.refresh()

  try:
    stage("start")
    yield stage
  except BaseException:
    bar.close()
    _BARS.pop(spec, None)
    tqdm.write(
      f"[ERROR] Terrain import failed: {trajectory} | {xml_path or '<flat>'}",
      file=sys.stderr,
    )
    raise
  else:
    bar.update(1)
    if bar.n == total:
      bar.close()
      _BARS.pop(spec, None)
      tqdm.write(
        "[INFO] Terrain specs ready. Scene/model compilation and simulation initialization follow.",
        file=sys.stderr,
      )
