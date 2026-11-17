"""Motion dataset QA, diversity analysis and diversity-based subsets.

  wbc-mjlab-motion-data qa DATASET [--report qa.csv] [--keep-dir DIR]
  wbc-mjlab-motion-data analyze DATASET [--heldout DATASET] [--list clips.txt]
  wbc-mjlab-motion-data select DATASET --out-dir DIR (--clips N | --hours H)

``--keep-dir`` / ``--out-dir`` write a dataset folder of links to the source clips, usable
as ``--motion-file``. See :mod:`wbc_mjlab.motion.diversity`.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _add_common(p: argparse.ArgumentParser) -> None:
  p.add_argument("dataset", type=Path, help="dataset folder (raw clips) or one clip")
  p.add_argument("--robot", default="g1", help="robot id (default: g1)")
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--format", dest="format_name", default=None, help="source format")
  p.add_argument("--input-fps", type=float, default=None, help="source rate override")
  p.add_argument("--floor", type=float, default=-0.03, help="lowest body z allowed (m)")
  p.add_argument(
    "--joint-speed", type=float, default=25.0, help="joint speed limit (rad/s)"
  )
  p.add_argument(
    "--base-jump", type=float, default=0.2, help="base step limit (m/frame)"
  )


def _scan(a):
  from wbc_mjlab.motion.diversity import scan_dataset

  scan = scan_dataset(
    a.dataset,
    robot_id=a.robot,
    device=a.device,
    format_name=a.format_name,
    input_fps=a.input_fps,
  )
  return scan, scan.passing(a.floor, a.joint_speed, a.base_jump)


def _subset(scan, ok, keep: np.ndarray):
  """The scan's windows / counts of the clips in ``keep`` (a boolean mask)."""
  owner = scan.window_clip()
  return scan.windows[keep[owner]], scan.count[keep]


def cmd_qa(a) -> None:
  from wbc_mjlab.motion.diversity import link_dataset

  scan, ok = _scan(a)
  rows = []
  for i, name in enumerate(scan.names):
    reasons = []
    if scan.min_body_z[i] < a.floor:
      reasons.append("below_floor")
    if scan.max_joint_speed[i] > a.joint_speed:
      reasons.append("joint_speed")
    if scan.max_base_jump[i] > a.base_jump:
      reasons.append("base_jump")
    if scan.count[i] == 0:
      reasons.append("too_short")
    rows.append(
      {
        "clip": name,
        "seconds": round(float(scan.seconds[i]), 3),
        "min_body_z": round(float(scan.min_body_z[i]), 4),
        "max_joint_speed": round(float(scan.max_joint_speed[i]), 3),
        "max_base_jump": round(float(scan.max_base_jump[i]), 4),
        "ok": bool(ok[i]),
        "reasons": " ".join(reasons),
      }
    )
  bad = [r for r in rows if not r["ok"]]
  print(
    f"{len(rows)} clips ({scan.seconds.sum() / 3600:.2f} h), {len(bad)} flagged "
    f"({scan.seconds[~ok].sum() / 3600:.2f} h)"
  )
  for r in bad[:20]:
    print(f"  {r['clip']}: {r['reasons']}")
  if len(bad) > 20:
    print(f"  ... {len(bad) - 20} more")
  if a.report:
    a.report.parent.mkdir(parents=True, exist_ok=True)
    with open(a.report, "w", newline="") as f:
      w = csv.DictWriter(f, fieldnames=list(rows[0]))
      w.writeheader()
      w.writerows(rows)
    print(f"Wrote {a.report}")
  if a.keep_dir:
    out = link_dataset(
      [s for s, k in zip(scan.sources, ok, strict=True) if k], a.keep_dir, copy=a.copy
    )
    print(f"Wrote {int(ok.sum())} passing clips to {out}")


def cmd_analyze(a) -> None:
  from wbc_mjlab.motion.diversity import analyze_windows, scan_dataset

  scan, ok = _scan(a)
  keep = ok.copy()
  if a.list:
    wanted = {line.strip() for line in open(a.list) if line.strip()}
    keep &= np.array([n in wanted for n in scan.names])
  windows, count = _subset(scan, ok, keep)
  heldout = None
  if a.heldout:
    heldout = scan_dataset(
      a.heldout,
      robot_id=a.robot,
      device=a.device,
      format_name=a.format_name,
      input_fps=a.input_fps,
    ).windows
  report = analyze_windows(
    windows, count, clusters=a.clusters, heldout=heldout, device=a.device
  )
  rarity = report.pop("rarity")
  names = [n for n, k in zip(scan.names, keep, strict=True) if k]
  order = np.argsort(rarity)
  report["clips"] = len(names)
  report["hours"] = float(scan.seconds[keep].sum() / 3600)
  report["rarest_clips"] = [names[i] for i in order[::-1][: a.top]]
  report["most_redundant_clips"] = [names[i] for i in order[: a.top]]
  print(json.dumps(report, indent=2))
  if a.output:
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2))
    print(f"Wrote {a.output}")


def cmd_select(a) -> None:
  from wbc_mjlab.motion.diversity import link_dataset, select_clips

  if a.clips is None and a.hours is None:
    raise SystemExit("give --clips, --hours or both")
  scan, ok = _scan(a)
  windows, count = _subset(scan, ok, ok)
  pool = np.flatnonzero(ok)
  order = select_clips(
    windows,
    count,
    scan.seconds[pool],
    num_clips=a.clips,
    hours=a.hours,
    clusters=a.clusters,
    tau=a.tau,
    device=a.device,
  )
  chosen = pool[order]
  print(
    f"selected {len(chosen)} of {len(pool)} passing clips, "
    f"{scan.seconds[chosen].sum() / 3600:.2f} of {scan.seconds[pool].sum() / 3600:.2f} h"
  )
  out = link_dataset([scan.sources[i] for i in chosen], a.out_dir, copy=a.copy)
  (out / "selection.txt").write_text("".join(f"{scan.names[i]}\n" for i in chosen))
  print(f"Wrote {out} (selection.txt in selection order)")


def main() -> None:
  parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  sub = parser.add_subparsers(dest="command", required=True)

  qa = sub.add_parser("qa", help="flag clips below the floor, too fast or jumping")
  _add_common(qa)
  qa.add_argument("--report", type=Path, default=None, help="per-clip CSV")
  qa.add_argument(
    "--keep-dir", type=Path, default=None, help="dataset of passing clips"
  )
  qa.add_argument("--copy", action="store_true", help="copy instead of linking")
  qa.set_defaults(func=cmd_qa)

  an = sub.add_parser("analyze", help="cluster coverage, rare / redundant clips")
  _add_common(an)
  an.add_argument("--list", type=Path, default=None, help="only these clip names")
  an.add_argument("--heldout", type=Path, default=None, help="dataset to cover")
  an.add_argument("--clusters", type=int, default=1024)
  an.add_argument("--top", type=int, default=10)
  an.add_argument("--output", type=Path, default=None, help="JSON report")
  an.set_defaults(func=cmd_analyze)

  se = sub.add_parser("select", help="diversity-based subset of the passing clips")
  _add_common(se)
  se.add_argument("--out-dir", type=Path, required=True)
  se.add_argument("--clips", type=int, default=None, help="number of clips")
  se.add_argument("--hours", type=float, default=None, help="total duration")
  se.add_argument("--clusters", type=int, default=4096)
  se.add_argument("--tau", type=float, default=0.5, help="cluster target exponent")
  se.add_argument("--copy", action="store_true", help="copy instead of linking")
  se.set_defaults(func=cmd_select)

  a = parser.parse_args()
  a.func(a)


if __name__ == "__main__":
  main()
