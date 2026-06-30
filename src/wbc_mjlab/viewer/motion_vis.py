"""Helpers for WBC motion-command Viser overlays."""

from __future__ import annotations

import html


def clip_name_for_trajectory(segment_names: tuple[str, ...], traj_id: int) -> str:
  if 0 <= traj_id < len(segment_names):
    return segment_names[traj_id]
  return f"traj_{traj_id}"


def error_to_rgba(
  error: float,
  *,
  max_error: float = 0.3,
  alpha: float = 0.85,
) -> tuple[float, float, float, float]:
  """Map tracking error to green (good) → red (bad)."""
  t = min(max(float(error) / max(max_error, 1.0e-6), 0.0), 1.0)
  return (t, 1.0 - t, 0.15, alpha)


def format_motion_context_html(
  *,
  env_idx: int,
  traj_id: int,
  clip_name: str,
  frame: int,
  phase: float,
  task_id: str | None,
  rsi_mode: str,
  rsi_strategy: str,
  anchor_body: str,
  num_bodies: int,
) -> str:
  task = html.escape(task_id or "—")
  clip = html.escape(clip_name)
  return f"""
<div style="font-size:0.82em;line-height:1.35;padding:0 0.25em 0.5em 0.25em;">
  <strong>Env</strong> #{env_idx}<br/>
  <strong>Clip</strong> {clip} <span style="opacity:0.7;">(traj {traj_id})</span><br/>
  <strong>Frame</strong> {frame} · <strong>Phase</strong> {phase:.2f}<br/>
  <strong>Task</strong> {task}<br/>
  <strong>RSI</strong> {html.escape(rsi_mode)} / {html.escape(rsi_strategy)}<br/>
  <strong>Tracked</strong> {num_bodies} bodies · anchor <code>{html.escape(anchor_body)}</code>
</div>
"""


def format_rsi_html(
  *,
  bin_idx: int,
  num_bins: int,
  failure: float,
  failure_levels: list[float],
  valid_mask: list[bool],
) -> str:
  if num_bins <= 0:
    return "<em>No RSI bins</em>"

  cells: list[str] = []
  for i, level in enumerate(failure_levels[:num_bins]):
    if not valid_mask[i]:
      continue
    border = "2px solid #fff" if i == bin_idx else "1px solid #444"
    cells.append(
      f'<span title="bin {i}: {level:.2f}" style="display:inline-block;width:10px;height:14px;'
      f"background:rgb({int(level*255)},{int((1-level)*180)},80);margin:1px;border:{border};"
      f'border-radius:2px;"></span>'
    )
  bar = "".join(cells) if cells else "—"
  return (
    f'<div style="font-size:0.82em;line-height:1.35;padding:0 0.25em 0.5em 0.25em;">'
    f"<strong>RSI bin</strong> {bin_idx}/{max(num_bins - 1, 0)} "
    f"· <strong>failure</strong> {failure:.3f}<br/>{bar}</div>"
  )
