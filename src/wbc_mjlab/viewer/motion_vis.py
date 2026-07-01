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


def _clamp01(value: float) -> float:
  return min(max(float(value), 0.0), 1.0)


def failure_rgb(level: float) -> str:
  """RdYlGn_r: green (easy) → red (hard)."""
  t = _clamp01(level)
  if t < 0.5:
    u = t * 2.0
    r = 26 + (255 - 26) * u
    g = 152 + (235 - 152) * u
    b = 80 + (132 - 80) * u
  else:
    u = (t - 0.5) * 2.0
    r = 255 + (215 - 255) * u
    g = 235 + (48 - 235) * u
    b = 132 + (39 - 132) * u
  return f"rgb({int(r)},{int(g)},{int(b)})"


def prob_rgb(probability: float, *, p_max: float) -> str:
  """Blues: light (rare) → dark (likely sampled)."""
  t = _clamp01(probability / max(p_max, 1.0e-9))
  r = int(247 * (1.0 - t) + 8 * t)
  g = int(251 * (1.0 - t) + 48 * t)
  b = int(255 * (1.0 - t) + 107 * t)
  return f"rgb({r},{g},{b})"


def visit_rgb(count: float, *, c_max: float) -> str:
  """Oranges: pale (rarely visited) → deep (often visited)."""
  t = _clamp01(count / max(c_max, 1.0e-9))
  r = int(255 * (1.0 - t) + 127 * t)
  g = int(245 * (1.0 - t) + 39 * t)
  b = int(235 * (1.0 - t) + 17 * t)
  return f"rgb({r},{g},{b})"


def assist_rgb(gain: float, *, beta_max: float) -> str:
  """Purples: low assist → high assist."""
  t = _clamp01(gain / max(beta_max, 1.0e-9))
  r = int(237 * (1.0 - t) + 106 * t)
  g = int(233 * (1.0 - t) + 27 * t)
  b = int(254 * (1.0 - t) + 154 * t)
  return f"rgb({r},{g},{b})"


_UNIFORM_RGB = "rgb(100, 100, 100)"


def _row_relative_scale(
  values: list[float],
  valid_mask: list[bool],
  *,
  eps: float = 1.0e-9,
) -> tuple[list[float] | None, float, float]:
  """Map a row to [0, 1] for display; None if all valid values are equal."""
  valid = [v for v, ok in zip(values, valid_mask, strict=False) if ok]
  if not valid:
    return None, 0.0, 0.0
  vmin, vmax = min(valid), max(valid)
  if vmax - vmin < eps:
    return None, vmin, vmax
  scaled = [
    ((v - vmin) / (vmax - vmin) if ok else 0.0)
    for v, ok in zip(values, valid_mask, strict=False)
  ]
  return scaled, vmin, vmax


def _bin_strip_html(
  *,
  label: str,
  raw_values: list[float],
  valid_mask: list[bool],
  current_bin: int,
  color_fn,
  value_fmt: str,
  bin_width_s: float,
  display_values: list[float] | None = None,
  uniform_gray: bool = False,
) -> str:
  cells: list[str] = []
  disp = display_values if display_values is not None else raw_values
  for i, raw in enumerate(raw_values):
    if not valid_mask[i]:
      cells.append(
        '<span style="display:inline-block;width:11px;height:16px;margin:1px;'
        'background:#555;border-radius:2px;" title="invalid bin"></span>'
      )
      continue
    border = "2px solid #fff" if i == current_bin else "1px solid #333"
    t0 = i * bin_width_s
    t1 = (i + 1) * bin_width_s
    title = html.escape(f"bin {i} [{t0:.1f}–{t1:.1f}s]: {value_fmt.format(raw)}")
    bg = _UNIFORM_RGB if uniform_gray else color_fn(disp[i])
    cells.append(
      f'<span title="{title}" style="display:inline-block;width:11px;height:16px;'
      f'margin:1px;background:{bg};border:{border};border-radius:2px;"></span>'
    )
  row = "".join(cells) if cells else "—"
  return (
    f'<div style="margin:0.15em 0;">'
    f'<span style="display:inline-block;width:7.5em;font-size:0.78em;opacity:0.9;">'
    f"{html.escape(label)}</span>{row}</div>"
  )


def format_rsi_panel_html(
  *,
  bin_idx: int,
  num_bins: int,
  bin_width_s: float,
  failure_levels: list[float],
  sampling_probs: list[float],
  visit_counts: list[float],
  assist_gains: list[float] | None,
  valid_mask: list[bool],
  failure_signal_label: str,
  show_assist: bool,
  beta_max: float,
) -> str:
  if num_bins <= 0:
    return "<em>No RSI bins</em>"

  mask = valid_mask[:num_bins]
  failures = failure_levels[:num_bins]
  probs = sampling_probs[:num_bins]
  visits = visit_counts[:num_bins]

  failure_scale, f_min, f_max = _row_relative_scale(failures, mask)
  prob_scale, p_min, p_max = _row_relative_scale(probs, mask)
  c_max = max((c for c, v in zip(visits, mask, strict=False) if v), default=1.0)

  t0 = bin_idx * bin_width_s
  t1 = (bin_idx + 1) * bin_width_s
  range_note = ""
  if failure_scale is None:
    range_note = (
      f'<br/><span style="opacity:0.75;">Failure flat at {f_min:.3f} '
      f"(no EMA yet — train or load rsi_bin_stats.npz)</span>"
    )
  else:
    range_note = (
      f'<br/><span style="opacity:0.75;">Failure range on clip: '
      f"{f_min:.3f} → {f_max:.3f}</span>"
    )
  header = (
    f'<div style="font-size:0.78em;opacity:0.85;margin-bottom:0.25em;">'
    f"<strong>Bin</strong> {bin_idx}/{max(num_bins - 1, 0)} "
    f"· <strong>time</strong> {t0:.1f}–{t1:.1f}s<br/>"
    f"<strong>Failure signal:</strong> {html.escape(failure_signal_label)}"
    f"{range_note}"
    f'<br/><span style="opacity:0.75;">Sample p renormalized within this clip</span>'
    f"</div>"
  )

  rows = [
    _bin_strip_html(
      label="Failure",
      raw_values=failures,
      valid_mask=mask,
      current_bin=bin_idx,
      color_fn=failure_rgb,
      display_values=failure_scale,
      uniform_gray=failure_scale is None,
      value_fmt="{:.3f}",
      bin_width_s=bin_width_s,
    ),
    _bin_strip_html(
      label="Sample p",
      raw_values=probs,
      valid_mask=mask,
      current_bin=bin_idx,
      color_fn=lambda v: prob_rgb(v, p_max=1.0),
      display_values=prob_scale,
      uniform_gray=prob_scale is None,
      value_fmt="{:.4f}",
      bin_width_s=bin_width_s,
    ),
    _bin_strip_html(
      label="Visits",
      raw_values=visits,
      valid_mask=mask,
      current_bin=bin_idx,
      color_fn=lambda v: visit_rgb(v, c_max=c_max),
      value_fmt="{:.0f}",
      bin_width_s=bin_width_s,
    ),
  ]
  if show_assist and assist_gains is not None:
    assist = assist_gains[:num_bins]
    assist_scale, _, _ = _row_relative_scale(assist, mask)
    rows.append(
      _bin_strip_html(
        label="Assist β",
        raw_values=assist,
        valid_mask=mask,
        current_bin=bin_idx,
        color_fn=lambda v: assist_rgb(v, beta_max=beta_max),
        display_values=assist_scale,
        uniform_gray=assist_scale is None,
        value_fmt="{:.3f}",
        bin_width_s=bin_width_s,
      )
    )

  legend = (
    '<div style="font-size:0.72em;opacity:0.7;margin-top:0.2em;">'
    "Colors are relative within this clip (green→red failure, pale→deep blue p). "
    "▌white border = playback bin · gray = invalid/uniform"
    "</div>"
  )
  return (
    f'<div style="font-size:0.82em;line-height:1.35;padding:0 0.25em 0.5em 0.25em;">'
    f"{header}{''.join(rows)}{legend}</div>"
  )


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


# Backward-compatible alias for older tests/callers.
format_rsi_html = format_rsi_panel_html
