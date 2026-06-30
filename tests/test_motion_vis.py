"""Tests for WBC Viser motion overlay helpers."""

from __future__ import annotations

from wbc_mjlab.viewer.motion_vis import (
  clip_name_for_trajectory,
  error_to_rgba,
  format_motion_context_html,
  format_rsi_html,
)


def test_clip_name_for_trajectory() -> None:
  names = ("walk1", "run2")
  assert clip_name_for_trajectory(names, 0) == "walk1"
  assert clip_name_for_trajectory(names, 5) == "traj_5"


def test_error_to_rgba() -> None:
  good = error_to_rgba(0.0)
  bad = error_to_rgba(1.0)
  assert good[1] > good[0]
  assert bad[0] > bad[1]


def test_format_motion_context_html_includes_clip() -> None:
  html = format_motion_context_html(
    env_idx=2,
    traj_id=1,
    clip_name="walk1_subject1",
    frame=10,
    phase=0.25,
    task_id="Wbc-G1-Zest",
    rsi_mode="adaptive",
    rsi_strategy="similarity_ema",
    anchor_body="torso_link",
    num_bodies=14,
  )
  assert "walk1_subject1" in html
  assert "Wbc-G1-Zest" in html


def test_format_rsi_html_marks_current_bin() -> None:
  html = format_rsi_html(
    bin_idx=2,
    num_bins=5,
    failure=0.42,
    failure_levels=[0.1, 0.2, 0.42, 0.8, 0.05],
    valid_mask=[True, True, True, True, True],
  )
  assert "0.420" in html
  assert "2px solid #fff" in html
