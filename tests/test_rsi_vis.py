"""Tests for RSI visualization helpers and sampling matrices."""

from __future__ import annotations

import torch

from wbc_mjlab.env.mdp.sampling import (
  compute_assist_gain_matrix,
  compute_sampling_prob_matrix,
  rsi_failure_signal_label,
  trajectory_conditional_prob_row,
)
from wbc_mjlab.viewer.motion_vis import (
  clip_name_for_trajectory,
  error_to_rgba,
  format_motion_context_html,
  format_rsi_panel_html,
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


def test_format_rsi_panel_html_separate_rows() -> None:
  html = format_rsi_panel_html(
    bin_idx=2,
    num_bins=5,
    bin_width_s=4.0,
    failure_levels=[0.1, 0.2, 0.42, 0.8, 0.05],
    sampling_probs=[0.05, 0.1, 0.5, 0.3, 0.05],
    visit_counts=[1, 2, 40, 10, 1],
    assist_gains=[0.0, 0.1, 0.4, 0.6, 0.0],
    valid_mask=[True, True, True, True, True],
    failure_signal_label="EMA(1 − mean motion-reward similarity)",
    show_assist=True,
    beta_max=0.6,
  )
  assert "Failure" in html
  assert "Sample p" in html
  assert "Visits" in html
  assert "Assist β" in html
  assert "2px solid #fff" in html
  assert "0.100" in html or "0.1" in html


def test_format_rsi_panel_html_uniform_is_gray() -> None:
  html = format_rsi_panel_html(
    bin_idx=0,
    num_bins=3,
    bin_width_s=4.0,
    failure_levels=[0.0, 0.0, 0.0],
    sampling_probs=[0.33, 0.33, 0.34],
    visit_counts=[0, 0, 0],
    assist_gains=None,
    valid_mask=[True, True, True],
    failure_signal_label="test",
    show_assist=False,
    beta_max=0.6,
  )
  assert "rgb(100, 100, 100)" in html
  assert "no EMA yet" in html


def test_trajectory_conditional_prob_row() -> None:
  prob = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
  valid = torch.tensor([[True, True], [True, True]])
  row = trajectory_conditional_prob_row(prob, valid, traj_id=0)
  assert abs(sum(row) - 1.0) < 1e-5
  assert row[0] < row[1]


def test_compute_sampling_prob_matrix_sums_to_one_on_valid() -> None:
  failure = torch.tensor([[0.1, 0.9], [0.5, 0.5]])
  valid = torch.tensor([[True, True], [True, False]])
  probs = compute_sampling_prob_matrix(
    failure,
    valid,
    temperature_base=1.0,
    uniform_ratio=0.0,
  )
  assert abs(probs.sum().item() - 1.0) < 1e-5
  assert probs[1, 1].item() == 0.0


def test_compute_assist_gain_matrix() -> None:
  failure = torch.tensor([[0.0, 1.0]])
  assist = compute_assist_gain_matrix(failure, eta=0.8, beta_max=0.6, enabled=True)
  assert assist[0, 0].item() == 0.0
  assert assist[0, 1].item() > assist[0, 0].item()


def test_rsi_failure_signal_label() -> None:
  assert "terminated" in rsi_failure_signal_label("binary_failure", similarity_from_rewards=False)
  assert "motion-reward" in rsi_failure_signal_label(
    "similarity_ema", similarity_from_rewards=True
  )
