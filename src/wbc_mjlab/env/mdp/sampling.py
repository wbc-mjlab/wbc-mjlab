"""RSI adaptive sampling helpers for ``MotionCommand``.

``adaptive_sampling_strategy`` selects the episode failure signal:

- ``binary_failure`` — failure on early termination (BeyondMimic-style RSI).
- ``similarity_ema`` — failure from mean per-step tracking similarity (Zest-style RSI).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch

from mjlab.utils.lab_api.math import sample_uniform

AdaptiveSimilarityTerm = Literal[
  "joint_pos",
  "anchor_pos",
  "anchor_ori",
  "body_pos",
  "body_ori",
  "body_lin_vel",
  "body_ang_vel",
]

AdaptiveSamplingStrategy = Literal["binary_failure", "similarity_ema"]

DEFAULT_SIMILARITY_STDS: dict[AdaptiveSimilarityTerm, float] = {
  "joint_pos": 1.0,
  "anchor_pos": 0.3,
  "anchor_ori": 0.4,
  "body_pos": 0.3,
  "body_ori": 0.4,
  "body_lin_vel": 1.0,
  "body_ang_vel": 3.14,
}


@dataclass
class AdaptiveSimilarityTermCfg:
  """One exp-kernel term in the per-step RSI similarity score ``s_k``."""

  term: AdaptiveSimilarityTerm
  weight: float = 1.0
  std: float | None = None


@dataclass
class TrackingSimilarityState:
  """Robot vs reference errors for one similarity step."""

  tracked_joint_pos_error: torch.Tensor
  anchor_pos_error: torch.Tensor
  anchor_ori_error: torch.Tensor
  body_pos_error: torch.Tensor
  body_ori_error: torch.Tensor
  body_lin_vel_error: torch.Tensor
  body_ang_vel_error: torch.Tensor


def wbc_joint_only_similarity_terms() -> tuple[AdaptiveSimilarityTermCfg, ...]:
  return (AdaptiveSimilarityTermCfg(term="joint_pos"),)


def whole_body_adaptive_similarity_terms() -> tuple[AdaptiveSimilarityTermCfg, ...]:
  return (
    AdaptiveSimilarityTermCfg(term="joint_pos", weight=1.0),
    AdaptiveSimilarityTermCfg(term="anchor_pos", weight=0.5),
    AdaptiveSimilarityTermCfg(term="anchor_ori", weight=0.5),
    AdaptiveSimilarityTermCfg(term="body_pos", weight=1.0),
    AdaptiveSimilarityTermCfg(term="body_ori", weight=1.0),
  )


def compile_similarity_terms(
  term_cfgs: tuple[AdaptiveSimilarityTermCfg, ...],
) -> tuple[list[tuple[AdaptiveSimilarityTerm, float, float]], float]:
  terms: list[tuple[AdaptiveSimilarityTerm, float, float]] = []
  weight_sum = 0.0
  for term_cfg in term_cfgs:
    if term_cfg.weight <= 0.0:
      continue
    std = term_cfg.std if term_cfg.std is not None else DEFAULT_SIMILARITY_STDS[term_cfg.term]
    terms.append((term_cfg.term, term_cfg.weight, std))
    weight_sum += term_cfg.weight
  return terms, weight_sum


def bin_index_for_frame(
  *,
  segment_start_idx: torch.Tensor,
  time_steps: torch.Tensor,
  trajectory_ids: torch.Tensor,
  bin_width_frames: int,
  bins_per_trajectory: int,
) -> torch.Tensor:
  seg_start = segment_start_idx[trajectory_ids]
  local_frames = torch.clamp(time_steps - seg_start, min=0)
  return torch.clamp(local_frames // bin_width_frames, max=bins_per_trajectory - 1)


def step_tracking_similarity(
  terms: list[tuple[AdaptiveSimilarityTerm, float, float]],
  weight_sum: float,
  state: TrackingSimilarityState,
  *,
  num_envs: int,
  device: torch.device | str,
) -> torch.Tensor:
  """Weighted mean of exp tracking kernels (per-step ``s_k``)."""
  if not terms:
    return torch.ones(num_envs, device=device)

  weight_sum = max(weight_sum, 1.0e-6)
  similarity = torch.zeros(num_envs, device=device)
  for term, weight, std in terms:
    std = max(std, 1.0e-6)
    if term == "joint_pos":
      error = torch.sum(torch.square(state.tracked_joint_pos_error), dim=-1)
      kernel = torch.exp(-error / std**2)
    elif term == "anchor_pos":
      error = torch.sum(torch.square(state.anchor_pos_error), dim=-1)
      kernel = torch.exp(-error / std**2)
    elif term == "anchor_ori":
      kernel = torch.exp(-state.anchor_ori_error**2 / std**2)
    elif term == "body_pos":
      kernel = torch.exp(-state.body_pos_error.mean(-1) / std**2)
    elif term == "body_ori":
      kernel = torch.exp(-state.body_ori_error.mean(-1) / std**2)
    elif term == "body_lin_vel":
      kernel = torch.exp(-state.body_lin_vel_error.mean(-1) / std**2)
    else:
      kernel = torch.exp(-state.body_ang_vel_error.mean(-1) / std**2)
    similarity += weight * kernel
  return similarity / weight_sum


def update_failure_ema(
  bin_failed_count: torch.Tensor,
  *,
  strategy: AdaptiveSamplingStrategy,
  bins_per_trajectory: int,
  alpha: float,
  traj_ids: torch.Tensor,
  start_bins: torch.Tensor,
  episode_terminated: torch.Tensor | None,
  episode_similarity_sum: torch.Tensor | None,
  episode_step_count: torch.Tensor | None,
) -> None:
  """EMA update of per-(trajectory, bin) failure level."""
  if strategy == "binary_failure":
    assert episode_terminated is not None
    if not torch.any(episode_terminated):
      return
    traj_ids = traj_ids[episode_terminated]
    start_bins = start_bins[episode_terminated]
    episode_failure = torch.ones(traj_ids.shape[0], device=bin_failed_count.device)
  else:
    assert episode_similarity_sum is not None and episode_step_count is not None
    episode_length = torch.clamp(episode_step_count.float(), min=1.0)
    mean_similarity = episode_similarity_sum / episode_length
    episode_failure = 1.0 - torch.clamp(mean_similarity, 0.0, 1.0)

  bin_count = bin_failed_count.numel()
  flat_idx = traj_ids * bins_per_trajectory + start_bins
  failure_sum = torch.zeros(bin_count, device=bin_failed_count.device)
  failure_count = torch.zeros(bin_count, device=bin_failed_count.device)
  failure_sum.scatter_add_(0, flat_idx, episode_failure)
  failure_count.scatter_add_(0, flat_idx, torch.ones_like(episode_failure))

  update_mask = failure_count > 0
  if not torch.any(update_mask):
    return

  mean_failure = failure_sum[update_mask] / failure_count[update_mask]
  flat_failed = bin_failed_count.view(-1)
  flat_failed[update_mask] = (
    (1.0 - alpha) * flat_failed[update_mask] + alpha * mean_failure
  )


def sample_adaptive_bins(
  bin_failed_count: torch.Tensor,
  valid_bin_indices: torch.Tensor,
  *,
  segment_length: torch.Tensor,
  segment_start_idx: torch.Tensor,
  bin_width_frames: int,
  temperature_base: float,
  uniform_ratio: float,
  num_samples: int,
  device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """Draw (trajectory, bin, frame) from failure-weighted bin distribution."""
  valid = valid_bin_indices
  temperature = temperature_base / math.log(1.0 + max(1, valid.shape[0]))
  logits = bin_failed_count[valid[:, 0], valid[:, 1]] / temperature
  probs_valid = torch.softmax(logits, dim=0)
  num_valid = max(1, valid.shape[0])
  probs_valid = (1.0 - uniform_ratio) * probs_valid + uniform_ratio / float(num_valid)

  sampled_valid = torch.multinomial(probs_valid, num_samples, replacement=True)
  traj_ids = valid[sampled_valid, 0]
  bins = valid[sampled_valid, 1]

  seg_lengths = segment_length[traj_ids].float()
  bin_starts = bins.float() * float(bin_width_frames)
  bin_spans = torch.minimum(
    torch.full_like(seg_lengths, float(bin_width_frames)),
    torch.clamp(seg_lengths - bin_starts, min=1.0),
  )
  local_frames = (
    bin_starts
    + sample_uniform(0.0, 1.0, (num_samples,), device=device) * bin_spans
  ).long()
  local_frames = torch.clamp(local_frames, max=(seg_lengths.long() - 1).clamp(min=0))
  time_steps = segment_start_idx[traj_ids] + local_frames
  return traj_ids, bins, time_steps, probs_valid
