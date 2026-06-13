"""RSI adaptive sampling helpers for ``MotionCommand``.

``RsiCfg.strategy`` selects the episode failure signal:

- ``binary_failure`` — failure on early termination (BeyondMimic RSI).
- ``similarity_ema`` — failure from mean per-step tracking similarity (Zest RSI).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch

from mjlab.utils.lab_api.math import sample_uniform

if TYPE_CHECKING:
  from wbc_mjlab.env.mdp.commands import MotionCommand

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


def joint_pos_similarity_preset() -> tuple[AdaptiveSimilarityTermCfg, ...]:
  """Hand-tuned RSI similarity: joint position only."""
  return (AdaptiveSimilarityTermCfg(term="joint_pos"),)


@dataclass
class RsiCfg:
  """Reference-state initialization (RSI) and adaptive bin sampling."""

  sampling_mode: Literal["adaptive", "uniform", "start"] = "adaptive"
  strategy: AdaptiveSamplingStrategy = "similarity_ema"
  similarity_terms: tuple[AdaptiveSimilarityTermCfg, ...] = field(
    default_factory=joint_pos_similarity_preset
  )
  bin_width_s: float = 4.0
  uniform_ratio: float = 0.15
  alpha: float = 0.005
  temperature_base: float = 1.0
  # Optional curriculum fidelity (off in ``BASE_RSI_CFG``; enable per task).
  similarity_norm_by_remaining_clip: bool = False
  min_bin_span_ratio: float = 0.0
  persist_failure_levels: bool = False
  failure_levels_filename: str = "rsi_bin_stats.npz"
  # When True, per-step similarity follows weighted ``motion_*`` reward terms
  # (same kernels and weights as training) instead of ``similarity_terms``.
  similarity_from_rewards: bool = False
  tracking_reward_prefix: str = "motion_"


def keybody_similarity_preset() -> tuple[AdaptiveSimilarityTermCfg, ...]:
  """Hand-tuned RSI similarity: anchor + keybody tracking terms."""
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


def resolve_tracking_reward_indices(
  reward_manager,
  *,
  name_prefix: str = "motion_",
) -> tuple[list[int], float]:
  """Reward-term indices and weight sum used for reward-aligned RSI similarity."""
  indices: list[int] = []
  weight_sum = 0.0
  for idx, (name, term_cfg) in enumerate(
    zip(reward_manager._term_names, reward_manager._term_cfgs, strict=False)
  ):
    if not name.startswith(name_prefix) or term_cfg.weight <= 0.0:
      continue
    indices.append(idx)
    weight_sum += term_cfg.weight
  return indices, weight_sum


def step_tracking_reward_similarity(
  step_reward: torch.Tensor,
  indices: list[int],
  weight_sum: float,
) -> torch.Tensor:
  """Weighted mean of per-step tracking reward values in ``[0, 1]``."""
  if not indices or weight_sum <= 0.0:
    return torch.zeros(step_reward.shape[0], device=step_reward.device)
  tracking_sum = step_reward[:, indices].sum(dim=1)
  return (tracking_sum / weight_sum).clamp(0.0, 1.0)


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


def build_bin_valid_mask(
  segment_length: torch.Tensor,
  *,
  bins_per_trajectory: int,
  bin_width_frames: int,
  min_bin_span_ratio: float,
  device: torch.device | str,
) -> torch.Tensor:
  """Mark valid (trajectory, bin) cells for adaptive sampling."""
  num_trajectories = int(segment_length.shape[0])
  mask = torch.zeros(
    num_trajectories, bins_per_trajectory, dtype=torch.bool, device=device
  )
  min_span_frames = 0
  if min_bin_span_ratio > 0.0:
    min_span_frames = max(1, int(round(min_bin_span_ratio * bin_width_frames)))

  for traj_idx in range(num_trajectories):
    seg_len = int(segment_length[traj_idx].item())
    if min_span_frames > 0 and seg_len < min_span_frames:
      continue
    n_bins = max(1, int(math.ceil(seg_len / float(bin_width_frames))))
    for bin_idx in range(min(n_bins, bins_per_trajectory)):
      if min_span_frames > 0:
        bin_start = bin_idx * bin_width_frames
        span = min(bin_width_frames, seg_len - bin_start)
        if span < min_span_frames:
          continue
      mask[traj_idx, bin_idx] = True
  return mask


def update_failure_ema(
  bin_failure_levels: torch.Tensor,
  *,
  strategy: AdaptiveSamplingStrategy,
  bins_per_trajectory: int,
  alpha: float,
  traj_ids: torch.Tensor,
  start_bins: torch.Tensor,
  episode_terminated: torch.Tensor | None,
  episode_similarity_sum: torch.Tensor | None,
  episode_step_count: torch.Tensor | None,
  similarity_denom: torch.Tensor | None = None,
  norm_by_remaining_clip: bool = False,
) -> None:
  """EMA update of per-(trajectory, bin) failure level."""
  if strategy == "binary_failure":
    assert episode_terminated is not None
    if not torch.any(episode_terminated):
      return
    traj_ids = traj_ids[episode_terminated]
    start_bins = start_bins[episode_terminated]
    episode_failure = torch.ones(traj_ids.shape[0], device=bin_failure_levels.device)
  else:
    assert episode_similarity_sum is not None and episode_step_count is not None
    if norm_by_remaining_clip:
      assert similarity_denom is not None
      denom = torch.clamp(similarity_denom.float(), min=1.0)
      mean_similarity = episode_similarity_sum / denom
    else:
      episode_length = torch.clamp(episode_step_count.float(), min=1.0)
      mean_similarity = episode_similarity_sum / episode_length
    episode_failure = 1.0 - torch.clamp(mean_similarity, 0.0, 1.0)

  bin_count = bin_failure_levels.numel()
  flat_idx = traj_ids * bins_per_trajectory + start_bins
  failure_sum = torch.zeros(bin_count, device=bin_failure_levels.device)
  failure_count = torch.zeros(bin_count, device=bin_failure_levels.device)
  failure_sum.scatter_add_(0, flat_idx, episode_failure)
  failure_count.scatter_add_(0, flat_idx, torch.ones_like(episode_failure))

  update_mask = failure_count > 0
  if not torch.any(update_mask):
    return

  mean_failure = failure_sum[update_mask] / failure_count[update_mask]
  flat_levels = bin_failure_levels.view(-1)
  flat_levels[update_mask] = (
    (1.0 - alpha) * flat_levels[update_mask] + alpha * mean_failure
  )


def sample_adaptive_bins(
  bin_failure_levels: torch.Tensor,
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
  logits = bin_failure_levels[valid[:, 0], valid[:, 1]] / temperature
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


def save_rsi_bin_stats(path: str | Path, command: MotionCommand) -> Path:
  """Write adaptive RSI bin failure levels to *path* (``.npz``)."""
  out = Path(path)
  out.parent.mkdir(parents=True, exist_ok=True)
  rsi = command.cfg.rsi
  levels = command.bin_failure_levels.detach().cpu().numpy()
  np.savez(
    out,
    failure_levels=levels,
    failure=levels,
    valid_mask=command.bin_valid_mask.detach().cpu().numpy(),
    bin_width_s=np.array([rsi.bin_width_s], dtype=np.float64),
    alpha=np.array([rsi.alpha], dtype=np.float64),
    uniform_ratio=np.array([rsi.uniform_ratio], dtype=np.float64),
    num_trajectories=np.array([command.motion.num_trajectories], dtype=np.int64),
    bins_per_trajectory=np.array([command.bins_per_trajectory], dtype=np.int64),
  )
  return out


def load_rsi_bin_stats(
  path: str | Path,
  command: MotionCommand,
  *,
  strict: bool = False,
) -> bool:
  """Restore bin failure levels from *path*. Returns False if the file is missing."""
  src = Path(path)
  if not src.is_file():
    return False

  data = np.load(src)
  if "failure_levels" in data:
    failure_levels = data["failure_levels"]
  elif "failure" in data:
    failure_levels = data["failure"]
  else:
    msg = f"RSI bin stats missing failure_levels array: {src}"
    if strict:
      raise ValueError(msg)
    print(f"[WARN] {msg}")
    return False
  valid = data["valid_mask"]

  if failure_levels.shape != command.bin_failure_levels.shape:
    msg = (
      f"RSI bin stats shape mismatch: file {failure_levels.shape}, "
      f"env {tuple(command.bin_failure_levels.shape)}"
    )
    if strict:
      raise ValueError(msg)
    print(f"[WARN] {msg}")
    return False

  command.bin_failure_levels.copy_(
    torch.as_tensor(failure_levels, dtype=torch.float32, device=command.device)
  )
  if valid.shape == command.bin_valid_mask.shape:
    command.bin_valid_mask.copy_(
      torch.as_tensor(valid, dtype=torch.bool, device=command.device)
    )
    command._valid_bin_indices = command.bin_valid_mask.nonzero(as_tuple=False)
  return True
