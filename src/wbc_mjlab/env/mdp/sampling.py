"""RSI adaptive sampling helpers for ``MotionCommand``.

``RsiCfg.strategy`` selects the episode failure signal:

- ``binary_failure`` — failure on early termination (BeyondMimic RSI).
- ``similarity_ema`` — failure from mean per-step tracking similarity (Zest RSI).
- ``discounted_return`` — failure from the discounted return earned from the start bin,
  relative to the most it could have earned (SuperMimic, arXiv:2610.28480, App. D.1).
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

AdaptiveSamplingStrategy = Literal[
  "binary_failure", "similarity_ema", "discounted_return"
]

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
  return_gamma: float = 0.99
  """``discounted_return``: discount of the episode return."""
  temperature_scale_by_bins: bool = True
  """Softmax temperature ``temperature_base / log(1 + bins)``; False: ``temperature_base``."""
  memory_ratio: float = 0.0
  """Share of starts drawn in proportion to each bin's running-average softmax weight, so a
  hard bin keeps some practice after it is learned (SuperMimic: 0.1)."""
  memory_alpha: float = 0.001
  """Rate of that running average, per sampling call."""


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
    std = (
      term_cfg.std
      if term_cfg.std is not None
      else DEFAULT_SIMILARITY_STDS[term_cfg.term]
    )
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
      kernel = torch.exp(-(state.anchor_ori_error**2) / std**2)
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
  episode_return: torch.Tensor | None = None,
  episode_return_max: torch.Tensor | None = None,
) -> None:
  """EMA update of per-(trajectory, bin) failure level."""
  if strategy == "binary_failure":
    assert episode_terminated is not None
    if not torch.any(episode_terminated):
      return
    traj_ids = traj_ids[episode_terminated]
    start_bins = start_bins[episode_terminated]
    episode_failure = torch.ones(traj_ids.shape[0], device=bin_failure_levels.device)
  elif strategy == "discounted_return":
    assert episode_return is not None and episode_return_max is not None
    ratio = episode_return / episode_return_max.clamp(min=1.0e-6)
    episode_failure = 1.0 - torch.clamp(ratio, 0.0, 1.0)
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
  flat_levels[update_mask] = (1.0 - alpha) * flat_levels[
    update_mask
  ] + alpha * mean_failure


def bin_probabilities(
  levels: torch.Tensor,
  *,
  temperature_base: float,
  uniform_ratio: float,
  temperature_scale_by_bins: bool = True,
  memory: torch.Tensor | None = None,
  memory_ratio: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Start probability of each valid bin from its failure level, and the softmax part.

  ``uniform_ratio`` uniform, ``memory_ratio`` in proportion to ``memory`` (running-average
  softmax weights), the rest ``softmax(level / temperature)``.
  """
  num_valid = max(1, levels.shape[0])
  temperature = temperature_base
  if temperature_scale_by_bins:
    temperature = temperature / math.log(1.0 + num_valid)
  softmax = torch.softmax(levels / temperature, dim=0)
  probs = (1.0 - uniform_ratio - memory_ratio) * softmax + uniform_ratio / num_valid
  if memory is not None and memory_ratio > 0.0:
    probs = probs + memory_ratio * memory / memory.sum().clamp(min=1.0e-12)
  return probs, softmax


def sample_adaptive_bins(
  probs_valid: torch.Tensor,
  valid_bin_indices: torch.Tensor,
  *,
  segment_length: torch.Tensor,
  segment_start_idx: torch.Tensor,
  bin_width_frames: int,
  num_samples: int,
  device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Draw (trajectory, bin, frame) from per-valid-bin probabilities (:func:`bin_probabilities`)."""
  valid = valid_bin_indices
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
    bin_starts + sample_uniform(0.0, 1.0, (num_samples,), device=device) * bin_spans
  ).long()
  local_frames = torch.clamp(local_frames, max=(seg_lengths.long() - 1).clamp(min=0))
  time_steps = segment_start_idx[traj_ids] + local_frames
  return traj_ids, bins, time_steps


def compute_sampling_prob_matrix(
  bin_failure_levels: torch.Tensor,
  bin_valid_mask: torch.Tensor,
  *,
  temperature_base: float,
  uniform_ratio: float,
  temperature_scale_by_bins: bool = True,
  memory: torch.Tensor | None = None,
  memory_ratio: float = 0.0,
) -> torch.Tensor:
  """Per-(trajectory, bin) sampling probability (same policy as the sampler)."""
  valid = bin_valid_mask.nonzero(as_tuple=False)
  probs = torch.zeros_like(bin_failure_levels)
  if valid.numel() == 0:
    return probs
  probs_valid, _ = bin_probabilities(
    bin_failure_levels[valid[:, 0], valid[:, 1]],
    temperature_base=temperature_base,
    uniform_ratio=uniform_ratio,
    temperature_scale_by_bins=temperature_scale_by_bins,
    memory=memory,
    memory_ratio=memory_ratio,
  )
  probs[valid[:, 0], valid[:, 1]] = probs_valid
  return probs


def trajectory_conditional_prob_row(
  prob_matrix: torch.Tensor,
  valid_mask: torch.Tensor,
  traj_id: int,
) -> list[float]:
  """Renormalize global bin probabilities over valid bins on one trajectory."""
  row = prob_matrix[traj_id]
  mask = valid_mask[traj_id]
  total = float(row[mask].sum().item())
  if total <= 0.0:
    return row.detach().cpu().tolist()
  out = row.clone()
  out[mask] = row[mask] / total
  return out.detach().cpu().tolist()


def compute_assist_gain_matrix(
  bin_failure_levels: torch.Tensor,
  *,
  eta: float,
  beta_max: float,
  enabled: bool,
) -> torch.Tensor:
  """Per-bin assistive wrench scale β from failure levels (Zest curriculum)."""
  if not enabled:
    return torch.zeros_like(bin_failure_levels)
  similarity = 1.0 - bin_failure_levels
  return torch.clamp(1.0 - similarity / max(eta, 1.0e-6), 0.0, beta_max)


def rsi_failure_signal_label(
  strategy: AdaptiveSamplingStrategy,
  *,
  similarity_from_rewards: bool,
) -> str:
  if strategy == "binary_failure":
    return "EMA(1 if early terminated)"
  if strategy == "discounted_return":
    return "EMA(1 − discounted return / attainable)"
  if similarity_from_rewards:
    return "EMA(1 − mean motion-reward similarity)"
  return "EMA(1 − mean tracking similarity)"


def sampling_state(command: MotionCommand) -> dict:
  """One state schema for PPO/WM checkpoints and exported RSI sidecars."""
  return {
    "schema": 1,
    "failure_levels": command.bin_failure_levels.detach().clone(),
    "memory": command._bin_memory.detach().clone(),
    "valid_mask": command.bin_valid_mask.detach().clone(),
    "visit_counts": command.bin_visit_counts.detach().clone(),
    "segment_names": tuple(command.motion.segment_names),
    "segment_lengths": command.motion.segment_length.detach().clone(),
    "bin_width_s": command.cfg.rsi.bin_width_s,
  }


def restore_sampling_state(command: MotionCommand, state, *, strict=False) -> bool:
  """Validate identity before copying anything; accept older schemas when compatible."""

  def fail(reason):
    if strict:
      raise ValueError(f"RSI sampling state: {reason}")
    print(f"[WARN] RSI sampling state: {reason}")
    return False

  def tensor(value, target):
    return torch.as_tensor(value, dtype=target.dtype, device=target.device)

  if "segment_names" in state and tuple(state["segment_names"]) != tuple(
    command.motion.segment_names
  ):
    return fail("motion segments differ")
  if "segment_lengths" in state and not torch.equal(
    tensor(state["segment_lengths"], command.motion.segment_length),
    command.motion.segment_length,
  ):
    return fail("motion segment lengths differ")
  if "bin_width_s" in state and not math.isclose(
    float(np.asarray(state["bin_width_s"]).reshape(-1)[0]), command.cfg.rsi.bin_width_s
  ):
    return fail("bin width differs")
  levels = state.get("failure_levels", state.get("failure"))
  if levels is None:
    return fail("failure levels are missing")
  targets = {
    "failure_levels": command.bin_failure_levels,
    "memory": command._bin_memory,
    "valid_mask": command.bin_valid_mask,
    "visit_counts": command.bin_visit_counts,
  }
  values = dict(state)
  values["failure_levels"] = levels
  for name, target in targets.items():
    if name in values and tuple(values[name].shape) != tuple(target.shape):
      return fail(f"{name} shape differs")
  for name, target in targets.items():
    if name in values:
      target.copy_(tensor(values[name], target))
  command._valid_bin_indices = command.bin_valid_mask.nonzero(as_tuple=False)
  return True


def save_rsi_bin_stats(path: str | Path, command: MotionCommand) -> Path:
  """Export the same sampler state stored in tracking checkpoints."""
  out = Path(path)
  out.parent.mkdir(parents=True, exist_ok=True)
  state = {
    k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v
    for k, v in sampling_state(command).items()
  }
  # Preserve names consumed by older plotting/export tools.
  state.update(
    failure=state["failure_levels"],
    alpha=command.cfg.rsi.alpha,
    uniform_ratio=command.cfg.rsi.uniform_ratio,
    num_trajectories=command.motion.num_trajectories,
    bins_per_trajectory=command.bins_per_trajectory,
  )
  np.savez(out, **state)
  return out


def load_rsi_bin_stats(
  path: str | Path, command: MotionCommand, *, strict=False
) -> bool:
  src = Path(path)
  if not src.is_file():
    return False
  with np.load(src) as state:
    return restore_sampling_state(command, dict(state), strict=strict)
