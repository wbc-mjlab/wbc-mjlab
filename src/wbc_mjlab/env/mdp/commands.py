from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import mujoco
import numpy as np
import torch

from mjlab.managers import CommandTerm
from mjlab.tasks.tracking.mdp import MotionCommandCfg as MjlabMotionCommandCfg
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  quat_apply_inverse,
  quat_error_magnitude,
  quat_from_euler_xyz,
  quat_inv,
  quat_mul,
  sample_uniform,
  yaw_quat,
)
from mjlab.viewer.debug_visualizer import DebugVisualizer

from wbc_mjlab.env.mdp.sampling import (
  AdaptiveSimilarityTermCfg,
  TrackingSimilarityState,
  bin_index_for_frame,
  compile_similarity_terms,
  sample_adaptive_bins,
  step_tracking_similarity,
  update_failure_ema,
  whole_body_adaptive_similarity_terms,
  wbc_joint_only_similarity_terms,
)

if TYPE_CHECKING:
  from collections.abc import Callable
  from typing import Any

  import viser

  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

_DESIRED_FRAME_COLORS = ((1.0, 0.5, 0.5), (0.5, 1.0, 0.5), (0.5, 0.5, 1.0))


class MotionLoader:
  """Load a flat motion NPZ, optionally with multi-trajectory segment metadata."""

  def __init__(
    self,
    motion_file: str,
    body_indexes: torch.Tensor,
    anchor_body_index: int,
    step_dt: float,
    device: str = "cpu",
  ) -> None:
    data = np.load(motion_file)
    self.joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
    self.joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
    self._body_pos_w = torch.tensor(
      data["body_pos_w"], dtype=torch.float32, device=device
    )
    self._body_quat_w = torch.tensor(
      data["body_quat_w"], dtype=torch.float32, device=device
    )
    self._body_lin_vel_w = torch.tensor(
      data["body_lin_vel_w"], dtype=torch.float32, device=device
    )
    self._body_ang_vel_w = torch.tensor(
      data["body_ang_vel_w"], dtype=torch.float32, device=device
    )
    self._body_indexes = body_indexes
    self.body_pos_w = self._body_pos_w[:, self._body_indexes]
    self.body_quat_w = self._body_quat_w[:, self._body_indexes]
    self.body_lin_vel_w = self._body_lin_vel_w[:, self._body_indexes]
    self.body_ang_vel_w = self._body_ang_vel_w[:, self._body_indexes]
    self.time_step_total = self.joint_pos.shape[0]

    if "segment_start_idx" in data and "segment_length" in data:
      self.segment_start_idx = torch.tensor(
        data["segment_start_idx"], dtype=torch.long, device=device
      )
      self.segment_length = torch.tensor(
        data["segment_length"], dtype=torch.long, device=device
      )
    else:
      self.segment_start_idx = torch.tensor([0], dtype=torch.long, device=device)
      self.segment_length = torch.tensor(
        [self.time_step_total], dtype=torch.long, device=device
      )

    self.num_trajectories = int(self.segment_start_idx.shape[0])
    self.segment_end_idx = self.segment_start_idx + self.segment_length

    anchor_lin_vel = self._body_lin_vel_w[:, anchor_body_index]
    anchor_ang_vel = self._body_ang_vel_w[:, anchor_body_index]
    self.anchor_lin_acc_w = torch.gradient(
      anchor_lin_vel, spacing=step_dt, dim=0
    )[0]
    self.anchor_ang_acc_w = torch.gradient(
      anchor_ang_vel, spacing=step_dt, dim=0
    )[0]


class MotionCommand(CommandTerm):
  cfg: MotionCommandCfg
  _env: ManagerBasedRlEnv

  def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.robot: Entity = env.scene[cfg.entity_name]
    if cfg.actuated_joint_names:
      tracked_joint_ids, _ = self.robot.find_joints(
        cfg.actuated_joint_names, preserve_order=True
      )
      self._tracked_joint_ids = torch.tensor(
        tracked_joint_ids, dtype=torch.long, device=self.device
      )
    else:
      self._tracked_joint_ids = None
    self.robot_anchor_body_index = self.robot.body_names.index(
      self.cfg.anchor_body_name
    )
    self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
    self.body_indexes = torch.tensor(
      self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0],
      dtype=torch.long,
      device=self.device,
    )

    self.motion = MotionLoader(
      self.cfg.motion_file,
      self.body_indexes,
      anchor_body_index=self.motion_anchor_body_index,
      step_dt=env.step_dt,
      device=self.device,
    )
    self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.trajectory_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.body_pos_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 3, device=self.device
    )
    self.body_quat_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 4, device=self.device
    )
    self.body_quat_relative_w[:, :, 0] = 1.0

    self.bin_width_frames = max(
      1, int(math.ceil(cfg.adaptive_bin_width_s / max(env.step_dt, 1.0e-6)))
    )
    segment_bins = torch.ceil(
      self.motion.segment_length.float() / float(self.bin_width_frames)
    ).long()
    self.bins_per_trajectory = max(1, int(segment_bins.max().item()))
    self.bin_valid_mask = torch.zeros(
      self.motion.num_trajectories,
      self.bins_per_trajectory,
      dtype=torch.bool,
      device=self.device,
    )
    for traj_idx in range(self.motion.num_trajectories):
      n_bins = int(segment_bins[traj_idx].item())
      self.bin_valid_mask[traj_idx, :n_bins] = True

    self.bin_failed_count = torch.zeros(
      self.motion.num_trajectories,
      self.bins_per_trajectory,
      dtype=torch.float,
      device=self.device,
    )
    self._valid_bin_indices = self.bin_valid_mask.nonzero(as_tuple=False)

    self._episode_similarity_sum = torch.zeros(self.num_envs, device=self.device)
    self._episode_step_count = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )
    self._episode_start_step = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )
    self._episode_start_bin = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )
    self.episode_assist_gain = torch.zeros(self.num_envs, device=self.device)
    self.assist_force_w = torch.zeros(self.num_envs, 3, device=self.device)
    self.assist_torque_w = torch.zeros(self.num_envs, 3, device=self.device)

    self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_anchor_lin_vel"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["error_anchor_ang_vel"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["assist_gain_mean"] = torch.zeros(self.num_envs, device=self.device)

    self._similarity_terms, similarity_weight_sum = compile_similarity_terms(
      cfg.adaptive_similarity_terms
    )
    if (
      cfg.adaptive_sampling_strategy == "similarity_ema"
      and similarity_weight_sum <= 0.0
    ):
      raise ValueError(
        "similarity_ema requires at least one adaptive_similarity_terms entry with weight > 0."
      )
    self._similarity_term_weight_sum = max(similarity_weight_sum, 1.0e-6)

    self._ghost_model: mujoco.MjModel | None = None
    self._ghost_color = np.array(cfg.viz.ghost_color, dtype=np.float32)

  @property
  def bin_count(self) -> int:
    return self.motion.num_trajectories * self.bins_per_trajectory

  def _update_failure_levels(self, env_ids: torch.Tensor) -> None:
    if self.cfg.sampling_mode != "adaptive" or env_ids.numel() == 0:
      return

    active_mask = self._episode_step_count[env_ids] > 0
    if not torch.any(active_mask):
      return
    env_ids = env_ids[active_mask]

    strategy = self.cfg.adaptive_sampling_strategy
    update_failure_ema(
      self.bin_failed_count,
      strategy=strategy,
      bins_per_trajectory=self.bins_per_trajectory,
      alpha=self.cfg.adaptive_alpha,
      traj_ids=self.trajectory_ids[env_ids],
      start_bins=self._episode_start_bin[env_ids],
      episode_terminated=(
        self._env.termination_manager.terminated[env_ids]
        if strategy == "binary_failure"
        else None
      ),
      episode_similarity_sum=(
        self._episode_similarity_sum[env_ids]
        if strategy == "similarity_ema"
        else None
      ),
      episode_step_count=(
        self._episode_step_count[env_ids]
        if strategy == "similarity_ema"
        else None
      ),
    )

  def _set_episode_assist_gain(
    self, env_ids: torch.Tensor, traj_ids: torch.Tensor, bins: torch.Tensor
  ) -> None:
    if not self.cfg.assistive_wrench_enabled:
      self.episode_assist_gain[env_ids] = 0.0
      return
    failure = self.bin_failed_count[traj_ids, bins]
    similarity = 1.0 - failure
    eta = max(self.cfg.assistive_eta, 1.0e-6)
    self.episode_assist_gain[env_ids] = torch.clamp(
      1.0 - similarity / eta, 0.0, self.cfg.assistive_beta_max
    )

  def _adaptive_sampling(self, env_ids: torch.Tensor):
    self._update_failure_levels(env_ids)

    traj_ids, bins, time_steps, probs_valid = sample_adaptive_bins(
      self.bin_failed_count,
      self._valid_bin_indices,
      segment_length=self.motion.segment_length,
      segment_start_idx=self.motion.segment_start_idx,
      bin_width_frames=self.bin_width_frames,
      temperature_base=self.cfg.adaptive_temperature_base,
      uniform_ratio=self.cfg.adaptive_uniform_ratio,
      num_samples=len(env_ids),
      device=self.device,
    )
    self.trajectory_ids[env_ids] = traj_ids
    self.time_steps[env_ids] = time_steps
    self._episode_start_bin[env_ids] = bins
    self._set_episode_assist_gain(env_ids, traj_ids, bins)

    num_valid = max(1, probs_valid.shape[0])
    H = -(probs_valid * (probs_valid + 1e-12).log()).sum()
    self.metrics["sampling_entropy"][:] = (
      float(H / math.log(num_valid)) if num_valid > 1 else 1.0
    )
    pmax, imax = probs_valid.max(dim=0)
    self.metrics["sampling_top1_prob"][:] = float(pmax)
    self.metrics["sampling_top1_bin"][:] = float(imax) / float(num_valid)

  def _uniform_sampling(self, env_ids: torch.Tensor):
    traj_ids = torch.randint(
      0, self.motion.num_trajectories, (len(env_ids),), device=self.device
    )
    seg_lengths = self.motion.segment_length[traj_ids]
    local_frames = (
      sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device)
      * (seg_lengths.float() - 1.0)
    ).long()
    self.trajectory_ids[env_ids] = traj_ids
    self.time_steps[env_ids] = self.motion.segment_start_idx[traj_ids] + local_frames
    start_bins = self._bin_index_for_frame(traj_ids, self.time_steps[env_ids])
    self._episode_start_bin[env_ids] = start_bins
    self._set_episode_assist_gain(env_ids, traj_ids, start_bins)

    num_valid = max(1, int(self.bin_valid_mask.sum().item()))
    self.metrics["sampling_entropy"][:] = 1.0
    self.metrics["sampling_top1_prob"][:] = 1.0 / num_valid
    self.metrics["sampling_top1_bin"][:] = 0.5

  def _tracking_similarity_state(self) -> TrackingSimilarityState:
    return TrackingSimilarityState(
      tracked_joint_pos_error=self._tracked_joint_pos_error(),
      anchor_pos_error=self.anchor_pos_w - self.robot_anchor_pos_w,
      anchor_ori_error=quat_error_magnitude(
        self.anchor_quat_w, self.robot_anchor_quat_w
      ),
      body_pos_error=torch.sum(
        torch.square(self.body_pos_relative_w - self.robot_body_pos_w), dim=-1
      ),
      body_ori_error=(
        quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w) ** 2
      ),
      body_lin_vel_error=torch.sum(
        torch.square(self.body_lin_vel_w - self.robot_body_lin_vel_w), dim=-1
      ),
      body_ang_vel_error=torch.sum(
        torch.square(self.body_ang_vel_w - self.robot_body_ang_vel_w), dim=-1
      ),
    )

  def _step_tracking_similarity(self) -> torch.Tensor:
    return step_tracking_similarity(
      self._similarity_terms,
      self._similarity_term_weight_sum,
      self._tracking_similarity_state(),
      num_envs=self.num_envs,
      device=self.device,
    )

  def _bin_index_for_frame(
    self, trajectory_ids: torch.Tensor, time_steps: torch.Tensor
  ):
    return bin_index_for_frame(
      segment_start_idx=self.motion.segment_start_idx,
      time_steps=time_steps,
      trajectory_ids=trajectory_ids,
      bin_width_frames=self.bin_width_frames,
      bins_per_trajectory=self.bins_per_trajectory,
    )

  # --- WBC reference features (Table S3); stacked in ``command`` for the actor. ---

  @property
  def ref_base_height(self) -> torch.Tensor:
    """Anchor height relative to env origin (z_I r̂_IB)."""
    return self.anchor_pos_w[:, 2:3] - self._env.scene.env_origins[:, 2:3]

  @property
  def ref_base_lin_vel_b(self) -> torch.Tensor:
    """Reference anchor linear velocity in anchor frame (B v̂_IB)."""
    return quat_apply_inverse(self.anchor_quat_w, self.anchor_lin_vel_w)

  @property
  def ref_base_ang_vel_b(self) -> torch.Tensor:
    """Reference anchor angular velocity in anchor frame (B ω̂_IB)."""
    return quat_apply_inverse(self.anchor_quat_w, self.anchor_ang_vel_w)

  @property
  def ref_gravity_b(self) -> torch.Tensor:
    """Reference gravity in anchor frame (B ĝ_I)."""
    return quat_apply_inverse(self.anchor_quat_w, self.robot.data.gravity_vec_w)

  @property
  def ref_base_lin_acc_b(self) -> torch.Tensor:
    return quat_apply_inverse(self.anchor_quat_w, self.anchor_lin_acc_w)

  @property
  def ref_base_ang_acc_b(self) -> torch.Tensor:
    return quat_apply_inverse(self.anchor_quat_w, self.anchor_ang_acc_w)

  @property
  def tracked_joint_pos(self) -> torch.Tensor:
    """Reference joint positions for actuated / tracked DoFs (absolute)."""
    if self._tracked_joint_ids is not None:
      return self.joint_pos[:, self._tracked_joint_ids]
    return self.joint_pos

  @property
  def tracked_joint_vel(self) -> torch.Tensor:
    """Reference joint velocities for actuated / tracked DoFs (absolute)."""
    if self._tracked_joint_ids is not None:
      return self.joint_vel[:, self._tracked_joint_ids]
    return self.joint_vel

  @property
  def command(self) -> torch.Tensor:
    """Legacy stacked WBC reference vector (same layout as default ref obs terms).

    Prefer configuring individual reference observation terms in ``wbc_env_cfg``.
    Kept for ONNX metadata, ``wbc_command_dim``, and legacy deploy bundles.
    """
    return torch.cat(
      [
        self.ref_base_height,
        self.ref_base_lin_vel_b,
        self.ref_base_ang_vel_b,
        self.ref_gravity_b,
        self.tracked_joint_pos,
      ],
      dim=-1,
    )

  @property
  def wbc_command_dim(self) -> int:
    return int(self.command.shape[-1])

  @property
  def joint_pos(self) -> torch.Tensor:
    return self.motion.joint_pos[self.time_steps]

  @property
  def joint_vel(self) -> torch.Tensor:
    return self.motion.joint_vel[self.time_steps]

  def _tracked_joint_pos_error(self) -> torch.Tensor:
    error = self.joint_pos - self.robot_joint_pos
    if self._tracked_joint_ids is not None:
      error = error[:, self._tracked_joint_ids]
    return error

  def _tracked_joint_vel_error(self) -> torch.Tensor:
    error = self.joint_vel - self.robot_joint_vel
    if self._tracked_joint_ids is not None:
      error = error[:, self._tracked_joint_ids]
    return error

  @property
  def body_pos_w(self) -> torch.Tensor:
    return (
      self.motion.body_pos_w[self.time_steps] + self._env.scene.env_origins[:, None, :]
    )

  @property
  def body_quat_w(self) -> torch.Tensor:
    return self.motion.body_quat_w[self.time_steps]

  @property
  def body_lin_vel_w(self) -> torch.Tensor:
    return self.motion.body_lin_vel_w[self.time_steps]

  @property
  def body_ang_vel_w(self) -> torch.Tensor:
    return self.motion.body_ang_vel_w[self.time_steps]

  @property
  def anchor_pos_w(self) -> torch.Tensor:
    return (
      self.motion.body_pos_w[self.time_steps, self.motion_anchor_body_index]
      + self._env.scene.env_origins
    )

  @property
  def anchor_quat_w(self) -> torch.Tensor:
    return self.motion.body_quat_w[self.time_steps, self.motion_anchor_body_index]

  @property
  def anchor_lin_vel_w(self) -> torch.Tensor:
    return self.motion.body_lin_vel_w[self.time_steps, self.motion_anchor_body_index]

  @property
  def anchor_ang_vel_w(self) -> torch.Tensor:
    return self.motion.body_ang_vel_w[self.time_steps, self.motion_anchor_body_index]

  @property
  def anchor_lin_acc_w(self) -> torch.Tensor:
    return self.motion.anchor_lin_acc_w[self.time_steps]

  @property
  def anchor_ang_acc_w(self) -> torch.Tensor:
    return self.motion.anchor_ang_acc_w[self.time_steps]

  @property
  def robot_joint_pos(self) -> torch.Tensor:
    return self.robot.data.joint_pos

  @property
  def robot_joint_vel(self) -> torch.Tensor:
    return self.robot.data.joint_vel

  @property
  def robot_body_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.body_indexes]

  @property
  def robot_body_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.body_indexes]

  @property
  def robot_body_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.body_indexes]

  @property
  def robot_body_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.body_indexes]

  @property
  def robot_anchor_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.robot_anchor_body_index]

  def _update_metrics(self):
    self.metrics["error_anchor_pos"] = torch.norm(
      self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1
    )
    self.metrics["error_anchor_rot"] = quat_error_magnitude(
      self.anchor_quat_w, self.robot_anchor_quat_w
    )
    self.metrics["error_anchor_lin_vel"] = torch.norm(
      self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1
    )
    self.metrics["error_anchor_ang_vel"] = torch.norm(
      self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1
    )

    self.metrics["error_body_pos"] = torch.norm(
      self.body_pos_relative_w - self.robot_body_pos_w, dim=-1
    ).mean(dim=-1)
    self.metrics["error_body_rot"] = quat_error_magnitude(
      self.body_quat_relative_w, self.robot_body_quat_w
    ).mean(dim=-1)

    self.metrics["error_body_lin_vel"] = torch.norm(
      self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1
    ).mean(dim=-1)
    self.metrics["error_body_ang_vel"] = torch.norm(
      self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1
    ).mean(dim=-1)

    self.metrics["error_joint_pos"] = torch.norm(
      self._tracked_joint_pos_error(), dim=-1
    )
    self.metrics["error_joint_vel"] = torch.norm(
      self._tracked_joint_vel_error(), dim=-1
    )
    self.metrics["assist_gain_mean"] = self.episode_assist_gain

  def _write_reference_state_to_sim(
    self,
    env_ids: torch.Tensor,
    root_pos: torch.Tensor,
    root_ori: torch.Tensor,
    root_lin_vel: torch.Tensor,
    root_ang_vel: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
  ) -> None:
    soft_limits = self.robot.data.soft_joint_pos_limits[env_ids]
    joint_pos = torch.clip(joint_pos, soft_limits[:, :, 0], soft_limits[:, :, 1])
    self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    root_state = torch.cat([root_pos, root_ori, root_lin_vel, root_ang_vel], dim=-1)
    self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    self.robot.reset(env_ids=env_ids)

  def _resample_command(self, env_ids: torch.Tensor):
    if self.cfg.sampling_mode == "start":
      self.trajectory_ids[env_ids] = 0
      self.time_steps[env_ids] = self.motion.segment_start_idx[0]
      self._episode_start_bin[env_ids] = 0
      start_traj = self.trajectory_ids[env_ids]
      start_bins = self._episode_start_bin[env_ids]
      self._set_episode_assist_gain(env_ids, start_traj, start_bins)
    elif self.cfg.sampling_mode == "uniform":
      self._uniform_sampling(env_ids)
    else:
      assert self.cfg.sampling_mode == "adaptive"
      self._adaptive_sampling(env_ids)

    root_pos = self.body_pos_w[env_ids, 0].clone()
    root_ori = self.body_quat_w[env_ids, 0].clone()
    root_lin_vel = self.body_lin_vel_w[env_ids, 0].clone()
    root_ang_vel = self.body_ang_vel_w[env_ids, 0].clone()

    range_list = [
      self.cfg.pose_range.get(key, (0.0, 0.0))
      for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=self.device)
    rand_samples = sample_uniform(
      ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
    )
    root_pos += rand_samples[:, 0:3]
    orientations_delta = quat_from_euler_xyz(
      rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5]
    )
    root_ori = quat_mul(orientations_delta, root_ori)
    range_list = [
      self.cfg.velocity_range.get(key, (0.0, 0.0))
      for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=self.device)
    rand_samples = sample_uniform(
      ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
    )
    root_lin_vel += rand_samples[:, :3]
    root_ang_vel += rand_samples[:, 3:]

    joint_pos = self.joint_pos[env_ids].clone()
    joint_vel = self.joint_vel[env_ids]

    joint_pos += sample_uniform(
      lower=self.cfg.joint_position_range[0],
      upper=self.cfg.joint_position_range[1],
      size=joint_pos.shape,
      device=joint_pos.device,
    )

    self._write_reference_state_to_sim(
      env_ids,
      root_pos,
      root_ori,
      root_lin_vel,
      root_ang_vel,
      joint_pos,
      joint_vel,
    )
    self._episode_start_step[env_ids] = self.time_steps[env_ids]
    self._episode_similarity_sum[env_ids] = 0.0
    self._episode_step_count[env_ids] = 0
    self.update_relative_body_poses()

  def update_relative_body_poses(self) -> None:
    anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )

    delta_pos_w = robot_anchor_pos_w_repeat
    delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
    delta_ori_w = yaw_quat(
      quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat))
    )

    self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
    self.body_pos_relative_w = delta_pos_w + quat_apply(
      delta_ori_w, self.body_pos_w - anchor_pos_w_repeat
    )

  def compute(self, dt: float) -> None:
    self._update_metrics()
    self.time_left -= dt
    resample_env_ids = (self.time_left <= 0.0).nonzero().flatten()
    if len(resample_env_ids) > 0:
      self._resample(resample_env_ids)
    self._update_command(advance_time=dt > 0.0)

  def _update_command(self, *, advance_time: bool = True):
    if self.cfg.adaptive_sampling_strategy == "similarity_ema" and advance_time:
      self._episode_similarity_sum += self._step_tracking_similarity()
      self._episode_step_count += 1

    if advance_time:
      self.time_steps += 1

    seg_end = self.motion.segment_end_idx[self.trajectory_ids]
    env_ids = torch.where(self.time_steps >= seg_end)[0]
    if env_ids.numel() > 0:
      self._resample_command(env_ids)

    self.update_relative_body_poses()

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    if self.cfg.viz.mode == "ghost":
      if self._ghost_model is None:
        self._ghost_model = copy.deepcopy(self._env.sim.mj_model)
        self._ghost_model.geom_rgba[:] = self._ghost_color

      entity: Entity = self._env.scene[self.cfg.entity_name]
      indexing = entity.indexing
      free_joint_q_adr = indexing.free_joint_q_adr.cpu().numpy()
      joint_q_adr = indexing.joint_q_adr.cpu().numpy()

      for batch in env_indices:
        qpos = np.zeros(self._env.sim.mj_model.nq)
        qpos[free_joint_q_adr[0:3]] = self.body_pos_w[batch, 0].cpu().numpy()
        qpos[free_joint_q_adr[3:7]] = self.body_quat_w[batch, 0].cpu().numpy()
        qpos[joint_q_adr] = self.joint_pos[batch].cpu().numpy()
        visualizer.add_ghost_mesh(qpos, model=self._ghost_model, label=f"ghost_{batch}")

    elif self.cfg.viz.mode == "frames":
      for batch in env_indices:
        desired_body_pos = self.body_pos_w[batch].cpu().numpy()
        desired_body_quat = self.body_quat_w[batch]
        desired_body_rotm = matrix_from_quat(desired_body_quat).cpu().numpy()

        current_body_pos = self.robot_body_pos_w[batch].cpu().numpy()
        current_body_quat = self.robot_body_quat_w[batch]
        current_body_rotm = matrix_from_quat(current_body_quat).cpu().numpy()

        for i, body_name in enumerate(self.cfg.body_names):
          visualizer.add_frame(
            position=desired_body_pos[i],
            rotation_matrix=desired_body_rotm[i],
            scale=0.08,
            label=f"desired_{body_name}_{batch}",
            axis_colors=_DESIRED_FRAME_COLORS,
          )
          visualizer.add_frame(
            position=current_body_pos[i],
            rotation_matrix=current_body_rotm[i],
            scale=0.12,
            label=f"current_{body_name}_{batch}",
          )

        desired_anchor_pos = self.anchor_pos_w[batch].cpu().numpy()
        desired_anchor_quat = self.anchor_quat_w[batch]
        desired_rotation_matrix = matrix_from_quat(desired_anchor_quat).cpu().numpy()
        visualizer.add_frame(
          position=desired_anchor_pos,
          rotation_matrix=desired_rotation_matrix,
          scale=0.1,
          label=f"desired_anchor_{batch}",
          axis_colors=_DESIRED_FRAME_COLORS,
        )

        current_anchor_pos = self.robot_anchor_pos_w[batch].cpu().numpy()
        current_anchor_quat = self.robot_anchor_quat_w[batch]
        current_rotation_matrix = matrix_from_quat(current_anchor_quat).cpu().numpy()
        visualizer.add_frame(
          position=current_anchor_pos,
          rotation_matrix=current_rotation_matrix,
          scale=0.15,
          label=f"current_anchor_{batch}",
        )

  def create_gui(
    self,
    name: str,
    server: viser.ViserServer,
    get_env_idx: Callable[[], int],
    on_change: Callable[[], None] | None = None,
    request_action: Callable[[str, Any], None] | None = None,
  ) -> None:
    max_frame = int(self.motion.time_step_total) - 1

    with server.gui.add_folder(name.capitalize()):
      scrubber = server.gui.add_slider(
        "Frame",
        min=0,
        max=max_frame,
        step=1,
        initial_value=0,
      )

      @scrubber.on_update
      def _(_) -> None:
        idx = get_env_idx()
        self.time_steps[idx] = int(scrubber.value)
        if on_change is not None:
          on_change()

      all_envs_cb = server.gui.add_checkbox("All envs", initial_value=True)
      start_btn = server.gui.add_button("Start Here")

      @start_btn.on_click
      def _(_) -> None:
        if request_action is not None:
          request_action(
            "CUSTOM",
            {"type": "gui_reset", "all_envs": all_envs_cb.value},
          )

    self._scrubber_handles = (scrubber, all_envs_cb, start_btn)
    self._set_scrubber_disabled(True)

  def _set_scrubber_disabled(self, disabled: bool) -> None:
    for handle in self._scrubber_handles:
      handle.disabled = disabled

  def on_viewer_pause(self, paused: bool) -> None:
    if hasattr(self, "_scrubber_handles"):
      self._set_scrubber_disabled(not paused)

  def apply_gui_reset(self, env_ids: torch.Tensor) -> bool:
    if not hasattr(self, "_scrubber_handles"):
      return False
    frame = int(self._scrubber_handles[0].value)
    self.reset_to_frame(env_ids, frame)
    self.update_relative_body_poses()
    return True

  def reset_to_frame(self, env_ids: torch.Tensor, frame: int) -> None:
    self.time_steps[env_ids] = frame
    traj_ids = torch.searchsorted(
      self.motion.segment_end_idx,
      torch.full((len(env_ids),), frame, device=self.device),
      right=False,
    )
    traj_ids = torch.clamp(traj_ids, max=self.motion.num_trajectories - 1)
    self.trajectory_ids[env_ids] = traj_ids
    self._write_reference_state_to_sim(
      env_ids,
      self.body_pos_w[env_ids, 0],
      self.body_quat_w[env_ids, 0],
      self.body_lin_vel_w[env_ids, 0],
      self.body_ang_vel_w[env_ids, 0],
      self.joint_pos[env_ids],
      self.joint_vel[env_ids],
    )


@dataclass(kw_only=True)
class MotionCommandCfg(MjlabMotionCommandCfg):
  motion_file: str
  anchor_body_name: str
  body_names: tuple[str, ...]
  entity_name: str
  actuated_joint_names: tuple[str, ...] = ()
  """If set, joint tracking metrics/RSI use only these DoFs (subset of the robot)."""
  pose_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  velocity_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  joint_position_range: tuple[float, float] = (-0.52, 0.52)
  adaptive_bin_width_s: float = 4.0
  adaptive_uniform_ratio: float = 0.15
  adaptive_alpha: float = 0.005
  adaptive_temperature_base: float = 1.0
  adaptive_sampling_strategy: Literal["binary_failure", "similarity_ema"] = (
    "similarity_ema"
  )
  adaptive_similarity_terms: tuple[AdaptiveSimilarityTermCfg, ...] = field(
    default_factory=wbc_joint_only_similarity_terms
  )
  sampling_mode: Literal["adaptive", "uniform", "start"] = "adaptive"
  assistive_wrench_enabled: bool = True
  assistive_beta_max: float = 0.6
  assistive_eta: float = 0.8

  @dataclass
  class VizCfg:
    mode: Literal["ghost", "frames"] = "ghost"
    ghost_color: tuple[float, float, float, float] = (0.5, 0.7, 0.5, 0.5)

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> MotionCommand:
    return MotionCommand(self, env)
