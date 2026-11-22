"""Viewer hooks for the motion command, including ghost drawing and playback widgets."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np
import torch
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_error_magnitude,
)
from mjlab.viewer.debug_visualizer import DebugVisualizer

from wbc_mjlab.env.mdp.sampling import (
  rsi_failure_signal_label,
  trajectory_conditional_prob_row,
)
from wbc_mjlab.viewer.motion_vis import (
  clip_name_for_trajectory,
  error_to_rgba,
  format_motion_context_html,
  format_rsi_panel_html,
)

if TYPE_CHECKING:
  from collections.abc import Callable
  from typing import Any

  import viser
  from mjlab.entity import Entity

_DESIRED_FRAME_COLORS = ((1.0, 0.5, 0.5), (0.5, 1.0, 0.5), (0.5, 0.5, 1.0))


class MotionViewerMixin:
  def _clip_name(self, traj_id: int) -> str:
    return clip_name_for_trajectory(self.motion.segment_names, traj_id)

  def _segment_phase(self, env_idx: int) -> float:
    traj_id = int(self.trajectory_ids[env_idx].item())
    seg_start = int(self.motion.segment_start_idx[traj_id].item())
    seg_len = max(int(self.motion.segment_length[traj_id].item()) - 1, 1)
    local = int(self.time_steps[env_idx].item()) - seg_start
    return float(np.clip(local / seg_len, 0.0, 1.0))

  def _body_tracking_errors(self, env_idx: int) -> tuple[np.ndarray, np.ndarray]:
    pos_err = torch.norm(
      self.body_pos_relative_w[env_idx] - self.robot_body_pos_w[env_idx], dim=-1
    )
    rot_err = quat_error_magnitude(
      self.body_quat_relative_w[env_idx], self.robot_body_quat_w[env_idx]
    )
    return pos_err.cpu().numpy(), rot_err.cpu().numpy()

  def update_viewer_gui(self, env_idx: int) -> None:
    if self._viewer_context_html is None:
      return
    traj_id = int(self.trajectory_ids[env_idx].item())
    frame = int(self.time_steps[env_idx].item())
    self._viewer_context_html.content = format_motion_context_html(
      env_idx=env_idx,
      traj_id=traj_id,
      clip_name=self._clip_name(traj_id),
      frame=frame,
      phase=self._segment_phase(env_idx),
      task_id=self._viewer_task_id,
      rsi_mode=self.cfg.rsi.sampling_mode,
      rsi_strategy=self.cfg.rsi.strategy,
      anchor_body=self.cfg.anchor_body_name,
      num_bodies=len(self.cfg.body_names),
    )
    if self._viewer_rsi_html is None:
      return
    bin_idx = int(
      self._bin_index_for_frame(
        self.trajectory_ids[env_idx : env_idx + 1],
        self.time_steps[env_idx : env_idx + 1],
      ).item()
    )
    failure, prob, visits, assist, valid = self._rsi_view_tensors()
    rsi = self.cfg.rsi
    traj_probs = trajectory_conditional_prob_row(prob, valid, traj_id)
    self._viewer_rsi_html.content = format_rsi_panel_html(
      bin_idx=bin_idx,
      num_bins=self.bins_per_trajectory,
      bin_width_s=rsi.bin_width_s,
      failure_levels=failure[traj_id].tolist(),
      sampling_probs=traj_probs,
      visit_counts=visits[traj_id].tolist(),
      assist_gains=assist[traj_id].tolist(),
      valid_mask=valid[traj_id].tolist(),
      failure_signal_label=rsi_failure_signal_label(
        rsi.strategy,
        similarity_from_rewards=rsi.similarity_from_rewards,
      ),
      show_assist=self.cfg.assistive_wrench_enabled,
      beta_max=self.cfg.assistive_beta_max,
    )

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
        if self._viewer_align_xy_yaw:
          root_pos = self.body_pos_relative_w[batch, 0].cpu().numpy()
          root_quat = self.body_quat_relative_w[batch, 0].cpu().numpy()
        else:
          root_pos = self.body_pos_w[batch, 0].cpu().numpy()
          root_quat = self.body_quat_w[batch, 0].cpu().numpy()
        qpos = np.zeros(self._env.sim.mj_model.nq)
        qpos[free_joint_q_adr[0:3]] = root_pos
        qpos[free_joint_q_adr[3:7]] = root_quat
        qpos[joint_q_adr] = self.joint_pos[batch].cpu().numpy()
        visualizer.add_ghost_mesh(qpos, model=self._ghost_model, label=f"ghost_{batch}")
        if self._viewer_color_bodies:
          self._add_body_error_markers(visualizer, batch)

    elif self.cfg.viz.mode == "frames":
      for batch in env_indices:
        if self._viewer_align_xy_yaw:
          desired_body_pos = self.body_pos_relative_w[batch].cpu().numpy()
          desired_body_quat = self.body_quat_relative_w[batch]
        else:
          desired_body_pos = self.body_pos_w[batch].cpu().numpy()
          desired_body_quat = self.body_quat_w[batch]
        desired_body_rotm = matrix_from_quat(desired_body_quat).cpu().numpy()

        current_body_pos = self.robot_body_pos_w[batch].cpu().numpy()
        current_body_quat = self.robot_body_quat_w[batch]
        current_body_rotm = matrix_from_quat(current_body_quat).cpu().numpy()

        pos_err, rot_err = self._body_tracking_errors(batch)
        for i, body_name in enumerate(self.cfg.body_names):
          err = float(pos_err[i] + rot_err[i])
          axis_colors = None
          if self._viewer_color_bodies:
            rgba = error_to_rgba(err)
            rgb = (rgba[0], rgba[1], rgba[2])
            axis_colors = (rgb, rgb, rgb)
          visualizer.add_frame(
            position=desired_body_pos[i],
            rotation_matrix=desired_body_rotm[i],
            scale=0.08,
            label=f"desired_{body_name}_{batch}",
            axis_colors=axis_colors or _DESIRED_FRAME_COLORS,
          )
          visualizer.add_frame(
            position=current_body_pos[i],
            rotation_matrix=current_body_rotm[i],
            scale=0.12,
            label=f"current_{body_name}_{batch}",
            axis_colors=axis_colors,
          )
          if self._viewer_color_bodies:
            visualizer.add_sphere(
              center=current_body_pos[i],
              radius=0.025,
              color=error_to_rgba(err),
              label=f"err_{body_name}_{batch}",
            )

        if self._viewer_align_xy_yaw:
          desired_anchor_pos = self.body_pos_relative_w[
            batch, self.motion_anchor_body_index
          ]
          desired_anchor_quat = self.body_quat_relative_w[
            batch, self.motion_anchor_body_index
          ]
        else:
          desired_anchor_pos = self.anchor_pos_w[batch]
          desired_anchor_quat = self.anchor_quat_w[batch]
        desired_rotation_matrix = matrix_from_quat(desired_anchor_quat).cpu().numpy()
        visualizer.add_frame(
          position=desired_anchor_pos.cpu().numpy(),
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

  def _add_body_error_markers(self, visualizer: DebugVisualizer, env_idx: int) -> None:
    pos_err, rot_err = self._body_tracking_errors(env_idx)
    body_pos = self.robot_body_pos_w[env_idx].cpu().numpy()
    for i, body_name in enumerate(self.cfg.body_names):
      err = float(pos_err[i] + rot_err[i])
      visualizer.add_sphere(
        center=body_pos[i],
        radius=0.03,
        color=error_to_rgba(err),
        label=f"err_{body_name}_{env_idx}",
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
      with server.gui.add_folder("Selected env", expand_by_default=True):
        self._viewer_context_html = server.gui.add_html("")

      with server.gui.add_folder("Adaptive sampling (RSI)", expand_by_default=False):
        self._viewer_rsi_html = server.gui.add_html("")

      align_cb = server.gui.add_checkbox(
        "Align xy/yaw to reference",
        initial_value=self._viewer_align_xy_yaw,
      )
      color_cb = server.gui.add_checkbox(
        "Color bodies by tracking error",
        initial_value=self._viewer_color_bodies,
      )

      @align_cb.on_update
      def _on_align(_) -> None:
        self._viewer_align_xy_yaw = bool(align_cb.value)
        if on_change is not None:
          on_change()

      @color_cb.on_update
      def _on_color(_) -> None:
        self._viewer_color_bodies = bool(color_cb.value)
        if on_change is not None:
          on_change()

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
        self.set_preview_frame(idx, int(scrubber.value))
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
    self.update_viewer_gui(get_env_idx())

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
