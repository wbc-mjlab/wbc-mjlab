"""Play WBC motion NPZ clips in Viser (browser viewer).

Uses mjlab's ``MjlabViserScene`` (same stack as ``viz-nan`` / ``ViserPlayViewer``)
and the mjviser ``motion_playback`` GUI pattern for timeline controls.

Usage::

  wbc-mjlab-vis-data --robot g1 --dataset lafan
  wbc-mjlab-vis-data --motion-file data/g1/lafan/npz/walk1_subject1.npz
  wbc-mjlab-vis-data --dataset-path data/g1/lafan
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import viser
from mjlab.viewer.viser import MjlabViserScene

from wbc_mjlab.motion.robot_assets import qpos_dof_joint_names, remap_dof_columns
from wbc_mjlab.motion.tyro_cli import cli as tyro_cli
from wbc_mjlab.robots.ids import RobotId, resolve_robot_id


@dataclass
class MotionTrajectory:
  qpos: np.ndarray
  qvel: np.ndarray
  fps: float

  @property
  def num_frames(self) -> int:
    return int(self.qpos.shape[0])

  @property
  def dt(self) -> float:
    return 1.0 / float(self.fps)


def _robot_xml_path(robot_id: RobotId) -> Path:
  if robot_id == "g1":
    from wbc_mjlab.robots.g1.constants import G1_XML

    return G1_XML
  raise ValueError(f"No default MuJoCo XML for robot {robot_id!r}")


def _resolve_motion_dir_and_npz(
  *,
  robot_id: RobotId,
  dataset: str | None,
  dataset_path: str | None,
  motion_file: str | None,
  cache_motion_bundle: bool,
) -> tuple[Path | None, Path]:
  from wbc_mjlab.data_paths import (
    resolve_dataset_root,
    resolve_motion_path,
    resolve_training_motion_file,
  )
  from wbc_mjlab.motion.stack_bundle import list_clip_npz_files

  if motion_file is not None:
    npz_path = resolve_motion_path(
      motion_file,
      robot_id=robot_id,
      cache_motion_bundle=cache_motion_bundle,
    )
    if npz_path.parent.name == "npz":
      clips = list_clip_npz_files(npz_path.parent.parent)
      if clips:
        return npz_path.parent, npz_path if npz_path in clips else clips[0]
    return None, npz_path

  if dataset is not None:
    root = resolve_dataset_root(robot_id, dataset)
    clips = list_clip_npz_files(root)
    if clips:
      return root / "npz", clips[0]
    npz_path = resolve_training_motion_file(
      robot_id,
      dataset=dataset,
      cache_motion_bundle=cache_motion_bundle,
    )
    return None, npz_path

  if dataset_path is not None:
    from wbc_mjlab.data_paths import resolve_dataset_root as _resolve_root

    root = _resolve_root(robot_id, dataset_path) if not Path(dataset_path).is_dir() else Path(dataset_path).resolve()
    if not root.is_dir():
      root = Path(dataset_path).expanduser().resolve()
    clips = list_clip_npz_files(root)
    if clips and not cache_motion_bundle:
      return root / "npz", clips[0]
    npz_path = resolve_training_motion_file(
      robot_id,
      dataset_path=dataset_path,
      cache_motion_bundle=cache_motion_bundle,
    )
    root = npz_path.parent
    if root.name == "npz":
      clips = list_clip_npz_files(root.parent)
      if clips:
        return root, npz_path if npz_path in clips else clips[0]
    if root.is_dir():
      clips = list_clip_npz_files(root)
      if clips:
        return root / "npz", npz_path if npz_path in clips else clips[0]
    return None, npz_path

  raise ValueError(
    "Provide --motion-file, --dataset <name>, or --dataset-path <dir>"
  )


def _qpos_v_addresses(
  model: mujoco.MjModel,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  free_q = free_v = None
  joint_q: list[int] = []
  joint_v: list[int] = []
  for j in range(model.njnt):
    jtype = int(model.jnt_type[j])
    adr_q = int(model.jnt_qposadr[j])
    adr_v = int(model.jnt_dofadr[j])
    if jtype == mujoco.mjtJoint.mjJNT_FREE:
      free_q = np.arange(adr_q, adr_q + 7, dtype=np.int64)
      free_v = np.arange(adr_v, adr_v + 6, dtype=np.int64)
      continue
    if j + 1 < model.njnt:
      next_q = int(model.jnt_qposadr[j + 1])
      next_v = int(model.jnt_dofadr[j + 1])
    else:
      next_q = model.nq
      next_v = model.nv
    joint_q.extend(range(adr_q, next_q))
    joint_v.extend(range(adr_v, next_v))
  if free_q is None or free_v is None:
    raise RuntimeError("Model has no free joint (expected floating base).")
  return free_q, np.asarray(joint_q, dtype=np.int64), free_v, np.asarray(joint_v, dtype=np.int64)


def _load_wbc_npz_trajectory(
  data: np.lib.npyio.NpzFile,
  model: mujoco.MjModel,
) -> tuple[np.ndarray, np.ndarray]:
  joint_names = [str(x) for x in data["joint_names"].tolist()]
  joint_pos = np.asarray(data["joint_pos"], dtype=np.float64)
  joint_vel = np.asarray(data["joint_vel"], dtype=np.float64)
  body_pos_w = np.asarray(data["body_pos_w"], dtype=np.float64)
  body_quat_w = np.asarray(data["body_quat_w"], dtype=np.float64)
  body_lin_vel_w = np.asarray(data["body_lin_vel_w"], dtype=np.float64)
  body_ang_vel_w = np.asarray(data["body_ang_vel_w"], dtype=np.float64)

  pelvis_body_index = 0
  model_joint_names = qpos_dof_joint_names(model, entity_name="robot")
  if list(joint_names) != list(model_joint_names):
    joint_pos = remap_dof_columns(joint_pos, joint_names, model_joint_names)
    joint_vel = remap_dof_columns(joint_vel, joint_names, model_joint_names)

  num_frames = int(joint_pos.shape[0])
  qpos = np.zeros((num_frames, model.nq), dtype=np.float64)
  qvel = np.zeros((num_frames, model.nv), dtype=np.float64)
  free_q, joint_q, free_v, joint_v = _qpos_v_addresses(model)

  qpos[:, free_q[0:3]] = body_pos_w[:, pelvis_body_index, :]
  qpos[:, free_q[3:7]] = body_quat_w[:, pelvis_body_index, :]
  qpos[:, joint_q] = joint_pos
  qvel[:, free_v[0:3]] = body_lin_vel_w[:, pelvis_body_index, :]
  qvel[:, free_v[3:6]] = body_ang_vel_w[:, pelvis_body_index, :]
  qvel[:, joint_v] = joint_vel
  return qpos, qvel


def load_motion_trajectory(
  npz_file: Path,
  model: mujoco.MjModel,
  *,
  fps_override: float | None,
) -> MotionTrajectory:
  data = np.load(npz_file, allow_pickle=True)
  if "qpos" in data:
    qpos = np.asarray(data["qpos"], dtype=np.float64)
    if qpos.ndim != 2 or qpos.shape[1] != model.nq:
      raise ValueError(
        f"NPZ qpos shape {qpos.shape} incompatible with model nq={model.nq}"
      )
    num_frames = int(qpos.shape[0])
    if "qvel" in data:
      qvel = np.asarray(data["qvel"], dtype=np.float64)
      if qvel.shape != (num_frames, model.nv):
        qvel = np.zeros((num_frames, model.nv), dtype=np.float64)
    else:
      qvel = np.zeros((num_frames, model.nv), dtype=np.float64)
  else:
    qpos, qvel = _load_wbc_npz_trajectory(data, model)

  fps = fps_override
  if fps is None:
    fps = float(data["fps"][0]) if "fps" in data else 50.0
  return MotionTrajectory(qpos=qpos, qvel=qvel, fps=float(fps))


class MotionNpzViewer:
  """NPZ motion scrubber/playback, modeled on ``mjlab.scripts.nan_viz.NanDumpViewer``."""

  def __init__(
    self,
    *,
    model: mujoco.MjModel,
    npz_path: Path,
    motion_dir: Path | None,
    fps_override: float | None,
  ) -> None:
    self.model = model
    self.npz_path = npz_path
    self.motion_dir = motion_dir
    self.fps_override = fps_override
    self.data = mujoco.MjData(model)
    self.traj = load_motion_trajectory(npz_path, model, fps_override=fps_override)

    self.server = viser.ViserServer(label="wbc-mjlab — motion NPZ")
    self.scene = MjlabViserScene(self.server, model, num_envs=1)
    self.scene.debug_visualization_enabled = True

    self.frame_idx = 0
    self.playing = True
    self.speed = 1.0
    self.looping = True
    self._accumulator = 0.0
    self._needs_render = True

  def setup(self) -> None:
    self.server.scene.add_grid(
      "/fixed_bodies/reference_grid",
      infinite_grid=True,
      fade_distance=50.0,
      shadow_opacity=0.2,
      plane_opacity=0.35,
    )
    tabs = self.scene.create_visualization_gui()

    with tabs.add_tab("Playback", icon=viser.Icon.PLAYER_PLAY):
      self._motion_dropdown = None
      if self.motion_dir is not None:
        motions = sorted(self.motion_dir.glob("*.npz"))
        labels = [p.name for p in motions]
        if len(labels) > 1:
          initial = self.npz_path.name if self.npz_path.name in labels else labels[0]
          self._motion_dropdown = self.server.gui.add_dropdown(
            "Motion",
            options=labels,
            initial_value=initial,
            hint=str(self.motion_dir),
          )

          @self._motion_dropdown.on_update
          def _(event) -> None:
            selected_path = self.motion_dir / event.target.value
            self._load_clip(selected_path)

      self._timeline = self.server.gui.add_slider(
        "Frame",
        min=0,
        max=max(0, self.traj.num_frames - 1),
        step=1,
        initial_value=0,
      )
      self._time_label = self.server.gui.add_html("")
      self._play_btn = self.server.gui.add_button(
        "Pause", icon=viser.Icon.PLAYER_PAUSE
      )

      @self._play_btn.on_click
      def _(_) -> None:
        self.playing = not self.playing
        self._play_btn.label = "Pause" if self.playing else "Play"
        self._play_btn.icon = (
          viser.Icon.PLAYER_PAUSE if self.playing else viser.Icon.PLAYER_PLAY
        )

      speed_btns = self.server.gui.add_button_group(
        "Speed", options=["0.25x", "0.5x", "1x", "2x", "4x"]
      )

      @speed_btns.on_click
      def _(event) -> None:
        self.speed = float(event.target.value.replace("x", ""))

      loop_cb = self.server.gui.add_checkbox("Loop", initial_value=True)

      @loop_cb.on_update
      def _(_) -> None:
        self.looping = bool(loop_cb.value)

      @self._timeline.on_update
      def _(_) -> None:
        self.frame_idx = int(self._timeline.value)
        self._needs_render = True

    self.render_frame(self.frame_idx)

  def _load_clip(self, npz_path: Path) -> None:
    try:
      self.traj = load_motion_trajectory(
        npz_path, self.model, fps_override=self.fps_override
      )
      self.npz_path = npz_path
    except Exception as exc:
      print(f"Failed to load {npz_path}: {exc}", file=sys.stderr)
      return
    self.frame_idx = 0
    self.playing = False
    self._play_btn.label = "Play"
    self._play_btn.icon = viser.Icon.PLAYER_PLAY
    self._timeline.max = max(0, self.traj.num_frames - 1)
    self._timeline.value = 0
    self._needs_render = True

  def _queue_root_velocity_arrows(self, mj_data: mujoco.MjData) -> None:
    self.scene.clear_debug_all()
    if not self.scene.debug_visualization_enabled:
      return

    origin = np.asarray(mj_data.qpos[0:3], dtype=np.float64)
    root_quat = np.asarray(mj_data.qpos[3:7], dtype=np.float64)
    v_world = np.asarray(mj_data.qvel[0:3], dtype=np.float64)
    w_local = np.asarray(mj_data.qvel[3:6], dtype=np.float64)

    conj = np.empty(4, dtype=np.float64)
    mujoco.mju_negQuat(conj, root_quat)
    v_local = np.empty(3, dtype=np.float64)
    mujoco.mju_rotVecQuat(v_local, v_world, conj)

    meansize = self.scene.meansize
    lin_gain = 0.8 * meansize
    ang_gain = 0.4 * meansize
    basis = np.eye(3, dtype=np.float64)
    dir_w = np.zeros((3, 3), dtype=np.float64)
    for i in range(3):
      mujoco.mju_rotVecQuat(dir_w[i], basis[i], root_quat)

    specs = (
      (float(v_local[0]), lin_gain, dir_w[0], (1.0, 0.2, 0.2, 0.9)),
      (float(v_local[1]), lin_gain, dir_w[1], (0.2, 1.0, 0.2, 0.9)),
      (float(w_local[2]), ang_gain, dir_w[2], (0.3, 0.5, 1.0, 0.9)),
    )
    for mag, gain, direction, color in specs:
      length = abs(mag) * gain
      if length < 1e-6:
        continue
      sign = 1.0 if mag >= 0.0 else -1.0
      end = origin + direction * sign * length
      self.scene.add_arrow(
        origin,
        end,
        color=color,
        width=0.03 * meansize,
      )

  def render_frame(self, idx: int) -> None:
    nf = self.traj.num_frames
    frame = max(0, min(int(idx), nf - 1))
    self.data.qpos[:] = self.traj.qpos[frame]
    self.data.qvel[:] = self.traj.qvel[frame]
    mujoco.mj_forward(self.model, self.data)
    self._queue_root_velocity_arrows(self.data)
    self.scene.update_from_mjdata(self.data)

    t = frame * self.traj.dt
    span = nf * self.traj.dt
    self._time_label.content = (
      f'<span style="font-size:0.85em">'
      f"{t:.2f}s / {span:.2f}s (frame {frame}/{nf - 1})"
      f"</span>"
    )

  def run(self) -> None:
    print("\nViser server running. Open the URL printed above. Ctrl+C to exit.\n")
    last_time = time.perf_counter()
    try:
      while True:
        now = time.perf_counter()
        wall_dt = now - last_time
        last_time = now

        if self.scene.needs_update:
          self.scene.refresh_visualization()

        nf = self.traj.num_frames
        dt = self.traj.dt
        if self.playing:
          self._accumulator += wall_dt * self.speed
          frames_to_advance = int(self._accumulator / dt)
          if frames_to_advance > 0:
            self._accumulator -= frames_to_advance * dt
            new_idx = self.frame_idx + frames_to_advance
            if new_idx >= nf:
              if self.looping:
                new_idx = new_idx % nf
              else:
                new_idx = nf - 1
                self.playing = False
                self._play_btn.label = "Play"
                self._play_btn.icon = viser.Icon.PLAYER_PLAY
            self.frame_idx = new_idx
            self._timeline.value = new_idx
            self.render_frame(new_idx)
        elif self._needs_render:
          self.render_frame(self.frame_idx)
          self._needs_render = False

        time.sleep(1.0 / 60.0)
    except KeyboardInterrupt:
      print("\nStopped.")
      self.server.stop()


def run_viewer(
  *,
  robot: str = "g1",
  dataset: str | None = None,
  dataset_path: str | None = None,
  motion_file: str | None = None,
  fps: float | None = None,
  cache_motion_bundle: bool = False,
) -> None:
  if dataset_path is not None and dataset is not None:
    raise ValueError("Use only one of --dataset or --dataset-path")

  robot_id = resolve_robot_id(robot)
  motion_dir, npz_path = _resolve_motion_dir_and_npz(
    robot_id=robot_id,
    dataset=dataset,
    dataset_path=dataset_path,
    motion_file=motion_file,
    cache_motion_bundle=cache_motion_bundle,
  )

  model_path = _robot_xml_path(robot_id)
  print(f"[INFO] robot={robot_id} model={model_path}")
  print(f"[INFO] npz={npz_path}")
  if motion_dir is not None:
    print(f"[INFO] motion_dir={motion_dir}")

  model = mujoco.MjModel.from_xml_path(str(model_path))
  viewer = MotionNpzViewer(
    model=model,
    npz_path=npz_path,
    motion_dir=motion_dir,
    fps_override=fps,
  )
  viewer.setup()
  viewer.run()


def main(
  robot: str = "g1",
  dataset: str | None = None,
  dataset_path: str | None = None,
  motion_file: str | None = None,
  fps: float | None = None,
  cache_motion_bundle: bool = False,
) -> None:
  run_viewer(
    robot=robot,
    dataset=dataset,
    dataset_path=dataset_path,
    motion_file=motion_file,
    fps=fps,
    cache_motion_bundle=cache_motion_bundle,
  )


def cli() -> None:
  tyro_cli(main, bool_shorthand=("cache_motion_bundle",))


if __name__ == "__main__":
  cli()
