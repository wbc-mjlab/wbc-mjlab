"""Convert retargeted CSV motions to a training motion bundle (offline).

CSV layout: ``root_pos(3) + root_rot_xyzw(4) + joint_pos(dof)`` per row.

With ``--dataset lafan`` (recommended), reads/writes under ``data/g1/lafan/``::

  data/g1/lafan/*.csv or data/g1/lafan/raw/*.csv
  data/g1/lafan/lafan.npz
  data/g1/lafan/npz/<clip>.npz
"""

from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from wbc_mjlab.motion.tyro_cli import cli as tyro_cli
from wbc_mjlab.motion.motion_z_debias import debias_motion_log_vertical
from wbc_mjlab.motion.motion_export_bundle import MotionClipExport, export_motion_training_bundle
from wbc_mjlab.motion.robot_assets import (
  conversion_scene_cfg,
  get_robot_motion_spec,
  peek_csv_dof_width,
  resolve_dof_joint_names,
  resolve_robot_id,
)
from mjlab.entity import Entity
from mjlab.scene import Scene
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.utils.lab_api.math import (
  axis_angle_from_quat,
  quat_conjugate,
  quat_mul,
  quat_slerp,
)
from mjlab.viewer.offscreen_renderer import OffscreenRenderer
from mjlab.viewer.viewer_config import ViewerConfig


class MotionLoader:
  def __init__(
    self,
    motion_file: str,
    input_fps: int,
    output_fps: int,
    device: torch.device | str,
    line_range: tuple[int, int] | None = None,
  ):
    self.motion_file = motion_file
    self.input_fps = input_fps
    self.output_fps = output_fps
    self.input_dt = 1.0 / self.input_fps
    self.output_dt = 1.0 / self.output_fps
    self.current_idx = 0
    self.device = device
    self.line_range = line_range
    self._load_motion()
    self._interpolate_motion()
    self._compute_velocities()

  def _load_motion(self):
    if self.line_range is None:
      motion = torch.from_numpy(np.loadtxt(self.motion_file, delimiter=","))
    else:
      motion = torch.from_numpy(
        np.loadtxt(
          self.motion_file,
          delimiter=",",
          skiprows=self.line_range[0] - 1,
          max_rows=self.line_range[1] - self.line_range[0] + 1,
        )
      )
    motion = motion.to(torch.float32).to(self.device)
    self.motion_base_poss_input = motion[:, :3]
    self.motion_base_rots_input = motion[:, 3:7]
    self.motion_base_rots_input = self.motion_base_rots_input[:, [3, 0, 1, 2]]
    self.motion_dof_poss_input = motion[:, 7:]

    self.input_frames = motion.shape[0]
    self.duration = (self.input_frames - 1) * self.input_dt

  def _interpolate_motion(self):
    times = torch.arange(
      0, self.duration, self.output_dt, device=self.device, dtype=torch.float32
    )
    self.output_frames = times.shape[0]
    index_0, index_1, blend = self._compute_frame_blend(times)
    self.motion_base_poss = self._lerp(
      self.motion_base_poss_input[index_0],
      self.motion_base_poss_input[index_1],
      blend.unsqueeze(1),
    )
    self.motion_base_rots = self._slerp(
      self.motion_base_rots_input[index_0],
      self.motion_base_rots_input[index_1],
      blend,
    )
    self.motion_dof_poss = self._lerp(
      self.motion_dof_poss_input[index_0],
      self.motion_dof_poss_input[index_1],
      blend.unsqueeze(1),
    )
    print(
      f"Motion interpolated, input frames: {self.input_frames}, "
      f"input fps: {self.input_fps}, "
      f"output frames: {self.output_frames}, "
      f"output fps: {self.output_fps}"
    )

  def _lerp(
    self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor
  ) -> torch.Tensor:
    return a * (1 - blend) + b * blend

  def _slerp(
    self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor
  ) -> torch.Tensor:
    slerped_quats = torch.zeros_like(a)
    for i in range(a.shape[0]):
      slerped_quats[i] = quat_slerp(a[i], b[i], float(blend[i]))
    return slerped_quats

  def _compute_frame_blend(
    self, times: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    phase = times / self.duration
    index_0 = (phase * (self.input_frames - 1)).floor().long()
    index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames - 1))
    blend = phase * (self.input_frames - 1) - index_0
    return index_0, index_1, blend

  def _compute_velocities(self):
    self.motion_base_lin_vels = torch.gradient(
      self.motion_base_poss, spacing=self.output_dt, dim=0
    )[0]
    self.motion_dof_vels = torch.gradient(
      self.motion_dof_poss, spacing=self.output_dt, dim=0
    )[0]
    self.motion_base_ang_vels = self._so3_derivative(
      self.motion_base_rots, self.output_dt
    )

  def _so3_derivative(self, rotations: torch.Tensor, dt: float) -> torch.Tensor:
    q_prev, q_next = rotations[:-2], rotations[2:]
    q_rel = quat_mul(q_next, quat_conjugate(q_prev))
    omega = axis_angle_from_quat(q_rel) / (2.0 * dt)
    omega = torch.cat([omega[:1], omega, omega[-1:]], dim=0)
    return omega

  def get_next_state(
    self,
  ) -> tuple[
    tuple[
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
    ],
    bool,
  ]:
    state = (
      self.motion_base_poss[self.current_idx : self.current_idx + 1],
      self.motion_base_rots[self.current_idx : self.current_idx + 1],
      self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
      self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
      self.motion_dof_poss[self.current_idx : self.current_idx + 1],
      self.motion_dof_vels[self.current_idx : self.current_idx + 1],
    )
    self.current_idx += 1
    reset_flag = False
    if self.current_idx >= self.output_frames:
      self.current_idx = 0
      reset_flag = True
    return state, reset_flag


def run_sim(
  sim: Simulation,
  scene: Scene,
  joint_names,
  input_file,
  input_fps,
  output_fps,
  render,
  line_range,
  renderer: OffscreenRenderer | None = None,
) -> dict[str, Any]:
  motion = MotionLoader(
    motion_file=input_file,
    input_fps=input_fps,
    output_fps=output_fps,
    device=sim.device,
    line_range=line_range,
  )

  robot: Entity = scene["robot"]
  robot_joint_indexes = robot.find_joints(joint_names, preserve_order=True)[0]

  log: dict[str, Any] = {
    "fps": [output_fps],
    "joint_pos": [],
    "joint_vel": [],
    "body_pos_w": [],
    "body_quat_w": [],
    "body_lin_vel_w": [],
    "body_ang_vel_w": [],
  }
  file_saved = False

  frames = []
  scene.reset()

  print(f"\nStarting simulation with {motion.output_frames} frames...")
  if render:
    print("Rendering enabled - generating video frames...")

  pbar = tqdm(
    total=motion.output_frames,
    desc="Processing frames",
    unit="frame",
    ncols=100,
    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
  )

  frame_count = 0
  while not file_saved:
    (
      (
        motion_base_pos,
        motion_base_rot,
        motion_base_lin_vel,
        motion_base_ang_vel,
        motion_dof_pos,
        motion_dof_vel,
      ),
      reset_flag,
    ) = motion.get_next_state()

    root_states = robot.data.default_root_state.clone()
    root_states[:, 0:3] = motion_base_pos
    root_states[:, :2] += scene.env_origins[:, :2]
    root_states[:, 3:7] = motion_base_rot
    root_states[:, 7:10] = motion_base_lin_vel
    root_states[:, 10:] = motion_base_ang_vel
    robot.write_root_state_to_sim(root_states)

    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    joint_pos[:, robot_joint_indexes] = motion_dof_pos
    joint_vel[:, robot_joint_indexes] = motion_dof_vel
    robot.write_joint_state_to_sim(joint_pos, joint_vel)

    sim.forward()
    scene.update(sim.mj_model.opt.timestep)
    if render and renderer is not None:
      renderer.update(sim.data)
      frames.append(renderer.render())

    if not file_saved:
      log["joint_pos"].append(robot.data.joint_pos[0, :].cpu().numpy().copy())
      log["joint_vel"].append(robot.data.joint_vel[0, :].cpu().numpy().copy())
      log["body_pos_w"].append(robot.data.body_link_pos_w[0, :].cpu().numpy().copy())
      log["body_quat_w"].append(robot.data.body_link_quat_w[0, :].cpu().numpy().copy())
      log["body_lin_vel_w"].append(
        robot.data.body_link_lin_vel_w[0, :].cpu().numpy().copy()
      )
      log["body_ang_vel_w"].append(
        robot.data.body_link_ang_vel_w[0, :].cpu().numpy().copy()
      )

      torch.testing.assert_close(
        robot.data.body_link_lin_vel_w[0, 0], motion_base_lin_vel[0]
      )
      torch.testing.assert_close(
        robot.data.body_link_ang_vel_w[0, 0], motion_base_ang_vel[0]
      )

      frame_count += 1
      pbar.update(1)
      if frame_count % 100 == 0:
        elapsed_time = frame_count / output_fps
        pbar.set_description(f"Processing frames (t={elapsed_time:.1f}s)")

      if reset_flag and not file_saved:
        file_saved = True
        pbar.close()
        print("\nStacking arrays...")
        for k in (
          "joint_pos",
          "joint_vel",
          "body_pos_w",
          "body_quat_w",
          "body_lin_vel_w",
          "body_ang_vel_w",
        ):
          log[k] = np.stack(log[k], axis=0)
  return log


def _resolve_input_csv_paths(input_path: str) -> list[Path]:
  path = Path(input_path).expanduser().resolve()
  if path.is_file():
    return [path]
  if path.is_dir():
    files = sorted(path.glob("*.csv"))
    if not files:
      raise ValueError(f"No .csv files found in directory: {path}")
    return files
  raise ValueError(f"Input path does not exist: {path}")


def main(
  input_path: str | None = None,
  output_dir: str | None = None,
  dataset: str | None = None,
  dataset_path: str | None = None,
  robot: str = "g1",
  input_fps: float = 30.0,
  output_fps: float = 50.0,
  device: str = "cuda:0",
  render: bool = False,
  line_range: tuple[int, int] | None = None,
  debias_z: bool = False,
  debias_foot_sole_z: float | None = None,
):
  if device.startswith("cuda") and not torch.cuda.is_available():
    print("[WARNING] CUDA is not available. Falling back to CPU.")
    device = "cpu"

  robot_id = resolve_robot_id(robot)
  _, motion_spec = get_robot_motion_spec(robot)
  if debias_z and (
    motion_spec.foot_body_names is None or motion_spec.foot_sole_z is None
  ):
    raise ValueError(f"Vertical debias is not configured for robot {robot_id!r}.")

  from wbc_mjlab.data_paths import resolve_conversion_paths

  input_path, output_dir = resolve_conversion_paths(
    robot_id=robot_id,
    dataset=dataset,
    dataset_path=dataset_path,
    input_path=input_path,
    output_dir=output_dir,
  )
  print(f"[INFO] dataset={dataset or '(custom)'} input={input_path} output={output_dir}")

  input_paths = _resolve_input_csv_paths(input_path)
  clips: list[MotionClipExport] = []

  sim_cfg = SimulationCfg()
  sim_cfg.mujoco.timestep = 1.0 / output_fps

  scene = Scene(conversion_scene_cfg(motion_spec), device=device)
  model = scene.compile()
  csv_dof_dim = peek_csv_dof_width(input_paths[0], line_range)
  joint_names, _ = resolve_dof_joint_names(model, csv_dof_dim)
  print(
    f"[INFO] robot={robot_id}, csv_dof={csv_dof_dim}, "
    f"joints={len(joint_names)} (model.nq={model.nq})"
  )

  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)
  robot_body_names = list(scene["robot"].body_names)

  renderer = None
  if render:
    viewer_cfg = ViewerConfig(
      height=480,
      width=640,
      origin_type=ViewerConfig.OriginType.ASSET_ROOT,
      entity_name="robot",
      distance=2.0,
      elevation=-5.0,
      azimuth=20,
    )
    renderer = OffscreenRenderer(model=sim.mj_model, cfg=viewer_cfg, scene=scene)
    renderer.initialize()

  for csv_path in input_paths:
    dof_dim = peek_csv_dof_width(csv_path, line_range)
    if dof_dim != len(joint_names):
      raise ValueError(
        f"{csv_path}: expected {len(joint_names)} joint columns for robot "
        f"{robot_id!r}, got {dof_dim}"
      )
    print(f"[INFO] Converting {csv_path} (robot={robot_id})")
    log = run_sim(
      sim=sim,
      scene=scene,
      joint_names=joint_names,
      input_fps=input_fps,
      input_file=str(csv_path),
      output_fps=output_fps,
      render=render,
      line_range=line_range,
      renderer=renderer,
    )

    if debias_z:
      assert motion_spec.foot_body_names is not None
      assert motion_spec.foot_sole_z is not None
      foot_sole_z = (
        debias_foot_sole_z
        if debias_foot_sole_z is not None
        else motion_spec.foot_sole_z
      )
      z_shift = debias_motion_log_vertical(
        log,
        robot_body_names=robot_body_names,
        foot_body_names=motion_spec.foot_body_names,
        foot_sole_z=foot_sole_z,
      )
      print(
        f"[INFO] Vertical debias {csv_path.name}: shift_z={z_shift:.6f} m, "
        f"foot_sole_z={foot_sole_z:.6f} m"
      )

    clips.append(
      MotionClipExport(log=log, source_path=csv_path, joint_names=joint_names)
    )

  export_motion_training_bundle(
    output_dir=output_dir,
    clips=clips,
    robot_id=robot_id,
    robot_body_names=list(scene["robot"].body_names),
    output_fps=output_fps,
  )


if __name__ == "__main__":
  tyro_cli(main, bool_shorthand=("debias_z",))
