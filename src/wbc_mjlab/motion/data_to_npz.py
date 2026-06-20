"""Convert motion source files (CSV, PKL, …) to training NPZ clips.

Source layout is defined in ``motion_formats.py`` (canonical fields:
``base_pos``, ``base_rot_xyzw``, ``dof_pos``, ``fps``). Format is inferred from
the file extension unless ``--format`` is set.

With ``--dataset lafan``, reads/writes under ``data/g1/lafan/``::

  data/g1/lafan/raw/*.csv or *.pkl
  data/g1/lafan/npz/<clip>.npz
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from wbc_mjlab.motion.motion_export_bundle import (
  MotionClipExport,
  export_motion_clip_npz,
  npz_output_dir,
)
from wbc_mjlab.motion.motion_formats import (
  MotionLoader,
  load_motion_clip,
  peek_dof_width,
  resolve_input_motion_paths,
)
from wbc_mjlab.motion.motion_z_debias import debias_motion_log_vertical
from wbc_mjlab.motion.robot_assets import (
  conversion_scene_cfg,
  get_robot_motion_spec,
  resolve_dof_joint_names,
  resolve_robot_id,
)
from wbc_mjlab.motion.tyro_cli import cli as tyro_cli
from mjlab.entity import Entity
from mjlab.scene import Scene
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.viewer.offscreen_renderer import OffscreenRenderer
from mjlab.viewer.viewer_config import ViewerConfig


def run_sim(
  sim: Simulation,
  scene: Scene,
  joint_names: list[str],
  raw_clip,
  output_fps: float,
  render: bool,
  renderer: OffscreenRenderer | None = None,
) -> dict[str, Any]:
  motion = MotionLoader(
    raw=raw_clip,
    output_fps=int(output_fps),
    device=sim.device,
    model_joint_names=joint_names,
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
      renderer.render()

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


def main(
  input_path: str | None = None,
  output_dir: str | None = None,
  dataset: str | None = None,
  dataset_path: str | None = None,
  robot: str = "g1",
  format: str | None = None,
  input_fps: float | None = None,
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

  input_paths = resolve_input_motion_paths(input_path, format_name=format)
  clips: list[MotionClipExport] = []
  npz_output_dir(output_dir).mkdir(parents=True, exist_ok=True)

  sim_cfg = SimulationCfg()
  sim_cfg.mujoco.timestep = 1.0 / output_fps

  scene = Scene(conversion_scene_cfg(motion_spec), device=device)
  model = scene.compile()
  probe = load_motion_clip(
    input_paths[0],
    format_name=format,
    input_fps=input_fps,
    line_range=line_range,
  )
  dof_dim = probe.dof_pos.shape[1]
  joint_names, _ = resolve_dof_joint_names(
    model,
    dof_dim,
    source_joint_names=probe.source_joint_names,
  )
  print(
    f"[INFO] robot={robot_id}, dof={dof_dim}, "
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

  for source_path in input_paths:
    path_dof = peek_dof_width(
      source_path,
      format_name=format,
      line_range=line_range,
    )
    if path_dof != len(joint_names):
      raise ValueError(
        f"{source_path}: expected {len(joint_names)} joint DOFs for robot "
        f"{robot_id!r}, got {path_dof}"
      )

    raw_clip = load_motion_clip(
      source_path,
      format_name=format,
      input_fps=input_fps,
      line_range=line_range,
    )
    print(
      f"[INFO] Converting {source_path} "
      f"(robot={robot_id}, format={format or 'auto'}, input_fps={raw_clip.fps})"
    )

    log = run_sim(
      sim=sim,
      scene=scene,
      joint_names=joint_names,
      raw_clip=raw_clip,
      output_fps=output_fps,
      render=render,
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
        f"[INFO] Vertical debias {source_path.name}: shift_z={z_shift:.6f} m, "
        f"foot_sole_z={foot_sole_z:.6f} m"
      )

    clip = MotionClipExport(
      log=log,
      source_path=source_path,
      joint_names=joint_names,
    )
    export_motion_clip_npz(
      output_dir=output_dir,
      clip=clip,
      robot_id=robot_id,
      robot_body_names=robot_body_names,
    )
    clips.append(clip)

  print(f"[INFO] Finished {len(clips)} clip(s) in {npz_output_dir(output_dir)}")


def cli() -> None:
  tyro_cli(main, bool_shorthand=("debias_z",))


if __name__ == "__main__":
  cli()
