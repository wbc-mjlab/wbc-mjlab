"""Convert retargeted PKL motions (GMR / bvh_to_robot) to a training motion bundle.

With ``--dataset my_clips``, reads/writes ``data/g1/my_clips/`` (see ``data/README.md``).

PKL layout (from ``retargeting_utils/bvh_to_robot.py``)::

  fps: int
  root_pos: (T, 3)
  root_rot: (T, 4)  # xyzw
  dof_pos: (T, nq - 7)
  local_body_pos: optional
  link_body_list: optional

Optional metadata keys (if present) override joint ordering:

  joint_names, dof_joint_names, joint_order

Robot joint order is otherwise taken from the compiled mjlab asset (``qpos[7:]``).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from wbc_mjlab.motion.tyro_cli import cli as tyro_cli
from wbc_mjlab.motion.csv_to_npz import MotionLoader
from wbc_mjlab.motion.motion_z_debias import debias_motion_log_vertical
from wbc_mjlab.motion.motion_export_bundle import MotionClipExport, export_motion_training_bundle
from wbc_mjlab.motion.robot_assets import (
  conversion_scene_cfg,
  get_robot_motion_spec,
  normalize_joint_name_list,
  remap_dof_columns,
  resolve_dof_joint_names,
  resolve_robot_id,
)
from mjlab.scene import Scene
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.viewer.offscreen_renderer import OffscreenRenderer
from mjlab.viewer.viewer_config import ViewerConfig

_GMR_PKL_KEYS = ("fps", "root_pos", "root_rot", "dof_pos")


def load_gmr_pkl(path: Path) -> dict[str, Any]:
  with path.open("rb") as handle:
    data = pickle.load(handle)
  if not isinstance(data, dict):
    raise TypeError(f"Expected dict in {path}, got {type(data)}")
  missing = [k for k in _GMR_PKL_KEYS if k not in data]
  if missing:
    raise KeyError(f"Missing keys {missing} in {path}")
  return data


def _pick_joint_names_from_pkl(data: dict[str, Any]) -> list[str] | None:
  for key in ("dof_joint_names", "joint_names", "joint_order"):
    raw = data.get(key)
    if raw is None:
      continue
    names = normalize_joint_name_list(list(raw))
    if names:
      return names
  link_names = data.get("link_body_list")
  if link_names:
    # Body list is not joint order; ignore unless explicitly useful later.
    return None
  return None


def resolve_joint_names_for_motion(
  *,
  data: dict[str, Any],
  model,
  entity_name: str = "robot",
) -> tuple[list[str], list[str] | None]:
  dof_dim = int(np.asarray(data["dof_pos"]).shape[1])
  return resolve_dof_joint_names(
    model,
    dof_dim,
    source_joint_names=_pick_joint_names_from_pkl(data),
    entity_name=entity_name,
  )


class PklMotionLoader(MotionLoader):
  """Resampled motion from a GMR-style pickle."""

  def __init__(
    self,
    motion_file: str,
    input_fps: int,
    output_fps: int,
    device: torch.device | str,
    *,
    pkl_joint_names: list[str] | None = None,
    model_joint_names: list[str] | None = None,
  ):
    self._pkl_joint_names = pkl_joint_names
    self._model_joint_names = model_joint_names
    super().__init__(
      motion_file=motion_file,
      input_fps=input_fps,
      output_fps=output_fps,
      device=device,
      line_range=None,
    )

  def _load_motion(self) -> None:
    data = load_gmr_pkl(Path(self.motion_file))
    root_pos = np.asarray(data["root_pos"], dtype=np.float32)
    root_rot_xyzw = np.asarray(data["root_rot"], dtype=np.float32)
    dof_pos = np.asarray(data["dof_pos"], dtype=np.float32)

    if self._pkl_joint_names is not None and self._model_joint_names is not None:
      dof_pos = remap_dof_columns(
        dof_pos, self._pkl_joint_names, self._model_joint_names
      )

    motion = np.concatenate(
      [root_pos, root_rot_xyzw, dof_pos],
      axis=1,
      dtype=np.float32,
    )
    motion = torch.from_numpy(motion).to(device=self.device)

    self.motion_base_poss_input = motion[:, :3]
    self.motion_base_rots_input = motion[:, 3:7]
    self.motion_base_rots_input = self.motion_base_rots_input[:, [3, 0, 1, 2]]
    self.motion_dof_poss_input = motion[:, 7:]

    self.input_frames = motion.shape[0]
    self.duration = (self.input_frames - 1) * self.input_dt


def _resolve_input_pkl_paths(input_path: str) -> list[Path]:
  path = Path(input_path).expanduser().resolve()
  if path.is_file():
    return [path]
  if path.is_dir():
    files = sorted(path.glob("*.pkl"))
    if not files:
      raise ValueError(f"No .pkl files found in directory: {path}")
    return files
  raise ValueError(f"Input path does not exist: {path}")


def run_sim_pkl(
  sim: Simulation,
  scene: Scene,
  joint_names: list[str],
  input_file: str,
  input_fps: float,
  output_fps: float,
  render: bool,
  renderer: OffscreenRenderer | None,
  *,
  pkl_joint_names: list[str] | None,
  model_joint_names: list[str] | None,
) -> dict[str, Any]:
  motion = PklMotionLoader(
    motion_file=input_file,
    input_fps=int(input_fps),
    output_fps=int(output_fps),
    device=sim.device,
    pkl_joint_names=pkl_joint_names,
    model_joint_names=model_joint_names,
  )

  robot = scene["robot"]
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

  pbar = tqdm(
    total=motion.output_frames,
    desc=f"Processing {Path(input_file).name}",
    unit="frame",
    ncols=100,
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
      _ = renderer.render()

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

      frame_count += 1
      pbar.update(1)

      if reset_flag and not file_saved:
        file_saved = True
        pbar.close()
        for key in (
          "joint_pos",
          "joint_vel",
          "body_pos_w",
          "body_quat_w",
          "body_lin_vel_w",
          "body_ang_vel_w",
        ):
          log[key] = np.stack(log[key], axis=0)

  return log


def main(
  input_path: str | None = None,
  output_dir: str | None = None,
  dataset: str | None = None,
  dataset_path: str | None = None,
  robot: str = "g1",
  input_fps: float | None = None,
  output_fps: float = 50.0,
  device: str = "cuda:0",
  render: bool = False,
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

  input_paths = _resolve_input_pkl_paths(input_path)
  clips: list[MotionClipExport] = []

  sim_cfg = SimulationCfg()
  sim_cfg.mujoco.timestep = 1.0 / output_fps

  scene = Scene(conversion_scene_cfg(motion_spec), device=device)
  model = scene.compile()
  joint_names, _ = resolve_joint_names_for_motion(
    data=load_gmr_pkl(input_paths[0]), model=model
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

  for pkl_path in input_paths:
    pkl_data = load_gmr_pkl(pkl_path)
    model_joint_names, pkl_joint_names = resolve_joint_names_for_motion(
      data=pkl_data, model=model
    )
    effective_input_fps = float(input_fps if input_fps is not None else pkl_data["fps"])
    print(
      f"[INFO] Converting {pkl_path} (robot={robot_id}, input_fps={effective_input_fps})"
    )

    log = run_sim_pkl(
      sim=sim,
      scene=scene,
      joint_names=model_joint_names,
      input_file=str(pkl_path),
      input_fps=effective_input_fps,
      output_fps=output_fps,
      render=render,
      renderer=renderer,
      pkl_joint_names=pkl_joint_names,
      model_joint_names=model_joint_names,
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
        f"[INFO] Vertical debias {pkl_path.name}: shift_z={z_shift:.6f} m, "
        f"foot_sole_z={foot_sole_z:.6f} m"
      )

    clips.append(
      MotionClipExport(
        log=log,
        source_path=pkl_path,
        joint_names=model_joint_names,
      )
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
