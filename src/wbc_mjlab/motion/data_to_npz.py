"""Convert motion source files (CSV, PKL, …) to training NPZ clips.

Source layout is defined in ``motion_formats.py`` (canonical fields:
``base_pos``, ``base_rot_xyzw``, ``dof_pos``, ``fps``). Format is inferred from
the file extension unless ``--format`` is set.

With ``--dataset lafan``, reads/writes under ``data/g1/lafan/``::

  data/g1/lafan/raw/*.csv or *.pkl
  data/g1/lafan/npz/<clip>.npz

Use ``--batch-size N`` for a GPU worker pool of ``N`` parallel FK envs. Idle envs
continuously pull the next clip from the queue; progress shows finished/total clips.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
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
  MotionPlayback,
  load_motion_clip,
  peek_dof_width,
  resample_motion_clips_batch,
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

_MotionState = tuple[
  torch.Tensor,
  torch.Tensor,
  torch.Tensor,
  torch.Tensor,
  torch.Tensor,
  torch.Tensor,
]


@dataclass
class _EnvSlot:
  source_path: Path | None = None
  motion: MotionPlayback | None = None
  log: dict[str, Any] | None = None
  cached_state: _MotionState | None = None

  @property
  def active(self) -> bool:
    return self.motion is not None


def _empty_log(output_fps: float) -> dict[str, Any]:
  return {
    "fps": [output_fps],
    "joint_pos": [],
    "joint_vel": [],
    "body_pos_w": [],
    "body_quat_w": [],
    "body_lin_vel_w": [],
    "body_ang_vel_w": [],
  }


def _finalize_log(log: dict[str, Any]) -> dict[str, Any]:
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


def _apply_motion_state(
  *,
  env_idx: int,
  state: _MotionState,
  root_states: torch.Tensor,
  joint_pos: torch.Tensor,
  joint_vel: torch.Tensor,
  env_origins_xy: torch.Tensor,
  robot_joint_indexes: list[int],
) -> None:
  (
    motion_base_pos,
    motion_base_rot,
    motion_base_lin_vel,
    motion_base_ang_vel,
    motion_dof_pos,
    motion_dof_vel,
  ) = state
  root_states[env_idx, 0:3] = motion_base_pos[0]
  root_states[env_idx, :2] += env_origins_xy[env_idx]
  root_states[env_idx, 3:7] = motion_base_rot[0]
  root_states[env_idx, 7:10] = motion_base_lin_vel[0]
  root_states[env_idx, 10:] = motion_base_ang_vel[0]
  joint_pos[env_idx, robot_joint_indexes] = motion_dof_pos[0]
  joint_vel[env_idx, robot_joint_indexes] = motion_dof_vel[0]


def _append_env_log(robot: Entity, env_idx: int, log: dict[str, Any]) -> None:
  log["joint_pos"].append(robot.data.joint_pos[env_idx, :].cpu().numpy().copy())
  log["joint_vel"].append(robot.data.joint_vel[env_idx, :].cpu().numpy().copy())
  log["body_pos_w"].append(robot.data.body_link_pos_w[env_idx, :].cpu().numpy().copy())
  log["body_quat_w"].append(robot.data.body_link_quat_w[env_idx, :].cpu().numpy().copy())
  log["body_lin_vel_w"].append(
    robot.data.body_link_lin_vel_w[env_idx, :].cpu().numpy().copy()
  )
  log["body_ang_vel_w"].append(
    robot.data.body_link_ang_vel_w[env_idx, :].cpu().numpy().copy()
  )


def _fill_idle_slots(
  slots: list[_EnvSlot],
  *,
  pending_paths: deque[Path],
  joint_names: list[str],
  output_fps: float,
  device: torch.device | str,
  format_name: str | None,
  input_fps: float | None,
  line_range: tuple[int, int] | None,
) -> int:
  """Load and GPU-resample clips for all idle envs in one batch."""
  idle_indices = [idx for idx, slot in enumerate(slots) if not slot.active]
  if not idle_indices or not pending_paths:
    return 0

  count = min(len(idle_indices), len(pending_paths))
  paths = [pending_paths.popleft() for _ in range(count)]
  raw_clips = [
    load_motion_clip(
      path,
      format_name=format_name,
      input_fps=input_fps,
      line_range=line_range,
    )
    for path in paths
  ]
  resampled = resample_motion_clips_batch(
    raw_clips,
    int(output_fps),
    device,
    model_joint_names=joint_names,
  )

  for env_idx, source_path, raw_clip, clip in zip(
    idle_indices[:count], paths, raw_clips, resampled, strict=True
  ):
    slots[env_idx].source_path = source_path
    slots[env_idx].motion = MotionPlayback(clip)
    slots[env_idx].log = _empty_log(output_fps)
    slots[env_idx].cached_state = None
    tqdm.write(
      f"[INFO]   {source_path.name} "
      f"(format={format_name or 'auto'}, input_fps={raw_clip.fps}, "
      f"frames={clip.output_frames})"
    )

  if count > 1:
    tqdm.write(f"[INFO] GPU-resampled {count} clip(s) in one batch")
  return count


def run_sim_pool(
  sim: Simulation,
  scene: Scene,
  joint_names: list[str],
  pending_paths: deque[Path],
  total_clips: int,
  output_fps: float,
  *,
  format_name: str | None,
  input_fps: float | None,
  line_range: tuple[int, int] | None,
  render: bool,
  renderer: OffscreenRenderer | None = None,
) -> Iterator[tuple[Path, dict[str, Any]]]:
  """FK clips on a fixed GPU worker pool; yields each clip as soon as it finishes."""
  num_envs = sim.num_envs
  robot: Entity = scene["robot"]
  robot_joint_indexes = robot.find_joints(joint_names, preserve_order=True)[0]
  env_origins_xy = scene.env_origins[:, :2]
  slots = [_EnvSlot() for _ in range(num_envs)]

  scene.reset()
  print(
    f"\nStarting FK pool: {num_envs} env(s), "
    f"{total_clips} clip(s) @ {output_fps} Hz"
  )
  if render:
    print("Rendering enabled - generating video frames...")

  clip_pbar = tqdm(
    total=total_clips,
    desc="Clips",
    unit="clip",
    ncols=100,
    position=0,
    bar_format=(
      "{l_bar}{bar}| {n_fmt}/{total_fmt} [{percentage:3.0f}%] "
      "{elapsed}<{remaining}, {rate_fmt}"
    ),
  )
  frame_pbar = tqdm(
    desc="FK frames",
    unit="frame",
    ncols=100,
    position=1,
    leave=False,
  )

  _fill_idle_slots(
    slots,
    pending_paths=pending_paths,
    joint_names=joint_names,
    output_fps=output_fps,
    device=sim.device,
    format_name=format_name,
    input_fps=input_fps,
    line_range=line_range,
  )

  while any(slot.active for slot in slots) or pending_paths:
    _fill_idle_slots(
      slots,
      pending_paths=pending_paths,
      joint_names=joint_names,
      output_fps=output_fps,
      device=sim.device,
      format_name=format_name,
      input_fps=input_fps,
      line_range=line_range,
    )

    if not any(slot.active for slot in slots):
      break

    stepped = [False] * num_envs
    finished_paths: list[tuple[int, Path, dict[str, Any]]] = []
    root_states = robot.data.default_root_state.clone()
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()

    for env_idx, slot in enumerate(slots):
      if not slot.active:
        continue

      assert slot.motion is not None
      state, done = slot.motion.advance_state()
      slot.cached_state = state
      stepped[env_idx] = True
      _apply_motion_state(
        env_idx=env_idx,
        state=state,
        root_states=root_states,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        env_origins_xy=env_origins_xy,
        robot_joint_indexes=robot_joint_indexes,
      )
      if done:
        assert slot.source_path is not None
        assert slot.log is not None
        finished_paths.append((env_idx, slot.source_path, slot.log))
        slot.source_path = None
        slot.motion = None
        slot.log = None
        slot.cached_state = state

    robot.write_root_state_to_sim(root_states)
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    sim.forward()
    scene.update(sim.mj_model.opt.timestep)

    if render and renderer is not None:
      renderer.update(sim.data)
      renderer.render()

    for env_idx in range(num_envs):
      if not stepped[env_idx]:
        continue
      log = next(
        (log for idx, _, log in finished_paths if idx == env_idx),
        slots[env_idx].log,
      )
      if log is not None:
        _append_env_log(robot, env_idx, log)

    frames_stepped = sum(stepped)
    if frames_stepped:
      frame_pbar.update(frames_stepped)

    for _, source_path, log in finished_paths:
      clip_pbar.update(1)
      clip_pbar.set_postfix(
        last=source_path.name,
        active=sum(1 for slot in slots if slot.active),
        queued=len(pending_paths),
        refresh=False,
      )
      yield source_path, _finalize_log(log)

  clip_pbar.close()
  frame_pbar.close()


def main(
  input_path: str | None = None,
  output_dir: str | None = None,
  dataset: str | None = None,
  dataset_path: str | None = None,
  robot: str = "g1",
  format: str | None = None,
  input_fps: float | None = None,
  output_fps: float = 50.0,
  batch_size: int = 1,
  device: str = "cuda:0",
  render: bool = False,
  line_range: tuple[int, int] | None = None,
  debias_z: bool = False,
  debias_foot_sole_z: float | None = None,
):
  if batch_size < 1:
    raise ValueError(f"batch_size must be >= 1, got {batch_size}")
  if render and batch_size > 1:
    raise ValueError("--render is only supported with --batch-size 1")

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
  pending_paths: deque[Path] = deque(input_paths)
  clips: list[MotionClipExport] = []
  npz_output_dir(output_dir).mkdir(parents=True, exist_ok=True)

  sim_cfg = SimulationCfg()
  sim_cfg.mujoco.timestep = 1.0 / output_fps

  scene = Scene(conversion_scene_cfg(motion_spec, num_envs=batch_size), device=device)
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
    f"joints={len(joint_names)} (model.nq={model.nq}), "
    f"pool_size={batch_size}, clips={len(input_paths)}"
  )

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

  sim = Simulation(num_envs=batch_size, cfg=sim_cfg, model=model, device=device)
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

  for source_path, log in run_sim_pool(
    sim=sim,
    scene=scene,
    joint_names=joint_names,
    pending_paths=pending_paths,
    total_clips=len(input_paths),
    output_fps=output_fps,
    format_name=format,
    input_fps=input_fps,
    line_range=line_range,
    render=render,
    renderer=renderer,
  ):
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
