"""Online motion library: packing / cache, resampling + FK against MuJoCo, mirror, debias,
and the training command on source clips."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

from wbc_mjlab.data_paths import repo_root

SAMPLES = repo_root() / "data" / "g1" / "samples"
CLIP = "walk1_subject1"
ROWS = 90  # source frames used (3 s at 30 Hz): keeps the CPU FK checks quick
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def baker():
  from wbc_mjlab.motion.library import Baker

  return Baker(
    SAMPLES,
    robot_id="g1",
    fps=50.0,
    mirror=True,
    device=DEVICE,
    batch=256,
    clips=[CLIP],
    line_range=(1, ROWS),
  )


def _scene_model() -> mujoco.MjModel:
  """The robot scene the library runs FK on (the conversion scene)."""
  from mjlab.scene import Scene

  from wbc_mjlab.motion.robot_assets import conversion_scene_cfg, get_robot_motion_spec

  _, spec = get_robot_motion_spec("g1")
  return Scene(conversion_scene_cfg(spec), device="cpu").compile()


def _joint_names() -> list[str]:
  from wbc_mjlab.motion.robot_assets import qpos_dof_joint_names

  return qpos_dof_joint_names(_scene_model())


# ---------------------------------------------------------------------------------- packing
def test_pack_cache_rebuilds_when_a_clip_changes(tmp_path: Path) -> None:
  from wbc_mjlab.motion.library import CACHE_DIR, has_sources, load_dataset

  for name in ("walk1_subject1", "dance"):
    shutil.copy(SAMPLES / f"{name}.csv", tmp_path / f"{name}.csv")
  assert has_sources(tmp_path) and not has_sources(tmp_path / "missing")
  names = _joint_names()

  packed = load_dataset(tmp_path, names, verbose=False)
  assert packed.names == ["dance", "walk1_subject1"]
  assert packed.frames.shape == (int(packed.length.sum()), 7 + len(names))
  assert np.allclose(np.linalg.norm(packed.frames[:, 3:7], axis=1), 1.0, atol=1e-5)
  caches = list((tmp_path / CACHE_DIR).glob("motion_*.npz"))
  assert len(caches) == 1

  again = load_dataset(tmp_path, names, verbose=False)  # served from the cache
  assert np.array_equal(again.frames, packed.frames)

  clip = tmp_path / "dance.csv"
  clip.write_text("\n".join(clip.read_text().splitlines()[:40]) + "\n")
  os.utime(clip, ns=(clip.stat().st_atime_ns, clip.stat().st_mtime_ns + 10**9))
  edited = load_dataset(tmp_path, names, verbose=False)
  assert int(edited.length[0]) == 40
  new = list((tmp_path / CACHE_DIR).glob("motion_*.npz"))
  assert len(new) == 1 and new[0] != caches[0]


def test_training_source_is_the_dataset_folder(tmp_path: Path) -> None:
  from wbc_mjlab.data_paths import resolve_motion_path
  from wbc_mjlab.motion.stack_bundle import resolve_training_motion_source

  shutil.copy(SAMPLES / f"{CLIP}.csv", tmp_path / f"{CLIP}.csv")
  assert resolve_motion_path(tmp_path) == tmp_path.resolve()
  assert resolve_training_motion_source(tmp_path, robot_id="g1") == tmp_path.resolve()


# ---------------------------------------------------------------------------------- baking
def _slerp(q0: np.ndarray, q1: np.ndarray, b: float) -> np.ndarray:
  q0, q1 = q0 / np.linalg.norm(q0), q1 / np.linalg.norm(q1)
  if np.dot(q0, q1) < 0.0:
    q1 = -q1
  angle = 2.0 * np.arctan2(np.linalg.norm(q0 - q1), np.linalg.norm(q0 + q1))
  if angle < 1e-12:
    return q0
  out = (np.sin((1 - b) * angle) * q0 + np.sin(b * angle) * q1) / np.sin(angle)
  return out / np.linalg.norm(out)


def _reference_mujoco(t: float, raw, model, bodies: list[int], fps_out: float) -> tuple:
  """Ground truth for one frame: numpy interpolation + central differences + CPU MuJoCo."""
  n = raw.base_pos.shape[0]
  end = (n - 1) / raw.fps
  quat = raw.base_rot_xyzw[:, [3, 0, 1, 2]].astype(np.float64)

  def coords(tt: float) -> np.ndarray:
    x = min(max(tt, 0.0), end) * raw.fps
    i0 = min(int(np.floor(x)), n - 1)
    i1 = min(i0 + 1, n - 1)
    b = x - i0
    pos = (1 - b) * raw.base_pos[i0] + b * raw.base_pos[i1]
    dof = (1 - b) * raw.dof_pos[i0] + b * raw.dof_pos[i1]
    return np.concatenate([pos, _slerp(quat[i0], quat[i1], b), dof])

  h = 1.0 / fps_out
  t0, t1 = max(t - h, 0.0), min(t + h, end)
  q0, q, q1 = coords(t0), coords(t), coords(t1)
  qvel = np.zeros(model.nv)
  qvel[:3] = (q1[:3] - q0[:3]) / (t1 - t0)
  dq = np.zeros(3)
  rel = np.zeros(4)
  conj = q0[3:7] * np.array([1, -1, -1, -1])
  mujoco.mju_mulQuat(rel, q1[3:7], conj)
  mujoco.mju_quat2Vel(dq, rel, t1 - t0)  # world-frame angular velocity
  rot = np.zeros(9)
  mujoco.mju_quat2Mat(rot, q[3:7])
  qvel[3:6] = rot.reshape(3, 3).T @ dq
  qvel[6:] = (q1[7:] - q0[7:]) / (t1 - t0)
  data = mujoco.MjData(model)
  data.qpos[:] = q
  data.qvel[:] = qvel
  mujoco.mj_kinematics(model, data)
  mujoco.mj_comPos(model, data)
  mujoco.mj_comVel(model, data)
  vel = np.zeros((len(bodies), 6))
  for i, b in enumerate(bodies):
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, b, vel[i], 0)
  return data.xpos[bodies].copy(), data.xquat[bodies].copy(), vel[:, 3:], vel[:, :3]


def test_bake_matches_mujoco_fk(baker) -> None:
  from wbc_mjlab.motion.motion_formats import load_motion_clip

  raw = load_motion_clip(SAMPLES / f"{CLIP}.csv", line_range=(1, ROWS))
  model = _scene_model()
  assert model.nq == 7 + len(baker.joint_names)  # free joint first, then the joints
  bodies = [
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"robot/{n}")
    for n in baker.body_names
  ]
  seg = baker.segment_names.index(CLIP)
  log = baker.bake(seg)
  frames = log["joint_pos"].shape[0]
  assert frames == torch.arange(0, (ROWS - 1) / 30.0, 1 / 50.0).numel()
  for k in (0, 1, 37, frames - 1):
    pos, quat, lin, ang = _reference_mujoco(k / 50.0, raw, model, bodies, 50.0)
    # float32 GPU FK vs float64 MuJoCo: 0.1 mm / 1e-4
    np.testing.assert_allclose(log["body_pos_w"][k], pos, atol=1e-4)
    q = log["body_quat_w"][k]
    assert np.minimum(np.abs(q - quat), np.abs(q + quat)).max() < 1e-4
    np.testing.assert_allclose(log["body_lin_vel_w"][k], lin, atol=1e-4)
    np.testing.assert_allclose(log["body_ang_vel_w"][k], ang, atol=1e-4)


def test_mirror_matches_body_level_mirroring(baker) -> None:
  """Mirroring the coordinates before FK = mirroring the bodies after (old converter)."""
  from wbc_mjlab.motion.motion_mirror import mirror_motion_log
  from wbc_mjlab.robots.symmetry import get_robot_symmetry_config

  sym = get_robot_symmetry_config("g1")
  names = baker.segment_names
  assert f"{CLIP}{sym.mirror_suffix}" in names
  orig = baker.bake(names.index(CLIP))
  mirrored = baker.bake(names.index(f"{CLIP}{sym.mirror_suffix}"))
  expected = mirror_motion_log(
    orig, joint_names=baker.joint_names, body_names=baker.body_names, config=sym
  )
  for key in (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
  ):
    np.testing.assert_allclose(mirrored[key], expected[key], atol=2e-3, err_msg=key)
  q, e = mirrored["body_quat_w"], expected["body_quat_w"]
  assert np.minimum(np.abs(q - e), np.abs(q + e)).max() < 2e-3


def test_debias_puts_the_lowest_foot_on_the_sole() -> None:
  from wbc_mjlab.motion.library import Baker
  from wbc_mjlab.motion.robot_assets import get_robot_motion_spec

  _, spec = get_robot_motion_spec("g1")
  b = Baker(
    SAMPLES,
    robot_id="g1",
    debias_z=True,
    device=DEVICE,
    batch=256,
    clips=[CLIP],
    line_range=(1, ROWS),
  )
  log = b.bake(0)
  feet = [b.body_names.index(f) for f in spec.foot_body_names]
  lowest = float(log["body_pos_w"][:, feet, 2].min())
  # min over the source frames is the sole height; baked frames sit between them
  assert spec.foot_sole_z - 1e-5 <= lowest <= spec.foot_sole_z + 5e-3


# ---------------------------------------------------------------------------------- training
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_command_plays_source_clips_online() -> None:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  from wbc_mjlab.motion.library import OnlineMotion
  from wbc_mjlab.tasks import prepare_wbc_run

  prepare_wbc_run(task_id="Wbc-G1")
  cfg = load_env_cfg("Wbc-G1")
  cfg.commands["motion"].motion_file = str(SAMPLES)
  cfg.commands["motion"].mirror = True
  cfg.scene.num_envs = 8
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  try:
    cmd = env.command_manager.get_term("motion")
    assert isinstance(cmd.motion, OnlineMotion)
    n_csv = len(list(SAMPLES.glob("*.csv")))
    assert cmd.motion.num_trajectories == 2 * n_csv
    act = torch.zeros(8, env.action_manager.total_action_dim, device="cuda:0")
    env.reset()
    for _ in range(40):
      env.step(act)
    # the cached / patched reference equals a fresh eager computation
    ref = cmd.reference()
    fresh = cmd.motion._frame(cmd.time_steps)
    for name in ("joint_pos", "body_pos_w", "body_lin_vel_w", "anchor_lin_acc_w"):
      a, b = getattr(ref, name), getattr(fresh, name)
      assert torch.isfinite(a).all()
      torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
    # graph replay at a reset-sized batch matches eager too
    some = cmd.time_steps[:3]
    torch.testing.assert_close(
      cmd.motion.frame(some).body_quat_w, cmd.motion._frame(some).body_quat_w
    )
  finally:
    env.close()


def test_run_motion_options_read_back(tmp_path: Path) -> None:
  from wbc_mjlab.motion.library import apply_run_motion_options, run_motion_options

  params = tmp_path / "params"
  params.mkdir()
  (params / "env.yaml").write_text(
    "scene:\n  size: !!python/tuple\n  - 1\n  - 2\n"
    "commands:\n  motion:\n    motion_file: /data/x\n    mirror: true\n"
    "    debias_z: true\n    source_fps: null\n    anchor_body_name: torso_link\n"
  )
  assert run_motion_options(tmp_path) == {
    "mirror": True,
    "debias_z": True,
    "source_fps": None,
  }
  assert run_motion_options(tmp_path / "missing") == {}
  cfg = SimpleNamespace(mirror=False, debias_z=False, source_fps=30.0)
  apply_run_motion_options(cfg, tmp_path)
  assert cfg.mirror and cfg.debias_z and cfg.source_fps is None
