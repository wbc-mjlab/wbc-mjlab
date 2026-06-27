"""Unit tests for the wbc_reference_stream_v1 exporter (issue wbc-mjlab-ow5).

CPU-only and asset-free: validates the reference-command math (term order/dims,
anchor frame, gravity, joint remap), the raw .bin byte length / round-trip, and
the index.json schema. The math is cross-checked against an independent
quaternion-rotation reference (the deploy C++ ``q.conjugate() * v`` convention).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from wbc_mjlab.export.web_reference import (
  REF_TERMS,
  REFERENCE_SCHEMA,
  ReferenceClip,
  command_dim,
  expected_bin_bytes,
  quat_apply_inverse,
  ref_terms_index,
  reference_command_from_npz,
  reference_terms_from_clip,
  write_reference_bin,
  write_reference_index,
)

# Toy 4-dof joint set so we can run without g1 assets. Real G1 is 29-dof / 39-dim.
_TOY_JOINTS = ["j0", "j1", "j2", "j3"]
_TOY_DIM = command_dim(_TOY_JOINTS)  # 1 + 3 + 3 + 3 + 4 = 14


def _quat_conjugate_rotate(quat_wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:
  """Independent reference: rotate ``vec`` by the conjugate of ``quat`` (per row).

  Uses the Hamilton-product form ``v' = q* (0,v) q`` with q* the conjugate,
  the exact analogue of deploy's ``q.conjugate() * Eigen::Vector3f(v)``.
  """
  out = np.empty_like(vec, dtype=np.float64)
  for i in range(quat_wxyz.shape[0]):
    w, x, y, z = quat_wxyz[i]
    # Conjugate quaternion.
    qc = np.array([w, -x, -y, -z])

    def qmul(a, b):
      aw, ax, ay, az = a
      bw, bx, by, bz = b
      return np.array(
        [
          aw * bw - ax * bx - ay * by - az * bz,
          aw * bx + ax * bw + ay * bz - az * by,
          aw * by - ax * bz + ay * bw + az * bx,
          aw * bz + ax * by - ay * bx + az * bw,
        ]
      )

    qv = np.array([0.0, *vec[i]])
    q = np.array([w, x, y, z])
    res = qmul(qmul(qc, qv), q)
    out[i] = res[1:4]
  return out


def test_ref_terms_layout_sums_to_command_dim() -> None:
  assert [name for name, _ in REF_TERMS] == [
    "ref_base_height",
    "ref_base_lin_vel_b",
    "ref_base_ang_vel_b",
    "ref_gravity_b",
    "ref_joint_pos",
  ]
  # G1: 1 + 3 + 3 + 3 + 29 = 39.
  assert command_dim([f"j{i}" for i in range(29)]) == 39
  terms = ref_terms_index([f"j{i}" for i in range(29)])
  assert sum(t["dim"] for t in terms) == 39
  assert terms[-1] == {"name": "ref_joint_pos", "dim": 29}


def test_quat_apply_inverse_matches_hamilton_reference() -> None:
  rng = np.random.default_rng(7)
  quat = rng.standard_normal((16, 4))
  quat /= np.linalg.norm(quat, axis=1, keepdims=True)
  vec = rng.standard_normal((16, 3))
  got = quat_apply_inverse(quat, vec)
  ref = _quat_conjugate_rotate(quat, vec)
  assert np.allclose(got, ref, atol=1e-10)


def test_identity_quat_passthrough_and_gravity() -> None:
  frames = 5
  quat = np.tile([1.0, 0.0, 0.0, 0.0], (frames, 1))
  pos = np.zeros((frames, 3))
  pos[:, 2] = 0.9  # anchor height
  lin = np.tile([0.1, 0.2, 0.3], (frames, 1))
  ang = np.tile([0.01, 0.02, 0.03], (frames, 1))
  joints = np.tile(np.arange(len(_TOY_JOINTS), dtype=np.float64), (frames, 1))

  cmd = reference_terms_from_clip(
    anchor_pos_w=pos,
    anchor_quat_w=quat,
    anchor_lin_vel_w=lin,
    anchor_ang_vel_w=ang,
    joint_pos=joints,
  )
  assert cmd.shape == (frames, _TOY_DIM)
  assert cmd.dtype == np.dtype("<f4")
  # Identity rotation: world == body frame.
  assert np.allclose(cmd[:, 0], 0.9)  # ref_base_height
  assert np.allclose(cmd[:, 1:4], lin)  # ref_base_lin_vel_b
  assert np.allclose(cmd[:, 4:7], ang)  # ref_base_ang_vel_b
  assert np.allclose(cmd[:, 7:10], [0.0, 0.0, -1.0])  # ref_gravity_b
  assert np.allclose(cmd[:, 10:14], joints)  # ref_joint_pos


def test_env_origin_z_offsets_base_height() -> None:
  pos = np.array([[0.0, 0.0, 1.2]])
  cmd = reference_terms_from_clip(
    anchor_pos_w=pos,
    anchor_quat_w=np.array([[1.0, 0.0, 0.0, 0.0]]),
    anchor_lin_vel_w=np.zeros((1, 3)),
    anchor_ang_vel_w=np.zeros((1, 3)),
    joint_pos=np.zeros((1, len(_TOY_JOINTS))),
    env_origin_z=0.5,
  )
  assert np.isclose(cmd[0, 0], 0.7)


def _fake_npz(frames: int, robot_body_names: list[str], joint_names: list[str]) -> dict:
  """A dict mimicking an np.load'd clip NPZ (only the fields we read)."""
  rng = np.random.default_rng(3)
  nb = len(robot_body_names)
  quat = rng.standard_normal((frames, nb, 4)).astype(np.float32)
  quat /= np.linalg.norm(quat, axis=2, keepdims=True)
  return {
    "fps": np.asarray([50.0], dtype=np.float32),
    "joint_names": np.asarray(joint_names, dtype=object),
    "body_pos_w": rng.standard_normal((frames, nb, 3)).astype(np.float32),
    "body_quat_w": quat,
    "body_lin_vel_w": rng.standard_normal((frames, nb, 3)).astype(np.float32),
    "body_ang_vel_w": rng.standard_normal((frames, nb, 3)).astype(np.float32),
    "joint_pos": rng.standard_normal((frames, len(joint_names))).astype(np.float32),
    "joint_vel": rng.standard_normal((frames, len(joint_names))).astype(np.float32),
  }


def test_reference_from_npz_picks_anchor_and_remaps_joints() -> None:
  frames = 6
  bodies = ["pelvis", "torso_link", "left_hand"]
  # NPZ joint order is REVERSED vs the config order to exercise the remap.
  npz_joints = list(reversed(_TOY_JOINTS))
  npz = _fake_npz(frames, bodies, npz_joints)

  cmd = reference_command_from_npz(
    npz,
    robot_body_names=bodies,
    anchor_body_name="torso_link",
    config_joint_names=_TOY_JOINTS,
  )
  assert cmd.shape == (frames, _TOY_DIM)

  # Anchor terms must come from the torso_link (index 1) columns.
  anchor = 1
  expected = reference_terms_from_clip(
    anchor_pos_w=npz["body_pos_w"][:, anchor],
    anchor_quat_w=npz["body_quat_w"][:, anchor],
    anchor_lin_vel_w=npz["body_lin_vel_w"][:, anchor],
    anchor_ang_vel_w=npz["body_ang_vel_w"][:, anchor],
    # joint_pos reordered NPZ -> config order (reverse the reversal).
    joint_pos=npz["joint_pos"][:, ::-1],
  )
  assert np.allclose(cmd, expected, atol=1e-6)


def test_unknown_anchor_body_raises() -> None:
  npz = _fake_npz(2, ["pelvis", "torso_link"], _TOY_JOINTS)
  with pytest.raises(ValueError, match="Anchor body"):
    reference_command_from_npz(
      npz,
      robot_body_names=["pelvis", "torso_link"],
      anchor_body_name="nonexistent_link",
      config_joint_names=_TOY_JOINTS,
    )


def test_bin_byte_length_and_round_trip(tmp_path) -> None:
  frames = 7
  cmd = reference_terms_from_clip(
    anchor_pos_w=np.zeros((frames, 3)),
    anchor_quat_w=np.tile([1.0, 0.0, 0.0, 0.0], (frames, 1)),
    anchor_lin_vel_w=np.zeros((frames, 3)),
    anchor_ang_vel_w=np.zeros((frames, 3)),
    joint_pos=np.arange(frames * len(_TOY_JOINTS)).reshape(frames, len(_TOY_JOINTS)),
  )
  bin_path = tmp_path / "clip.bin"
  written = write_reference_bin(bin_path, cmd)
  expected = expected_bin_bytes(frames, _TOY_DIM)
  assert written == expected
  assert bin_path.stat().st_size == expected
  assert expected == frames * _TOY_DIM * 4

  loaded = np.fromfile(bin_path, dtype="<f4").reshape(frames, _TOY_DIM)
  assert np.array_equal(loaded, cmd)


def test_reference_index_json(tmp_path) -> None:
  clips = [
    ReferenceClip(
      id="walk1_subject1",
      name="Walk1 Subject1",
      file="walk1_subject1.bin",
      frames=120,
      duration_sec=2.4,
      tags=("walk", "locomotion"),
    )
  ]
  joint_names = [f"j{i}" for i in range(29)]
  out = write_reference_index(
    tmp_path, robot_id="g1", fps=50.0, joint_names=joint_names, clips=clips
  )
  doc = json.loads(out.read_text())
  assert doc["schema"] == REFERENCE_SCHEMA
  assert doc["robot"] == "g1"
  assert doc["commandDim"] == 39
  assert doc["fps"] == 50.0
  assert [t["name"] for t in doc["refTerms"]] == [
    "ref_base_height",
    "ref_base_lin_vel_b",
    "ref_base_ang_vel_b",
    "ref_gravity_b",
    "ref_joint_pos",
  ]
  assert sum(t["dim"] for t in doc["refTerms"]) == 39
  entry = doc["clips"][0]
  assert entry["id"] == "walk1_subject1"
  assert entry["frames"] == 120
  assert entry["file"] == "walk1_subject1.bin"
