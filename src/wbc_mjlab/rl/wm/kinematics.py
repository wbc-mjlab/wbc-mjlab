"""Differentiable forward kinematics of a floating-base robot (torch).

Mirrors MuJoCo's ``mj_kinematics`` for a tree of one free joint (the root) and at most one
hinge per body, plus body-origin velocities (mjlab ``body_link_*_vel_w``). Bodies at the same
tree depth are computed together, so one call is a few kernels per depth level, not per body.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
import torch
from mjlab.utils.lab_api.math import quat_apply, quat_from_angle_axis, quat_mul


@dataclass
class RobotState:
  """Floating-base state in the env frame (mjlab conventions, quaternions wxyz)."""

  root_pos: torch.Tensor  # (N, 3)
  root_quat: torch.Tensor  # (N, 4)
  root_lin_vel: torch.Tensor  # (N, 3) world frame
  root_ang_vel: torch.Tensor  # (N, 3) world frame
  joint_pos: torch.Tensor  # (N, J)
  joint_vel: torch.Tensor  # (N, J)

  @staticmethod
  def from_vector(x: torch.Tensor, num_joints: int) -> RobotState:
    j = num_joints
    return RobotState(
      x[:, 0:3], x[:, 3:7], x[:, 7:10], x[:, 10:13], x[:, 13 : 13 + j], x[:, 13 + j :]
    )

  def vector(self) -> torch.Tensor:
    return torch.cat(
      [
        self.root_pos,
        self.root_quat,
        self.root_lin_vel,
        self.root_ang_vel,
        self.joint_pos,
        self.joint_vel,
      ],
      dim=-1,
    )


@dataclass
class BodyKinematics:
  """Poses and origin velocities of every robot body, world (env) frame."""

  pos: torch.Tensor  # (N, B, 3)
  quat: torch.Tensor  # (N, B, 4)
  lin_vel: torch.Tensor  # (N, B, 3)
  ang_vel: torch.Tensor  # (N, B, 3)


class RobotKinematics:
  """Forward kinematics for one robot entity of a compiled MuJoCo model.

  Args:
    mj_model: The scene model.
    body_ids: Global body ids of the robot, in entity order (root first).
    joint_qpos_adr: ``qpos`` address of each robot hinge joint, in entity joint order.
    device: Torch device.
  """

  def __init__(
    self,
    mj_model: mujoco.MjModel,
    body_ids: list[int],
    joint_qpos_adr: list[int],
    device: torch.device | str,
  ):
    body_ids = [int(b) for b in body_ids]
    local = {b: i for i, b in enumerate(body_ids)}
    root = body_ids[0]
    if mj_model.body_jntnum[root] != 1 or mj_model.jnt_type[
      mj_model.body_jntadr[root]
    ] != (mujoco.mjtJoint.mjJNT_FREE):
      raise ValueError("the robot root body must have exactly one free joint")
    joint_of_qpos = {int(a): k for k, a in enumerate(joint_qpos_adr)}

    depth = [0] * len(body_ids)
    parent = [-1] * len(body_ids)
    joint = [-1] * len(
      body_ids
    )  # entity joint index of the body's hinge (-1: fixed body)
    axis = np.zeros((len(body_ids), 3))
    anchor = np.zeros((len(body_ids), 3))
    for i, b in enumerate(body_ids[1:], start=1):
      p = int(mj_model.body_parentid[b])
      if p not in local:
        raise ValueError(f"body {mj_model.body(b).name!r} is not in the robot subtree")
      parent[i], depth[i] = local[p], depth[local[p]] + 1
      n = int(mj_model.body_jntnum[b])
      if n > 1:
        raise ValueError(
          f"body {mj_model.body(b).name!r} has {n} joints (at most 1 hinge)"
        )
      if n == 1:
        j = int(mj_model.body_jntadr[b])
        if mj_model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
          raise ValueError(f"joint {mj_model.joint(j).name!r} is not a hinge")
        joint[i] = joint_of_qpos[int(mj_model.jnt_qposadr[j])]
        axis[i], anchor[i] = mj_model.jnt_axis[j], mj_model.jnt_pos[j]

    def t(x, dtype=torch.float32):
      return torch.as_tensor(np.asarray(x), dtype=dtype, device=device)

    self.num_bodies = len(body_ids)
    self.num_joints = len(joint_qpos_adr)
    self._body_pos = t(mj_model.body_pos[body_ids])
    self._body_quat = t(mj_model.body_quat[body_ids])
    self._axis, self._anchor = t(axis), t(anchor)
    # Fixed bodies read a zero column appended after the joints.
    self._joint = t([j if j >= 0 else self.num_joints for j in joint], torch.long)
    self._levels = [
      t([i for i in range(len(body_ids)) if depth[i] == d], torch.long)
      for d in range(1, max(depth) + 1)
    ]
    # Bodies are computed level by level into arrays in level order: column of each body
    # there, its parent's column, and the permutation back to entity order.
    level_order = [0] + [i for lv in self._levels for i in lv.tolist()]
    column = {b: c for c, b in enumerate(level_order)}
    self._parent_col = [
      t([column[parent[i]] for i in lv.tolist()], torch.long) for lv in self._levels
    ]
    self._entity_order = t([column[i] for i in range(len(body_ids))], torch.long)
    self._mj_model = mj_model
    self._local = local
    self._device = device

  def site_frame(self, site_id: int) -> tuple[int, torch.Tensor, torch.Tensor]:
    """(entity body index, local pos, local quat) of a site on the robot."""
    m = self._mj_model
    b = self._local.get(int(m.site_bodyid[site_id]))
    if b is None:
      raise ValueError(f"site {m.site(site_id).name!r} is not on the robot")
    pos = torch.as_tensor(m.site_pos[site_id], dtype=torch.float32, device=self._device)
    quat = torch.as_tensor(
      m.site_quat[site_id], dtype=torch.float32, device=self._device
    )
    return b, pos, quat

  def bodies(self, s: RobotState) -> BodyKinematics:
    """Poses and origin velocities of every body (entity order)."""
    n = s.root_pos.shape[0]
    q = torch.cat([s.joint_pos, s.joint_pos.new_zeros(n, 1)], dim=-1)
    qd = torch.cat([s.joint_vel, s.joint_vel.new_zeros(n, 1)], dim=-1)
    pos, quat = s.root_pos.unsqueeze(1), s.root_quat.unsqueeze(1)
    lin, ang = s.root_lin_vel.unsqueeze(1), s.root_ang_vel.unsqueeze(1)
    for level, pc in zip(self._levels, self._parent_col, strict=True):
      k = level.numel()
      p_pos, p_quat, p_lin, p_ang = pos[:, pc], quat[:, pc], lin[:, pc], ang[:, pc]
      body_axis = self._axis[level].expand(n, k, 3)
      body_anchor = self._anchor[level].expand(n, k, 3)
      # Body frame before its joint, then the hinge about its anchor (mj_kinematics).
      b_quat = quat_mul(p_quat, self._body_quat[level].expand(n, k, 4))
      b_pos = p_pos + quat_apply(p_quat, self._body_pos[level].expand(n, k, 3))
      x_anchor = b_pos + quat_apply(b_quat, body_anchor)
      x_axis = quat_apply(b_quat, body_axis)
      ji = self._joint[level]
      b_quat = quat_mul(b_quat, quat_from_angle_axis(q[:, ji], body_axis))
      b_pos = x_anchor - quat_apply(b_quat, body_anchor)
      # Origin velocities: the parent's motion carried to the body, plus the hinge's.
      w_joint = x_axis * qd[:, ji].unsqueeze(-1)
      b_ang = p_ang + w_joint
      b_lin = (
        p_lin
        + torch.cross(p_ang, b_pos - p_pos, dim=-1)
        + torch.cross(w_joint, b_pos - x_anchor, dim=-1)
      )
      pos, quat = torch.cat([pos, b_pos], 1), torch.cat([quat, b_quat], 1)
      lin, ang = torch.cat([lin, b_lin], 1), torch.cat([ang, b_ang], 1)
    o = self._entity_order
    return BodyKinematics(pos[:, o], quat[:, o], lin[:, o], ang[:, o])
