"""Dynamics-only sagittal augmentation using the repo's robot symmetry metadata.

After BlinkMimic d264658, mirror.py (MIT; THIRD_PARTY_NOTICES):
https://github.com/fuyichen0725/blinkmimic/blob/d264658d6c9ce61fb52f9f0f8985a369f3ccc409/src/blinkmimic/mirror.py
SuperMimic world-model loss: https://arxiv.org/abs/2610.28480.
Goals/reward/termination heads keep the original, unreflected windows.
"""

from __future__ import annotations

import torch
from mjlab.envs.mdp import dr
from torch import nn

from wbc_mjlab.motion.motion_mirror import build_joint_mirror_map
from wbc_mjlab.rl.wm.kinematics import RobotState
from wbc_mjlab.robots.symmetry import get_robot_symmetry_config


class DynamicsMirror(nn.Module):
  def __init__(self, im, robot_id):
    super().__init__()
    config = get_robot_symmetry_config(robot_id)
    if config is None:
      raise ValueError(f"no dynamics symmetry metadata for {robot_id}")
    covered = {name for pair in config.joints for name in (pair.name_a, pair.name_b)}
    if set(im.joint_names) - covered:
      raise ValueError("dynamics mirroring needs explicit symmetry for every joint")
    perm, sign = build_joint_mirror_map(im.joint_names, config)
    dev = im.env.device
    self.register_buffer("joint_perm", torch.as_tensor(perm, device=dev))
    self.register_buffer(
      "joint_sign", torch.as_tensor(sign, device=dev, dtype=torch.float32)
    )
    self.register_buffer("polar", torch.tensor([1.0, -1.0, 1.0], device=dev))
    self.register_buffer("quat_sign", torch.tensor([1.0, -1.0, 1.0, -1.0], device=dev))
    # Context order is defined by Imagination._read_context; build the same order.
    indices, scales, offset = [], [], 0
    for name in im.context_names:
      cfg = im.env.event_manager.get_term_cfg(name)
      if cfg.func is dr.encoder_bias:
        idx, scale = perm.tolist(), sign.tolist()
      elif cfg.func is dr.geom_friction:
        if not cfg.params.get("shared_random"):
          raise ValueError("dynamics mirroring needs shared geom friction")
        n = len(cfg.params.get("axes") or [0])
        idx, scale = list(range(n)), [1.0] * n
      elif cfg.func is dr.body_com_offset:
        body_ids = cfg.params["asset_cfg"].body_ids
        names = [im.body_names[i] for i in body_ids]
        swaps = [
          n.replace("left", "RIGHT").replace("right", "left").replace("RIGHT", "right")
          for n in names
        ]
        if set(swaps) != set(names):
          raise ValueError("COM randomization must cover mirrored body pairs")
        idx = [3 * names.index(n) + k for n in swaps for k in range(3)]
        scale = [1.0, -1.0, 1.0] * len(names)
      else:
        raise ValueError(f"no reflection rule for context {name}")
      indices.extend(offset + i for i in idx)
      scales.extend(scale)
      offset += len(idx)
    self.register_buffer(
      "context_perm", torch.tensor(indices, dtype=torch.long, device=dev)
    )
    self.register_buffer("context_sign", torch.tensor(scales, device=dev))
    if offset != im._context.shape[-1]:
      raise ValueError("symmetry context layout differs from world-model context")

  def joints(self, x):
    return x[..., self.joint_perm] * self.joint_sign

  def state(self, x):
    j = len(self.joint_perm)
    s = RobotState.from_vector(x.reshape(-1, x.shape[-1]), j)
    mirrored = RobotState(
      s.root_pos * self.polar,
      s.root_quat * self.quat_sign,
      s.root_lin_vel * self.polar,
      -s.root_ang_vel * self.polar,
      self.joints(s.joint_pos),
      self.joints(s.joint_vel),
    )
    return mirrored.vector().reshape_as(x)

  def windows(self, states, successors, setpoints, context, probability):
    mask = torch.rand(states.shape[1], device=states.device) < probability
    select = mask[None, :, None]
    return (
      torch.where(select, self.state(states), states),
      torch.where(select, self.state(successors), successors),
      torch.where(select, self.joints(setpoints), setpoints),
      torch.where(
        mask[:, None], context[..., self.context_perm] * self.context_sign, context
      ),
    )
