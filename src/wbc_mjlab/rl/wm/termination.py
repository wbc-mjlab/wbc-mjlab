"""Tracking-based survival for the shared imagination path.

Sigmoid relaxation follows BlinkMimic (Chen, 2026), mdp.py, and its xMimic
reimplementation (Bohlinger, 2026), xmimic/mdp.py at 54860e1 (MIT):
https://github.com/fuyichen0725/blinkmimic/blob/d264658d6c9ce61fb52f9f0f8985a369f3ccc409/src/blinkmimic/mdp.py
https://github.com/nico-bohlinger/xmimic/blob/54860e100251fa80143627565568d5e6b0ff2c2b/xmimic/mdp.py
This adapter uses WBC's configured anchor, error definitions and thresholds rather
than importing upstream's robot/task constants. Other failures retain SuperMimic's
learned termination head (https://arxiv.org/abs/2610.28480).
"""

import math

import torch
from mjlab.utils.lab_api.math import quat_apply_inverse

from wbc_mjlab.env.mdp import terminations as terms


class TrackingTermination:
  """Soft OR of supported physical tracking failures; timeouts remain clip masks.

  Upstream widths are 0.02 m for height and 0.04 for projected-gravity mismatch;
  the full-position variant uses the same 0.02 m width.
  Unsupported terms continue to be represented by the learned head. At least one
  supported term is required so selecting hybrid cannot silently have no effect.
  """

  def __init__(self, configured):
    self.terms = []
    supported = (
      terms.bad_anchor_pos,
      terms.bad_anchor_pos_z_only,
      terms.bad_anchor_ori,
    )
    for cfg in configured.values():
      if cfg is None or cfg.time_out or cfg.func not in supported:
        continue
      threshold = cfg.params["threshold"]
      if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError(
          "hybrid termination requires positive finite tracking thresholds"
        )
      self.terms.append((cfg.func, dict(cfg.params)))
    if not self.terms:
      raise ValueError(
        "hybrid termination needs an anchor position/height/orientation term"
      )

  def __call__(self, env):
    survive = torch.ones(env.num_envs, device=env.device)
    for func, params in self.terms:
      command = env.command_manager.get_term(params["command_name"])
      if func is terms.bad_anchor_ori:
        gravity = env.scene[params["asset_cfg"].name].data.gravity_vec_w
        ref = quat_apply_inverse(command.anchor_quat_w, gravity)
        actual = quat_apply_inverse(command.robot_anchor_quat_w, gravity)
        error, width = (ref[:, 2] - actual[:, 2]).abs(), 0.04
      else:
        delta = command.anchor_pos_w - command.robot_anchor_pos_w
        error = (
          delta[:, 2].abs()
          if func is terms.bad_anchor_pos_z_only
          else delta.norm(dim=-1)
        )
        width = 0.02
      survive = survive * (1 - torch.sigmoid((error - params["threshold"]) / width))
    return 1 - survive
