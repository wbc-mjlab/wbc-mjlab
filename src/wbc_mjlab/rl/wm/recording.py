"""Pre-reset terminal-state capture for world-model replay.

After SuperMimic, Algorithm A.1, https://arxiv.org/abs/2610.28480, and the
pre-reset recorder / clean-action convention in CuboiLeo/SuperMimic-Reproduced
(ba61053), src/supermimic/{data/recorder,envs/actions}.py:
https://github.com/CuboiLeo/SuperMimic-Reproduced/tree/ba6105349069bf040f1fdad60873d13138d505a2
Uses entity indexing and the repo's RobotState, without assuming a G1 state layout.
"""

import torch
from mjlab.managers.recorder_manager import RecorderTerm
from mjlab.utils.lab_api.math import quat_apply

from wbc_mjlab.rl.wm.kinematics import RobotState

RECORDER = "wm_terminal"


def simulator_state(env, robot) -> RobotState:
  """Read integrated qpos/qvel; link caches are one substep stale before auto-reset."""
  idx = robot.indexing
  q = env.sim.data.qpos[:, idx.free_joint_q_adr]
  v = env.sim.data.qvel[:, idx.free_joint_v_adr]
  return RobotState(
    q[:, :3] - env.scene.env_origins,
    q[:, 3:7],
    v[:, :3],
    quat_apply(q[:, 3:7], v[:, 3:6]),
    env.sim.data.qpos[:, idx.joint_q_adr],
    env.sim.data.qvel[:, idx.joint_v_adr],
  )


class TerminalStateRecorder(RecorderTerm):
  """Only a small staging buffer; the algorithm owns the time-major replay."""

  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self.robot = env.command_manager.get_term(cfg.params["command_name"]).robot
    self.state = torch.zeros(
      env.num_envs, 13 + 2 * len(self.robot.joint_names), device=env.device
    )
    self.captured = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

  def record_pre_reset(self, env_ids):
    self.state[env_ids] = simulator_state(self._env, self.robot).vector()[env_ids]
    self.captured[env_ids] = True
