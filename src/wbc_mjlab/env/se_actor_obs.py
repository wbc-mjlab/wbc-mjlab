"""State-estimation actor observation layout (task configs only).

SE tasks drop height/gravity reference proxies and add anchor pose tracking error
(world position + ``quat_error_magnitude``) plus ``base_lin_vel``.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

import wbc_mjlab.env.mdp as mdp

_MOTION = "motion"
_MOTION_PARAMS = {"command_name": _MOTION}

_SE_REMOVED_TERMS = (
  "ref_base_height",
  "ref_gravity_b",
  "projected_gravity",
)

# Anchor (``MotionCommand.anchor_*`` vs ``robot_anchor_*``), not pelvis root — matches
# tracking rewards/terminations; ``base_lin_vel`` remains root-frame IMU velocity.
_SE_ADDED_TERMS: dict[str, ObservationTermCfg] = {
  "motion_anchor_pos_error_w": ObservationTermCfg(
    func=mdp.motion_anchor_pos_error_w,
    params=_MOTION_PARAMS,
    noise=Unoise(n_min=-0.1, n_max=0.1),
  ),
  "motion_anchor_ori_error": ObservationTermCfg(
    func=mdp.motion_anchor_ori_error,
    params=_MOTION_PARAMS,
    noise=Unoise(n_min=-0.02, n_max=0.02),
  ),
  "base_lin_vel": ObservationTermCfg(
    func=mdp.builtin_sensor,
    params={"sensor_name": ""},
    noise=Unoise(n_min=-0.25, n_max=0.25),
  ),
}


def configure_state_estimation_actor_obs(cfg: ManagerBasedRlEnvCfg) -> None:
  """Drop height/gravity proxies; add anchor pose error + base lin vel."""
  actor = cfg.observations["actor"]
  for key in _SE_REMOVED_TERMS:
    actor.terms.pop(key, None)
  actor.terms.update(_SE_ADDED_TERMS)
