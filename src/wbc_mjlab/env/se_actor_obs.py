"""State-estimation actor observation layout (task configs only).

The base ``make_base_wbc_env_cfg`` template has reference command + proprio only.
SE tasks call ``configure_state_estimation_actor_obs`` to add measurements and swap
reference terms to full anchor/keybody pose.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

import wbc_mjlab.env.mdp as mdp

_MOTION = "motion"
_MOTION_PARAMS = {"command_name": _MOTION}

# Non-SE reference proxies dropped when SE layout is applied.
_SE_REMOVED_TERMS = (
  "ref_base_height",
  "ref_gravity_b",
  "projected_gravity",
)

# Full motion reference command on the actor.
_SE_REF_TERMS: dict[str, ObservationTermCfg] = {
  "ref_anchor_pos_w": ObservationTermCfg(
    func=mdp.ref_anchor_pos_w,
    params=_MOTION_PARAMS,
    noise=Unoise(n_min=-0.01, n_max=0.01),
  ),
  "ref_anchor_ori_6d": ObservationTermCfg(
    func=mdp.ref_anchor_ori_6d,
    params=_MOTION_PARAMS,
    noise=Unoise(n_min=-0.02, n_max=0.02),
  ),
  "ref_body_pos": ObservationTermCfg(
    func=mdp.ref_body_pos_b,
    params=_MOTION_PARAMS,
  ),
  "ref_body_ori": ObservationTermCfg(
    func=mdp.ref_body_ori_b,
    params=_MOTION_PARAMS,
  ),
}

# Robot-side SE measurements (full anchor pose tracking error + base velocity).
_SE_MEASUREMENT_TERMS: dict[str, ObservationTermCfg] = {
  "motion_anchor_pos_b": ObservationTermCfg(
    func=mdp.motion_anchor_pos_b,
    params=_MOTION_PARAMS,
    noise=Unoise(n_min=-0.25, n_max=0.25),
  ),
  "motion_anchor_ori_b": ObservationTermCfg(
    func=mdp.motion_anchor_ori_b,
    params=_MOTION_PARAMS,
    noise=Unoise(n_min=-0.02, n_max=0.02),
  ),
  "base_lin_vel": ObservationTermCfg(
    func=mdp.builtin_sensor,
    params={"sensor_name": ""},
    noise=Unoise(n_min=-0.5, n_max=0.5),
  ),
}


def configure_state_estimation_actor_obs(cfg: ManagerBasedRlEnvCfg) -> None:
  """Add SE measurements and full ref command; drop non-SE reference proxies."""
  actor = cfg.observations["actor"]
  for key in _SE_REMOVED_TERMS:
    actor.terms.pop(key, None)
  actor.terms.update(_SE_REF_TERMS)
  actor.terms.update(_SE_MEASUREMENT_TERMS)
