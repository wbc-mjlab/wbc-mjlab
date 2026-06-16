"""State-estimation actor observation layout (task configs only).

The base ``make_base_wbc_env_cfg`` template has reference command + proprio only.
SE tasks call ``configure_state_estimation_actor_obs`` to swap height/gravity refs for
full anchor xyz + ori, and add root pose + base velocity measurements.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

import wbc_mjlab.env.mdp as mdp

_MOTION = "motion"
_MOTION_PARAMS = {"command_name": _MOTION}
_ROBOT = SceneEntityCfg("robot")

# Dropped when SE layout is applied (replaced by ref anchor xyz/ori).
_SE_REMOVED_TERMS = (
  "ref_base_height",
  "ref_gravity_b",
  "projected_gravity",
)

# Anchor reference command (replaces ref_base_height + ref_gravity_b / projected_gravity).
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
}

# Robot-side SE measurements (root pose + base velocity; no motion command handle).
_SE_MEASUREMENT_TERMS: dict[str, ObservationTermCfg] = {
  "root_pos_w": ObservationTermCfg(
    func=mdp.root_pos_w,
    params={"asset_cfg": _ROBOT},
    noise=Unoise(n_min=-0.01, n_max=0.01),
  ),
  "root_ori_6d": ObservationTermCfg(
    func=mdp.root_ori_6d,
    params={"asset_cfg": _ROBOT},
    noise=Unoise(n_min=-0.02, n_max=0.02),
  ),
  "base_lin_vel": ObservationTermCfg(
    func=mdp.builtin_sensor,
    params={"sensor_name": ""},
    noise=Unoise(n_min=-0.5, n_max=0.5),
  ),
}


def configure_state_estimation_actor_obs(cfg: ManagerBasedRlEnvCfg) -> None:
  """Swap height/gravity refs for anchor xyz/ori; add root pose + base lin vel."""
  actor = cfg.observations["actor"]
  for key in _SE_REMOVED_TERMS:
    actor.terms.pop(key, None)
  actor.terms.update(_SE_REF_TERMS)
  actor.terms.update(_SE_MEASUREMENT_TERMS)
