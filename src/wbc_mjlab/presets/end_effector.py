"""Actor overlay: existing keybody pose refs instead of joint references.

Compose on top of ``apply_wbc``. Does **not** reweight rewards, RSI, or
terminations, and does not add MDP terms. Actor drops ``ref_joint_*`` and
reuses ``ref_body_pos`` / ``ref_body_ori`` already defined for the critic.
Critic keeps joint refs. Actions leave the motion joint residual so deploy
does not need ``q_ref``.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

import wbc_mjlab.env.mdp as mdp
from wbc_mjlab.env.mdp.actions import DefaultOffsetJointPositionActionCfg

_MOTION = "motion"
_MOTION_PARAMS = {"command_name": _MOTION}
_PROPRIO_HEAD = (
  "base_ang_vel",
  "projected_gravity",
  "joint_pos",
  "joint_vel",
  "actions",
  "motion_anchor_pos_error_w",
  "motion_anchor_ori_error",
  "base_lin_vel",
)


def _insert_terms_before_proprio(
  terms: dict[str, ObservationTermCfg],
  new_terms: dict[str, ObservationTermCfg],
) -> None:
  """Insert command terms just before the first proprio / SE measurement term."""
  rebuilt: dict[str, ObservationTermCfg] = {}
  inserted = False
  for name, term in terms.items():
    if not inserted and name in _PROPRIO_HEAD:
      rebuilt.update(new_terms)
      inserted = True
    rebuilt[name] = term
  if not inserted:
    rebuilt.update(new_terms)
  terms.clear()
  terms.update(rebuilt)


def _use_default_offset_joint_action(cfg: ManagerBasedRlEnvCfg) -> None:
  """q_cmd = q_default + scale · a; RSI seeds a from q_ref (train only)."""
  old = cfg.actions["joint_pos"]
  cfg.actions["joint_pos"] = DefaultOffsetJointPositionActionCfg(
    entity_name=getattr(old, "entity_name", "robot"),
    actuator_names=getattr(old, "actuator_names", (".*",)),
    scale=old.scale,
    use_default_offset=True,
    command_name="motion",
  )


def apply_end_effector(cfg: ManagerBasedRlEnvCfg) -> None:
  """Swap actor joint-reference command for existing keybody pose refs.

  Leaves rewards, RSI, terminations, and ``observations.py`` untouched.
  Critic keeps ``ref_joint_pos`` and already has ``ref_body_*``.
  """
  ref_terms = {
    "ref_body_pos": ObservationTermCfg(
      func=mdp.ref_body_pos_b,
      params=dict(_MOTION_PARAMS),
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "ref_body_ori": ObservationTermCfg(
      func=mdp.ref_body_ori_b,
      params=dict(_MOTION_PARAMS),
      noise=Unoise(n_min=-0.02, n_max=0.02),
    ),
  }

  actor = cfg.observations["actor"]
  actor.terms.pop("ref_joint_pos", None)
  actor.terms.pop("ref_joint_vel", None)
  _insert_terms_before_proprio(actor.terms, ref_terms)

  _use_default_offset_joint_action(cfg)
