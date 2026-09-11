"""Actor overlay: keybody pose refs instead of joint references.

Compose on top of ``apply_wbc``. Does **not** reweight rewards, RSI, or
terminations. Actor drops ``ref_joint_*`` and uses ``ref_body_pos`` /
``ref_body_ori`` (optionally restricted to a body subset, e.g. end-effectors).
Critic keeps joint refs and full keybody refs. Actions leave the motion joint
residual so deploy does not need ``q_ref``.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from wbc_mjlab.env.mdp.actions import DefaultOffsetJointPositionActionCfg
from wbc_mjlab.env.mdp.observations import ref_body_ori_b, ref_body_pos_b

_MOTION = "motion"


def apply_end_effector(
  cfg: ManagerBasedRlEnvCfg,
  *,
  body_names: tuple[str, ...] | None = None,
) -> None:
  """Swap actor joint-reference command for keybody pose refs.

  Leaves rewards, RSI, and terminations untouched. Critic keeps
  ``ref_joint_pos`` and full ``ref_body_*``.

  Args:
    body_names: Bodies included in actor ``ref_body_pos`` / ``ref_body_ori``.
      ``None`` uses all motion-command bodies. Pass end-effector names to
      restrict the actor command to EEs only.
  """
  params: dict = {"command_name": _MOTION}
  if body_names is not None:
    params["body_names"] = body_names

  actor = cfg.observations["actor"]
  actor.terms.pop("ref_joint_pos", None)
  actor.terms.pop("ref_joint_vel", None)
  actor.terms.update(
    {
      "ref_body_pos": ObservationTermCfg(
        func=ref_body_pos_b,
        params=params,
        noise=Unoise(n_min=-0.05, n_max=0.05),
      ),
      "ref_body_ori": ObservationTermCfg(
        func=ref_body_ori_b,
        params=params,
        noise=Unoise(n_min=-0.02, n_max=0.02),
      ),
    }
  )

  old = cfg.actions["joint_pos"]
  cfg.actions["joint_pos"] = DefaultOffsetJointPositionActionCfg(
    entity_name=getattr(old, "entity_name", "robot"),
    actuator_names=getattr(old, "actuator_names", (".*",)),
    scale=old.scale,
    use_default_offset=True,
    command_name=_MOTION,
  )
