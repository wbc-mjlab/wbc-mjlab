"""Write WBC motion-tracking policy parameters to YAML (checkpoint artifact).

Stores the trained policy interface: joint PD, action scales, actor observation
layout, and tracking command metadata. Downstream runtimes read this file on
their own; training does not encode any particular deploy stack.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from mjlab.utils.string import resolve_expr

SCHEMA_VERSION = "wbc_tracking_params_v1"
PARAMS_FILENAME = "wbc_tracking_params.yaml"
MANIFEST_FILENAME = "wbc_tracking_params_manifest.yaml"


def _policy_step_seconds(cfg: Any) -> float:
  return float(cfg.sim.mujoco.timestep) * int(cfg.decimation)


def _joint_position_action(cfg: Any):
  for term in cfg.actions.values():
    if type(term).__name__ in (
      "JointPositionActionCfg",
      "ReferenceJointPositionActionCfg",
    ):
      return term
  raise RuntimeError("WBC tracking params: no joint position action in cfg.actions")


def _action_mode(action) -> str:
  name = type(action).__name__
  if name == "ReferenceJointPositionActionCfg":
    return "reference_residual"
  if name == "JointPositionActionCfg":
    return "default_relative"
  raise RuntimeError(f"WBC tracking params: unsupported joint action {name!r}")


def _actor_term_names(cfg: Any, *, has_state_estimation: bool) -> list[str]:
  terms = list(cfg.observations["actor"].terms.keys())
  if not has_state_estimation:
    terms = [t for t in terms if t not in ("motion_anchor_pos_b", "base_lin_vel")]
  return terms


def _actor_history_length(cfg: Any) -> int:
  """Effective actor observation history length for deploy (1 = no stacking)."""
  actor = cfg.observations["actor"]
  if actor.history_length is not None and actor.history_length > 0:
    return int(actor.history_length)
  max_term_history = max(
    (term.history_length for term in actor.terms.values() if term is not None),
    default=0,
  )
  return max(1, max_term_history)


def _resolve_scales(action, joint_names: tuple[str, ...]) -> list[float]:
  sc = action.scale
  if isinstance(sc, (int, float)):
    return [float(sc)] * len(joint_names)
  if isinstance(sc, dict):
    return list(resolve_expr(sc, joint_names, default_val=1.0))
  raise TypeError(type(sc))


def _observation_dim(name: str, *, joint_count: int, wbc_command_dim: int) -> int:
  if name == "command":
    return wbc_command_dim
  if name in ("base_ang_vel", "projected_gravity", "motion_anchor_pos_b", "base_lin_vel"):
    return 3
  if name in ("joint_pos", "joint_vel", "actions"):
    return joint_count
  raise KeyError(f"unexpected actor term {name!r}")


def _build_robot_entity(cfg: Any):
  return cfg.scene.entities["robot"].build()


def _joint_names_from_cfg(cfg: Any) -> tuple[str, ...]:
  return _build_robot_entity(cfg).joint_names


def _match_actuator(joint_name: str, entity: Any) -> Any:
  for act in entity.actuators:
    if joint_name in act.target_names:
      return act
  raise KeyError(f"No actuator matches joint {joint_name!r}")


def _actuator_gain(act: Any, joint_name: str, field: str) -> float:
  val = getattr(act.cfg, field)
  if isinstance(val, dict):
    return float(resolve_expr(val, (joint_name,), default_val=0.0)[0])
  return float(val)


def _pd_from_robot(cfg: Any, joint_names: tuple[str, ...]) -> tuple[list[float], list[float], list[float]]:
  robot_cfg = cfg.scene.entities["robot"]
  entity = _build_robot_entity(cfg)
  jp = robot_cfg.init_state.joint_pos or {}
  default_pos = [float(x) for x in resolve_expr(jp, joint_names, default_val=0.0)]
  stiffness: list[float] = []
  damping: list[float] = []
  for jn in joint_names:
    act = _match_actuator(jn, entity)
    stiffness.append(_actuator_gain(act, jn, "stiffness"))
    damping.append(_actuator_gain(act, jn, "damping"))
  return stiffness, damping, default_pos


def build_wbc_tracking_params(
  cfg: Any,
  *,
  robot_id: str,
  has_state_estimation: bool,
) -> dict[str, Any]:
  joint_names = _joint_names_from_cfg(cfg)
  action = _joint_position_action(cfg)
  action_scales = _resolve_scales(action, joint_names)
  policy_step_dt = _policy_step_seconds(cfg)
  wbc_command_dim = 10 + len(joint_names)
  actor_names = _actor_term_names(cfg, has_state_estimation=has_state_estimation)
  actor_history_length = _actor_history_length(cfg)

  actor_observations: dict[str, Any] = {}
  for name in actor_names:
    dim = _observation_dim(name, joint_count=len(joint_names), wbc_command_dim=wbc_command_dim)
    actor_observations[name] = {
      "dim": dim,
      "scale": [1.0] * dim,
      "params": {"command_name": "motion"} if name == "command" else {},
    }

  motion_cmd = cfg.commands["motion"]
  stiffness, damping, default_pos = _pd_from_robot(cfg, joint_names)

  return {
    "schema_version": SCHEMA_VERSION,
    "robot_id": robot_id,
    "policy_step_dt": policy_step_dt,
    "joint_names": list(joint_names),
    "default_joint_pos": default_pos,
    "stiffness": stiffness,
    "damping": damping,
    "action": {
      "action_mode": _action_mode(action),
      "scale": action_scales,
      "command_name": "motion",
    },
    "actor_observations": actor_observations,
    "tracking": {
      "anchor_body_name": motion_cmd.anchor_body_name,
      "has_state_estimation": has_state_estimation,
      "wbc_command_dim": wbc_command_dim,
      "actor_observation_names": actor_names,
      "actor_history_length": actor_history_length,
    },
  }


def write_wbc_tracking_params_yaml(
  path: Path,
  cfg: Any,
  *,
  robot_id: str,
  has_state_estimation: bool,
) -> dict[str, Any]:
  doc = build_wbc_tracking_params(
    cfg, robot_id=robot_id, has_state_estimation=has_state_estimation
  )
  path.parent.mkdir(parents=True, exist_ok=True)
  header = (
    f"# {SCHEMA_VERSION} robot={robot_id!r} "
    f"has_state_estimation={has_state_estimation}\n"
  )
  path.write_text(
    header + yaml.safe_dump(doc, sort_keys=False, default_flow_style=None),
    encoding="utf-8",
  )
  return doc


def write_wbc_tracking_params_bundle(
  directory: Path,
  cfg: Any,
  *,
  robot_id: str,
  has_state_estimation: bool,
) -> dict[str, Any]:
  directory.mkdir(parents=True, exist_ok=True)
  doc = write_wbc_tracking_params_yaml(
    directory / PARAMS_FILENAME,
    cfg,
    robot_id=robot_id,
    has_state_estimation=has_state_estimation,
  )
  manifest = {
    "schema_version": SCHEMA_VERSION,
    "robot_id": robot_id,
    "has_state_estimation": has_state_estimation,
    "policy_step_dt": doc["policy_step_dt"],
    "wbc_command_dim": doc["tracking"]["wbc_command_dim"],
    "params_file": PARAMS_FILENAME,
    "onnx_metadata_keys": [
      "policy_only_export",
      "action_mode",
      "joint_names",
      "default_joint_pos",
      "action_scale",
      "observation_names",
      "anchor_body_name",
      "wbc_command_dim",
      "actor_history_length",
    ],
  }
  (directory / MANIFEST_FILENAME).write_text(
    yaml.safe_dump(manifest, sort_keys=False),
    encoding="utf-8",
  )
  return doc
