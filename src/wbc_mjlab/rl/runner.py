from __future__ import annotations

import os
from pathlib import Path
from typing import cast

import torch
import wandb
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointPositionAction
from mjlab.rl.exporter_utils import attach_metadata_to_onnx
from mjlab.rl.runner import MjlabOnPolicyRunner
from wbc_mjlab.env.mdp.commands import MotionCommand
from mjlab.tasks.tracking.rl.runner import MotionTrackingOnPolicyRunner

from wbc_mjlab.deploy_paths import PLAY_ONNX_LATEST_NAME


def _tracking_policy_onnx_metadata(
  env: ManagerBasedRlEnv,
  joint_action: JointPositionAction,
  run_path: str,
  *,
  action_mode: str,
) -> dict[str, list | str | float]:
  robot: Entity = env.scene["robot"]
  target_ids = joint_action._target_ids
  actuated_joint_names = [robot.joint_names[i] for i in target_ids]
  default_joint_pos = robot.data.default_joint_pos[0, target_ids].cpu().tolist()
  scale = joint_action._scale
  if isinstance(scale, torch.Tensor):
    action_scale: list | float = (
      scale[0].item() if scale.numel() == 1 else scale.flatten().cpu().tolist()
    )
  else:
    action_scale = scale
  return {
    "run_path": run_path,
    "policy_only_export": "true",
    "action_mode": action_mode,
    "joint_names": actuated_joint_names,
    "default_joint_pos": default_joint_pos,
    "action_scale": action_scale,
    "command_names": list(env.command_manager.active_terms),
    "observation_names": list(env.observation_manager.active_terms["actor"]),
  }


class PolicyOnlyMotionTrackingRunner(MotionTrackingOnPolicyRunner):
  """Tracking runner that exports a policy-only ONNX (obs -> actions).

  mjlab's default ``MotionTrackingOnPolicyRunner`` wraps the actor in
  ``_OnnxMotionModel``, which registers the full motion clip as ONNX constants.
  That makes the file scale with training motion length (~hundreds of MB for
  large libraries). At runtime the reference should come from an external
  source, not from inside the policy graph).
  """

  def export_policy_to_onnx(
    self, path: str, filename: str = "policy.onnx", verbose: bool = False
  ) -> None:
    """Export actor + obs normalizer only (no motion buffers in the graph)."""
    onnx_model = self.alg.get_policy().as_onnx(verbose=verbose)
    onnx_model.to("cpu")
    onnx_model.eval()
    os.makedirs(path, exist_ok=True)
    dummy_inputs = onnx_model.get_dummy_inputs()  # type: ignore[operator]
    # Do not pass dynamic_axes={}: empty dict drops static shapes in some ORT builds.
    torch.onnx.export(
      onnx_model,
      dummy_inputs,
      os.path.join(path, filename),
      export_params=True,
      opset_version=18,
      verbose=verbose,
      input_names=list(onnx_model.input_names),  # type: ignore[arg-type]
      output_names=list(onnx_model.output_names),  # type: ignore[arg-type]
      dynamo=False,
    )

  def build_policy_export_metadata(self, run_path: str) -> dict[str, list | str | float]:
    """ONNX metadata (actor obs layout + WBC motion fields)."""
    env = cast(ManagerBasedRlEnv, self.env.unwrapped)
    joint_action = env.action_manager.get_term("joint_pos")
    if isinstance(joint_action, JointPositionAction):
      metadata = _tracking_policy_onnx_metadata(
        env, joint_action, run_path, action_mode="default_relative"
      )
    elif type(joint_action).__name__ == "ReferenceJointPositionAction":
      metadata = _tracking_policy_onnx_metadata(
        env, joint_action, run_path, action_mode="reference_residual"
      )
    else:
      raise TypeError(
        f"Unsupported joint_pos action type for ONNX metadata: {type(joint_action).__name__}"
      )

    motion_term = cast(MotionCommand, env.command_manager.get_term("motion"))
    metadata.update(
      {
        "anchor_body_name": motion_term.cfg.anchor_body_name,
        "body_names": list(motion_term.cfg.body_names),
        "wbc_command_dim": motion_term.wbc_command_dim,
      }
    )
    return metadata

  def save(self, path: str, infos=None) -> None:
    """Checkpoint save + policy-only ONNX (not the motion-embedded graph)."""
    MjlabOnPolicyRunner.save(self, path, infos)

    policy_dir, filename, onnx_path = self._get_export_paths(path)
    latest_path = policy_dir / PLAY_ONNX_LATEST_NAME
    try:
      self.export_policy_to_onnx(str(policy_dir), filename)
      self.export_policy_to_onnx(str(policy_dir), PLAY_ONNX_LATEST_NAME)

      run_name: str = (
        wandb.run.name if self.logger.logger_type == "wandb" and wandb.run else "local"
      )
      metadata = self.build_policy_export_metadata(run_name)
      attach_metadata_to_onnx(str(onnx_path), metadata)
      attach_metadata_to_onnx(str(latest_path), metadata)

      if self.logger.logger_type in ["wandb"] and self.cfg["upload_model"]:
        wandb.save(str(onnx_path), base_path=str(policy_dir))
        wandb.save(str(latest_path), base_path=str(policy_dir))
        if self.registry_name is not None:
          wandb.run.use_artifact(self.registry_name)  # type: ignore
          self.registry_name = None
    except Exception as e:
      print(f"[WARN] Policy-only ONNX export failed (training continues): {e}")

    try:
      self._export_wbc_tracking_params_yaml(path)
    except Exception as e:
      print(f"[WARN] WBC tracking params YAML failed (training continues): {e}")

  def _export_wbc_tracking_params_yaml(self, log_dir: str) -> None:
    if int(os.environ.get("RANK", "0")) != 0:
      return
    cfg = getattr(self.env.unwrapped, "cfg", None)
    if cfg is None:
      return

    from wbc_mjlab.export.tracking_params_yaml import write_wbc_tracking_params_bundle
    from wbc_mjlab.tasks import last_registered_robot_id

    meta = self.build_policy_export_metadata("export")
    obs_raw = meta.get("observation_names", "")
    if isinstance(obs_raw, str):
      obs_names = [x.strip() for x in obs_raw.split(",") if x.strip()]
    else:
      obs_names = list(obs_raw)
    has_se = "motion_anchor_pos_b" in obs_names

    params_dir = Path(log_dir) / "params" / "wbc_tracking"
    write_wbc_tracking_params_bundle(
      params_dir,
      cfg,
      robot_id=last_registered_robot_id(),
      has_state_estimation=has_se,
    )
    print(f"[INFO] Wrote WBC tracking params -> {params_dir}")
