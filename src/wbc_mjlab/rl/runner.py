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
from wbc_mjlab.env.mdp.sampling import load_rsi_bin_stats
from mjlab.tasks.tracking.rl.runner import MotionTrackingOnPolicyRunner

from wbc_mjlab.deploy_paths import PLAY_PARAMS_SUBDIR, PLAY_POLICY_ONNX_NAME


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

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
    registry_name: str | None = None,
  ):
    super().__init__(env, train_cfg, log_dir, device)
    self.registry_name = registry_name

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

    log_dir = Path(path).parent
    params_dir = log_dir / PLAY_PARAMS_SUBDIR
    onnx_path = params_dir / PLAY_POLICY_ONNX_NAME
    try:
      params_dir.mkdir(parents=True, exist_ok=True)
      self.export_policy_to_onnx(str(params_dir), PLAY_POLICY_ONNX_NAME)

      run_name: str = (
        wandb.run.name if self.logger.logger_type == "wandb" and wandb.run else "local"
      )
      metadata = self.build_policy_export_metadata(run_name)
      attach_metadata_to_onnx(str(onnx_path), metadata)

      if self.logger.logger_type in ["wandb"] and self.cfg["upload_model"]:
        wandb.save(str(onnx_path), base_path=str(params_dir))
        if self.registry_name is not None:
          wandb.run.use_artifact(self.registry_name)  # type: ignore
          self.registry_name = None
    except Exception as e:
      print(f"[WARN] Policy-only ONNX export failed (training continues): {e}")

    try:
      self._export_deploy_params(log_dir)
    except Exception as e:
      print(f"[WARN] Deploy params export failed (training continues): {e}")

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    infos = super().load(path, load_cfg=load_cfg, strict=strict, map_location=map_location)
    self._maybe_load_rsi_bin_stats(path)
    return infos

  def _motion_command(self) -> MotionCommand | None:
    env = cast(ManagerBasedRlEnv, self.env.unwrapped)
    if "motion" not in env.command_manager.active_terms:
      return None
    return cast(MotionCommand, env.command_manager.get_term("motion"))

  def _export_deploy_params(self, log_dir: Path) -> None:
    if int(os.environ.get("RANK", "0")) != 0:
      return
    cfg = getattr(self.env.unwrapped, "cfg", None)
    if cfg is None:
      return

    from wbc_mjlab.export.policy_bundle import export_deploy_params
    from wbc_mjlab.tasks import last_registered_robot_id

    yaml_path, rsi_path, _manifest_path = export_deploy_params(
      log_dir,
      cfg,
      cast(ManagerBasedRlEnv, self.env.unwrapped),
      robot_id=last_registered_robot_id(),
    )
    print(f"[INFO] Wrote WBC tracking params -> {yaml_path}")
    if rsi_path is not None:
      print(f"[INFO] Wrote RSI bin stats -> {rsi_path}")

  def _maybe_load_rsi_bin_stats(self, checkpoint_path: str) -> None:
    motion_cmd = self._motion_command()
    if motion_cmd is None or not motion_cmd.cfg.rsi.persist_failure_levels:
      return

    log_dir = Path(checkpoint_path).parent
    candidates = (
      log_dir / PLAY_PARAMS_SUBDIR / motion_cmd.cfg.rsi.failure_levels_filename,
      log_dir / motion_cmd.cfg.rsi.failure_levels_filename,
    )
    for src in candidates:
      if load_rsi_bin_stats(src, motion_cmd):
        print(f"[INFO] Restored RSI bin stats from {src}")
        return
