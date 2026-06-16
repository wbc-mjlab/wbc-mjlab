"""Write deploy-facing artifacts under ``<run_dir>/params/``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from mjlab.envs import ManagerBasedRlEnv

from wbc_mjlab.deploy_paths import PLAY_CONFIG_NAME, PLAY_PARAMS_SUBDIR
from wbc_mjlab.env.mdp.commands import MotionCommand
from wbc_mjlab.env.mdp.sampling import save_rsi_bin_stats
from wbc_mjlab.export.tracking_params_yaml import write_wbc_tracking_params_yaml
from wbc_mjlab.motion.manifest import (
  build_motion_library_manifest,
  export_motion_library_manifest,
)


def actor_has_state_estimation(cfg: Any) -> bool:
  terms = cfg.observations["actor"].terms
  return (
    "root_pos_w" in terms
    and "root_ori_6d" in terms
    and "base_lin_vel" in terms
  )


def export_tracking_params_yaml(
  params_dir: Path,
  cfg: Any,
  *,
  robot_id: str,
) -> Path:
  """Write ``params/config.yaml`` alongside policy ONNX."""
  params_dir.mkdir(parents=True, exist_ok=True)
  out = params_dir / PLAY_CONFIG_NAME
  write_wbc_tracking_params_yaml(
    out,
    cfg,
    robot_id=robot_id,
    has_state_estimation=actor_has_state_estimation(cfg),
  )
  return out


def export_rsi_bin_stats(params_dir: Path, env: ManagerBasedRlEnv) -> Path | None:
  """Write ``params/rsi_bin_stats.npz`` when adaptive RSI persistence is enabled."""
  if "motion" not in env.command_manager.active_terms:
    return None

  motion_cmd = cast(MotionCommand, env.command_manager.get_term("motion"))
  if not motion_cmd.cfg.rsi.persist_failure_levels:
    return None

  params_dir.mkdir(parents=True, exist_ok=True)
  return save_rsi_bin_stats(
    params_dir / motion_cmd.cfg.rsi.failure_levels_filename,
    motion_cmd,
  )


def export_motion_library_manifest_from_env(
  params_dir: Path, env: ManagerBasedRlEnv
) -> Path | None:
  """Write ``params/motion_library.yaml`` from the loaded motion library."""
  if "motion" not in env.command_manager.active_terms:
    return None

  motion_cmd = cast(MotionCommand, env.command_manager.get_term("motion"))
  if not motion_cmd.cfg.motion_file:
    return None

  params_dir.mkdir(parents=True, exist_ok=True)
  manifest = build_motion_library_manifest(motion_cmd.cfg.motion_file)
  return export_motion_library_manifest(params_dir, manifest)


def export_deploy_params(
  log_dir: Path,
  cfg: Any,
  env: ManagerBasedRlEnv,
  *,
  robot_id: str,
  write_motion_library: bool = False,
) -> tuple[Path, Path | None, Path | None]:
  """Write deploy params next to ``policy.onnx`` (config, RSI stats, optional manifest)."""
  params_dir = log_dir / PLAY_PARAMS_SUBDIR
  yaml_path = export_tracking_params_yaml(params_dir, cfg, robot_id=robot_id)
  rsi_path = export_rsi_bin_stats(params_dir, env)
  manifest_path = (
    export_motion_library_manifest_from_env(params_dir, env)
    if write_motion_library
    else None
  )
  return yaml_path, rsi_path, manifest_path
