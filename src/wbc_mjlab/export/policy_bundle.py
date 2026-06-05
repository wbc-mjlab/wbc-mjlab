"""Write deploy-facing artifacts next to ``latest.onnx`` under ``params/``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from wbc_mjlab.deploy_paths import PLAY_TRACKING_PARAMS_NAME
from wbc_mjlab.export.tracking_params_yaml import write_wbc_tracking_params_yaml


def actor_has_state_estimation(cfg: Any) -> bool:
  return "motion_anchor_pos_b" in cfg.observations["actor"].terms


def export_tracking_params_yaml(
  params_dir: Path,
  cfg: Any,
  *,
  robot_id: str,
) -> Path:
  """Write ``params/wbc_tracking_params.yaml`` alongside policy ONNX."""
  params_dir.mkdir(parents=True, exist_ok=True)
  out = params_dir / PLAY_TRACKING_PARAMS_NAME
  write_wbc_tracking_params_yaml(
    out,
    cfg,
    robot_id=robot_id,
    has_state_estimation=actor_has_state_estimation(cfg),
  )
  return out
