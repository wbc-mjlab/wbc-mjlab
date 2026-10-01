"""Curriculum terms for WBC motion tracking."""

from __future__ import annotations

import torch
from mjlab.envs import ManagerBasedRlEnv


def terrain_levels_motion(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
) -> dict[str, torch.Tensor]:
  """Adjust terrain difficulty from the outcome of the previous episode.

  Environments that reached the time limit move up one terrain row. Environments
  that terminated early move down one row. The initial reset leaves the levels
  selected by ``max_init_terrain_level`` unchanged.
  """
  terrain = env.scene.terrain
  if terrain is None or terrain.terrain_origins is None:
    return {}

  move_up = env.termination_manager.time_outs[env_ids].clone()
  move_down = env.termination_manager.terminated[env_ids].clone() & ~move_up

  if env.common_step_counter == 0:
    move_up.zero_()
    move_down.zero_()

  terrain.update_env_origins(env_ids, move_up, move_down)

  levels = terrain.terrain_levels.float()
  return {
    "mean": levels.mean(),
    "max": levels.max(),
    "promoted": move_up.float().mean(),
    "demoted": move_down.float().mean(),
  }
