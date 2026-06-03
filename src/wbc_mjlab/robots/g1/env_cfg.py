"""G1 WBC environment configuration (instance of the shared ``Wbc`` env)."""

from __future__ import annotations

from typing import Literal

from mjlab.envs import ManagerBasedRlEnvCfg

from wbc_mjlab.env.mdp.commands import (
  AdaptiveSimilarityTermCfg,
  whole_body_adaptive_similarity_terms,
)
from wbc_mjlab.env.wbc_env_cfg import make_base_wbc_env_cfg
from wbc_mjlab.robots.g1.instance import configure_g1_wbc_env

# Default RSI preset for G1 (override per run via env cfg kwargs if needed).
DEFAULT_ADAPTIVE_SIMILARITY_TERMS = whole_body_adaptive_similarity_terms()


def g1_wbc_env_cfg(
  *,
  play: bool = False,
  has_state_estimation: bool = True,
  use_reference_residual_action: bool = True,
  sampling_strategy: Literal["binary_failure", "similarity_ema"] | None = None,
  adaptive_similarity_terms: tuple[AdaptiveSimilarityTermCfg, ...] | None = None,
) -> ManagerBasedRlEnvCfg:
  """Full G1 tracking env: shared WBC template + G1 bodies/sensors/scales."""
  del use_reference_residual_action  # always ref-residual on G1 today
  cfg = make_base_wbc_env_cfg(use_reference_residual_action=True)
  return configure_g1_wbc_env(
    cfg,
    play=play,
    has_state_estimation=has_state_estimation,
    sampling_strategy=sampling_strategy,
    adaptive_similarity_terms=adaptive_similarity_terms or DEFAULT_ADAPTIVE_SIMILARITY_TERMS,
  )
