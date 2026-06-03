"""Shared WBC task preset type (one mjlab task id per preset)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from wbc_mjlab.robots.ids import RobotId

SamplingStrategy = Literal["binary_failure", "similarity_ema"]
SimilarityPreset = Literal["whole_body", "joint_only"]


@dataclass(frozen=True)
class WbcTaskPreset:
  """One registered mjlab task: env knobs + separate log experiment name."""

  task_id: str
  robot_id: RobotId
  description: str
  experiment_name: str
  has_state_estimation: bool = True
  sampling_strategy: SamplingStrategy | None = None
  adaptive_similarity_preset: SimilarityPreset | None = "whole_body"

  def env_kwargs(self) -> dict:
    from wbc_mjlab.env.mdp.commands import (
      whole_body_adaptive_similarity_terms,
      wbc_joint_only_similarity_terms,
    )

    kwargs: dict = {"has_state_estimation": self.has_state_estimation}
    if self.sampling_strategy is not None:
      kwargs["sampling_strategy"] = self.sampling_strategy
    preset = self.adaptive_similarity_preset
    if preset == "joint_only":
      kwargs["adaptive_similarity_terms"] = wbc_joint_only_similarity_terms()
    elif preset == "whole_body":
      kwargs["adaptive_similarity_terms"] = whole_body_adaptive_similarity_terms()
    return kwargs
