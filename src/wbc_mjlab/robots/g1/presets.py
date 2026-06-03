"""G1 WBC task presets (BeyondMimic / Zest-style RSI and deploy obs)."""

from __future__ import annotations

from wbc_mjlab.tasks.preset import WbcTaskPreset

G1_WBC_TASK_PRESETS: tuple[WbcTaskPreset, ...] = (
  WbcTaskPreset(
    task_id="Wbc-G1",
    robot_id="g1",
    description="Full actor obs, whole-body RSI (similarity EMA).",
    experiment_name="wbc_g1",
    has_state_estimation=True,
    sampling_strategy="similarity_ema",
    adaptive_similarity_preset="whole_body",
  ),
  WbcTaskPreset(
    task_id="Wbc-G1-NoSE",
    robot_id="g1",
    description="Deploy-style policy obs (no anchor position), whole-body RSI.",
    experiment_name="wbc_g1_nose",
    has_state_estimation=False,
    sampling_strategy="similarity_ema",
    adaptive_similarity_preset="whole_body",
  ),
  WbcTaskPreset(
    task_id="Wbc-G1-Zest",
    robot_id="g1",
    description="No state estimation, joint-only RSI (Zest-style adaptive sampling).",
    experiment_name="wbc_g1_zest",
    has_state_estimation=False,
    sampling_strategy="similarity_ema",
    adaptive_similarity_preset="joint_only",
  ),
  WbcTaskPreset(
    task_id="Wbc-G1-BinaryFailure",
    robot_id="g1",
    description="Full obs, whole-body RSI with binary failure resampling (BeyondMimic-style).",
    experiment_name="wbc_g1_binary",
    has_state_estimation=True,
    sampling_strategy="binary_failure",
    adaptive_similarity_preset="whole_body",
  ),
)
