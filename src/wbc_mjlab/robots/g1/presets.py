"""G1 WBC task presets (BeyondMimic / Zest-style RSI and deploy obs)."""

from __future__ import annotations

from wbc_mjlab.tasks.preset import WbcTaskPreset

G1_WBC_TASK_PRESETS: tuple[WbcTaskPreset, ...] = (
  WbcTaskPreset(
    task_id="Wbc-G1",
    robot_id="g1",
    description="Deploy-style obs, joint-only RSI, actor history length 10.",
    experiment_name="wbc_g1",
    env_cfg="wbc",
  ),
  WbcTaskPreset(
    task_id="Wbc-G1-NoSE",
    robot_id="g1",
    description="Deploy-style obs (no anchor position), same RSI as Wbc-G1 without history.",
    experiment_name="wbc_g1_nose",
    env_cfg="nose",
  ),
  WbcTaskPreset(
    task_id="Wbc-G1-Zest",
    robot_id="g1",
    description="No state estimation, joint-only RSI (Zest-style adaptive sampling).",
    experiment_name="wbc_g1_zest",
    env_cfg="zest",
  ),
  WbcTaskPreset(
    task_id="Wbc-G1-BinaryFailure",
    robot_id="g1",
    description="Full obs, whole-body RSI with binary failure resampling (BeyondMimic-style).",
    experiment_name="wbc_g1_binary",
    env_cfg="binary_failure",
  ),
)
