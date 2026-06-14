"""G1 WBC task configs: one entry per mjlab task (metadata + env builder)."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg

from wbc_mjlab.env.mdp.commands import MotionCommandCfg
from wbc_mjlab.robots.g1.configs.binary_failure import g1_wbc_binary_failure_env_cfg
from wbc_mjlab.robots.g1.configs.wbc import g1_wbc_env_cfg
from wbc_mjlab.robots.g1.configs.zest import g1_wbc_zest_env_cfg, g1_wbc_zest_se_env_cfg
from wbc_mjlab.tasks.config import WbcTaskConfig

G1_WBC_TASKS: tuple[WbcTaskConfig, ...] = (
  WbcTaskConfig(
    task_id="Wbc-G1",
    robot_id="g1",
    description=(
      "Zest Table S4 tracking + RSI, EE z resets, light foot slip / anti-shake, history=1."
    ),
    experiment_name="wbc_g1",
    build_env_cfg=g1_wbc_env_cfg,
  ),
  WbcTaskConfig(
    task_id="Wbc-G1-Zest",
    robot_id="g1",
    description="Zest paper repro: no SE, reward-aligned RSI.",
    experiment_name="wbc_g1_zest",
    build_env_cfg=g1_wbc_zest_env_cfg,
  ),
  WbcTaskConfig(
    task_id="Wbc-G1-Zest-SE",
    robot_id="g1",
    description="Zest rewards/RSI with motion_anchor_pos_b + base_lin_vel only.",
    experiment_name="wbc_g1_zest_se",
    build_env_cfg=g1_wbc_zest_se_env_cfg,
  ),
  WbcTaskConfig(
    task_id="Wbc-G1-BinaryFailure",
    robot_id="g1",
    description="Full obs, whole-body RSI with binary failure (BeyondMimic paper).",
    experiment_name="wbc_g1_binary",
    build_env_cfg=g1_wbc_binary_failure_env_cfg,
  ),
)

G1_TASK_BY_ID: dict[str, WbcTaskConfig] = {t.task_id: t for t in G1_WBC_TASKS}

DEFAULT_G1_TASK_ID = "Wbc-G1"


def get_g1_task_config(task_id: str = DEFAULT_G1_TASK_ID) -> WbcTaskConfig:
  try:
    return G1_TASK_BY_ID[task_id]
  except KeyError as exc:
    known = ", ".join(sorted(G1_TASK_BY_ID))
    raise KeyError(f"Unknown G1 task {task_id!r}. Known: {known}") from exc


def make_g1_wbc_env_cfg(
  *,
  play: bool = False,
  task_id: str = DEFAULT_G1_TASK_ID,
  **kwargs,
) -> ManagerBasedRlEnvCfg:
  if kwargs:
    unknown = ", ".join(sorted(kwargs))
    raise TypeError(
      f"Unknown env cfg kwargs for G1: {unknown}. Pass task_id=<Wbc-G1-…>."
    )
  cfg = get_g1_task_config(task_id).build_env_cfg()

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.curriculum = {}
    cfg.events.pop("push_robot", None)
    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
    motion_cmd.assistive_wrench_enabled = False
    if "assistive_wrench" in cfg.events:
      cfg.events["assistive_wrench"].params["enabled"] = False

  return cfg


__all__ = [
  "DEFAULT_G1_TASK_ID",
  "G1_TASK_BY_ID",
  "G1_WBC_TASKS",
  "get_g1_task_config",
  "g1_wbc_binary_failure_env_cfg",
  "g1_wbc_env_cfg",
  "g1_wbc_zest_env_cfg",
  "g1_wbc_zest_se_env_cfg",
  "make_g1_wbc_env_cfg",
]
