"""Tests for external robot/task registration API."""

from __future__ import annotations

from wbc_mjlab.extension import WbcRobotSpec, register_robot, register_wbc_extension
from wbc_mjlab.motion.robot_assets import RobotMotionSpec, get_robot_motion_spec
from wbc_mjlab.robots.env import make_wbc_env_cfg, make_wbc_rl_cfg
from wbc_mjlab.robots.ids import known_robot_ids, resolve_robot_id
from wbc_mjlab.tasks import get_task_config, list_wbc_task_ids, register_wbc_tasks
from wbc_mjlab.tasks.config import WbcTaskConfig


def _dummy_env_cfg(*, play: bool = False, task_id: str = "Wbc-Testbot", **kwargs):
  del play, task_id, kwargs
  return make_wbc_env_cfg("g1", task_id="Wbc-G1")


def _dummy_rl_cfg():
  return make_wbc_rl_cfg("g1")


def test_register_external_robot_and_task() -> None:
  register_robot(
    WbcRobotSpec(
      robot_id="testbot",
      aliases=("tb",),
      make_env_cfg=_dummy_env_cfg,
      make_rl_cfg=_dummy_rl_cfg,
      motion_spec=RobotMotionSpec(scene_cfg_fn=lambda: make_wbc_env_cfg("g1").scene),
    )
  )
  register_wbc_tasks(
    WbcTaskConfig(
      task_id="Wbc-Testbot",
      robot_id="testbot",
      description="smoke",
      experiment_name="wbc_testbot",
      build_env_cfg=lambda: _dummy_env_cfg(),
    )
  )

  assert "testbot" in known_robot_ids()
  assert resolve_robot_id("tb") == "testbot"
  _, motion_spec = get_robot_motion_spec("testbot")
  assert motion_spec.scene_cfg_fn is not None
  assert "Wbc-Testbot" in list_wbc_task_ids()
  assert get_task_config("Wbc-Testbot").robot_id == "testbot"


def test_register_wbc_extension_helper() -> None:
  register_wbc_extension(
    WbcRobotSpec(
      robot_id="testbot2",
      make_env_cfg=_dummy_env_cfg,
      make_rl_cfg=_dummy_rl_cfg,
    ),
    WbcTaskConfig(
      task_id="Wbc-Testbot2",
      robot_id="testbot2",
      description="smoke",
      experiment_name="wbc_testbot2",
      build_env_cfg=lambda: _dummy_env_cfg(),
    ),
  )
  assert resolve_robot_id("testbot2") == "testbot2"
  assert get_task_config("Wbc-Testbot2").experiment_name == "wbc_testbot2"
