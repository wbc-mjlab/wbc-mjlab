"""Cfg-level checks for the end-effector tracking preset."""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "disable")

from wbc_mjlab.env.mdp.actions import (
  DefaultOffsetJointPositionActionCfg,
  ReferenceJointPositionActionCfg,
)
from wbc_mjlab.robots.g1.constants import G1_MOTION_BODY_NAMES
from wbc_mjlab.robots.g1.tasks import (
  g1_wbc_ee_env_cfg,
  g1_wbc_ee_se_env_cfg,
  g1_wbc_env_cfg,
)
from wbc_mjlab.tasks import list_wbc_task_ids


def test_ee_task_ids_registered() -> None:
  ids = list_wbc_task_ids()
  assert "Wbc-G1-EE" in ids
  assert "Wbc-G1-EE-SE" in ids


def test_ee_preset_replaces_actor_joint_refs_keeps_wbc_rewards() -> None:
  cfg = g1_wbc_ee_env_cfg()
  wbc = g1_wbc_env_cfg()
  actor = cfg.observations["actor"].terms
  critic = cfg.observations["critic"].terms

  assert "ref_joint_pos" not in actor
  assert "ref_joint_vel" not in actor
  assert "ref_joint_pos" in critic
  assert "ee_pos_b" not in actor
  assert "ee_ori_b" not in actor
  for name in ("ref_body_pos", "ref_body_ori"):
    assert name in actor
    assert actor[name].func is critic[name].func
    assert actor[name].params == {"command_name": "motion"}
  assert list(actor).index("ref_body_pos") < list(actor).index("joint_pos")

  assert isinstance(cfg.actions["joint_pos"], DefaultOffsetJointPositionActionCfg)
  assert not isinstance(cfg.actions["joint_pos"], ReferenceJointPositionActionCfg)
  assert cfg.actions["joint_pos"].use_default_offset is True

  for name, term in wbc.rewards.items():
    ee_term = cfg.rewards[name]
    assert ee_term.weight == term.weight
    assert ee_term.params == term.params
  assert cfg.rewards["motion_joint_pos"].weight == 1.0
  assert cfg.rewards["motion_body_pos"].params["body_names"] == G1_MOTION_BODY_NAMES
  assert cfg.terminations.keys() == wbc.terminations.keys()


def test_ee_se_keeps_body_ref_command_and_adds_anchor_error() -> None:
  cfg = g1_wbc_ee_se_env_cfg()
  actor = cfg.observations["actor"].terms
  assert "ref_body_pos" in actor
  assert "ref_joint_pos" not in actor
  assert "ref_base_height" not in actor
  assert "motion_anchor_pos_error_w" in actor
  assert "base_lin_vel" in actor
  assert isinstance(cfg.actions["joint_pos"], DefaultOffsetJointPositionActionCfg)


def test_ref_body_observation_dims() -> None:
  from wbc_mjlab.export.tracking_params_yaml import _observation_dim

  n_bodies = len(G1_MOTION_BODY_NAMES)
  assert _observation_dim("ref_body_pos", joint_count=29, body_count=n_bodies) == 3 * n_bodies
  assert _observation_dim("ref_body_ori", joint_count=29, body_count=n_bodies) == 6 * n_bodies
