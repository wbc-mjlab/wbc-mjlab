"""Robots service: registered robots (G1 + extensions), their MJCF, WBC wiring and pose.

A robot is whatever ``register_wbc_extension`` registered (or the built-in G1): an id, an
MJCF, env / RL builders and tasks. The Studio shows a card per robot (model size, tasks,
datasets, runs), the tracked keybodies / anchor / feet of its motion command on the model,
and joint sliders over the default pose.
"""

from __future__ import annotations

import re
from functools import lru_cache

import mujoco
import numpy as np

from wbc_mjlab.studio.scene import Overlay

OWNER = "robots"
KEYBODY_COLOR = np.array([54, 207, 224], np.uint8)  # --color-accent
ANCHOR_COLOR = np.array([230, 162, 60], np.uint8)  # --color-amber
FOOT_COLOR = np.array([79, 199, 138], np.uint8)  # --color-success


@lru_cache(maxsize=16)
def task_env_cfg(task_id: str):
  """Env config of a task (cached: building one costs a few seconds the first time)."""
  from wbc_mjlab.tasks import get_task_config

  return get_task_config(task_id).build_env_cfg()


def robot_tasks(robot_id: str) -> list[dict]:
  from wbc_mjlab.tasks import all_wbc_tasks

  return [
    {"id": t.task_id, "description": t.description, "experiment": t.experiment_name}
    for t in all_wbc_tasks()
    if t.robot_id == robot_id
  ]


@lru_cache(maxsize=16)
def robot_wiring(robot_id: str) -> dict:
  """Motion-command wiring of the robot's base task: tracked bodies, anchor, feet, init pose."""
  tasks = robot_tasks(robot_id)
  if not tasks:
    return {}
  base = next(
    (t["id"] for t in tasks if t["id"].lower() == f"wbc-{robot_id}"), tasks[0]["id"]
  )
  cfg = task_env_cfg(base)
  out: dict = {"task": base}
  motion = cfg.commands.get("motion") if hasattr(cfg.commands, "get") else None
  if motion is not None:
    out["anchor"] = motion.anchor_body_name
    out["bodies"] = list(motion.body_names)
  try:
    from wbc_mjlab.motion.robot_assets import get_robot_motion_spec

    _, spec = get_robot_motion_spec(robot_id)
    out["feet"] = list(spec.foot_body_names or ())
  except Exception:  # noqa: BLE001
    out["feet"] = []
  try:
    init = cfg.scene.entities["robot"].init_state
    out["init_pos"] = list(init.pos)
    out["init_rot"] = list(init.rot)
    out["init_joint_pos"] = dict(init.joint_pos or {})
  except Exception:  # noqa: BLE001
    pass
  out["episode_length_s"] = float(getattr(cfg, "episode_length_s", 0.0))
  out["step_dt"] = float(cfg.decimation * cfg.sim.mujoco.timestep)
  out["rewards"] = list(cfg.rewards.keys())
  out["terminations"] = list(cfg.terminations.keys())
  return out


def _body_id(model: mujoco.MjModel, name: str) -> int:
  return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


class RobotsService:
  def __init__(self, core):
    self.c = core
    self.qpos: np.ndarray | None = None
    self.show_wiring = True

  # ------------------------------------------------------------------ catalogue
  def catalogue(self) -> list[dict]:
    from wbc_mjlab.data_paths import list_datasets
    from wbc_mjlab.robots import custom
    from wbc_mjlab.studio import jobs

    runs = jobs.list_runs()
    added = {s.id for s in custom.list_specs()}
    out = []
    for rid in self.c.robot_names():
      tasks = robot_tasks(rid)
      experiments = {t["experiment"] for t in tasks}
      try:
        model = _model_summary(rid)
      except Exception as exc:  # noqa: BLE001
        model = {"error": str(exc)}
      out.append(
        {
          "id": rid,
          "selected": rid == self.c.robot_id,
          "tasks": len(tasks),
          "datasets": len(list_datasets(rid)),
          "runs": sum(r.experiment in experiments for r in runs),
          "builtin": rid == "g1",
          "custom": rid in added,
          **model,
        }
      )
    return out

  def info(self) -> dict:
    m = self.c.model
    if m is None:
      return {"robot": ""}
    joints = []
    q = self.qpos if self.qpos is not None else m.qpos0
    for j in range(m.njnt):
      if m.jnt_type[j] not in (
        mujoco.mjtJoint.mjJNT_HINGE,
        mujoco.mjtJoint.mjJNT_SLIDE,
      ):
        continue
      adr = int(m.jnt_qposadr[j])
      lo, hi = (m.jnt_range[j] if m.jnt_limited[j] else (-np.pi, np.pi)).tolist()
      joints.append(
        {
          "name": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j),
          "value": float(q[adr]),
          "min": float(lo),
          "max": float(hi),
          "group": _joint_group(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)),
        }
      )
    from wbc_mjlab.motion.robot_assets import get_robot_mjcf_path

    return {
      "robot": self.c.robot_id,
      "mjcf": str(get_robot_mjcf_path(self.c.robot_id)),
      "joints": joints,
      "tasks": robot_tasks(self.c.robot_id),
      "wiring": robot_wiring(self.c.robot_id),
      "show_wiring": self.show_wiring,
      **_model_summary(self.c.robot_id),
    }

  # ------------------------------------------------------------------ actions
  def select(self, name: str) -> bool:
    return self.c.run_task(f"load robot {name}", lambda: self.c.select_robot(name))

  def on_robot(self) -> None:
    self.qpos = self.default_qpos()
    self.show_pose()

  def default_qpos(self) -> np.ndarray:
    """``qpos0`` with the base task's initial joint pose and root height applied."""
    m = self.c.model
    q = m.qpos0.copy()
    try:
      w = robot_wiring(self.c.robot_id)
    except Exception:  # noqa: BLE001
      w = {}
    patterns = w.get("init_joint_pos", {})
    for j in range(m.njnt):
      if m.jnt_type[j] not in (
        mujoco.mjtJoint.mjJNT_HINGE,
        mujoco.mjtJoint.mjJNT_SLIDE,
      ):
        continue
      name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
      for pat, val in patterns.items():
        if re.fullmatch(pat, name):
          q[int(m.jnt_qposadr[j])] = float(val)
    if m.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE and "init_pos" in w:
      q[0:3] = w["init_pos"]
      q[3:7] = w.get("init_rot", (1, 0, 0, 0))
    return q

  def set_joint(self, name: str, value: float) -> None:
    m = self.c.model
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
      raise ValueError(f"no joint {name!r}")
    if self.qpos is None:
      self.qpos = self.default_qpos()
    self.qpos[int(m.jnt_qposadr[jid])] = float(value)
    self.show_pose()

  def reset_pose(self, zero: bool = False) -> None:
    self.qpos = self.c.model.qpos0.copy() if zero else self.default_qpos()
    if zero and self.c.model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE:
      self.qpos[0:3] = self.default_qpos()[0:3]
    self.show_pose()

  def set_options(self, show_wiring: bool | None = None) -> None:
    if show_wiring is not None:
      self.show_wiring = bool(show_wiring)
    self.show_pose()

  def show_pose(self) -> None:
    """Hand the scene to the robot view: one instance in the slider pose + wiring markers."""
    if self.c.model is None:
      return
    self.c.take_scene(OWNER)
    if self.qpos is None:
      self.qpos = self.default_qpos()
    self.c.show_qpos(self.qpos, OWNER)
    self.c.scene.set_hud({})
    self._wiring_overlays()

  def _wiring_overlays(self) -> None:
    keys = ("keybodies", "anchor", "feet")
    if not self.show_wiring:
      for k in keys:
        self.c.overlay(OWNER, k, None)
      return
    try:
      w = robot_wiring(self.c.robot_id)
    except Exception as exc:  # noqa: BLE001
      self.c.log(f"wiring: {exc}", "warn")
      return
    m = self.c.model
    pos = self.c.scene.body_pos[0]
    for key, names, color, size in (
      ("keybodies", w.get("bodies", []), KEYBODY_COLOR, 0.045),
      ("anchor", [w.get("anchor", "")], ANCHOR_COLOR, 0.085),
      ("feet", w.get("feet", []), FOOT_COLOR, 0.06),
    ):
      ids = [b for b in (_body_id(m, n) for n in names if n) if b >= 0]
      if ids:
        self.c.overlay(
          OWNER, key, Overlay("points", pos[ids], color, size, opacity=0.9, on_top=True)
        )
      else:
        self.c.overlay(OWNER, key, None)


def _joint_group(name: str) -> str:
  n = name.lower()
  for key, group in (
    ("wrist", "arms"),
    ("elbow", "arms"),
    ("shoulder", "arms"),
    ("waist", "torso"),
    ("torso", "torso"),
    ("head", "torso"),
    ("hip", "legs"),
    ("knee", "legs"),
    ("ankle", "legs"),
  ):
    if key in n:
      return group
  return "other"


@lru_cache(maxsize=16)
def _model_summary(robot_id: str) -> dict:
  from wbc_mjlab.studio.core import load_robot_model

  m = load_robot_model(robot_id)
  hinge = sum(
    int(t in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE))
    for t in m.jnt_type
  )
  return {
    "nq": int(m.nq),
    "nv": int(m.nv),
    "nbody": int(m.nbody),
    "dof": hinge,
    "nu": int(m.nu),
    "mass": float(m.body_subtreemass[0]),
    "floating": bool(m.njnt and m.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE),
  }


__all__ = ["RobotsService", "robot_tasks", "robot_wiring", "task_env_cfg"]
