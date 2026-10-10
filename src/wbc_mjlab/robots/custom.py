"""Robots from a model file + a YAML spec, without code (the Studio's *Add robot*).

Each robot is a folder in ``<config>/robots/<id>/`` (:func:`wbc_mjlab.user_config.config_dir`,
``~/.config/wbc-mjlab``)::

  robot.yaml       the spec: model file, anchor / keybodies / end-effectors / feet, PD gains
                   per joint, initial pose (:class:`CustomRobotSpec`)
  <model>.xml|urdf the uploaded model and its meshes, untouched
  wbc_robot.xml    the model as wbc-mjlab uses it, written next to the source on save:
                   actuators, keyframes and world geoms removed (wbc-mjlab adds its own PD
                   actuators and terrain), a free joint on the root, and the sensors the WBC
                   tasks read (``imu_ang_vel`` / ``imu_lin_vel`` IMU, ``root_angmom``) when the
                   model has none

:func:`register_all` (run by the ``mjlab.tasks`` entry point, so in every CLI and the Studio)
registers each robot like an extension: robot id ``<id>``, tasks ``Wbc-<Id>`` (the default WBC
stack), ``Wbc-<Id>-SE`` (state-estimation actor) and ``Wbc-<Id>-Terrain-Scan`` (curriculum
terrain + height scan), data in ``data/<id>/``.
"""

from __future__ import annotations

import dataclasses
import math
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
import yaml

SPEC_FILE = "robot.yaml"
PREPARED = "wbc_robot.xml"
ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
MODEL_SUFFIXES = (".xml", ".urdf", ".mjcf")
IMU_SITE = "imu"
GYRO, VELOCIMETER, ANGMOM = "imu_ang_vel", "imu_lin_vel", "root_angmom"
NATURAL_FREQ_HZ = 10.0  # PD design as in the H2 extension: ω = 10 Hz, ζ = 2
DAMPING_RATIO = 2.0
DEFAULT_ARMATURE = 0.01
DEFAULT_EFFORT = 100.0


@dataclass
class JointGains:
  effort: float = DEFAULT_EFFORT
  """Peak torque magnitude (N·m)."""
  stiffness: float = 0.0
  damping: float = 0.0
  armature: float = DEFAULT_ARMATURE
  """Reflected rotor inertia (kg·m²)."""


@dataclass
class CustomRobotSpec:
  id: str
  model: str
  """Source model file, relative to the robot folder."""
  anchor_body: str
  keybodies: list[str]
  """Bodies the motion command tracks (rewards, RSI, terminations)."""
  ee_bodies: list[str]
  """End-effectors (EE height termination, the EE task's actor command)."""
  feet_bodies: list[str]
  foot_sites: list[str] = field(default_factory=list)
  """Sites under the feet for the foot-slip reward (none: the reward is dropped)."""
  foot_geoms: list[str] = field(default_factory=list)
  """Named collision geoms of the feet for friction randomization (none: not randomized)."""
  foot_sole_z: float = 0.0
  """Sole below the foot bodies (m), for z debias in conversion."""
  init_height: float = 1.0
  init_joint_pos: dict[str, float] = field(default_factory=dict)
  joints: dict[str, JointGains] = field(default_factory=dict)
  action_scale: float = 0.25
  """Action = scale × effort / stiffness per joint (as G1 / H2)."""
  description: str = ""
  nconmax: int = 64
  njmax: int = 400

  # ------------------------------------------------------------------ paths
  @property
  def folder(self) -> Path:
    return robots_dir() / self.id

  @property
  def source(self) -> Path:
    return self.folder / self.model

  @property
  def prepared(self) -> Path:
    return self.source.parent / PREPARED

  @property
  def task_id(self) -> str:
    return task_id_for(self.id)

  def to_dict(self) -> dict:
    d = dataclasses.asdict(self)
    d["joints"] = {k: dataclasses.asdict(v) for k, v in self.joints.items()}
    return d

  @classmethod
  def from_dict(cls, d: dict) -> CustomRobotSpec:
    d = dict(d)
    d["joints"] = {k: JointGains(**v) for k, v in (d.get("joints") or {}).items()}
    known = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in known})


def robots_dir() -> Path:
  from wbc_mjlab.user_config import config_dir

  return config_dir() / "robots"


def task_id_for(robot_id: str) -> str:
  """``h1`` → ``Wbc-H1``, ``h2x`` → ``Wbc-H2X``, ``my_bot`` → ``Wbc-My-Bot``."""
  parts = [
    p.upper() if len(p) <= 3 and any(c.isdigit() for c in p) else p.capitalize()
    for p in robot_id.split("_")
    if p
  ]
  return "Wbc-" + "-".join(parts)


def check_id(robot_id: str) -> str:
  if not ID_RE.match(robot_id or ""):
    raise ValueError(
      "robot ids: lowercase letters, digits and '_', starting with a letter"
    )
  return robot_id


# ---------------------------------------------------------------------- model preparation
def _is_urdf(path: Path) -> bool:
  return path.suffix.lower() == ".urdf"


MESH_FORMATS = (".stl", ".obj", ".msh")


def _mesh_setup(path: Path, spec: mujoco.MjSpec) -> None:
  """URDF meshes come as ``package://pkg/…``, absolute or relative paths: point MuJoCo at the
  folder (around the URDF) they resolve from, or strip paths and use the folder holding them.
  MuJoCo reads STL / OBJ / MSH only."""
  names = re.findall(r'filename="([^"]+)"', path.read_text(errors="replace"))
  if not names:
    return
  bad = sorted({Path(n).suffix.lower() for n in names} - set(MESH_FORMATS))
  if bad:
    raise ValueError(
      f"MuJoCo can't read {', '.join(bad)} meshes: convert them to .stl or .obj (e.g. with "
      "Blender or trimesh), or use the robot's MJCF"
    )
  rel = names[0].split("://", 1)[-1]
  if "://" in names[0]:
    rel = rel.split("/", 1)[-1] if "/" in rel else rel  # drop the package name
  roots = [path.parent, *path.parents[:3]]
  for root in roots:
    if (root / rel).is_file():
      spec.meshdir = str(root)
      if "://" in names[0]:
        spec.strippath = False
      return
  base = Path(rel).name
  for root in roots:
    hit = next(root.rglob(base), None)
    if hit is not None:
      spec.meshdir = str(hit.parent)
      spec.strippath = True
      return


def _root_body(spec: mujoco.MjSpec):
  """The robot: the world body's child with the largest subtree (scenes may add props)."""

  def size(b) -> int:
    return 1 + sum(size(c) for c in b.bodies)

  bodies = list(spec.worldbody.bodies)
  if not bodies:
    raise ValueError("the model has no bodies")
  return max(bodies, key=size)


def _actuator_efforts(spec: mujoco.MjSpec) -> dict[str, float]:
  """Peak torque per joint from the model's own actuators (before they are removed)."""
  out: dict[str, float] = {}
  for a in spec.actuators:
    if a.trntype != mujoco.mjtTrn.mjTRN_JOINT or not a.target:
      continue
    gear = abs(float(a.gear[0])) or 1.0
    if a.forcelimited == mujoco.mjtLimited.mjLIMITED_TRUE or (
      a.forcerange[0] != 0 or a.forcerange[1] != 0
    ):
      eff = max(abs(a.forcerange[0]), abs(a.forcerange[1]))
    elif a.biastype == mujoco.mjtBias.mjBIAS_NONE and (
      a.ctrlrange[0] or a.ctrlrange[1]
    ):
      eff = max(abs(a.ctrlrange[0]), abs(a.ctrlrange[1])) * gear  # motor
    else:
      continue
    if eff > 0:
      out[a.target] = float(eff)
  for j in spec.joints:
    if j.actfrcrange[0] or j.actfrcrange[1]:
      out.setdefault(j.name, float(max(abs(j.actfrcrange[0]), abs(j.actfrcrange[1]))))
  return out


def load_source(path: Path) -> mujoco.MjSpec:
  path = Path(path)
  spec = mujoco.MjSpec.from_file(str(path))
  if _is_urdf(path):
    _mesh_setup(path, spec)
    spec.compiler.discardvisual = False
  return spec


def prepare(path: Path) -> tuple[mujoco.MjSpec, dict]:
  """The model as wbc-mjlab uses it (see the module docstring) and what was changed."""
  spec = load_source(path)
  notes: list[str] = []
  efforts = _actuator_efforts(spec)
  if spec.actuators:
    notes.append(
      f"removed {len(spec.actuators)} actuators (wbc-mjlab adds PD actuators)"
    )
    for a in list(spec.actuators):
      spec.delete(a)
  if spec.keys:
    notes.append(f"removed {len(spec.keys)} keyframes")
    for k in list(spec.keys):
      spec.delete(k)
  root = _root_body(spec)
  extra = [g for g in spec.worldbody.geoms] + [
    b for b in spec.worldbody.bodies if b.name != root.name
  ]
  if extra:
    notes.append(f"removed {len(extra)} world geoms / bodies (floor, props)")
    for e in extra:
      spec.delete(e)
  joints = list(root.joints)
  if not joints:
    root.add_freejoint(name="floating_base_joint")
    notes.append(f"added a free joint to {root.name}")
  elif joints[0].type != mujoco.mjtJoint.mjJNT_FREE:
    raise ValueError(
      f"the root body {root.name} has a non-free joint: fixed-base robots aren't supported"
    )
  have = {(s.type, s.name) for s in spec.sensors}
  names = {s.name for s in spec.sensors}
  if not any(t == mujoco.mjtSensor.mjSENS_GYRO for t, _ in have) or GYRO not in names:
    site = spec.site(IMU_SITE)  # the model's own IMU site, wherever it is
    if site is None:
      site = root.add_site(name=IMU_SITE, pos=[0, 0, 0], size=[0.01, 0.01, 0.01])
    if GYRO not in names:
      spec.add_sensor(
        name=GYRO,
        type=mujoco.mjtSensor.mjSENS_GYRO,
        objtype=mujoco.mjtObj.mjOBJ_SITE,
        objname=site.name,
      )
    if VELOCIMETER not in names:
      spec.add_sensor(
        name=VELOCIMETER,
        type=mujoco.mjtSensor.mjSENS_VELOCIMETER,
        objtype=mujoco.mjtObj.mjOBJ_SITE,
        objname=site.name,
      )
    notes.append(f"added IMU sensors {GYRO} / {VELOCIMETER} on {root.name}")
  if ANGMOM not in names:
    spec.add_sensor(
      name=ANGMOM,
      type=mujoco.mjtSensor.mjSENS_SUBTREEANGMOM,
      objtype=mujoco.mjtObj.mjOBJ_BODY,
      objname=root.name,
    )
    notes.append(f"added {ANGMOM} sensor on {root.name}")
  return spec, {"notes": notes, "efforts": efforts, "root": root.name}


# ---------------------------------------------------------------------- inspection
_HINGE = (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)


def _name(m, kind, i) -> str:
  return mujoco.mj_id2name(m, kind, i) or ""


def _subtree(m: mujoco.MjModel, b: int) -> list[int]:
  return [i for i in range(m.nbody) if _is_descendant(m, i, b)]


def _is_descendant(m: mujoco.MjModel, i: int, b: int) -> bool:
  while i > 0:
    if i == b:
      return True
    i = int(m.body_parentid[i])
  return b == 0


def _geom_bottom(m: mujoco.MjModel, d: mujoco.MjData, g: int) -> float:
  """Lowest world z of a geom (exact for meshes, boxes, spheres, capsules, cylinders)."""
  t, size = m.geom_type[g], m.geom_size[g]
  rot, pos = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g]
  if t == mujoco.mjtGeom.mjGEOM_MESH:
    mesh = int(m.geom_dataid[g])
    a, n = int(m.mesh_vertadr[mesh]), int(m.mesh_vertnum[mesh])
    return float((m.mesh_vert[a : a + n] @ rot.T + pos)[:, 2].min())
  if t == mujoco.mjtGeom.mjGEOM_SPHERE:
    return float(pos[2] - size[0])
  if t == mujoco.mjtGeom.mjGEOM_BOX:
    return float(pos[2] - np.abs(rot[2]) @ size)
  if t in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
    axis_z = abs(rot[2, 2]) * size[1]
    radial = size[0] * (
      1.0
      if t == mujoco.mjtGeom.mjGEOM_CAPSULE
      else math.sqrt(max(0.0, 1 - rot[2, 2] ** 2))
    )
    return float(pos[2] - axis_z - radial)
  if t == mujoco.mjtGeom.mjGEOM_PLANE:
    return float("inf")
  return float(pos[2] - m.geom_rbound[g])


def _default_qpos(
  m: mujoco.MjModel, keys: dict[str, np.ndarray]
) -> tuple[np.ndarray, str]:
  for k in ("home", "stand", "standing", "init", "default", "knees_bent"):
    if k in keys:
      return keys[k].copy(), k
  q = m.qpos0.copy()
  for j in range(m.njnt):
    if m.jnt_type[j] in _HINGE and m.jnt_limited[j]:
      lo, hi = m.jnt_range[j]
      adr = m.jnt_qposadr[j]
      q[adr] = min(max(q[adr], lo), hi)
  return q, ""


# PD defaults when the model has no armature / effort: G1's per joint class (armature of its
# motors, peak torque), scaled by robot mass; gains then follow the H2 rule (ω = 10 Hz, ζ = 2).
G1_MASS = 35.0
_JOINT_CLASSES = (
  (r"knee", 0.0251, 139.0),
  (r"hip", 0.0177, 88.0),
  (r"ankle|foot|toe", 0.0072, 50.0),
  (r"waist|torso|spine|trunk|chest", 0.0087, 88.0),
  (r"wrist|hand|finger|grip", 0.0043, 5.0),
  (r"shoulder|elbow|arm", 0.0036, 25.0),
  (r"head|neck", 0.0036, 10.0),
)


def joint_class(name: str) -> tuple[float, float]:
  """(armature, peak torque) of a G1-sized joint of this kind."""
  for pat, arm, eff in _JOINT_CLASSES:
    if re.search(pat, name, re.I):
      return arm, eff
  return 0.0072, 50.0


def _pick(names: list[str], *patterns: str) -> list[str]:
  return [n for n in names if any(re.search(p, n, re.I) for p in patterns)]


def suggest(m: mujoco.MjModel, d: mujoco.MjData, root: int) -> dict:
  """Best-guess wiring from body names and the kinematic tree."""
  bodies = [_name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)]
  children = {i: [] for i in range(m.nbody)}
  for i in range(1, m.nbody):
    children[int(m.body_parentid[i])].append(i)
  leaves = [i for i in range(1, m.nbody) if not children[i] and m.body_mass[i] > 0]
  root_name = bodies[root]
  torso = _pick(
    bodies, r"^torso", r"torso_link$", r"chest", r"^trunk$", r"^waist_?link"
  )
  anchor = torso[0] if torso else root_name

  def lowest(ids: list[int]) -> list[int]:
    return sorted(ids, key=lambda i: d.xpos[i][2])

  def chain(i: int) -> list[int]:
    out = []
    while i > 0:
      out.append(i)
      i = int(m.body_parentid[i])
    return out[::-1]

  # feet: named ankle / foot bodies, deepest per side; else the two lowest leaves
  named = _pick(bodies, r"ankle_roll", r"foot", r"ankle", r"toe")
  feet: list[str] = []
  for side in (r"(^|_)l(eft)?(_|$)|^left", r"(^|_)r(ight)?(_|$)|^right"):
    cands = [n for n in named if re.search(side, n, re.I)]
    if cands:
      feet.append(max(cands, key=lambda n: len(chain(bodies.index(n)))))
  if len(feet) < 2:
    feet = [bodies[i] for i in lowest(leaves)[:2]]
  hand_named = _pick(bodies, r"wrist_yaw", r"hand", r"palm", r"wrist", r"gripper")
  hands: list[str] = []
  for side in (r"(^|_)l(eft)?(_|$)|^left", r"(^|_)r(ight)?(_|$)|^right"):
    cands = [n for n in hand_named if re.search(side, n, re.I)]
    if cands:
      hands.append(max(cands, key=lambda n: len(chain(bodies.index(n)))))
  if len(hands) < 2:
    # the deepest leaf on each side (y > 0 left, y < 0 right) that isn't a foot or a head
    not_hand = re.compile(r"head|neck|camera|lidar|imu|sensor|eye", re.I)
    cands = [
      i for i in leaves if bodies[i] not in feet and not not_hand.search(bodies[i])
    ]
    hands = []
    for sign in (1, -1):
      side = [i for i in cands if sign * (d.xpos[i][1] - d.xpos[root][1]) > 0.02]
      if side:
        hands.append(bodies[max(side, key=lambda i: (len(chain(i)), -d.xpos[i][2]))])
  ees = [*feet, *hands]
  # keybodies: root, anchor and 3 bodies per limb — proximal (hip / shoulder), middle
  # (knee / elbow), the end-effector; by name, else evenly along the chain
  keys = [root_name] + ([anchor] if anchor != root_name else [])
  stages = (
    (r"hip_roll", r"shoulder_roll", r"hip", r"shoulder", r"thigh", r"upper_?arm"),
    (r"knee", r"elbow", r"shin", r"calf", r"fore_?arm", r"lower_?arm"),
  )
  for ee in ees:
    c = [i for i in chain(bodies.index(ee)) if i != root and bodies[i] != anchor]
    picks = []
    for k, pats in enumerate(stages):
      hit = next((i for p in pats for i in c if re.search(p, bodies[i], re.I)), None)
      picks.append(
        hit
        if hit is not None
        else c[min(len(c) - 1, (k + 1) * len(c) // 3 - 1)]
        if c
        else None
      )
    picks.append(bodies.index(ee))
    for i in picks:
      if i is not None and bodies[i] not in keys:
        keys.append(bodies[i])
  sites = [_name(m, mujoco.mjtObj.mjOBJ_SITE, s) for s in range(m.nsite)]
  foot_ids = [bodies.index(f) for f in feet]
  foot_sites = [
    s
    for i, s in enumerate(sites)
    if s
    and any(_is_descendant(m, int(m.site_bodyid[i]), f) for f in foot_ids)
    and s != IMU_SITE
  ]
  foot_geoms = []
  sole = 0.0
  for f in foot_ids:
    gs = [
      g
      for g in range(m.ngeom)
      if _is_descendant(m, int(m.geom_bodyid[g]), f)
      and (m.geom_contype[g] or m.geom_conaffinity[g])
    ]
    foot_geoms += [n for n in (_name(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in gs) if n]
    if gs:
      sole = max(sole, float(d.xpos[f][2] - min(_geom_bottom(m, d, g) for g in gs)))
  return {
    "anchor_body": anchor,
    "keybodies": keys,
    "ee_bodies": ees,
    "feet_bodies": feet,
    "foot_sites": foot_sites[: len(feet)]
    if len(foot_sites) >= len(feet)
    else foot_sites,
    "foot_geoms": foot_geoms,
    "foot_sole_z": round(sole, 4),
  }


def inspect_model(path: Path) -> dict:
  """What the Studio's *Add robot* form needs: the tree, joints with suggested PD gains,
  sites and a suggested wiring (all computed on the prepared model)."""
  spec, info = prepare(path)
  m = spec.compile()
  keys = {}
  src = load_source(path)
  try:
    sm = src.compile()
    for k in range(sm.nkey):
      if sm.nq == m.nq:
        keys[_name(sm, mujoco.mjtObj.mjOBJ_KEY, k)] = sm.key_qpos[k].copy()
  except Exception:  # noqa: BLE001
    pass
  q, key_name = _default_qpos(m, keys)
  d = mujoco.MjData(m)
  q[0:3] = 0.0
  q[3:7] = (1, 0, 0, 0)
  d.qpos[:] = q
  mujoco.mj_kinematics(m, d)
  bottoms = [
    _geom_bottom(m, d, g)
    for g in range(m.ngeom)
    if m.geom_contype[g] or m.geom_conaffinity[g]
  ]
  if not bottoms:
    bottoms = [_geom_bottom(m, d, g) for g in range(m.ngeom)]
  floor = min(bottoms) if bottoms else 0.0
  tops = [float(d.xpos[i][2]) for i in range(1, m.nbody)]
  root = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, info["root"])
  w = 2.0 * math.pi * NATURAL_FREQ_HZ
  scale = float(m.body_mass.sum()) / G1_MASS
  joints = []
  for j in range(m.njnt):
    if m.jnt_type[j] not in _HINGE:
      continue
    name = _name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
    cls_arm, cls_eff = joint_class(name)
    arm = float(m.dof_armature[m.jnt_dofadr[j]]) or cls_arm * scale
    eff = float(info["efforts"].get(name, round(cls_eff * scale, 1)))
    lo, hi = m.jnt_range[j] if m.jnt_limited[j] else (-math.pi, math.pi)
    joints.append(
      {
        "name": name,
        "range": [float(lo), float(hi)],
        "default": float(q[m.jnt_qposadr[j]]),
        "effort": eff,
        "armature": arm,
        "armature_from_model": bool(m.dof_armature[m.jnt_dofadr[j]]),
        "effort_from_model": name in info["efforts"],
        "stiffness": arm * w * w,
        "damping": 2.0 * DAMPING_RATIO * arm * w,
      }
    )
  bodies = []
  for i in range(1, m.nbody):
    bodies.append(
      {
        "name": _name(m, mujoco.mjtObj.mjOBJ_BODY, i),
        "parent": _name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.body_parentid[i])),
        "mass": float(m.body_mass[i]),
        "z": float(d.xpos[i][2] - floor),
      }
    )
  ncollide = sum(
    1 for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]
  )
  return {
    "model": str(path),
    "notes": info["notes"],
    "root": info["root"],
    "bodies": bodies,
    "joints": joints,
    "sites": [_name(m, mujoco.mjtObj.mjOBJ_SITE, s) for s in range(m.nsite)],
    "keyframe": key_name,
    "mass": float(m.body_mass.sum()),
    "height": float(max(tops) - floor) if tops else 0.0,
    "init_height": round(float(-floor) + 0.01, 3),
    "dof": len(joints),
    "collision_geoms": ncollide,
    "pd": {"natural_freq_hz": NATURAL_FREQ_HZ, "damping_ratio": DAMPING_RATIO},
    "suggested": suggest(m, d, root)
    | {"nconmax": max(35, 4 * ncollide), "njmax": max(250, 12 * ncollide)},
  }


# ---------------------------------------------------------------------- check
def settle_test(spec: CustomRobotSpec, seconds: float = 0.5) -> dict:
  """Drop the robot in its initial pose onto a floor and hold the pose with the spec's PD
  gains (CPU MuJoCo, no policy) for ``seconds``. Plain PD doesn't balance a humanoid for long
  (G1 with its own gains sags after ~1 s), so this checks the setup, not standing:
  stable simulation, no start inside the floor, lands on its feet, holds the pose, torque
  limits not saturated. Returns ``{"ok", "checks": [{name, ok, text, warn_only?}], …}``
  (``ok``: no failed check; a ``warn_only`` check only warns)."""
  ms, _ = prepare(spec.source)
  ms.worldbody.add_geom(
    name="wbc_floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05]
  )
  ms.option.timestep = 0.005
  m = ms.compile()
  d = mujoco.MjData(m)
  kp, kd, eff, target, qa, va = [], [], [], [], [], []
  q0 = m.qpos0.copy()
  q0[0:3] = (0.0, 0.0, spec.init_height)
  q0[3:7] = (1.0, 0.0, 0.0, 0.0)
  for j in range(m.njnt):
    if m.jnt_type[j] not in _HINGE:
      continue
    name = _name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
    g = spec.joints.get(name) or _default_gains()
    adr, dof = int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j])
    q0[adr] = spec.init_joint_pos.get(name, q0[adr])
    m.dof_armature[dof] = g.armature
    qa.append(adr)
    va.append(dof)
    kp.append(g.stiffness)
    kd.append(g.damping)
    eff.append(g.effort)
    target.append(q0[adr])
  kp, kd, eff, target = map(np.asarray, (kp, kd, eff, target))
  qa, va = np.asarray(qa, dtype=int), np.asarray(va, dtype=int)
  d.qpos[:] = q0
  mujoco.mj_forward(m, d)
  floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "wbc_floor")

  def floor_contacts() -> tuple[set[int], float]:
    bodies, depth = set(), 0.0
    for c in d.contact[: d.ncon]:
      if floor in (c.geom1, c.geom2):
        other = c.geom2 if c.geom1 == floor else c.geom1
        bodies.add(int(m.geom_bodyid[other]))
        depth = max(depth, -float(c.dist))
    return bodies, depth

  _, penetration = floor_contacts()
  z0, saturated, unstable = float(d.qpos[2]), 0.0, False
  steps = max(1, int(seconds / m.opt.timestep))
  for _ in range(steps):
    tau = kp * (target - d.qpos[qa]) - kd * d.qvel[va]
    saturated += float(np.mean(np.abs(tau) > eff))
    d.qfrc_applied[:] = 0.0
    d.qfrc_applied[va] = np.clip(tau, -eff, eff)
    mujoco.mj_step(m, d)
    if (
      not np.isfinite(d.qpos).all()
      or d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number
    ):
      unstable = True
      break
  if unstable:
    return {
      "ok": False,
      "checks": [
        {
          "name": "stable",
          "ok": False,
          "text": "the simulation blew up: lower the gains or raise the armature",
        }
      ],
    }
  z1 = float(d.qpos[2])
  rot = np.zeros(9)
  mujoco.mju_quat2Mat(rot, d.qpos[3:7])
  tilt = math.degrees(math.acos(float(np.clip(rot[8], -1.0, 1.0))))
  touching, _ = floor_contacts()
  feet = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f) for f in spec.feet_bodies}
  on_feet = {b for b in touching if any(_is_descendant(m, b, f) for f in feet)}
  others = sorted(_name(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in touching - on_feet)
  sag = float(np.max(np.abs(d.qpos[qa] - target))) if len(qa) else 0.0
  sat = saturated / steps
  checks = [
    {
      "name": "stable",
      "ok": True,
      "text": f"{seconds:g} s simulated without blowing up",
    },
    {
      "name": "start",
      "ok": penetration < 0.02,
      "text": f"starts {penetration * 100:.1f} cm inside the floor: raise the init height"
      if penetration >= 0.02
      else "starts on or just above the floor",
    },
    {
      "name": "feet",
      "ok": bool(on_feet) and not others,
      "text": "lands on its feet"
      if on_feet and not others
      else f"touches the floor with {', '.join(others) or 'nothing'}: check the init height, pose and feet",
    },
    {
      "name": "holds",
      "ok": z1 > 0.8 * z0 and tilt < 20.0 and sag < 0.6,
      "warn_only": True,  # plain PD isn't expected to balance for long
      "text": f"root {z0:.2f} → {z1:.2f} m, tilt {tilt:.0f}°, joints within {math.degrees(sag):.0f}° of the pose"
      + (
        ""
        if z1 > 0.8 * z0 and tilt < 20.0 and sag < 0.6
        else ": sags fast — stiffer gains or a bent-knee pose may help"
      ),
    },
    {
      "name": "torque",
      "ok": sat < 0.2,
      "text": f"torque limits saturated {sat * 100:.0f}% of the time"
      + (": raise the efforts" if sat >= 0.2 else ""),
    },
  ]
  return {
    "ok": all(c["ok"] or c.get("warn_only") for c in checks),
    "checks": checks,
    "root_start": z0,
    "root_end": z1,
    "tilt_deg": tilt,
  }


# ---------------------------------------------------------------------- storage
def list_specs() -> list[CustomRobotSpec]:
  out = []
  d = robots_dir()
  for p in sorted(d.glob(f"*/{SPEC_FILE}")) if d.is_dir() else []:
    try:
      out.append(load_spec(p.parent.name))
    except Exception as exc:  # noqa: BLE001
      print(f"[wbc-mjlab] custom robot {p.parent.name}: {exc}")
  return out


def load_spec(robot_id: str) -> CustomRobotSpec:
  path = robots_dir() / check_id(robot_id) / SPEC_FILE
  return CustomRobotSpec.from_dict(yaml.safe_load(path.read_text()) or {})


def save_spec(spec: CustomRobotSpec) -> CustomRobotSpec:
  """Validate, write ``wbc_robot.xml`` + ``robot.yaml`` and (re)register the robot."""
  check_id(spec.id)
  from wbc_mjlab.robots.ids import known_robot_ids

  if spec.id in ("g1",) or (
    spec.id in known_robot_ids() and spec.id not in {s.id for s in list_specs()}
  ):
    raise ValueError(f"{spec.id!r} is already a robot of wbc-mjlab or an extension")
  if not spec.source.is_file():
    raise ValueError(f"model not found: {spec.source}")
  prepared, _ = prepare(spec.source)
  m = prepared.compile()
  bodies = {_name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)}
  for label, names in (
    ("anchor", [spec.anchor_body]),
    ("keybodies", spec.keybodies),
    ("end-effectors", spec.ee_bodies),
    ("feet", spec.feet_bodies),
  ):
    missing = [n for n in names if n not in bodies]
    if missing:
      raise ValueError(f"{label}: no body {', '.join(missing)} in the model")
  if not spec.keybodies or not spec.feet_bodies:
    raise ValueError("pick at least the keybodies and the feet")
  sites = {_name(m, mujoco.mjtObj.mjOBJ_SITE, s) for s in range(m.nsite)}
  spec.foot_sites = [s for s in spec.foot_sites if s in sites]
  geoms = {_name(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in range(m.ngeom)}
  spec.foot_geoms = [g for g in spec.foot_geoms if g in geoms]
  joints = {
    _name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
    for j in range(m.njnt)
    if m.jnt_type[j] in _HINGE
  }
  unknown = set(spec.joints) - joints
  if unknown:
    raise ValueError(f"gains for unknown joints: {', '.join(sorted(unknown))}")
  for j in joints - set(spec.joints):
    spec.joints[j] = _default_gains()
  for g in spec.joints.values():
    if g.stiffness <= 0:
      raise ValueError("every joint needs a stiffness > 0")
  spec.folder.mkdir(parents=True, exist_ok=True)
  spec.prepared.write_text(prepared.to_xml())
  (spec.folder / SPEC_FILE).write_text(yaml.safe_dump(spec.to_dict(), sort_keys=False))
  register(spec)
  return spec


def delete(robot_id: str) -> None:
  folder = robots_dir() / check_id(robot_id)
  if folder.is_dir():
    shutil.rmtree(folder)
  _unregister(robot_id)


def _default_gains() -> JointGains:
  w = 2.0 * math.pi * NATURAL_FREQ_HZ
  return JointGains(
    effort=DEFAULT_EFFORT,
    stiffness=DEFAULT_ARMATURE * w * w,
    damping=2.0 * DAMPING_RATIO * DEFAULT_ARMATURE * w,
    armature=DEFAULT_ARMATURE,
  )


# ---------------------------------------------------------------------- env config
def _entity_cfg(spec: CustomRobotSpec):
  from mjlab.actuator.pd_actuator import IdealPdActuatorCfg
  from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

  path = str(spec.prepared)
  actuators = tuple(
    IdealPdActuatorCfg(
      target_names_expr=(f"^{re.escape(name)}$",),
      stiffness=float(g.stiffness),
      damping=float(g.damping),
      effort_limit=float(g.effort),
      armature=float(g.armature),
    )
    for name, g in sorted(spec.joints.items())
  )
  return EntityCfg(
    init_state=EntityCfg.InitialStateCfg(
      pos=(0.0, 0.0, float(spec.init_height)),
      joint_pos={f"^{re.escape(k)}$": float(v) for k, v in spec.init_joint_pos.items()}
      or {".*": 0.0},
      joint_vel={".*": 0.0},
    ),
    spec_fn=lambda: mujoco.MjSpec.from_file(path),
    articulation=EntityArticulationInfoCfg(
      actuators=actuators, soft_joint_pos_limit_factor=0.9
    ),
  )


def base_env_cfg(spec: CustomRobotSpec):
  """The shared WBC template wired to this robot (what ``g1_base_cfg`` does for G1)."""
  from mjlab.managers.observation_manager import ObservationTermCfg
  from mjlab.managers.scene_entity_config import SceneEntityCfg
  from mjlab.sensor import ContactMatch, ContactSensorCfg

  import wbc_mjlab.env.mdp as mdp
  from wbc_mjlab.env.wbc_env_cfg import make_base_wbc_env_cfg

  cfg = make_base_wbc_env_cfg(use_reference_residual_action=True)
  cfg.scene.entities = {"robot": _entity_cfg(spec)}
  feet = "|".join(re.escape(f) for f in spec.feet_bodies)
  root = _root_name(spec)
  keys = tuple(spec.keybodies)
  cfg.scene.sensors = (
    ContactSensorCfg(
      name="feet_ground_contact",
      primary=ContactMatch(mode="subtree", pattern=rf"^({feet})$", entity="robot"),
      secondary=ContactMatch(mode="body", pattern="terrain"),
      fields=("found", "force"),
      reduce="netforce",
      num_slots=1,
      track_air_time=True,
    ),
    ContactSensorCfg(
      name="self_collision",
      primary=ContactMatch(mode="subtree", pattern=re.escape(root), entity="robot"),
      secondary=ContactMatch(mode="subtree", pattern=re.escape(root), entity="robot"),
      fields=("found", "force"),
      reduce="none",
      num_slots=1,
      history_length=4,
    ),
    ContactSensorCfg(
      name="keybodies_ground_contact",
      primary=ContactMatch(
        mode="body",
        pattern=r"^(" + "|".join(re.escape(k) for k in keys) + r")$",
        entity="robot",
      ),
      secondary=ContactMatch(mode="body", pattern="terrain"),
      fields=("found", "force"),
      reduce="netforce",
      num_slots=len(keys),
      track_air_time=False,
    ),
  )
  cfg.actions["joint_pos"].scale = {
    f"^{re.escape(n)}$": spec.action_scale * g.effort / g.stiffness
    for n, g in spec.joints.items()
  }
  for group in ("actor", "critic"):
    terms = cfg.observations[group].terms
    for name, sensor in (("base_ang_vel", GYRO), ("base_lin_vel", VELOCIMETER)):
      t = terms.get(name)
      if t is not None and "sensor_name" in (t.params or {}):
        t.params["sensor_name"] = f"robot/{sensor}"
  motion = cfg.commands["motion"]
  motion.anchor_body_name = spec.anchor_body
  motion.body_names = keys
  anchor = (spec.anchor_body,)
  ev = cfg.events
  ev["assistive_wrench"].params["asset_cfg"].body_names = anchor
  ev["assistive_wrench"].params["body_name"] = spec.anchor_body
  ev["pull_robot"].params["asset_cfg"].body_names = anchor
  ev["base_com"].params["asset_cfg"].body_names = anchor
  if spec.foot_geoms:
    ev["foot_friction"].params["asset_cfg"].geom_names = tuple(
      f"^{re.escape(g)}$" for g in spec.foot_geoms
    )
  else:
    ev.pop("foot_friction", None)
  cfg.terminations["ee_body_pos"].params["body_names"] = tuple(spec.ee_bodies)
  cfg.observations["critic"].terms["keybody_contact_forces"] = ObservationTermCfg(
    func=mdp.keybody_contact_forces,
    params={"sensor_name": "keybodies_ground_contact"},
  )
  if spec.foot_sites:
    cfg.rewards["foot_slip"].params["asset_cfg"] = SceneEntityCfg(
      "robot", site_names=tuple(spec.foot_sites)
    )
  else:
    cfg.rewards.pop("foot_slip", None)
  cfg.sim.nconmax = int(spec.nconmax)
  cfg.sim.njmax = int(spec.njmax)
  cfg.viewer.body_name = spec.anchor_body
  cfg.viewer.distance = max(2.5, 2.2 * float(spec.init_height) + 1.0)
  cfg.viewer.elevation = -5.0
  cfg.viewer.azimuth = 120.0
  return cfg


def _root_name(spec: CustomRobotSpec) -> str:
  text = spec.prepared.read_text() if spec.prepared.is_file() else ""
  m = re.search(r"<worldbody>\s*<body[^>]*name=\"([^\"]+)\"", text)
  return m.group(1) if m else spec.anchor_body


def env_cfg(
  spec: CustomRobotSpec, *, se: bool = False, terrain: bool = False, play: bool = False
):
  """``Wbc-<Id>`` (``se``: ``-SE``, ``terrain``: ``-Terrain-Scan`` — procedural curriculum
  terrain, a height scan under the root body and terrain-aware terminations, as G1)."""
  from wbc_mjlab.presets.se_actor import apply_se_actor
  from wbc_mjlab.presets.terrain import apply_terrain_scan
  from wbc_mjlab.presets.wbc import apply_wbc

  cfg = base_env_cfg(spec)
  apply_wbc(
    cfg,
    motion_body_names=tuple(spec.keybodies),
    ee_termination_bodies=tuple(spec.ee_bodies),
  )
  if se:
    apply_se_actor(cfg)
    for group in ("actor", "critic"):
      t = cfg.observations[group].terms.get("base_lin_vel")
      if t is not None and "sensor_name" in (t.params or {}):
        t.params["sensor_name"] = f"robot/{VELOCIMETER}"
  if terrain:
    apply_terrain_scan(
      cfg,
      scan_body_name=_root_name(spec),
      scan_offset=float(spec.init_height),  # root height standing on flat ground
      ee_termination_bodies=tuple(spec.ee_bodies),
    )
    cfg.sim.nconmax = max(cfg.sim.nconmax, int(spec.nconmax))
    cfg.sim.njmax = max(cfg.sim.njmax, int(spec.njmax))
  if play:
    if terrain:  # no curriculum in play: spread envs over every difficulty row
      cfg.scene.terrain.max_init_terrain_level = None
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.curriculum = {}
    cfg.events.pop("push_robot", None)
    motion = cfg.commands["motion"]
    motion.pose_range = {}
    motion.velocity_range = {}
    motion.assistive_wrench_enabled = False
    if "assistive_wrench" in cfg.events:
      cfg.events["assistive_wrench"].params["enabled"] = False
  return cfg


# ---------------------------------------------------------------------- registration
def tasks_for(spec: CustomRobotSpec):
  from wbc_mjlab.tasks.config import WbcTaskConfig

  tid = spec.task_id
  base = spec.description or f"custom robot {spec.id}"
  return (
    WbcTaskConfig(
      task_id=tid,
      robot_id=spec.id,
      description=f"{base}: default WBC tracking + RSI.",
      experiment_name=f"wbc_{spec.id}",
      build_env_cfg=lambda: env_cfg(spec),
    ),
    WbcTaskConfig(
      task_id=f"{tid}-Terrain-Scan",
      robot_id=spec.id,
      description=f"{tid} on curriculum terrain + actor height scan; terrain-aware terminations.",
      experiment_name=f"wbc_{spec.id}_terrain_scan",
      build_env_cfg=lambda: env_cfg(spec, terrain=True),
    ),
    WbcTaskConfig(
      task_id=f"{tid}-SE",
      robot_id=spec.id,
      description=f"{tid} + SE obs (anchor pose tracking error, IMU base lin vel).",
      experiment_name=f"wbc_{spec.id}_se",
      build_env_cfg=lambda: env_cfg(spec, se=True),
    ),
  )


def _unregister(robot_id: str) -> None:
  from mjlab.tasks import registry as mj_registry

  from wbc_mjlab import tasks as wbc_tasks

  for tid in [t for t, c in wbc_tasks._TASK_BY_ID.items() if c.robot_id == robot_id]:  # noqa: SLF001
    wbc_tasks._TASK_BY_ID.pop(tid, None)  # noqa: SLF001
    wbc_tasks._MJLAB_REGISTERED.discard(tid)  # noqa: SLF001
    mj_registry._REGISTRY.pop(tid, None)  # noqa: SLF001


def register(spec: CustomRobotSpec) -> None:
  """Register (or re-register after an edit) one custom robot and its tasks."""
  from wbc_mjlab.extension import WbcRobotSpec, register_wbc_extension
  from wbc_mjlab.motion.robot_assets import RobotMotionSpec
  from wbc_mjlab.robots.g1.rl_cfg import g1_wbc_rl_cfg

  _unregister(spec.id)
  by_id = {t.task_id: t for t in tasks_for(spec)}

  def make_env_cfg(*, play: bool = False, task_id: str = spec.task_id, **kwargs):
    if kwargs:
      raise TypeError(f"unknown env cfg kwargs for {spec.id}: {', '.join(kwargs)}")
    if task_id not in by_id:
      raise KeyError(f"unknown task {task_id!r} for {spec.id}: {', '.join(by_id)}")
    return env_cfg(
      spec,
      se=task_id.endswith("-SE"),
      terrain=task_id.endswith("-Terrain-Scan"),
      play=play,
    )

  register_wbc_extension(
    WbcRobotSpec(
      robot_id=spec.id,
      make_env_cfg=make_env_cfg,
      make_rl_cfg=g1_wbc_rl_cfg,
      motion_spec=RobotMotionSpec(
        scene_cfg_fn=lambda: env_cfg(spec).scene,
        foot_body_names=tuple(spec.feet_bodies),
        foot_sole_z=float(spec.foot_sole_z),
        mjcf_path=spec.prepared,
      ),
      mjcf_path=spec.prepared,
    ),
    tuple(by_id.values()),
  )


def register_all() -> list[str]:
  """Register every custom robot (a broken one is reported and skipped)."""
  done = []
  for spec in list_specs():
    try:
      if not spec.prepared.is_file():
        raise FileNotFoundError(f"{spec.prepared} is missing: re-save the robot")
      register(spec)
      done.append(spec.id)
    except Exception as exc:  # noqa: BLE001
      print(f"[wbc-mjlab] custom robot {spec.id} not registered: {exc}")
  return done


__all__ = [
  "CustomRobotSpec",
  "JointGains",
  "delete",
  "env_cfg",
  "inspect_model",
  "list_specs",
  "load_spec",
  "prepare",
  "register",
  "register_all",
  "robots_dir",
  "save_spec",
  "task_id_for",
]
