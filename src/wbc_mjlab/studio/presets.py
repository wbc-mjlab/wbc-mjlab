"""Custom presets: a registered WBC task (the *base*) plus overrides edited in the Studio.

A preset changes what the base task's builder produced, without code:

  observations  actor (deployed) / critic terms: remove, add from the robot's term library,
                noise scale, history (steps back), frames ahead and their stride (reference
                terms), keybody subset; group history / corruption
  rewards       weight, numeric params, on / off per term
  terminations  numeric params, on / off per term
  command       reference / RSI / assist knobs of the motion command (``rsi.bin_width_s`` …)
  events        domain-randomization ranges, on / off per event
  terrain       procedural terrain knobs (``-Terrain-Scan`` tasks)

How a preset is trained (the method and its settings) is a *recipe*, picked in Train
(:mod:`wbc_mjlab.studio.recipes`). Presets saved before recipes may still hold PPO settings
(``agent``): they keep applying to PPO's runner config until moved to a recipe
(:func:`move_agent_to_recipe`).

*Modules* are reusable partial overrides without a base (``se``: the state-estimation actor
layout, derived from the built-in ``<task>-SE`` vs ``<task>`` pair; your own in
``modules/``). A preset applies its modules in order, then its own overrides.

Everything lives outside the repo in :func:`config_dir` (``~/.config/wbc-mjlab``, or
``$WBC_MJLAB_CONFIG_DIR``): ``presets/<name>.yaml``, ``modules/<name>.yaml``.
:func:`register` turns a preset into a real mjlab task ``<base>--<name>`` logging to
``<base experiment>--<name>``, so train, play and run discovery treat it like a built-in task
(in this process; training jobs register it via ``python -m wbc_mjlab.studio.train_preset``,
which also stores the resolved preset in the run: ``params/studio_preset.yaml``, used to replay
the run whatever happens to the preset later).
"""

from __future__ import annotations

import copy
import dataclasses
import functools
import inspect
import re
import time
import typing
from pathlib import Path
from typing import Any

import yaml

SEP = "--"
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,47}$")
GROUPS = (
  "observations",
  "rewards",
  "terminations",
  "command",
  "events",
  "terrain",
  "agent",  # legacy: PPO settings, now recipes
)
OBS_GROUPS = ("actor", "critic")
SNAPSHOT = "studio_preset.yaml"
BUILTIN_MODULES = {
  "se": (
    "-SE",
    "State estimation: the actor drops height / gravity references, adds anchor pose "
    "error and IMU base velocity",
  ),
}

# Motion-command fields that are wiring, not tuning knobs.
_COMMAND_SKIP = {
  "motion_file",
  "anchor_body_name",
  "body_names",
  "entity_name",
  "debug_vis",
  "viz",
  "viz.mode",
  "actuated_joint_names",
  "sampling_mode",  # legacy top-level field; rsi.sampling_mode is the live one
  "adaptive_kernel_size",
  "adaptive_lambda",
  "adaptive_uniform_ratio",
  "adaptive_alpha",
  "rsi.similarity_terms",
  "rsi.failure_levels_filename",
  "rsi.tracking_reward_prefix",
  "rsi.persist_failure_levels",
}
HELP = {
  "rsi.sampling_mode": "where episodes start: clip start, uniform, or adaptive (failure-weighted bins)",
  "rsi.strategy": "failure signal for adaptive sampling",
  "rsi.bin_width_s": "length of one RSI bin",
  "rsi.uniform_ratio": "share of starts drawn uniformly",
  "rsi.alpha": "EMA rate of the failure levels",
  "rsi.temperature_base": "softmax temperature over bin failure levels",
  "assistive_wrench_enabled": "virtual wrench that helps on hard bins (curriculum)",
  "assistive_beta_max": "max assist gain on the hardest bins",
  "joint_position_range": "noise added to the reference joints at reset",
}


def config_dir() -> Path:
  """Per-user Studio configs (not in the repo): see :mod:`wbc_mjlab.user_config`."""
  from wbc_mjlab.user_config import config_dir as user_config_dir

  return user_config_dir()


def presets_dir() -> Path:
  return config_dir() / "presets"


def modules_dir() -> Path:
  return config_dir() / "modules"


# ---------------------------------------------------------------------- schema
def _num(v: Any) -> bool:
  return isinstance(v, (int, float)) and not isinstance(v, bool)


def _field(key: str, value: Any, *, choices=None) -> dict | None:
  """One editable value ``{key, kind, value, …}`` or None when it is not a tuning knob."""
  if isinstance(value, bool):
    return {"key": key, "kind": "bool", "value": value}
  if _num(value):
    return {
      "key": key,
      "kind": "int" if isinstance(value, int) else "number",
      "value": value,
    }
  if choices and isinstance(value, str):
    return {"key": key, "kind": "choice", "value": value, "choices": list(choices)}
  if isinstance(value, (tuple, list)) and all(isinstance(x, str) for x in value):
    return {"key": key, "kind": "strings", "value": list(value)}
  if isinstance(value, (tuple, list)) and value and all(_num(x) for x in value):
    if len(value) == 2 and not all(isinstance(x, int) for x in value):
      return {"key": key, "kind": "range", "value": [float(x) for x in value]}
    if all(isinstance(x, int) for x in value):
      return {"key": key, "kind": "ints", "value": [int(x) for x in value]}
  return None


def _param_fields(params: dict | None) -> list[dict]:
  out = []
  for k, v in (params or {}).items():
    if (
      isinstance(v, dict)
      and v
      and all(
        isinstance(x, (tuple, list)) and len(x) == 2 and all(_num(y) for y in x)
        for x in v.values()
      )
    ):
      out += [
        {"key": f"{k}.{sk}", "kind": "range", "value": [float(a) for a in sv]}
        for sk, sv in v.items()
      ]
      continue
    f = _field(k, v)
    if f is not None:
      out.append(f)
  return out


def _literal_choices(cls, name: str):
  try:
    hint = typing.get_type_hints(cls).get(name)
  except Exception:  # noqa: BLE001
    return None
  if typing.get_origin(hint) is typing.Literal:
    return typing.get_args(hint)
  return None


def _walk_dataclass(obj, prefix: str = "", depth: int = 0) -> list[dict]:
  out = []
  for f in dataclasses.fields(obj):
    key = f"{prefix}{f.name}"
    value = getattr(obj, f.name)
    if dataclasses.is_dataclass(value) and depth < 1:
      out += _walk_dataclass(value, f"{key}.", depth + 1)
      continue
    if (
      isinstance(value, dict)
      and value
      and all(isinstance(x, (tuple, list)) and len(x) == 2 for x in value.values())
    ):
      out += [
        {"key": f"{key}.{sk}", "kind": "range", "value": [float(a) for a in sv]}
        for sk, sv in value.items()
      ]
      continue
    item = _field(key, value, choices=_literal_choices(type(obj), f.name))
    if item is not None:
      out.append(item | dict(f.metadata))
  return out


def _func_name(term) -> str:
  fn = getattr(term, "func", None)
  return getattr(fn, "__name__", type(fn).__name__ if fn is not None else "")


def describe(env_cfg) -> dict:
  """Every editable knob of a task's env config, with its current (default) value."""
  rewards = [
    {
      "name": name,
      "func": _func_name(t),
      "weight": float(t.weight),
      "params": _param_fields(t.params),
    }
    for name, t in env_cfg.rewards.items()
  ]
  terminations = [
    {
      "name": name,
      "func": _func_name(t),
      "time_out": bool(getattr(t, "time_out", False)),
      "params": _param_fields(t.params),
    }
    for name, t in env_cfg.terminations.items()
  ]
  events = [
    {
      "name": name,
      "func": _func_name(t),
      "mode": str(getattr(t, "mode", "")),
      "params": _param_fields(t.params),
    }
    for name, t in env_cfg.events.items()
  ]
  command = []
  motion = env_cfg.commands.get("motion") if hasattr(env_cfg.commands, "get") else None
  if motion is not None:
    command = [f for f in _walk_dataclass(motion) if f["key"] not in _COMMAND_SKIP]
  command.append(
    {
      "key": "env.episode_length_s",
      "kind": "number",
      "value": float(env_cfg.episode_length_s),
    }
  )
  for f in command:
    if f["key"] in HELP:
      f["help"] = HELP[f["key"]]
  return {
    "terrain": _describe_terrain(env_cfg),
    "observations": _describe_observations(env_cfg),
    "rewards": rewards,
    "terminations": terminations,
    "command": command,
    "events": events,
  }


# ---------------------------------------------------------------------- terrain
_TERRAIN_SKIP = {
  "size",
  "border_width",
  "border_height",
  "horizontal_scale",
  "vertical_scale",
  "slope_threshold",
  "resolution",
  "base_thickness_ratio",
  "flat_patch_sampling",
  "merge_similar_heights",
  "height_merge_threshold",
  "max_merge_distance",
}
TERRAIN_HELP = {
  "curriculum.start_row": "row new envs start on (0 = easiest); they move up as they track well",
  "curriculum.num_rows": "difficulty levels (rows) of the curriculum",
  "curriculum.difficulty_range": "difficulty of the first and the last row",
}


def _generator(env_cfg):
  t = getattr(env_cfg.scene, "terrain", None)
  if t is None or getattr(t, "terrain_type", "") != "generator":
    return None, None
  return t, t.terrain_generator


def _describe_terrain(env_cfg) -> list[dict]:
  """Knobs of a procedural terrain (``-Terrain-Scan`` tasks): curriculum and the mix and
  ranges of each sub-terrain (``<sub>.proportion``, ``rough.noise_range`` …)."""
  t, gen = _generator(env_cfg)
  if gen is None:
    return []
  out = []
  if t.max_init_terrain_level is not None:
    out.append(
      {
        "key": "curriculum.start_row",
        "kind": "int",
        "value": int(t.max_init_terrain_level),
      }
    )
  for key in ("num_rows", "difficulty_range"):
    f = _field(f"curriculum.{key}", getattr(gen, key))
    if f is not None:
      out.append(f)
  for name, sub in gen.sub_terrains.items():
    for f in _walk_dataclass(sub):
      if f["key"].split(".")[0] in _TERRAIN_SKIP:
        continue
      out.append(f | {"key": f"{name}.{f['key']}"})
  for f in out:
    if f["key"] in TERRAIN_HELP:
      f["help"] = TERRAIN_HELP[f["key"]]
  return out


def _apply_terrain(env_cfg, ov: dict) -> None:
  t, gen = _generator(env_cfg)
  if gen is None:
    return
  for key, value in (ov or {}).items():
    if key == "curriculum.start_row":
      t.max_init_terrain_level = int(value)
    elif key.startswith("curriculum."):
      _set_path(gen, key.removeprefix("curriculum."), value)
    else:
      sub, _, rest = key.partition(".")
      if sub in gen.sub_terrains:
        _set_path(gen.sub_terrains[sub], rest, value)


# ---------------------------------------------------------------------- observations
def _obs_kind(name: str) -> str:
  """``reference``: what the robot is asked to track (from the motion); ``state``: measured
  robot state; ``privileged``: sim-only signals (contacts, assist, curriculum)."""
  if name.startswith(
    ("assist", "keybody_contact", "motion_segment", "motion_tracking")
  ):
    return "privileged"
  if name.startswith(("ref_", "motion_")) or name == "command":
    return "reference"
  return "state"


def _noise_text(noise) -> str:
  if noise is None:
    return ""
  lo, hi = getattr(noise, "n_min", None), getattr(noise, "n_max", None)
  if hi is None:
    return type(noise).__name__

  def f(v):
    if isinstance(v, (tuple, list)):
      return "(" + ", ".join(f"{x:g}" for x in v) + ")"
    return f"{v:g}"

  neg = (tuple(-x for x in hi) if isinstance(hi, (tuple, list)) else -hi) == (
    tuple(lo) if isinstance(lo, (tuple, list)) else lo
  )
  return f"±{f(hi)}" if neg else f"[{f(lo)}, {f(hi)}]"


def _scale_noise(noise, factor: float):
  if noise is None or factor == 1:
    return noise
  if factor <= 0:
    return None
  lo, hi = getattr(noise, "n_min", None), getattr(noise, "n_max", None)
  if hi is None:
    return noise

  def mul(v):
    return tuple(x * factor for x in v) if isinstance(v, (tuple, list)) else v * factor

  return dataclasses.replace(noise, n_min=mul(lo), n_max=mul(hi))


def _takes_bodies(term) -> bool:
  try:
    return "body_names" in inspect.signature(term.func).parameters
  except (TypeError, ValueError):
    return False


# critic-only terms the export table doesn't cover: (per keybody, constant)
_EXTRA_DIMS = {
  "ref_body_lin_vel": (3, 0),
  "ref_body_ang_vel": (3, 0),
  "motion_body_lin_vel": (3, 0),
  "motion_body_ang_vel": (3, 0),
  "ref_base_lin_acc": (0, 3),
  "ref_base_ang_acc": (0, 3),
  "assistive_force": (0, 3),
  "assistive_torque": (0, 3),
  "assistive_gain": (0, 1),
  "motion_segment_phase": (0, 1),
}


def _obs_dim(
  name: str, term, joints: int, bodies: int, tracking: int = 0, env_cfg=None
) -> int | None:
  """Width of a term (None if unknown without building the env; ``tracking``: number of
  ``motion_*`` reward terms; ``env_cfg``: sensor layouts, for ``height_scan``)."""
  from wbc_mjlab.export.tracking_params_yaml import (
    _height_scan_params,
    _observation_dim,
  )

  if name == "height_scan":
    try:
      d = _height_scan_params(env_cfg, term)["num_rays"]
    except Exception:  # noqa: BLE001 — no cfg / sensor at hand
      return None
    return d * max(1, int(getattr(term, "history_length", 0) or 0))

  sub = (term.params or {}).get("body_names")
  nb = len(sub) if sub else bodies
  try:
    d = _observation_dim(name, joint_count=joints, body_count=nb)
  except KeyError:
    if name in _EXTRA_DIMS:
      per_body, const = _EXTRA_DIMS[name]
      d = per_body * nb + const
    elif name == "motion_tracking_step_rewards" and tracking:
      d = tracking
    else:
      return None
  frames = int((term.params or {}).get("frames", 1))
  return d * frames * max(1, int(getattr(term, "history_length", 0) or 0))


def deployable(name: str) -> bool:
  """Actor terms the exporter (``tracking_params.yaml``) knows how to rebuild on the robot."""
  from wbc_mjlab.export.tracking_params_yaml import _observation_dim

  # height_scan: exported with its grid layout (elevation map on the robot)
  if name == "height_scan":
    return True

  try:
    _observation_dim(name, joint_count=1, body_count=1)
    return True
  except KeyError:
    return False


def _robot_sizes(env_cfg) -> tuple[int, int, list[str]]:
  motion = env_cfg.commands.get("motion") if hasattr(env_cfg.commands, "get") else None
  bodies = list(getattr(motion, "body_names", ()) or ())
  try:
    from wbc_mjlab.export.tracking_params_yaml import _joint_names_from_cfg

    joints = len(_joint_names_from_cfg(env_cfg))
  except Exception:  # noqa: BLE001
    joints = 0
  return joints, len(bodies), bodies


def _describe_observations(env_cfg) -> dict:
  from wbc_mjlab.env.mdp.observations import takes_frames

  joints, nbodies, bodies = _robot_sizes(env_cfg)
  tracking = sum(n.startswith("motion_") for n in env_cfg.rewards)
  groups = []
  for g in OBS_GROUPS:
    grp = env_cfg.observations.get(g)
    if grp is None:
      continue
    terms = []
    for name, t in grp.terms.items():
      sub = (t.params or {}).get("body_names")
      terms.append(
        {
          "name": name,
          "func": _func_name(t),
          "kind": _obs_kind(name),
          "dim": _obs_dim(name, t, joints, nbodies, tracking, env_cfg),
          "noise": _noise_text(t.noise),
          "history_length": int(t.history_length or 0),
          "frames": int((t.params or {}).get("frames", 1)),
          "stride": int((t.params or {}).get("stride", 1)),
          "takes_frames": takes_frames(t),
          "bodies": list(sub) if sub else None,
          "takes_bodies": _takes_bodies(t),
          "deployable": deployable(name) if g == "actor" else True,
        }
      )
    groups.append(
      {
        "name": g,
        "history_length": int(grp.history_length or 0),
        "enable_corruption": bool(grp.enable_corruption),
        "terms": terms,
      }
    )
  return {"groups": groups, "bodies": bodies, "joints": joints}


@functools.lru_cache(maxsize=8)
def _library(robot_id: str) -> dict[str, dict]:
  """Every observation term the robot's built-in tasks define: ``name → {group: term}``
  (the actor variant carries the deploy noise, the critic one is noise-free)."""
  from wbc_mjlab.robots.env import make_wbc_env_cfg
  from wbc_mjlab.tasks import all_wbc_tasks, register_all_wbc_tasks

  register_all_wbc_tasks()
  lib: dict[str, dict] = {}
  for t in all_wbc_tasks():
    if t.robot_id != robot_id or SEP in t.task_id:
      continue
    cfg = make_wbc_env_cfg(robot_id, task_id=t.task_id)
    for g in OBS_GROUPS:
      grp = cfg.observations.get(g)
      for name, term in (grp.terms if grp else {}).items():
        lib.setdefault(name, {}).setdefault(g, term)
  return lib


def _missing_sensor(term, env_cfg) -> str:
  """The scene sensor a term reads that ``env_cfg`` doesn't have ("" if none is missing;
  ``robot/…`` sensors come with the robot model)."""
  sensor = (term.params or {}).get("sensor_name") or ""
  if not sensor or sensor.startswith("robot/") or env_cfg is None:
    return ""
  return "" if sensor in {s.name for s in env_cfg.scene.sensors or ()} else sensor


def library(
  robot_id: str, joints: int = 0, bodies: int = 0, env_cfg=None
) -> list[dict]:
  """The term library as rows for the UI (``dim`` when known; ``bodies`` = keybody subset
  of the variant a preset gets when it adds the term to the actor; ``requires``: a scene
  sensor ``env_cfg`` lacks, e.g. the height scan of the ``-Terrain-Scan`` tasks)."""
  from wbc_mjlab.env.mdp.observations import takes_frames

  out = []
  for name, v in sorted(_library(robot_id).items()):
    term = v.get("actor") or next(iter(v.values()))
    sub = (term.params or {}).get("body_names")
    out.append(
      {
        "name": name,
        "kind": _obs_kind(name),
        "dim": _obs_dim(name, term, joints, bodies, env_cfg=env_cfg),
        "requires": _missing_sensor(term, env_cfg),
        "takes_bodies": _takes_bodies(term),
        "takes_frames": takes_frames(term),
        "bodies": list(sub) if sub else None,
        "func": _func_name(term),
        "deployable": deployable(name),
        "noise": _noise_text(term.noise),
      }
    )
  return out


def _new_term(robot_id: str, name: str, group: str):
  variants = _library(robot_id).get(name)
  if not variants:
    raise ValueError(f"no observation term {name!r} for robot {robot_id}")
  term = copy.deepcopy(variants.get(group) or next(iter(variants.values())))
  if group != "actor":
    term.noise = None
  return term


def _apply_observations(env_cfg, ov: dict, robot_id: str | None) -> None:
  from wbc_mjlab.env.mdp.observations import look_ahead, takes_frames

  for g, o in (ov or {}).items():
    grp = env_cfg.observations.get(g)
    if grp is None or not isinstance(o, dict):
      continue
    if "history_length" in o:
      grp.history_length = int(o["history_length"]) or None
    if "enable_corruption" in o:
      grp.enable_corruption = bool(o["enable_corruption"])
    for name, t in (o.get("terms") or {}).items():
      if t.get("enabled") is False:
        grp.terms.pop(name, None)
        continue
      if name not in grp.terms:
        if not t.get("add"):
          continue
        if robot_id is None:
          raise ValueError(f"adding {name!r} needs the robot")
        term = _new_term(robot_id, name, g)
        missing = _missing_sensor(term, env_cfg)
        if missing:
          raise ValueError(
            f"{name} reads the {missing!r} sensor, which this task doesn't have: start "
            "from a task that has it (e.g. a -Terrain-Scan task)"
          )
        grp.terms[name] = term
      term = grp.terms[name]
      if "noise_scale" in t:
        term.noise = _scale_noise(term.noise, float(t["noise_scale"]))
      if "history_length" in t:
        term.history_length = int(t["history_length"])
      if ("frames" in t or "stride" in t) and takes_frames(term):
        p = term.params or {}
        term = grp.terms[name] = look_ahead(
          term, t.get("frames", p.get("frames", 1)), t.get("stride", p.get("stride", 1))
        )
      if "bodies" in t and _takes_bodies(term):
        term.params = dict(term.params or {})
        if t["bodies"]:
          term.params["body_names"] = tuple(t["bodies"])
        else:
          term.params.pop("body_names", None)


def observation_diff(a, b) -> dict:
  """Observation overrides turning env cfg ``a`` into ``b`` (terms removed / added)."""
  out: dict = {}
  for g in OBS_GROUPS:
    ta, tb = a.observations.get(g), b.observations.get(g)
    if ta is None or tb is None:
      continue
    terms = {n: {"enabled": False} for n in ta.terms if n not in tb.terms}
    terms |= {n: {"add": True} for n in tb.terms if n not in ta.terms}
    if terms:
      out[g] = {"terms": terms}
  return out


# ---------------------------------------------------------------------- apply
def _coerce(old: Any, new: Any) -> Any:
  if isinstance(old, bool):
    return bool(new)
  if isinstance(old, int) and not isinstance(old, bool):
    return int(round(float(new)))
  if isinstance(old, float):
    return float(new)
  if isinstance(old, tuple):
    return (
      tuple(_coerce(o, n) for o, n in zip(old, new, strict=True))
      if len(old) == len(new)
      else tuple(new)
    )
  if isinstance(old, list):
    return list(new)
  return new


def _set_param(params: dict, key: str, value: Any) -> None:
  head, _, sub = key.partition(".")
  if sub and isinstance(params.get(head), dict):
    d = params[head]
    k = next((x for x in d if str(x) == sub), None)
    if k is not None:
      d[k] = _coerce(d[k], value)
    return
  if key in params:
    params[key] = _coerce(params[key], value)


def _set_path(obj, path: str, value: Any) -> None:
  parts = path.split(".")
  for p in parts[:-1]:
    nxt = getattr(obj, p, None)
    if nxt is None:
      return
    if isinstance(nxt, dict):  # dict of ranges (pose_range.x)
      k = next((x for x in nxt if str(x) == parts[-1]), None)
      if k is not None:
        nxt[k] = _coerce(nxt[k], value)
      return
    obj = nxt
  if hasattr(obj, parts[-1]):
    setattr(obj, parts[-1], _coerce(getattr(obj, parts[-1]), value))


def apply(env_cfg, agent_cfg, overrides: dict, robot_id: str | None = None) -> None:
  """Apply preset ``overrides`` to an env config and a runner config in place
  (``robot_id``: where added observation terms come from)."""
  ov = overrides or {}
  _apply_observations(env_cfg, ov.get("observations"), robot_id)
  _apply_terrain(env_cfg, ov.get("terrain"))
  for group in ("rewards", "terminations", "events"):
    terms = getattr(env_cfg, group)
    for name, o in (ov.get(group) or {}).items():
      if name not in terms:
        continue
      if o.get("enabled") is False:
        del terms[name]
        continue
      term = terms[name]
      if "weight" in o and hasattr(term, "weight"):
        term.weight = float(o["weight"])
      if o.get("params"):
        term.params = dict(term.params or {})
        for k, v in o["params"].items():
          _set_param(term.params, k, v)
  motion = env_cfg.commands.get("motion") if hasattr(env_cfg.commands, "get") else None
  for key, value in (ov.get("command") or {}).items():
    if key.startswith("env."):
      _set_path(env_cfg, key[4:], value)
    elif motion is not None:
      _set_path(motion, key, value)
  # legacy PPO settings (now recipes); the world model's runner config derives from PPO's
  if agent_cfg is not None:
    for key, value in (ov.get("agent") or {}).items():
      _set_path(agent_cfg, key, value)


def count_changes(overrides: dict) -> int:
  n = 0
  for o in (overrides.get("observations") or {}).values():
    n += sum(k != "terms" for k in o) + sum(
      len(t) for t in (o.get("terms") or {}).values()
    )
  for group in ("rewards", "terminations", "events"):
    for o in (overrides.get(group) or {}).values():
      n += (
        int(o.get("enabled") is False) + int("weight" in o) + len(o.get("params") or {})
      )
  return n + sum(len(overrides.get(g) or {}) for g in ("command", "terrain", "agent"))


# ---------------------------------------------------------------------- storage
def _check(name: str) -> str:
  if not NAME_RE.match(name or ""):
    raise ValueError("names: letters, digits, '_', '-', '.' (max 48)")
  return name


def _path(name: str) -> Path:
  return presets_dir() / f"{_check(name)}.yaml"


def _module_path(name: str) -> Path:
  return modules_dir() / f"{_check(name)}.yaml"


def _clean(overrides: dict) -> dict:
  return {g: overrides[g] for g in GROUPS if (overrides or {}).get(g)}


def task_id(preset: dict) -> str:
  return f"{preset['base']}{SEP}{preset['name']}"


def list_presets(robot: str | None = None) -> list[dict]:
  out = []
  d = presets_dir()
  for p in sorted(d.glob("*.yaml")) if d.is_dir() else []:
    try:
      pr = load(p.stem)
    except Exception:  # noqa: BLE001
      continue
    if robot and pr.get("robot") and pr["robot"] != robot:
      continue
    out.append(
      pr | {"task": task_id(pr), "changes": count_changes(pr.get("overrides") or {})}
    )
  return out


def load(name: str) -> dict:
  data = yaml.safe_load(_path(name).read_text()) or {}
  data["name"] = name
  data.setdefault("overrides", {})
  data["modules"] = list(data.get("modules") or [])
  return data


def save(
  name: str,
  base: str,
  overrides: dict,
  description: str = "",
  modules: list[str] | None = None,
) -> dict:
  from wbc_mjlab.tasks import get_task_config

  cfg = get_task_config(base)
  modules = list(modules or [])
  module_overrides(modules, cfg.robot_id)  # every module must exist
  data = {
    "base": cfg.task_id,
    "robot": cfg.robot_id,
    "description": description,
    "updated": time.strftime("%Y-%m-%d %H:%M"),
    "modules": modules,
    "overrides": _clean(overrides),
  }
  p = _path(name)
  p.parent.mkdir(parents=True, exist_ok=True)
  p.write_text(yaml.safe_dump(data, sort_keys=False))
  return data | {"name": name}


def delete(name: str) -> None:
  _path(name).unlink(missing_ok=True)


def move_agent_to_recipe(name: str) -> dict:
  """Move preset ``name``'s PPO settings (``agent``, from before recipes) to a PPO recipe of
  the same name; returns the recipe."""
  from wbc_mjlab.studio import recipes

  preset = load(name)
  settings = preset["overrides"].pop("agent", None) or {}
  if not settings:
    raise ValueError(f"preset {name!r} has no PPO settings")
  recipe = recipes.save(
    name, "ppo", settings, f"PPO settings of preset {name}, from before recipes"
  )
  save(
    name,
    preset["base"],
    preset["overrides"],
    preset.get("description", ""),
    preset["modules"],
  )
  return recipe


# ---------------------------------------------------------------------- modules
@functools.lru_cache(maxsize=8)
def _builtin_modules(robot_id: str) -> dict[str, dict]:
  """Modules read off the built-in tasks: ``se`` = ``<task>-SE`` minus ``<task>``."""
  from wbc_mjlab.robots.env import make_wbc_env_cfg
  from wbc_mjlab.tasks import all_wbc_tasks, register_all_wbc_tasks

  register_all_wbc_tasks()
  ids = {t.task_id for t in all_wbc_tasks() if t.robot_id == robot_id}
  out = {}
  for name, (suffix, text) in BUILTIN_MODULES.items():
    pair = next(
      (
        (t[: -len(suffix)], t)
        for t in sorted(ids, key=lambda x: (len(x), x))
        if t.endswith(suffix) and t[: -len(suffix)] in ids
      ),
      None,
    )
    if pair is None:
      continue
    a, b = (make_wbc_env_cfg(robot_id, task_id=t) for t in pair)
    out[name] = {
      "name": name,
      "builtin": True,
      "description": text,
      "source": f"{pair[1]} vs {pair[0]}",
      "overrides": {"observations": observation_diff(a, b)},
    }
  return out


def list_modules(robot: str | None = None) -> list[dict]:
  out = [
    m | {"changes": count_changes(m["overrides"])}
    for m in (_builtin_modules(robot).values() if robot else ())
  ]
  d = modules_dir()
  for p in sorted(d.glob("*.yaml")) if d.is_dir() else []:
    try:
      m = load_module(p.stem)
    except Exception:  # noqa: BLE001
      continue
    if robot and m.get("robot") and m["robot"] != robot:
      continue
    out.append(m | {"changes": count_changes(m["overrides"])})
  return out


def load_module(name: str, robot: str | None = None) -> dict:
  if robot and name in _builtin_modules(robot):
    return _builtin_modules(robot)[name]
  data = yaml.safe_load(_module_path(name).read_text()) or {}
  data["name"] = name
  data["builtin"] = False
  data.setdefault("overrides", {})
  return data


def save_module(
  name: str, overrides: dict, description: str = "", robot: str = ""
) -> dict:
  if name in BUILTIN_MODULES:
    raise ValueError(f"{name!r} is a built-in module: pick another name")
  data = {
    "description": description,
    "robot": robot,  # body / term names are robot-specific
    "updated": time.strftime("%Y-%m-%d %H:%M"),
    "overrides": _clean(overrides),
  }
  p = _module_path(name)
  p.parent.mkdir(parents=True, exist_ok=True)
  p.write_text(yaml.safe_dump(data, sort_keys=False))
  return data | {"name": name, "builtin": False}


def delete_module(name: str) -> None:
  if name in BUILTIN_MODULES:
    raise ValueError("built-in modules can't be deleted")
  _module_path(name).unlink(missing_ok=True)


def module_overrides(names: list[str], robot_id: str) -> list[dict]:
  out = []
  for n in names or ():
    try:
      out.append({"name": n, "overrides": load_module(n, robot_id)["overrides"]})
    except OSError as exc:
      raise ValueError(f"module {n!r} not found") from exc
  return out


# ---------------------------------------------------------------------- resolution
def resolve(name: str) -> dict:
  """Preset ``name`` with its modules inlined: all it takes to rebuild its configs (the run
  snapshot)."""
  from wbc_mjlab.tasks import get_task_config, register_all_wbc_tasks

  register_all_wbc_tasks()
  preset = load(name)
  robot = get_task_config(preset["base"]).robot_id
  return preset | {"resolved_modules": module_overrides(preset["modules"], robot)}


def snapshot_of(run_dir: Path) -> dict | None:
  p = Path(run_dir) / "params" / SNAPSHOT
  if not p.is_file():
    return None
  try:
    return yaml.safe_load(p.read_text()) or None
  except Exception:  # noqa: BLE001
    return None


def split_task_id(tid: str) -> tuple[str, str] | None:
  """``"<base>--<name>"`` → ``(base, name)`` for preset task ids (None otherwise)."""
  if SEP not in tid:
    return None
  base, name = tid.rsplit(SEP, 1)
  return base, name


def task_for_experiment(experiment: str) -> str:
  """Preset task id of a run's experiment folder ``<base experiment>--<name>`` ("" if none)."""
  if SEP not in experiment:
    return ""
  _, name = experiment.rsplit(SEP, 1)
  try:
    return task_id(load(name))
  except (OSError, ValueError):
    return ""


# ---------------------------------------------------------------------- registration
_REGISTERED: dict[str, str] = {}


def build(
  base: str,
  overrides: dict,
  *,
  play: bool = False,
  modules: list[dict] | list[str] = (),
):
  """Env + runner configs of ``base`` with ``modules`` (names, or resolved
  ``{name, overrides}``) then ``overrides`` applied (fresh copies)."""
  from wbc_mjlab.robots.env import make_wbc_env_cfg
  from wbc_mjlab.tasks import _rl_cfg_for_task, get_task_config

  cfg = get_task_config(base)
  env_cfg = make_wbc_env_cfg(cfg.robot_id, play=play, task_id=cfg.task_id)
  agent_cfg = copy.deepcopy(_rl_cfg_for_task(cfg))
  if modules and isinstance(modules[0], str):
    modules = module_overrides(list(modules), cfg.robot_id)
  for m in modules:
    apply(env_cfg, agent_cfg, m["overrides"], cfg.robot_id)
  apply(env_cfg, agent_cfg, overrides, cfg.robot_id)
  if (
    play and "actor" in env_cfg.observations
  ):  # overrides can't turn play noise back on
    env_cfg.observations["actor"].enable_corruption = False
  if play:  # play spreads envs over every terrain row (no curriculum), as the task does
    t, gen = _generator(env_cfg)
    if gen is not None:
      t.max_init_terrain_level = None
  return env_cfg, agent_cfg


def register(name: str, snapshot: dict | None = None) -> str:
  """Register preset ``name`` (or a run's resolved ``snapshot`` of it) as the mjlab / WBC task
  ``<base>--<name>``, re-registering when the content changed; returns the task id."""
  import mjlab  # noqa: F401
  from mjlab.tasks.registry import register_mjlab_task

  from wbc_mjlab import tasks as wbc_tasks
  from wbc_mjlab.rl.runner import PolicyOnlyMotionTrackingRunner
  from wbc_mjlab.tasks.config import WbcTaskConfig

  wbc_tasks.register_all_wbc_tasks()
  preset = snapshot or resolve(name)
  tid = task_id(preset)
  stamp = yaml.safe_dump(preset, sort_keys=True)
  if _REGISTERED.get(name) == stamp:
    return tid
  base = wbc_tasks.get_task_config(preset["base"])
  overrides = preset.get("overrides") or {}
  modules = preset.get("resolved_modules") or []
  env_cfg, agent_cfg = build(base.task_id, overrides, modules=modules)
  play_cfg, _ = build(base.task_id, overrides, play=True, modules=modules)
  experiment = f"{base.experiment_name}{SEP}{name}"
  agent_cfg.experiment_name = experiment
  from mjlab.tasks import registry as mj_registry

  mj_registry._REGISTRY.pop(tid, None)  # noqa: SLF001 — the preset changed: re-register
  register_mjlab_task(
    task_id=tid,
    env_cfg=env_cfg,
    play_env_cfg=play_cfg,
    rl_cfg=agent_cfg,
    runner_cls=PolicyOnlyMotionTrackingRunner,
  )
  mods = [m["name"] for m in modules]
  wbc_tasks._TASK_BY_ID[tid] = WbcTaskConfig(  # noqa: SLF001
    task_id=tid,
    robot_id=base.robot_id,
    description=preset.get("description")
    or " + ".join([base.task_id, *mods, f"{count_changes(overrides)} change(s)"]),
    experiment_name=experiment,
    build_env_cfg=lambda: build(base.task_id, overrides, modules=modules)[0],
  )
  wbc_tasks._MJLAB_REGISTERED.add(tid)  # noqa: SLF001
  _REGISTERED[name] = stamp
  return tid


__all__ = [
  "GROUPS",
  "SEP",
  "apply",
  "build",
  "config_dir",
  "count_changes",
  "delete",
  "delete_module",
  "describe",
  "library",
  "list_modules",
  "list_presets",
  "load",
  "load_module",
  "move_agent_to_recipe",
  "observation_diff",
  "register",
  "resolve",
  "save",
  "save_module",
  "snapshot_of",
  "split_task_id",
  "task_for_experiment",
  "task_id",
]
