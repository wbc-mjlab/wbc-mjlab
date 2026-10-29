"""Shared saved-run reconstruction for resume, Play and deterministic evaluation.

Registered Python owns classes and functions; ``params/*.yaml`` restores their data.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnvCfg

  from wbc_mjlab.rl.policy import TrackingRunnerCfg


class _Loader(yaml.SafeLoader):
  pass


def _python_data(loader, suffix, node):
  """Read mjlab's Python YAML tags as data, with a single built-in slice whitelist."""
  if suffix.startswith("name:"):
    return suffix[5:]
  if suffix == "object/apply:builtins.slice":
    return slice(*loader.construct_sequence(node))
  if suffix == "tuple":
    return tuple(loader.construct_sequence(node))
  if isinstance(node, yaml.SequenceNode):
    return loader.construct_sequence(node)
  if isinstance(node, yaml.MappingNode):
    return loader.construct_mapping(node)
  return loader.construct_scalar(node)


_Loader.add_multi_constructor("tag:yaml.org,2002:python/", _python_data)


def read_run_params(run_dir: Path | str, name: str) -> dict:
  """``params/<name>.yaml`` of a run; empty when missing or unreadable."""
  path = Path(run_dir) / "params" / f"{name}.yaml"
  if not path.is_file():
    return {}
  try:
    doc = yaml.load(path.read_text(errors="replace"), Loader=_Loader)  # noqa: S506
  except yaml.YAMLError:
    return {}
  return doc if isinstance(doc, dict) else {}


def restore_config(target, values, path=""):
  """Restore data into an existing registered config; never import YAML callables.

  Manager term dictionaries retain the saved order and enabled set. Code/function
  identities must still match the registered task or its saved Studio preset.
  """
  for field in fields(target):
    key = field.name
    if key not in values:
      continue
    value, current = values[key], getattr(target, key)
    location = f"{path}.{key}" if path else key
    if key in ("func", "spec_fn") or callable(current):
      identity = (
        f"{current.__module__}.{current.__qualname__}"
        if callable(current) and hasattr(current, "__qualname__")
        else ""
      )
      if value and isinstance(value, str) and identity and value != identity:
        raise ValueError(
          f"{location}: saved callable {value!r} differs from registered {identity!r}"
        )
      continue
    if isinstance(value, str) and value.startswith("<non-serializable:"):
      continue
    if isinstance(current, Enum):
      # Older yaml.dump represents enum constructors as one-element sequences.
      setattr(
        target,
        key,
        type(current)(value[0] if isinstance(value, (tuple, list)) else value),
      )
    elif is_dataclass(current) and isinstance(value, dict):
      restore_config(current, value, location)
    elif (
      isinstance(current, dict)
      and isinstance(value, dict)
      and any(is_dataclass(v) for v in current.values())
    ):
      restored = {}
      for name, item in value.items():
        if item is None:
          continue
        if name not in current:
          raise ValueError(
            f"{location}.{name}: saved term is missing from the registered task"
          )
        obj = copy.deepcopy(current[name])
        if is_dataclass(obj) and isinstance(item, dict):
          restore_config(obj, item, f"{location}.{name}")
        else:
          obj = copy.deepcopy(item)
        restored[name] = obj
      setattr(target, key, restored)
    elif (
      isinstance(current, (tuple, list))
      and isinstance(value, (tuple, list))
      and any(is_dataclass(v) for v in current)
    ):
      if len(current) != len(value):
        raise ValueError(
          f"{location}: saved config sequence needs its registered preset"
        )
      restored = []
      for i, (obj, item) in enumerate(zip(current, value, strict=True)):
        obj = copy.deepcopy(obj)
        if is_dataclass(obj) and isinstance(item, dict):
          restore_config(obj, item, f"{location}[{i}]")
        else:
          obj = copy.deepcopy(item)
        restored.append(obj)
      setattr(target, key, tuple(restored) if isinstance(current, tuple) else restored)
    elif isinstance(current, tuple) and isinstance(value, (tuple, list)):
      setattr(target, key, tuple(value))
    elif current is None and isinstance(value, dict):
      raise ValueError(
        f"{location}: saved config needs a registered preset to rebuild it"
      )
    else:
      setattr(target, key, copy.deepcopy(value))


def run_task_id(run_dir: Path | str) -> str:
  """Task identity from saved metadata, with compatibility for older run layouts."""
  from wbc_mjlab.studio.presets import snapshot_of, task_for_experiment, task_id
  from wbc_mjlab.tasks import all_wbc_tasks

  root = Path(run_dir)
  saved = read_run_params(root, "run").get("task_id")
  if saved:
    return saved
  snapshot = snapshot_of(root)
  if snapshot:
    return task_id(snapshot)
  experiment = read_run_params(root, "agent").get("experiment_name", root.parent.name)
  if root.name == "wbc_g1" and root.parent.name in ("demo", "demos"):
    return "Wbc-G1"
  for task in all_wbc_tasks():
    if task.experiment_name == experiment:
      return task.task_id
  return task_for_experiment(experiment)


@dataclass
class RunConfig:
  task_id: str
  env: ManagerBasedRlEnvCfg
  agent: TrackingRunnerCfg
  run_dir: Path


def load_run_config(
  run_dir: Path | str, task_id: str | None = None, *, play: bool = True
) -> RunConfig:
  """Shared saved-task reconstruction for CLI/Studio Play, evaluation and resume.

  Build registered Python code first, restore saved data second, then apply explicit
  Play overrides. Evaluation applies its protocol after this function returns.
  """
  from mjlab.tasks.registry import load_env_cfg

  from wbc_mjlab.rl.algos import env_cfg, run_algo, run_rl_cfg
  from wbc_mjlab.studio.presets import register, snapshot_of, split_task_id
  from wbc_mjlab.tasks import prepare_wbc_run

  root = Path(run_dir)
  saved_task = run_task_id(root)
  if task_id and saved_task and task_id != saved_task:
    raise ValueError(f"run was trained with {saved_task}, not {task_id}")
  selected = task_id or saved_task
  if not selected:
    raise ValueError("cannot infer this legacy run's task; provide --task")
  preset = split_task_id(selected)
  if preset:
    register(preset[1], snapshot=snapshot_of(root))
  prepare_wbc_run(task_id=selected)
  cfg = env_cfg(load_env_cfg(selected), run_algo(root))
  saved_env = read_run_params(root, "env")
  # Robot definitions and sensor classes come from registered Python/preset code;
  # task behavior and terrain parameters come from the saved run.
  for name in (
    "observations",
    "rewards",
    "terminations",
    "events",
    "commands",
    "actions",
    "curriculum",
    "sim",
    "seed",
    "metrics",
    "decimation",
    "is_finite_horizon",
    "scale_rewards_by_dt",
    "episode_length_s",
  ):
    if name in saved_env:
      restore_config(cfg, {name: saved_env[name]})
  scene = saved_env.get("scene", {})
  restore_config(
    cfg.scene,
    {k: v for k, v in scene.items() if k not in ("entities", "sensors", "spec_fn")},
  )
  change = actor_layout_change(root, cfg)
  if change:
    raise ValueError(change)
  from wbc_mjlab.motion.library import apply_run_motion_options

  apply_run_motion_options(cfg.commands["motion"], root)
  if play:
    cfg.episode_length_s = 1e9
    cfg.curriculum = {}
    cfg.observations["actor"].enable_corruption = False
    for name in ("push_robot", "pull_robot", "assistive_wrench"):
      cfg.events.pop(name, None)
    command = cfg.commands["motion"]
    command.pose_range = {}
    command.velocity_range = {}
    command.assistive_wrench_enabled = False
    if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_type == "generator":
      cfg.scene.terrain.max_init_terrain_level = None
  return RunConfig(selected, cfg, run_rl_cfg(selected, root), root)


# Term parameters that change what a term's values mean or where they sit.
_LAYOUT_PARAMS = ("frames", "stride", "num_frames", "body_names")


def _layout(group: dict) -> list[tuple]:
  hist = group.get("history_length")
  out = []
  for name, term in (group.get("terms") or {}).items():
    if term is None:
      continue
    params = term.get("params") or {}
    shape = tuple(
      (k, tuple(v) if isinstance(v, (list, tuple)) else v)
      for k, v in sorted(params.items())
      if k in _LAYOUT_PARAMS
    )
    out.append(
      (name, shape, int(hist if hist is not None else term.get("history_length") or 0))
    )
  return out


def actor_layout_change(run_dir: Path | str, env_cfg) -> str:
  """What differs between the actor observation run ``run_dir`` was trained on
  (``params/env.yaml``) and the one ``env_cfg`` builds: term order, frames ahead, history
  ("" when they match or the run has no record). A policy fed another layout of the same
  size would load and act on scrambled inputs."""
  import dataclasses

  trained = read_run_params(run_dir, "env").get("observations", {}).get("actor")
  if not trained:
    return ""
  group = env_cfg.observations["actor"]
  now = {
    "history_length": group.history_length,
    "terms": {
      n: {"params": t.params, "history_length": t.history_length}
      for n, t in group.terms.items()
      if dataclasses.is_dataclass(t)
    },
  }
  a, b = _layout(trained), _layout(now)
  if a == b:
    return ""
  if [x[0] for x in a] != [x[0] for x in b]:
    return (
      f"trained with actor terms {[x[0] for x in a]}, the task now builds "
      f"{[x[0] for x in b]}"
    )
  diff = [f"{x[0]}: {x[1:]} → {y[1:]}" for x, y in zip(a, b, strict=True) if x != y]
  return "the actor's terms changed since training: " + "; ".join(diff)
