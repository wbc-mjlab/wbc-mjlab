"""Training recipes: how a preset learns — the method (PPO or the world model) and its
settings — saved by name, independent of any preset.

A preset is the task (what the policy sees, how it is rewarded, where episodes start); a recipe
is how it is trained. Train picks one of each, so any recipe trains any preset. A job passes
the recipe as ``--algo <method> --agent.<key> <value>`` to ``wbc-mjlab-train``: the command
alone reproduces the run.

Built-in recipes: ``ppo`` and ``wm`` (each method with the task's own settings) and
``wm-24gb`` (2048 imagined rollouts, what fits a 24 GB GPU). Yours live in
``<config dir>/recipes/<name>.yaml`` (:func:`wbc_mjlab.studio.presets.config_dir`).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import yaml

from wbc_mjlab.rl.algos import ALGORITHMS, Algo, check_algo, rl_cfg

# Runner config keys a recipe can set (prefixes; the rest is wiring).
_KEEP = (
  "seed",
  "num_steps_per_env",
  "save_interval",
  "algorithm",
  "actor.hidden_dims",
  "actor.activation",
  "critic.hidden_dims",
)
_SKIP = {"algorithm.class_name", "algorithm.optimizer", "algorithm.share_cnn_encoders"}

ARCHITECTURE = (
  "actor.hidden_dims",
  "actor.activation",
  "critic.hidden_dims",
  "algorithm.model_hidden_dims",
  "algorithm.heads_hidden_dims",
)
"""Settings a checkpoint fixes: continuing a run keeps that run's values."""

SECTIONS: dict[Algo, tuple[tuple[str, str], ...]] = {
  "ppo": (("algorithm.", "Optimisation"),),
  "wm": (("algorithm.", "World model & imagination"),),
}
_COMMON_SECTIONS = (
  ("actor.", "Policy network"),
  ("critic.", "Value network"),
  ("", "Rollouts"),
)

HELP = {
  "algorithm.learning_rate": "Adam learning rate",
  "algorithm.entropy_coef": "exploration bonus",
  "actor.hidden_dims": "the policy both methods train and export; checkpoints of other sizes won't load",
  "critic.hidden_dims": "value network",
  "algorithm.imagined_rollouts": "rollouts per policy update; 2048 fits a 24 GB GPU",
  "algorithm.horizon": "imagined steps per rollout; the critic bootstraps beyond",
  "algorithm.window": "steps of the windows the world model is fit on",
  "algorithm.model_hidden_dims": "dynamics network; smaller is faster, less accurate",
  "algorithm.updates_per_iteration": "updates per iteration of num_steps_per_env env steps",
  "algorithm.action_noise": "std of the exploration noise on actions in the simulator",
  "algorithm.replay_steps": "steps of every env kept in the replay",
  "algorithm.compile": "torch.compile the update (first update compiles for minutes)",
  "algorithm.cuda_graphs": "run each update as one CUDA graph (with compile)",
  "num_steps_per_env": "env steps of every env per iteration",
}

BUILTIN: dict[str, dict] = {
  "ppo": {
    "algo": "ppo",
    "label": "PPO",
    "description": "PPO with the preset's settings",
    "settings": {},
  },
  "wm": {
    "algo": "wm",
    "label": "World model · paper",
    "description": "SuperMimic's settings: 4096 imagined rollouts (needs > 24 GB)",
    "settings": {},
  },
  "wm-24gb": {
    "algo": "wm",
    "label": "World model · 24 GB GPU",
    "description": "2048 imagined rollouts: fits a 24 GB GPU (RTX 4090)",
    "settings": {"algorithm.imagined_rollouts": 2048},
  },
}


def recipes_dir() -> Path:
  from wbc_mjlab.studio.presets import config_dir

  return config_dir() / "recipes"


def _path(name: str) -> Path:
  from wbc_mjlab.studio.presets import _check

  return recipes_dir() / f"{_check(name)}.yaml"


# ---------------------------------------------------------------------- settings
def fields(task: str, algo: Algo) -> list[dict]:
  """Every setting of ``algo`` for ``task``: ``{key, kind, value, section, main, help}``
  with the task's value as ``value`` (``main``: the method's settings shown first)."""
  from wbc_mjlab.studio.presets import _walk_dataclass

  check_algo(algo)
  sections = (*SECTIONS[algo], *_COMMON_SECTIONS)
  out = []
  for f in _walk_dataclass(rl_cfg(task, algo)):
    key = f["key"]
    if key in _SKIP or not any(key == k or key.startswith(k + ".") for k in _KEEP):
      continue
    section = next(title for prefix, title in sections if key.startswith(prefix))
    main = key in ALGORITHMS[algo].settings
    out.append(f | {"section": section, "main": main, "help": HELP.get(key, "")})
  order = [title for _, title in sections]
  return sorted(out, key=lambda f: order.index(f["section"]))


def check_settings(task: str, algo: Algo, settings: dict | None) -> dict:
  """``settings`` validated against ``algo``'s settings for ``task`` (unknown keys raise)."""
  known = {f["key"] for f in fields(task, algo)}
  for key in settings or {}:
    if key not in known:
      raise ValueError(f"{key} is not a setting of {ALGORITHMS[algo].label}")
  return dict(settings or {})


def architecture_of(run_dir: Path | str) -> dict:
  """The :data:`ARCHITECTURE` settings run ``run_dir`` was trained with (those its
  ``params/agent.yaml`` holds)."""
  from wbc_mjlab.run_params import read_run_params

  agent = read_run_params(run_dir, "agent")
  out = {}
  for key in ARCHITECTURE:
    value: Any = agent
    for part in key.split("."):
      value = value.get(part) if isinstance(value, dict) else None
    if value is not None:
      out[key] = list(value) if isinstance(value, (list, tuple)) else value
  return out


# ---------------------------------------------------------------------- storage
def list_recipes() -> list[dict]:
  """Built-in recipes, then yours: ``{name, algo, label, description, settings, builtin}``."""
  out = [r | {"name": n, "builtin": True} for n, r in BUILTIN.items()]
  d = recipes_dir()
  for p in sorted(d.glob("*.yaml")) if d.is_dir() else []:
    try:
      out.append(load(p.stem))
    except (OSError, ValueError, yaml.YAMLError):
      continue
  return out


def load(name: str) -> dict:
  if name in BUILTIN:
    return BUILTIN[name] | {"name": name, "builtin": True}
  data = yaml.safe_load(_path(name).read_text()) or {}
  check_algo(data.get("algo", ""))
  return {
    "name": name,
    "algo": data["algo"],
    "label": name,
    "description": data.get("description", ""),
    "settings": dict(data.get("settings") or {}),
    "builtin": False,
    "updated": data.get("updated", ""),
  }


def save(name: str, algo: str, settings: dict, description: str = "") -> dict:
  if name in BUILTIN:
    raise ValueError(f"{name!r} is a built-in recipe: pick another name")
  check_algo(algo)
  data = {
    "algo": algo,
    "description": description,
    "updated": time.strftime("%Y-%m-%d %H:%M"),
    "settings": {k: _plain(v) for k, v in (settings or {}).items()},
  }
  p = _path(name)
  p.parent.mkdir(parents=True, exist_ok=True)
  p.write_text(yaml.safe_dump(data, sort_keys=False))
  return load(name)


def delete(name: str) -> None:
  if name in BUILTIN:
    raise ValueError("built-in recipes can't be deleted")
  _path(name).unlink(missing_ok=True)


def default(algo: Algo) -> str:
  """The recipe Train starts ``algo`` with: on a GPU under 40 GB the world model's 24 GB one
  (4096 imagined rollouts ran out of memory on 24 GB)."""
  if algo == "wm":
    import torch

    if torch.cuda.is_available():
      gb = torch.cuda.get_device_properties(0).total_memory / 2**30
      if gb < 40:
        return "wm-24gb"
  return algo


def _plain(v: Any) -> Any:
  return list(v) if isinstance(v, tuple) else v


__all__ = [
  "ARCHITECTURE",
  "BUILTIN",
  "architecture_of",
  "check_settings",
  "default",
  "delete",
  "fields",
  "list_recipes",
  "load",
  "recipes_dir",
  "save",
]
