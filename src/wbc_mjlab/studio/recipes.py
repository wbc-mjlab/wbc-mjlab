"""Training recipes: how a preset learns — the method (PPO or the world model) and its
settings — saved by name, independent of any preset.

A preset is the task (what the policy sees, how it is rewarded, where episodes start); a recipe
is how it is trained. Train picks one of each; world-model recipes share the
trainer's observation, action and environment support constraints.
An unedited fresh job passes ``--recipe <name>`` to resolve defaults in the worker;
custom edits use ``--agent.<key> <value>``. Resolved configs are saved in the run
for reproducibility and resume.

Built-in recipes: ``ppo`` and ``wm`` (task defaults), ``wm-24gb`` (2048 imagined
rollouts), ``wm-encoder`` and ``wm-blink`` (optional experiments). Yours live in
``<config dir>/recipes/<name>.yaml`` (:func:`wbc_mjlab.studio.presets.config_dir`).
"""

from __future__ import annotations

import time
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any

import yaml

from wbc_mjlab.rl.algos import ALGORITHMS, Algo, check_algo, rl_cfg
from wbc_mjlab.rl.policy import OLD_ACTOR_KEYS, PolicyCfg, migrate_actor_settings
from wbc_mjlab.rl.wm.config import WmAlgorithmCfg

# Runner config keys a recipe can set (prefixes; the rest is wiring).
_KEEP = (
  "seed",
  "num_steps_per_env",
  "save_interval",
  "light_checkpoints",
  "clip_actions",
  "algorithm",
  "actor.hidden_dims",
  "actor.activation",
  "critic.hidden_dims",
) + tuple(
  "actor." + f.name for f in dataclass_fields(PolicyCfg) if f.metadata.get("setting")
)
_SKIP = {"algorithm.class_name", "algorithm.optimizer", "algorithm.share_cnn_encoders"}

ARCHITECTURE = (
  (
    "actor.hidden_dims",
    "actor.activation",
    "critic.hidden_dims",
  )
  + tuple(
    "actor." + f.name
    for f in dataclass_fields(PolicyCfg)
    if f.metadata.get("checkpoint")
  )
  + tuple(
    "algorithm." + f.name
    for f in dataclass_fields(WmAlgorithmCfg)
    if f.metadata.get("checkpoint")
  )
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
  "algorithm.learning_rate": "base learning rate (actor multiplier configured separately)",
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
  "algorithm.dynamics_mode": "full state increments, integrated pose residuals, or velocity-only integration",
  "algorithm.separate_action_noise": "keep clean actions in history and smoothing reward; noise only at execution",
  "algorithm.diagnostic_interval": "updates between independent replay diagnostics; 0 disables",
  "num_steps_per_env": "env steps of every env per iteration",
}

_OFFICIAL_OPTIONS = {
  "algorithm.dynamics_mode": "velocity",
  "algorithm.velocity_blend": None,
  "algorithm.dynamics_loss_scale": "physical",
  "algorithm.critic_inputs": "structured",
  "algorithm.critic_layer_norm": True,
  "algorithm.critic_activation": "gelu",
  "algorithm.critic_norm_eps": 0.01,
  "algorithm.bound_critic": True,
  "algorithm.termination_mode": "hybrid",
  "algorithm.skip_nonfinite_updates": True,
  "algorithm.model_optimizer": "muon",
  "algorithm.actor_optimizer": "muon",
  "algorithm.fused_optimizer": True,
  "actor.architecture": "motion_encoder",
  "actor.reference_residual": False,
  "actor.hidden_dims": [2048, 1024, 512, 256],
  "actor.activation": "gelu",
  "algorithm.imagined_rollouts": 2048,
  "clip_actions": 20.0,
  "save_interval": 250,
  "light_checkpoints": False,
}

BUILTIN: dict[str, dict] = {
  "wm-supermimic": {
    "algo": "wm",
    "label": "World model · official-inspired",
    "description": "Experimental velocity-only dynamics, fixed physical loss and structured critic. Shares the existing trainer; not a paper reproduction.",
    "settings": dict(_OFFICIAL_OPTIONS),
  },
  "wm-blink": {
    "algo": "wm",
    "label": "World model · BlinkMimic",
    "description": "Experimental BlinkMimic learner for motion presets. Optional joint-reference shortcut and symmetry augmentation; use Wbc-G1-Blink for contact shaping. Single GPU.",
    "settings": {
      "algorithm.model_hidden_dims": [1024, 1024, 1024],
      "algorithm.model_steps": 5,
      "algorithm.imagined_rollouts": 2048,
      "algorithm.dynamics_mode": "integrated",
      "algorithm.velocity_blend": None,
      "algorithm.dynamics_loss_scale": "increment",
      "algorithm.model_optimizer": "muon",
      "algorithm.actor_optimizer": "muon",
      "algorithm.fused_optimizer": True,
      "algorithm.skip_nonfinite_updates": True,
      "algorithm.policy_ema": 0.995,
      "algorithm.critic_norm_eps": 0.01,
      "algorithm.bound_critic": True,
      "algorithm.termination_mode": "hybrid",
      "clip_actions": 20.0,
      "algorithm.mirror_probability": 0.0,
      "algorithm.learning_rate": 4e-4,
      "algorithm.actor_lr_multiplier": 4.0,
      "algorithm.actor_lr_decay_start": 1400,
      "algorithm.actor_lr_decay_updates": 300,
      "algorithm.actor_lr_return_threshold": 0.75,
      "algorithm.recent_failure_share": 0.3,
      "algorithm.probe_interval": 125,
      "algorithm.probe_start": 1000,
      "algorithm.probe_share": 0.15,
      "algorithm.compile_collection": True,
      "algorithm.collection_cuda_graphs": True,
      "algorithm.probe_cuda_graphs": True,
      "actor.architecture": "motion_encoder",
      # Task-dependent features remain explicit opt-ins: EE observations need no
      # joint reference, and custom robots need not define a reflection mapping.
      "actor.reference_residual": False,
      "actor.hidden_dims": [2048, 1024, 512, 256],
      "actor.activation": "silu",
      "save_interval": 250,
      "light_checkpoints": False,
    },
  },
  "ppo": {
    "algo": "ppo",
    "label": "PPO",
    "description": "PPO with the preset's settings",
    "settings": {},
  },
  "wm": {
    "algo": "wm",
    "label": "World model · default",
    "description": "SuperMimic's settings: 4096 imagined rollouts (needs > 24 GB)",
    "settings": {},
  },
  "wm-24gb": {
    "algo": "wm",
    "label": "World model · 24 GB GPU",
    "description": "2048 imagined rollouts: fits a 24 GB GPU (RTX 4090)",
    "settings": {"algorithm.imagined_rollouts": 2048},
  },
  "wm-encoder": {
    "algo": "wm",
    "label": "World model · combined experiment",
    "description": "Optional SuperMimic encoder/decoder and integrated residual experiment",
    "settings": {
      "actor.architecture": "motion_encoder",
      "algorithm.dynamics_mode": "integrated",
      "algorithm.separate_action_noise": True,
      "algorithm.imagined_rollouts": 2048,
      "actor.hidden_dims": [2048, 1024, 512, 256],
      "actor.activation": "silu",
    },
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
  cfg = rl_cfg(task, algo)
  for f in _walk_dataclass(cfg):
    key = f["key"]
    if key in _SKIP or not any(key == k or key.startswith(k + ".") for k in _KEEP):
      continue
    section = next(title for prefix, title in sections if key.startswith(prefix))
    main = key in ALGORITHMS[algo].settings
    if key == "actor.goal_terms":
      from mjlab.tasks.registry import load_env_cfg

      from wbc_mjlab.rl.policy import goal_term_names

      terms = load_env_cfg(task).observations["actor"].terms
      try:
        inferred = ", ".join(goal_term_names(terms))
      except ValueError:
        inferred = "select explicit terms for this preset"
      f["help"] += "; inferred: " + inferred
    out.append(
      f | {"section": section, "main": main, "help": HELP.get(key, f.get("help", ""))}
    )
  order = [title for _, title in sections]
  return sorted(out, key=lambda f: order.index(f["section"]))


def check_settings(task: str, algo: Algo, settings: dict | None) -> dict:
  """``settings`` validated against ``algo``'s settings for ``task`` (unknown keys raise)."""
  settings = migrate_recipe_settings(settings or {})
  known = {f["key"] for f in fields(task, algo)}
  for key in settings or {}:
    if key not in known:
      raise ValueError(f"{key} is not a setting of {ALGORITHMS[algo].label}")
  actor = rl_cfg(task, algo).actor
  architecture = settings.get("actor.architecture", actor.architecture)
  if architecture == "motion_encoder":
    from mjlab.tasks.registry import load_env_cfg

    from wbc_mjlab.rl.policy import goal_term_names

    goal_term_names(
      load_env_cfg(task).observations["actor"].terms,
      settings.get("actor.goal_terms", actor.goal_terms),
    )
  if settings.get("actor.reference_residual", actor.reference_residual):
    from mjlab.tasks.registry import load_env_cfg

    from wbc_mjlab.env.mdp.actions import ReferenceJointPositionActionCfg
    from wbc_mjlab.env.mdp.observations import ref_joint_pos

    env = load_env_cfg(task)
    terms = [
      t
      for t in env.observations["actor"].terms.values()
      if t is not None and t.func is ref_joint_pos
    ]
    if (
      len(terms) != 1
      or terms[0].params.get("frames", 1) < 2
      or isinstance(env.actions.get("joint_pos"), ReferenceJointPositionActionCfg)
    ):
      raise ValueError(
        "reference residual policy requires joint lookahead and default-relative actions"
      )
  return dict(settings or {})


def migrate_recipe_settings(settings):
  aliases = {"algorithm." + old: "actor." + new for old, new in OLD_ACTOR_KEYS.items()}
  out = {key: value for key, value in settings.items() if key not in aliases}
  for old, new in aliases.items():
    if old in settings:
      out.setdefault(new, settings[old])
  return out


def architecture_of(run_dir: Path | str) -> dict:
  """The :data:`ARCHITECTURE` settings run ``run_dir`` was trained with (those its
  ``params/agent.yaml`` holds)."""
  from wbc_mjlab.run_params import read_run_params

  agent = migrate_actor_settings(read_run_params(run_dir, "agent"))
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
    "settings": migrate_recipe_settings(data.get("settings") or {}),
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
