"""Training algorithms of a run: ``ppo`` (default) or ``wm`` (world model, :mod:`wbc_mjlab.rl.wm`).

Every algorithm trains the same tasks and presets (env, motion options) and exports the same
policy; it brings its own runner config (``params/agent.yaml``, which Play reads back) and
may adapt the env. :data:`ALGORITHMS` describes them for the CLI and the Studio (whose
recipes, :mod:`wbc_mjlab.studio.recipes`, are an algorithm and its settings).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

Algo = Literal["ppo", "wm"]
ALGOS: tuple[Algo, ...] = ("ppo", "wm")


@dataclass(frozen=True)
class AlgoInfo:
  """What an algorithm is, for listings and the Studio's Train form."""

  id: Algo
  label: str
  summary: str
  settings: tuple[str, ...] = ()
  """Runner config keys worth setting per run (``algorithm.imagined_rollouts`` …)."""
  iterations: tuple[int, ...] = (1000, 5000, 30000, 100000)
  """Typical ``max_iterations`` budgets (an iteration's length differs per algorithm)."""
  notes: str = ""


ALGORITHMS: dict[Algo, AlgoInfo] = {
  "ppo": AlgoInfo(
    id="ppo",
    settings=("actor.architecture",),
    label="PPO",
    summary=(
      "On-policy RL in the simulator: robust, works with every preset. "
      "An iteration is num_steps_per_env (24) env steps of every env and one PPO update."
    ),
  ),
  "wm": AlgoInfo(
    id="wm",
    label="World model",
    summary=(
      "SuperMimic: a world model learns the robot's dynamics from the simulator and the "
      "policy is trained by backpropagation through imagined rollouts. "
      "An iteration is 5 env steps and 2 updates."
    ),
    settings=(
      "actor.architecture",
      "algorithm.imagined_rollouts",
      "algorithm.horizon",
      "algorithm.model_hidden_dims",
      "algorithm.compile",
    ),
    iterations=(5000, 20000, 60000, 160000),
    notes=(
      "Pushes, pulls and the assistive wrench are off (a deterministic model cannot "
      "predict them). The first update compiles for a few minutes."
    ),
  ),
}
"""Every training algorithm (what ``--algo`` accepts), by id."""


def check_algo(algo: str) -> Algo:
  if algo not in ALGORITHMS:
    raise ValueError(f"unknown algorithm {algo!r}; choose from {', '.join(ALGOS)}")
  return algo  # type: ignore[return-value]


def runner_cfg(ppo_cfg, algo: Algo):
  """``algo``'s runner config derived from a task's PPO runner config ``ppo_cfg`` (the
  policy network, experiment and logger carry over)."""
  check_algo(algo)
  if algo == "wm":
    from wbc_mjlab.rl.wm.config import wm_runner_cfg

    return wm_runner_cfg(ppo_cfg)
  from wbc_mjlab.rl.policy import tracking_runner_cfg

  return tracking_runner_cfg(ppo_cfg)


def rl_cfg(task_id: str, algo: Algo = "ppo"):
  """Runner config of ``task_id`` for ``algo``."""
  from mjlab.tasks.registry import load_rl_cfg

  return runner_cfg(load_rl_cfg(task_id), algo)


def env_cfg(cfg, algo: Algo):
  """The env ``algo`` trains (and plays) ``cfg`` in: ``wm`` turns off external forces."""
  check_algo(algo)
  if algo == "wm":
    from wbc_mjlab.rl.wm.config import wm_env_cfg

    return wm_env_cfg(cfg)
  return cfg


def select_algo(task_id: str, algo: Algo) -> None:
  """Make ``algo`` the task's runner (and env) config in this process, before mjlab's
  train CLI reads them."""
  from mjlab.tasks import registry

  check_algo(algo)
  if algo != "ppo":
    entry = registry._REGISTRY[task_id]  # mjlab has no public re-registration
    registry._REGISTRY[task_id] = replace(
      entry, env_cfg=env_cfg(entry.env_cfg, algo), rl_cfg=rl_cfg(task_id, algo)
    )


def run_algo(run_dir: Path | str) -> Algo:
  """Algorithm a run was trained with (``params/agent.yaml``; PPO when absent)."""
  from wbc_mjlab.rl.wm.config import WmAlgorithmCfg
  from wbc_mjlab.run_params import read_run_params

  algorithm = read_run_params(run_dir, "agent").get("algorithm") or {}
  return "wm" if algorithm.get("class_name") == WmAlgorithmCfg.class_name else "ppo"


def run_rl_cfg(task_id: str, run_dir: Path | str):
  """Rebuild a run's architecture for Play/evaluation, including optional encoders.

  Apply known dataclass fields only: YAML is data, never imported or evaluated.
  """
  from wbc_mjlab.rl.policy import migrate_actor_settings
  from wbc_mjlab.run_params import read_run_params, restore_config

  cfg = rl_cfg(task_id, run_algo(run_dir))

  restore_config(cfg, migrate_actor_settings(read_run_params(run_dir, "agent")))
  return cfg


def check_env(task_id: str, algo: Algo, motion_file: str, *, num_envs: int = 2) -> dict:
  """Build ``task_id``'s env for ``algo`` (``num_envs`` envs on ``motion_file``) and check
  that ``algo`` can train it: ``{"ok", "error", "details"}``. For ``wm``, ``details`` names
  the rewards computed exactly in imagination, those learned, and the randomization the
  model is given as context."""
  import torch
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  check_algo(algo)
  cfg = env_cfg(load_env_cfg(task_id), algo)
  cfg.scene.num_envs = num_envs
  cfg.commands["motion"].motion_file = motion_file
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  try:
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
  except Exception as e:  # noqa: BLE001
    return {"ok": False, "error": f"env: {e}", "details": {}}
  try:
    if algo != "wm":
      return {"ok": True, "error": "", "details": {}}
    from wbc_mjlab.rl.wm.imagined import Imagination

    env.reset()
    try:
      im = Imagination(env)
    except ValueError as e:
      return {"ok": False, "error": str(e), "details": {}}
    return {
      "ok": True,
      "error": "",
      "details": {
        "exact_rewards": [n for n, _ in im.analytic_rewards],
        "learned_rewards": [n for _, n, _ in im.learned_rewards],
        "context": list(im.context_names),
      },
    }
  finally:
    env.close()


__all__ = [
  "ALGORITHMS",
  "ALGOS",
  "Algo",
  "AlgoInfo",
  "check_algo",
  "check_env",
  "env_cfg",
  "rl_cfg",
  "run_algo",
  "run_rl_cfg",
  "runner_cfg",
  "select_algo",
]
