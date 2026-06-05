"""Wrapper for mjlab's play script that auto-registers wbc_mjlab tasks.

Adds auto-checkpoint discovery: when no ``--checkpoint-file``,
``--wandb-run-path``, or dummy ``--agent`` flag is given, the script
looks in ``logs/rsl_rl/<experiment_name>/`` for the latest run folder and
picks the highest-iteration ``model_<N>.pt`` checkpoint there.

For **any** registered WBC tracking task, policy ONNX and
``wbc_tracking_params.yaml`` are written to ``<checkpoint_run_dir>/params/`` **before**
the play viewer opens (see ``wbc_mjlab.deploy_paths``).

``run_play`` below mirrors ``mjlab.scripts.play.run_play`` with that hook inserted;
keep it aligned when upgrading mjlab.

Usage:
  wbc-mjlab-play --robot g1 --dataset lafan
  wbc-mjlab-play --robot g1 --dataset-path /path/to/lafan
  wbc-mjlab-play --robot g1 --no-state-estimation --checkpoint-file /path/to/model.pt
"""

from __future__ import annotations

import os
import sys
import time as _time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
from mjlab.scripts.play import PlayConfig
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.rl.runner import MotionTrackingOnPolicyRunner
from wbc_mjlab.rl.runner import PolicyOnlyMotionTrackingRunner
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
from mjlab.viewer.viser.viewer import CheckpointManager, format_time_ago

from wbc_mjlab.deploy_paths import PLAY_ONNX_LATEST_NAME, PLAY_PARAMS_SUBDIR
from wbc_mjlab.env.mdp.commands import MotionCommand, MotionCommandCfg


def _parse_wandb_dt(value: str | datetime) -> datetime:
  """Parse a W&B datetime string (or pass through a datetime object)."""
  if isinstance(value, str):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))
  return value


def _export_onnx_pre_viewer(
  cfg: PlayConfig,
  runner: MjlabOnPolicyRunner,
  log_dir: Path | None,
) -> None:
  """Write ``params/latest.onnx`` using the loaded play runner (before viewer)."""
  if cfg.agent in ("zero", "random"):
    return
  if log_dir is None:
    return

  params_dir = log_dir / PLAY_PARAMS_SUBDIR
  onnx_path = params_dir / PLAY_ONNX_LATEST_NAME
  try:
    params_dir.mkdir(parents=True, exist_ok=True)
    runner.export_policy_to_onnx(str(params_dir), PLAY_ONNX_LATEST_NAME)
    if isinstance(runner, PolicyOnlyMotionTrackingRunner):
      metadata = runner.build_policy_export_metadata(log_dir.name)
    else:
      metadata = get_base_metadata(runner.env.unwrapped, log_dir.name)
      if isinstance(runner, MotionTrackingOnPolicyRunner):
        motion_term = cast(
          MotionCommand, runner.env.unwrapped.command_manager.get_term("motion")
        )
        metadata.update(
          {
            "anchor_body_name": motion_term.cfg.anchor_body_name,
            "body_names": list(motion_term.cfg.body_names),
          }
        )
    attach_metadata_to_onnx(str(onnx_path), metadata)
    print(f"[INFO] Pre-viewer ONNX written to {onnx_path.resolve()}")
  except Exception as e:
    print(f"[WARN] Pre-viewer ONNX export failed: {e}")
    return

  if isinstance(runner, PolicyOnlyMotionTrackingRunner):
    try:
      from wbc_mjlab.export.policy_bundle import export_tracking_params_yaml
      from wbc_mjlab.tasks import last_registered_robot_id

      yaml_path = export_tracking_params_yaml(
        params_dir,
        runner.env.unwrapped.cfg,
        robot_id=last_registered_robot_id(),
      )
      print(f"[INFO] WBC tracking params written to {yaml_path.resolve()}")
    except Exception as e:
      print(f"[WARN] WBC tracking params export failed: {e}")


def _find_latest_checkpoint(experiment_name: str) -> Path | None:
  """Return the highest-iteration .pt file from the most recent run folder."""
  log_root = Path("logs") / "rsl_rl" / experiment_name
  if not log_root.exists():
    return None

  run_dirs = sorted(
    (d for d in log_root.iterdir() if d.is_dir() and not d.name.startswith(".")),
    key=lambda d: d.name,
  )
  if not run_dirs:
    return None

  for run_dir in reversed(run_dirs):
    checkpoints = [f for f in run_dir.glob("model_*.pt")]
    if not checkpoints:
      continue

    def _iteration(p: Path) -> int:
      try:
        return int(p.stem.split("_", 1)[1])
      except (IndexError, ValueError):
        return -1

    return max(checkpoints, key=_iteration)

  return None


def _needs_auto_checkpoint() -> bool:
  """Return True when the user has not already specified how to load weights."""
  argv = sys.argv[1:]
  explicit_flags = {
    "--checkpoint-file",
    "--checkpoint_file",
    "--wandb-run-path",
    "--wandb_run_path",
  }
  for flag in explicit_flags:
    if flag in argv:
      return False
  for i, arg in enumerate(argv):
    if arg in ("--agent", "--agent=zero", "--agent=random"):
      if arg.startswith("--agent="):
        return False
      if i + 1 < len(argv) and argv[i + 1] in ("zero", "random"):
        return False
  return True


def run_play(task_id: str, cfg: PlayConfig) -> None:
  """Same as ``mjlab.scripts.play.run_play`` plus pre-viewer ONNX export."""

  configure_torch_backends()

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)

  DUMMY_MODE = cfg.agent in {"zero", "random"}
  TRAINED_MODE = not DUMMY_MODE

  if cfg.no_terminations:
    env_cfg.terminations = {}
    print("[INFO]: Terminations disabled")

  is_tracking_task = "motion" in env_cfg.commands and isinstance(
    env_cfg.commands["motion"], MotionCommandCfg
  )

  if is_tracking_task and cfg._demo_mode:
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.sampling_mode = "uniform"

  if is_tracking_task:
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)

    if cfg.motion_file is not None and Path(cfg.motion_file).exists():
      print(f"[INFO]: Using local motion file: {cfg.motion_file}")
      motion_cmd.motion_file = cfg.motion_file
    elif DUMMY_MODE:
      if not cfg.registry_name:
        raise ValueError(
          "Tracking tasks require either:\n"
          "  --motion-file /path/to/motion.npz (local file)\n"
          "  --registry-name your-org/motions/motion-name (download from WandB)"
        )
      registry_name = cfg.registry_name
      if ":" not in registry_name:
        registry_name = registry_name + ":latest"
      import wandb

      api = wandb.Api()
      artifact = api.artifact(registry_name)
      motion_cmd.motion_file = str(Path(artifact.download()) / "motion.npz")
    else:
      if cfg.motion_file is not None:
        print(f"[INFO]: Using motion file from CLI: {cfg.motion_file}")
        motion_cmd.motion_file = cfg.motion_file
      else:
        import wandb

        api = wandb.Api()
        if cfg.wandb_run_path is None and cfg.checkpoint_file is not None:
          raise ValueError(
            "Tracking tasks require `motion_file` when using `checkpoint_file`, "
            "or provide `wandb_run_path` so the motion artifact can be resolved."
          )
        if cfg.wandb_run_path is not None:
          wandb_run = api.run(str(cfg.wandb_run_path))
          art = next(
            (a for a in wandb_run.used_artifacts() if a.type == "motions"), None
          )
          if art is None:
            raise RuntimeError("No motion artifact found in the run.")
          motion_cmd.motion_file = str(Path(art.download()) / "motion.npz")

  log_dir: Path | None = None
  resume_path: Path | None = None
  if TRAINED_MODE:
    log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
    if cfg.checkpoint_file is not None:
      resume_path = Path(cfg.checkpoint_file)
      if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
      print(f"[INFO]: Loading checkpoint: {resume_path.name}")
    else:
      if cfg.wandb_run_path is None:
        raise ValueError(
          "`wandb_run_path` is required when `checkpoint_file` is not provided."
        )
      resume_path, was_cached = get_wandb_checkpoint_path(
        log_root_path, Path(cfg.wandb_run_path), cfg.wandb_checkpoint_name
      )
      run_id = resume_path.parent.name
      checkpoint_name = resume_path.name
      cached_str = "cached" if was_cached else "downloaded"
      print(
        f"[INFO]: Loading checkpoint: {checkpoint_name} (run: {run_id}, {cached_str})"
      )
    log_dir = resume_path.parent

  env_cfg.scene.num_envs = cfg.num_envs if cfg.num_envs is not None else 1
  if cfg.video_height is not None:
    env_cfg.viewer.height = cfg.video_height
  if cfg.video_width is not None:
    env_cfg.viewer.width = cfg.video_width

  render_mode = "rgb_array" if (TRAINED_MODE and cfg.video) else None
  if cfg.video and DUMMY_MODE:
    print(
      "[WARN] Video recording with dummy agents is disabled (no checkpoint/log_dir)."
    )
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

  if TRAINED_MODE and cfg.video:
    print("[INFO] Recording videos during play")
    assert log_dir is not None
    env = VideoRecorder(
      env,
      video_folder=log_dir / "videos" / "play",
      step_trigger=lambda step: step == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )

  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  if DUMMY_MODE:
    action_shape: tuple[int, ...] = env.unwrapped.action_space.shape
    if cfg.agent == "zero":

      class PolicyZero:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return torch.zeros(action_shape, device=env.unwrapped.device)

      policy = PolicyZero()
    else:

      class PolicyRandom:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1

      policy = PolicyRandom()
  else:
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(
      str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device
    )
    policy = runner.get_inference_policy(device=device)
    _export_onnx_pre_viewer(cfg, runner, log_dir)

  ckpt_manager: CheckpointManager | None = None
  if TRAINED_MODE and resume_path is not None:
    _ckpt_runner = runner  # pyright: ignore[reportPossiblyUnboundVariable]

    def _reload_policy(path: str):
      _ckpt_runner.load(
        path,
        load_cfg={"actor": True},
        strict=True,
        map_location=device,
      )
      return _ckpt_runner.get_inference_policy(device=device)

    if cfg.wandb_run_path is None:
      ckpt_dir = resume_path.parent

      def fetch_available_local() -> list[tuple[str, str]]:
        now = _time.time()
        entries: list[tuple[str, str, int]] = []
        for f in sorted(ckpt_dir.glob("*.pt")):
          try:
            step = int(f.stem.split("_")[1])
          except (IndexError, ValueError):
            step = 0
          ago = format_time_ago(int(now - f.stat().st_mtime))
          entries.append((f.name, ago, step))
        entries.sort(key=lambda x: x[2])
        return [(name, t) for name, t, _ in entries]

      ckpt_manager = CheckpointManager(
        current_name=resume_path.name,
        fetch_available=fetch_available_local,
        load_checkpoint=lambda name: _reload_policy(str(ckpt_dir / name)),
      )
    else:
      import wandb

      api = wandb.Api()
      run_path = str(cfg.wandb_run_path)
      wandb_run = api.run(run_path)
      _log_root = log_root_path  # pyright: ignore[reportPossiblyUnboundVariable]

      def fetch_available_wandb() -> list[tuple[str, str]]:
        wandb_run.load()
        now = datetime.now(tz=timezone.utc)
        entries: list[tuple[str, str, int]] = []
        for f in wandb_run.files():
          if not f.name.endswith(".pt"):
            continue
          try:
            step = int(f.name.split("_")[1].split(".")[0])
          except (IndexError, ValueError):
            step = 0
          ago = format_time_ago(
            int((now - _parse_wandb_dt(f.updated_at)).total_seconds())
          )
          entries.append((f.name, ago, step))
        entries.sort(key=lambda x: x[2])
        return [(name, t) for name, t, _ in entries]

      ckpt_manager = CheckpointManager(
        current_name=resume_path.name,
        fetch_available=fetch_available_wandb,
        load_checkpoint=lambda name: _reload_policy(
          str(get_wandb_checkpoint_path(_log_root, Path(run_path), name)[0])
        ),
        run_name=_parse_wandb_dt(wandb_run.created_at).strftime("%Y-%m-%d_%H-%M-%S"),
        run_url=wandb_run.url,
        run_status=wandb_run.state,
      )

  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
    del has_display
  else:
    resolved_viewer = cfg.viewer

  if resolved_viewer == "native":
    NativeMujocoViewer(env, policy).run()
  elif resolved_viewer == "viser":
    ViserPlayViewer(env, policy, checkpoint_manager=ckpt_manager).run()
  else:
    raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")

  env.close()


def main() -> None:
  from wbc_mjlab.tasks import prepare_wbc_run
  from wbc_mjlab.scripts.wbc_cli import (
    apply_dataset_motion_file,
    ensure_task_id,
    parse_wbc_argv,
  )

  prog = sys.argv[0]
  rest, robot, task_id, no_se, legacy, dataset, dataset_path, cache_motion_bundle = (
    parse_wbc_argv(sys.argv[1:])
  )
  if legacy:
    print("[WARN] Legacy task id mapped to --task / --no-state-estimation")
  prepare_wbc_run(task_id=task_id)
  rest = apply_dataset_motion_file(
    rest,
    robot=robot,
    dataset=dataset,
    dataset_path=dataset_path,
    cache_motion_bundle=cache_motion_bundle,
  )
  sys.argv = [prog, *ensure_task_id(rest, task_id)]

  if _needs_auto_checkpoint():
    positional = [a for a in sys.argv[1:] if not a.startswith("-")]
    if positional:
      task_id = positional[0]
      from mjlab.tasks.registry import load_rl_cfg

      try:
        rl_cfg = load_rl_cfg(task_id)
        experiment_name: str = rl_cfg.experiment_name
      except Exception:
        experiment_name = ""

      if experiment_name:
        checkpoint = _find_latest_checkpoint(experiment_name)
        if checkpoint is not None:
          print(f"[INFO] Auto-detected checkpoint: {checkpoint}")
          sys.argv.extend(["--checkpoint-file", str(checkpoint)])
        else:
          print(
            f"[WARN] No checkpoint found in logs/rsl_rl/{experiment_name}/. "
            "Pass --checkpoint-file explicitly or use --agent zero/random."
          )

  import mjlab.scripts.play as mjlab_play

  mjlab_play.run_play = run_play
  mjlab_play.main()


if __name__ == "__main__":
  main()
