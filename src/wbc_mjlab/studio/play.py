"""Play service: a checkpoint on N parallel envs over a motion library, in the Studio process.

The policy runs exactly as in ``wbc-mjlab-play`` (the task's play env config, uniform RSI) and
the Studio draws every env with its **reference ghost** (the clip frame the policy tracks),
keybody targets coloured by tracking error and a HUD with the focused env's reward terms and
tracking errors. The focused env can be given a clip (it loops it from the start); every other
env plays random clips. ``zero`` runs the zero action: for residual action modes that is PD
on the reference — the open-loop baseline the policy improves on.

Changes apply in place where they can: another checkpoint of the same run is loaded into the
running policy, and the env count changes instantly up to the session's capacity (envs are
built in powers of two, at least ``MIN_CAPACITY``). Another run or library, the terminations
switch or more envs than the capacity rebuild the session in the background.
"""

from __future__ import annotations

import threading
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from wbc_mjlab.studio import jobs as jobmod
from wbc_mjlab.studio.core import CONTROL_FPS
from wbc_mjlab.studio.plots import PlayRecorder
from wbc_mjlab.studio.scene import Overlay, error_colors, grid_offsets
from wbc_mjlab.studio.terrain import TerrainMap

OWNER = "play"
ZERO = "zero action (PD on reference)"
MAX_STEPS_PER_TICK = 4
MIN_CAPACITY = 16
GRID_SPACING = 2.0
TERRAIN_SPACING = 5.0  # envs on terrain: room for each env's terrain patch


def capacity_for(num_envs: int) -> int:
  return max(MIN_CAPACITY, 1 << max(0, int(num_envs) - 1).bit_length())


class Session:
  """One play env + policy (the body of ``wbc_mjlab.scripts.play.run_play``, headless)."""

  def __init__(
    self,
    task_id: str,
    checkpoint: Path | None,
    motion_file: str,
    *,
    num_envs: int,
    device: str,
    terminations: bool,
    preset: dict | None = None,
  ):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

    from wbc_mjlab.studio.presets import register, split_task_id
    from wbc_mjlab.tasks import prepare_wbc_run

    if split_task_id(task_id):  # a run's snapshot when it has one, else the preset file
      register(split_task_id(task_id)[1], snapshot=preset)
    prepare_wbc_run(task_id=task_id)
    env_cfg = load_env_cfg(task_id, play=True)
    agent_cfg = load_rl_cfg(task_id)
    motion = env_cfg.commands["motion"]
    motion.motion_file = motion_file
    motion.rsi.sampling_mode = "uniform"
    if not terminations:
      env_cfg.terminations = {}
    env_cfg.scene.num_envs = num_envs
    self.task = task_id
    self.env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    self.wrapped = RslRlVecEnvWrapper(self.env, clip_actions=agent_cfg.clip_actions)
    self.cmd = self.env.command_manager.get_term("motion")
    self.robot = self.env.scene["robot"]
    self.device = device
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    self.runner = runner_cls(self.wrapped, asdict(agent_cfg), device=device)
    self.policy = None
    self.load_policy(checkpoint)
    self.obs = self.wrapped.get_observations()

  def load_policy(self, checkpoint: Path | None) -> None:
    if checkpoint is None:
      shape = self.env.action_space.shape
      zeros = torch.zeros(shape, device=self.device)
      self.policy = lambda obs: zeros
      return
    self.runner.load(
      str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=self.device
    )
    self.policy = self.runner.get_inference_policy(device=self.device)

  @property
  def num_envs(self) -> int:
    return self.env.num_envs

  def clip_names(self) -> list[str]:
    return list(self.cmd.motion.segment_names)

  @torch.no_grad()  # not inference_mode: pin() / push() write env buffers outside of it
  def step(self) -> dict:
    actions = self.policy(self.obs)
    self.obs, reward, dones, _ = self.wrapped.step(actions)
    return {"reward": reward, "dones": dones}

  def pin(self, env_ids: torch.Tensor, clips: torch.Tensor) -> None:
    """Restart ``env_ids`` at frame 0 of ``clips`` (reference state written to the sim)."""
    cmd = self.cmd
    starts = cmd.motion.segment_start_idx[clips]
    cmd.trajectory_ids[env_ids] = clips
    cmd.time_steps[env_ids] = starts
    cmd._write_reference_state_to_sim(  # noqa: SLF001
      env_ids,
      cmd.body_pos_w[env_ids, 0],
      cmd.body_quat_w[env_ids, 0],
      cmd.body_lin_vel_w[env_ids, 0],
      cmd.body_ang_vel_w[env_ids, 0],
      cmd.joint_pos[env_ids],
      cmd.joint_vel[env_ids],
    )
    cmd._episode_start_step[env_ids] = starts  # noqa: SLF001
    self.env.episode_length_buf[env_ids] = 0
    cmd.update_relative_body_poses()
    self.env.scene.write_data_to_sim()
    self.env.sim.forward()
    self.obs = self.wrapped.get_observations()

  def close(self) -> None:
    try:
      self.env.close()
    except Exception:  # noqa: BLE001
      pass


class PlayService:
  def __init__(self, core):
    self.c = core
    self.session: Session | None = None
    self._tmap: TerrainMap | None = None  # terrain of the session (-Terrain-Scan tasks)
    self._patches: dict | None = None
    self.recorder: PlayRecorder | None = None  # the envs' records (Plots window)
    self.run_label = ""
    self.checkpoint = ""
    self.motion = ""  # motion library (dataset folder or bundle) of the session
    self.num_envs = 1
    self.terminations = False
    self.robot = ""
    self.playing = False
    self.speed = 1.0
    self.show_ghost = True
    self.show_targets = True
    self.ghost_mode = (
      "aligned"  # reference re-anchored to the robot (xy + yaw) | "world"
    )
    self.pinned: dict[int, int] = {}  # env → clip it loops; others draw random clips
    self._expect: dict[int, int] = {}  # pinned env → time step it should be at
    self._lock = threading.RLock()
    self._acc = 0.0
    self._returns = np.zeros(1)
    self._last: dict | None = None
    self._grid_cache: tuple | None = None
    core.on("focus", self._on_focus)

  # ------------------------------------------------------------------ choices
  def runs(self) -> list[dict]:
    return self.c.training.runs(robot=self.c.robot_id)

  def libraries(self, run: str = "") -> list[dict]:
    """Motion libraries to play on: the run's own (if still on disk) + converted datasets."""
    out = []
    r = jobmod.find_run(run) if run else None
    if r is not None and r.motion_source():
      out.append(
        {
          "value": r.motion_source(),
          "label": f"trained on · {Path(r.motion_source()).name}",
        }
      )
    for d in self.c.data.datasets():
      if d["npz"]:
        out.append(
          {"value": str(Path(d["path"])), "label": f"{d['name']} · {d['npz']} clips"}
        )
    return out

  def clips(self) -> list[str]:
    return self.session.clip_names() if self.session is not None else []

  # ------------------------------------------------------------------ state
  def state(self) -> dict:
    s = self.session
    focus = self.c.scene.focus
    clip = ""
    if s is not None and focus < s.num_envs:
      clip = s.clip_names()[int(s.cmd.trajectory_ids[focus])]
    return {
      "active": s is not None,
      "robot": self.robot,
      "run": self.run_label,
      "checkpoint": self.checkpoint,
      "motion": self.motion,
      "num_envs": self.num_envs,
      "capacity": s.num_envs if s else 0,
      "terminations": self.terminations,
      "playing": self.playing,
      "speed": self.speed,
      "show_ghost": self.show_ghost,
      "show_targets": self.show_targets,
      "ghost_mode": self.ghost_mode,
      "focus": focus,
      "focus_clip": clip,
      "focus_pinned": focus in self.pinned,
      "task": s.task if s else "",
      "clips_loaded": len(s.clip_names()) if s else 0,
      "zero": ZERO,
    }

  # ------------------------------------------------------------------ session
  def configure(
    self,
    run: str,
    checkpoint: str,
    *,
    motion: str = "",
    num_envs: int = 1,
    terminations: bool = False,
  ) -> str:
    """Apply a run / checkpoint / library / env count / terminations choice.

    Returns ``"swapped"`` (checkpoint loaded into the running policy), ``"resized"``,
    ``"unchanged"``, ``"rebuilding"`` (a new session loads in the background) or ``"busy"``."""
    num_envs = max(1, int(num_envs))
    s = self.session
    r = jobmod.find_run(run)
    if r is None:
      raise ValueError("pick a run")
    motion = motion or r.motion_source() or self._default_library()
    if (
      s is None
      or run != self.run_label
      or motion != self.motion
      or bool(terminations) != self.terminations
      or num_envs > s.num_envs
    ):
      ok = self._rebuild(run, checkpoint, motion, num_envs, bool(terminations))
      return "rebuilding" if ok else "busy"
    changed = []
    with self._lock:
      if (checkpoint or ZERO) != self.checkpoint:
        s.load_policy(None if checkpoint in ("", ZERO) else r.path / checkpoint)
        self.checkpoint = checkpoint or ZERO
        self._returns[:] = 0.0
        changed.append("swapped")
      if num_envs != self.num_envs:
        self.num_envs = num_envs
        self.pinned = {e: c for e, c in self.pinned.items() if e < num_envs}
        if self.c.scene.focus >= num_envs:
          self.c.set_focus(0)
        changed.append("resized")
      self._render()
    if "swapped" in changed:
      self.c.log(f"play: switched to {self.checkpoint} (same envs, no reload)", "ok")
    self.c.emit("play", **self.state())
    return changed[0] if changed else "unchanged"

  def _default_library(self) -> str:
    libs = self.libraries()
    if not libs:
      raise RuntimeError(
        "no converted dataset for this robot: convert one in Data first"
      )
    pref = next((x for x in libs if Path(x["value"]).name == "samples"), libs[0])
    return pref["value"]

  def _rebuild(
    self, run: str, checkpoint: str, motion: str, num_envs: int, terminations: bool
  ) -> bool:
    def job():
      r = jobmod.find_run(run)
      task = r.task_id()
      if not task:
        raise RuntimeError(f"{run}: unknown task for experiment {r.experiment!r}")
      robot = r.robot_id()
      keep = (
        self.session is not None and run == self.run_label and motion == self.motion
      )
      pinned_names = {}
      if keep:
        names = self.session.clip_names()
        pinned_names = {e: names[c] for e, c in self.pinned.items() if e < num_envs}
      focus = min(self.c.scene.focus, num_envs - 1) if keep else 0
      with self._lock:
        self._close()
      if robot and robot != self.c.robot_id:
        self.c.select_robot(robot)
      ckpt = None if checkpoint in ("", ZERO) else r.path / checkpoint
      capacity = capacity_for(num_envs)
      self.c.progress(f"building {capacity} envs · {task}", 0, 1)
      session = Session(
        task,
        ckpt,
        motion,
        num_envs=capacity,
        device=self.c.device,
        terminations=terminations,
        preset=r.preset_snapshot(),
      )
      try:  # procedural terrain (-Terrain-Scan tasks): drawn under the robots
        self._tmap = TerrainMap.from_model(session.env.sim.mj_model)
      except Exception as exc:  # noqa: BLE001
        self._tmap = None
        self.c.log(f"terrain view unavailable: {exc}", "warn")
      self._patches = None
      recorder = PlayRecorder(session, self.c.model, fps=CONTROL_FPS)
      with self._lock:
        self.session = session
        self.recorder = recorder
        self.robot, self.run_label, self.checkpoint = robot, run, checkpoint or ZERO
        self.motion, self.num_envs, self.terminations = motion, num_envs, terminations
        names = session.clip_names()
        self.pinned = {e: names.index(n) for e, n in pinned_names.items() if n in names}
        self._grid_cache = None
        self.c.take_scene(OWNER)
        self.c.set_focus(focus)
        self.restart()
        self.playing = True
      self.c.log(
        f"play {task}: {ckpt.name if ckpt else 'zero action'}, {num_envs} env(s) of "
        f"{capacity}, {len(names)} clip(s)",
        "ok",
      )
      self.c.emit("play", **self.state())

    return self.c.run_task("load policy", job)

  def _close(self) -> None:
    self.__dict__.pop("_reward_defaults", None)
    self._tmap, self._patches = None, None
    self.recorder = None
    self.c.scene.set_terrain(None)
    if self.session is not None:
      self.session.close()
      self.session = None
      if torch.cuda.is_available():
        torch.cuda.empty_cache()
    self.playing = False
    self._last = None

  def stop(self) -> None:
    with self._lock:
      self._close()
    self.c.scene.remove_overlays(prefix=f"{OWNER}/")
    self.c.scene.set_hud({})
    self.c.robots.show_pose()
    self.c.emit("play", **self.state())

  def restart(self) -> None:
    with self._lock:
      s = self.session
      if s is None:
        return
      self._returns = np.zeros(s.num_envs)
      s.env.reset()
      s.obs = s.wrapped.get_observations()
      self._repin(list(self.pinned))
      self._last = None
      if self.recorder is not None:
        self.recorder.clear()
      self.c.take_scene(OWNER)
      self._render()
    self.c.emit("play", **self.state())

  def _repin(self, envs: list[int]) -> None:
    s = self.session
    if not envs or s is None:
      return
    ids = torch.as_tensor(envs, device=s.device, dtype=torch.long)
    clips = torch.as_tensor(
      [self.pinned[e] for e in envs], device=s.device, dtype=torch.long
    )
    s.pin(ids, clips)
    starts = s.cmd.motion.segment_start_idx[clips].cpu().tolist()
    for e, st in zip(envs, starts, strict=True):
      self._expect[e] = int(st)
      self._returns[e] = 0.0

  def set_clip(self, clip: str, env: int | None = None) -> None:
    """Pin ``clip`` to ``env`` (default: the focused env) and restart it there; the env loops
    that clip. ``clip=""`` unpins it (it draws random clips again)."""
    with self._lock:
      s = self.session
      if s is None:
        raise RuntimeError("start a session first")
      env = self.c.scene.focus if env is None else int(env)
      if not 0 <= env < self.num_envs:
        raise ValueError(f"env {env} is not shown ({self.num_envs} env(s))")
      if not clip:
        self.pinned.pop(env, None)
        self._expect.pop(env, None)
      else:
        names = s.clip_names()
        if clip not in names:
          raise ValueError(f"unknown clip {clip!r}")
        self.pinned[env] = names.index(clip)
        self._repin([env])
      self.c.take_scene(OWNER)
      self._render()
    self.c.emit("play", **self.state())

  def set_playing(self, playing: bool) -> None:
    self.playing = bool(playing) and self.session is not None
    if self.playing:
      self.c.take_scene(OWNER)
    self.c.emit("play", **self.state())

  def set_options(self, **opts) -> None:
    if opts.get("ghost_mode") in ("aligned", "world"):
      self.ghost_mode = opts["ghost_mode"]
    for k in ("speed", "show_ghost", "show_targets"):
      if opts.get(k) is not None:
        setattr(self, k, type(getattr(self, k))(opts[k]))
    if opts.get("focus") is not None:
      self.c.set_focus(min(max(0, int(opts["focus"])), self.num_envs - 1))
    with self._lock:
      self._render()
    self.c.emit("play", **self.state())

  def step_once(self) -> None:
    with self._lock:
      if self.session is None:
        return
      self.c.take_scene(OWNER)
      self._advance(1)
      self._render()

  def apply_rewards(self, rewards: dict) -> int:
    """Preset reward overrides (``{name: {weight, enabled, params}}``) into the running env:
    the live reward bars and returns change at once; training is not affected."""
    from wbc_mjlab.studio.presets import _set_param

    with self._lock:
      s = self.session
      if s is None:
        raise RuntimeError("no Play session")
      rm = s.env.reward_manager
      if not hasattr(self, "_reward_defaults"):
        self._reward_defaults = {}
      changed = 0
      for name in rm.active_terms:
        term = rm.get_term_cfg(name)
        base = self._reward_defaults.setdefault(
          name, (term.weight, dict(term.params or {}))
        )
        o = rewards.get(name, {})
        weight = 0.0 if o.get("enabled") is False else float(o.get("weight", base[0]))
        params = dict(base[1])
        for k, v in (o.get("params") or {}).items():
          _set_param(params, k, v)
        changed += int(weight != term.weight or params != term.params)
        term.weight = weight
        term.params = params
    self.c.log(f"play: {changed} reward term(s) updated live", "ok")
    return changed

  def push(self, env: int | None = None, speed: float = 1.0) -> None:
    """A random horizontal velocity kick on the root of ``env`` (default: focused)."""
    with self._lock:
      s = self.session
      if s is None:
        return
      e = self.c.scene.focus if env is None else int(env)
      vel = s.robot.data.root_link_vel_w[e : e + 1].clone()
      ang = np.random.uniform(0, 2 * np.pi)
      vel[0, 0] += speed * float(np.cos(ang))
      vel[0, 1] += speed * float(np.sin(ang))
      s.robot.write_root_link_velocity_to_sim(
        vel, env_ids=torch.tensor([e], device=s.device)
      )
    self.c.log(f"pushed env {e} at {speed:.1f} m/s")

  def _on_focus(self, focus: int) -> None:
    """Another env focused: its HUD (and the Plots window's record) now, even while paused."""
    if self.session is None or self.c.owner != OWNER:
      return
    if self.recorder is not None:
      self.recorder.follow(focus)
    with self._lock:
      self._render()

  def plot_data(
    self, since: int = 0, epoch: int | None = None, env: int | None = None
  ) -> dict:
    """``env``'s record (default: the focused env) since row ``since`` (Plots window).
    The recorder has its own lock: a fetch never waits for a whole tick of steps."""
    rec = self.recorder
    if rec is None:
      raise RuntimeError("start a Play session first")
    return rec.fetch(self.c.scene.focus if env is None else env, since, epoch)

  # ------------------------------------------------------------------ stepping
  def _advance(self, steps: int) -> None:
    s = self.session
    for _ in range(steps):
      out = s.step()
      self._last = out
      self._returns += out["reward"].cpu().numpy()
      done = out["dones"].cpu().numpy().astype(bool)
      self._reset_blown_up(done)
      self._returns[done] = 0.0
      if self.pinned:
        ts = s.cmd.time_steps.cpu().numpy()
        redo = []
        for e in self.pinned:
          self._expect[e] = self._expect.get(e, int(ts[e])) + 1
          if done[e] or int(ts[e]) != self._expect[e]:
            redo.append(e)
        self._repin(redo)
      if self.recorder is not None:
        self.recorder.follow(self.c.scene.focus)
        self.recorder.record()

  def _reset_blown_up(self, done: np.ndarray) -> None:
    """Envs whose state went non-finite (a pile-up with terminations off) restart; ``done`` is
    marked for them so pinned envs re-pin."""
    s = self.session
    d = s.robot.data
    bad = ~torch.isfinite(d.root_link_pos_w).all(-1) | ~torch.isfinite(d.joint_pos).all(
      -1
    )
    if not bool(bad.any()):
      return
    ids = torch.nonzero(bad).flatten()
    s.env.reset(env_ids=ids)
    s.obs = s.wrapped.get_observations()
    done[ids.cpu().numpy()] = True
    self.c.log(
      f"play: env(s) {ids.cpu().tolist()} blew up (non-finite state) → reset", "warn"
    )

  def tick(self, dt: float) -> None:
    if self.session is None or not self.playing or self.c.busy or self.c.owner != OWNER:
      return
    self._acc += dt * self.speed
    steps = min(round(self._acc * CONTROL_FPS), MAX_STEPS_PER_TICK)
    if steps <= 0:
      return
    self._acc = 0.0 if steps == MAX_STEPS_PER_TICK else self._acc - steps / CONTROL_FPS
    with self._lock:
      if self.session is None:
        return
      self._advance(steps)
      self._render()

  # ------------------------------------------------------------------ rendering
  def _grid(self, n: int) -> torch.Tensor:
    spacing = TERRAIN_SPACING if self._tmap is not None else GRID_SPACING
    key = (n, self.session.device, spacing)
    if self._grid_cache is None or self._grid_cache[0] != key:
      grid = torch.as_tensor(grid_offsets(n, spacing), device=self.session.device)
      self._grid_cache = (key, grid)
    return self._grid_cache[1]

  def _render(self) -> None:
    s = self.session
    if s is None or self.c.owner != OWNER:
      return
    n = min(self.num_envs, s.num_envs)
    d, cmd = s.robot.data, s.cmd
    shift = self._grid(n) - s.env.scene.env_origins[:n]
    q = self.c.qpos_from_named(
      (d.root_link_pos_w[:n] + shift).cpu().numpy(),
      d.root_link_quat_w[:n].cpu().numpy(),
      s.robot.joint_names,
      d.joint_pos[:n].cpu().numpy(),
    )
    ghost = None
    if self.show_ghost:
      # aligned: the reference in the frame the body rewards use (anchored to the robot's
      # xy + yaw); world: where the clip is (non-SE policies drift away from it over time).
      aligned = self.ghost_mode == "aligned"
      root_pos = cmd.body_pos_relative_w[:n, 0] if aligned else cmd.body_pos_w[:n, 0]
      root_quat = cmd.body_quat_relative_w[:n, 0] if aligned else cmd.body_quat_w[:n, 0]
      ghost = self.c.qpos_from_named(
        (root_pos + shift).cpu().numpy(),
        root_quat.cpu().numpy(),
        s.robot.joint_names,
        cmd.joint_pos[:n].cpu().numpy(),
      )
    self._update_terrain(n, shift, d.root_link_pos_w[:n])
    names = s.clip_names()
    labels = (
      [names[int(i)] for i in cmd.trajectory_ids[:n].cpu().tolist()] if n > 1 else []
    )
    self.c.show_qpos(q, OWNER, ghost=ghost, labels=labels)
    focus = min(self.c.scene.focus, n - 1)
    if self.show_targets:
      target = (cmd.body_pos_relative_w[focus] + shift[focus]).cpu().numpy()
      err = torch.norm(
        cmd.body_pos_relative_w[focus] - cmd.robot_body_pos_w[focus], dim=-1
      )
      self.c.overlay(
        OWNER,
        "targets",
        Overlay(
          "points", target, error_colors(err.cpu().numpy(), 0.25), 0.05, on_top=True
        ),
      )
    else:
      self.c.overlay(OWNER, "targets", None)
    self.c.scene.set_hud(self._hud(focus))

  def _update_terrain(self, n: int, shift, root) -> None:
    """Terrain patches around the shown robots, rebuilt when one walks off its patch, the
    env count changes or the scene dropped them (another step took it)."""
    tmap = getattr(self, "_tmap", None)
    if tmap is None:
      return
    root = root[:, :2].cpu().numpy()
    shift = shift.cpu().numpy() if hasattr(shift, "cpu") else np.asarray(shift)
    prev = self._patches
    fresh = (
      prev is None
      or self.c.scene.terrain is None
      or len(prev["centers"]) != n
      or not np.allclose(prev["shift"], shift)
      or np.abs(root - prev["centers"]).max() > 1.2
    )
    if not fresh:
      return
    half = TERRAIN_SPACING / 2 - 0.2  # patches of neighbouring envs don't overlap
    res = 0.1 if n <= 16 else 0.2
    centers = np.round(root / 0.5) * 0.5
    self.c.scene.set_terrain(tmap.mesh(centers, shift, half, res))
    self._patches = {"centers": centers, "shift": shift.copy()}

  def _hud(self, focus: int) -> dict:
    s = self.session
    cmd = s.cmd
    tid = int(cmd.trajectory_ids[focus])
    start = int(cmd.motion.segment_start_idx[tid])
    length = int(cmd.motion.segment_length[tid])
    frame = int(cmd.time_steps[focus]) - start
    hud = {
      "owner": OWNER,
      "frame": frame,
      "frames": length,
      "time": frame / CONTROL_FPS,
      "duration": length / CONTROL_FPS,
      "playing": self.playing,
      "title": s.clip_names()[tid],
      "focus": focus,
      "num_envs": self.num_envs,
      "pinned": focus in self.pinned,
      "return": float(self._returns[focus]),
      "mean_return": float(self._returns[: self.num_envs].mean()),
      "episode_step": int(s.env.episode_length_buf[focus]),
    }
    metrics = {}
    for key in (
      "error_anchor_pos",
      "error_anchor_rot",
      "error_body_pos",
      "error_joint_pos",
    ):
      if key in cmd.metrics:
        metrics[key] = float(cmd.metrics[key][focus])
    hud["metrics"] = metrics
    if self._last is not None:
      rm = s.env.reward_manager
      vals = rm._step_reward[focus].cpu().tolist()  # noqa: SLF001 — one sync
      hud["terms"] = dict(zip(rm.active_terms, vals, strict=True))
    return hud


__all__ = ["ZERO", "PlayService", "Session"]
